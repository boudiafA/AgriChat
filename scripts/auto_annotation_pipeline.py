"""
auto_annotation_pipeline.py
===========================
End-to-end AgriMM auto-annotation pipeline for GitHub release.

Overview
--------
This script orchestrates the three-stage Vision-to-Verified-Knowledge pipeline
used to prepare instruction data for AgriChat:

1. Stage I   : image caption generation with Gemma 3
2. Stage II  : class-level knowledge retrieval with Gemini 3 Pro + Google Search grounding
3. Stage III : QA synthesis with Llama 3.1 Instruct

Design Notes
------------
- The implementation is intentionally split into reusable utility modules under
  `auto_annotation_utils/` so each stage can be maintained and tested in
  isolation.
- Every stage writes JSONL outputs incrementally and supports resume mode.
- The pipeline assumes a class-organised dataset by default:

      image_root/
      ├── class_a/
      │   ├── image_001.jpg
      │   └── ...
      └── class_b/
          └── ...

- If your dataset is not organised by class folders, you can still run Stage I.
  For Stage II, provide `--class-names-file` with one class name per line.

Environment Requirements
------------------------
- Stage I / Stage III require Hugging Face model access and GPU-backed PyTorch.
- Stage II requires the Google GenAI SDK and a valid `GEMINI_API_KEY`.

Example Usage
-------------
Run the full pipeline:

    python scripts/auto_annotation_pipeline.py \
        --image-root ./dataset/images \
        --output-dir ./auto_annotation_outputs \
        --resume

Run only Stage II and Stage III:

    python scripts/auto_annotation_pipeline.py \
        --image-root ./dataset/images \
        --output-dir ./auto_annotation_outputs \
        --stages knowledge qa \
        --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from auto_annotation_utils.captioning_stage import run_captioning_stage
from auto_annotation_utils.knowledge_stage import DEFAULT_KNOWLEDGE_MODEL, run_knowledge_stage
from auto_annotation_utils.qa_generation_stage import DEFAULT_QA_MODEL, run_qa_generation_stage

DEFAULT_CAPTION_MODEL = "google/gemma-3-12b-it"
DEFAULT_STAGES = ("captions", "knowledge", "qa")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the AgriMM auto-annotation pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help="Root directory containing dataset images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where stage JSONL outputs will be written.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEFAULT_STAGES,
        default=list(DEFAULT_STAGES),
        help="Subset of pipeline stages to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip records already present in existing stage outputs.",
    )
    parser.add_argument(
        "--class-names-file",
        type=Path,
        default=None,
        help="Optional text file with one class name per line for Stage II.",
    )

    stage1 = parser.add_argument_group("stage i - captioning")
    stage1.add_argument(
        "--caption-model",
        type=str,
        default=DEFAULT_CAPTION_MODEL,
        help="Gemma captioning model identifier.",
    )
    stage1.add_argument(
        "--caption-max-new-tokens",
        type=int,
        default=300,
        help="Maximum generated tokens for each caption.",
    )

    stage2 = parser.add_argument_group("stage ii - knowledge")
    stage2.add_argument(
        "--knowledge-model",
        type=str,
        default=DEFAULT_KNOWLEDGE_MODEL,
        help="Google GenAI model identifier for class knowledge generation.",
    )

    stage3 = parser.add_argument_group("stage iii - qa synthesis")
    stage3.add_argument(
        "--qa-model",
        type=str,
        default=DEFAULT_QA_MODEL,
        help="Text model used to synthesize QA pairs.",
    )
    stage3.add_argument(
        "--num-qa-pairs",
        type=int,
        default=5,
        help="Number of QA pairs to synthesize per image.",
    )
    stage3.add_argument(
        "--qa-max-new-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens for the QA stage.",
    )
    stage3.add_argument(
        "--qa-batch-size",
        type=int,
        default=1,
        help="Batch size for the QA synthesis model.",
    )
    stage3.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the QA model using 4-bit quantization.",
    )
    stage3.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the QA model using 8-bit quantization.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI argument combinations."""
    if not args.image_root.exists():
        raise FileNotFoundError(f"Image root does not exist: {args.image_root}")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit and --load-in-8bit cannot be used together.")
    if args.num_qa_pairs < 1:
        raise ValueError("--num-qa-pairs must be at least 1.")
    if args.qa_batch_size < 1:
        raise ValueError("--qa-batch-size must be at least 1.")


def main() -> None:
    """Execute the requested pipeline stages."""
    args = parse_args()
    validate_args(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    captions_path = args.output_dir / "stage1_captions.jsonl"
    knowledge_path = args.output_dir / "stage2_class_knowledge.jsonl"
    qa_path = args.output_dir / "stage3_qa_pairs.jsonl"

    summaries: list[dict[str, int | str]] = []

    if "captions" in args.stages:
        summaries.append(
            run_captioning_stage(
                image_root=args.image_root,
                output_path=captions_path,
                model_id=args.caption_model,
                max_new_tokens=args.caption_max_new_tokens,
                resume=args.resume,
            )
        )

    if "knowledge" in args.stages:
        summaries.append(
            run_knowledge_stage(
                image_root=args.image_root,
                class_names_file=args.class_names_file,
                output_path=knowledge_path,
                model_id=args.knowledge_model,
                resume=args.resume,
            )
        )

    if "qa" in args.stages:
        if not captions_path.exists():
            raise FileNotFoundError(
                f"Stage III requires captions at {captions_path}. Run Stage I first or provide the file."
            )
        if not knowledge_path.exists():
            raise FileNotFoundError(
                f"Stage III requires class knowledge at {knowledge_path}. Run Stage II first or provide the file."
            )

        summaries.append(
            run_qa_generation_stage(
                image_root=args.image_root,
                captions_path=captions_path,
                knowledge_path=knowledge_path,
                output_path=qa_path,
                model_id=args.qa_model,
                num_qa_pairs=args.num_qa_pairs,
                max_new_tokens=args.qa_max_new_tokens,
                batch_size=args.qa_batch_size,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
                resume=args.resume,
            )
        )

    print("\nPipeline summary")
    print("=" * 80)
    for summary in summaries:
        stage_name = summary.get("stage", "unknown")
        print(f"{stage_name}:")
        for key, value in summary.items():
            if key == "stage":
                continue
            print(f"  {key}: {value}")
    print("=" * 80)


if __name__ == "__main__":
    main()
