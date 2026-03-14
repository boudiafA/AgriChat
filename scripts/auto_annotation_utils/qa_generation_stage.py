"""
Stage III: instruction and QA synthesis.

This stage combines visual captions with verified class knowledge and asks a
text model to emit structured, diverse QA pairs for each image.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .common import collect_images, infer_class_name, load_jsonl_index

DEFAULT_QA_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

QA_SYSTEM_PROMPT = (
    "You are creating high-quality training data for an agricultural multimodal assistant. "
    "Generate grounded, professional question-answer pairs only from the provided inputs."
)

QA_USER_TEMPLATE = """
You will receive:
1. a visual caption generated from the image
2. verified class knowledge for the image category

Class name:
{class_name}

Verified knowledge:
{knowledge}

Visual caption:
{caption}

Instructions:
- Generate exactly {num_qa_pairs} QA pairs as a JSON array.
- Keep all questions grounded in the class knowledge and caption.
- Do not invent diseases, symptoms, locations, or cultivar names that are not supported.
- Use diverse question types such as identification, visual reasoning, health status, cultivation, and morphology.
- Answers must be complete sentences and professionally phrased.
- Return JSON only in the format:
  [
    {{"question": "...", "answer": "..."}}
  ]
""".strip()


def load_generation_pipeline(
    model_id: str,
    load_in_4bit: bool,
    load_in_8bit: bool,
):
    """Create a text-generation pipeline for Stage III."""
    import torch
    import transformers

    if load_in_4bit and load_in_8bit:
        raise ValueError("--load-in-4bit and --load-in-8bit are mutually exclusive.")

    model_kwargs: dict[str, Any]
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs = {
            "device_map": "auto",
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        }
    elif load_in_8bit:
        model_kwargs = {
            "device_map": "auto",
            "load_in_8bit": True,
        }
    else:
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }

    return transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs=model_kwargs,
    )


def parse_qa_pairs(raw_text: str) -> list[dict[str, str]] | None:
    """Parse a JSON array of QA pairs from model output."""
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, list) else None


def extract_generated_text(result: Any) -> str:
    """Handle both plain-string and chat-template pipeline outputs."""
    candidate = result[0] if isinstance(result, list) else result
    generated_text = candidate.get("generated_text", "")

    if isinstance(generated_text, list) and generated_text:
        last_message = generated_text[-1]
        if isinstance(last_message, dict):
            return str(last_message.get("content", "")).strip()
    return str(generated_text).strip()


def load_caption_index(captions_path: Path) -> dict[str, dict[str, Any]]:
    """Index caption records by image path."""
    return load_jsonl_index(captions_path, "image_path")


def load_knowledge_index(knowledge_path: Path) -> dict[str, dict[str, Any]]:
    """Index knowledge records by class name."""
    return load_jsonl_index(knowledge_path, "class_name")


def build_qa_records(
    image_root: Path,
    captions_path: Path,
    knowledge_path: Path,
) -> list[dict[str, str]]:
    """Join Stage I and Stage II outputs into per-image generation records."""
    captions = load_caption_index(captions_path)
    knowledge = load_knowledge_index(knowledge_path)
    records: list[dict[str, str]] = []

    for image_path in collect_images(image_root=image_root, recursive=True):
        image_key = str(image_path)
        caption_record = captions.get(image_key)
        if not caption_record or not caption_record.get("caption"):
            continue

        class_name = caption_record.get("class_name") or infer_class_name(image_root, image_path)
        if not class_name:
            continue

        knowledge_record = knowledge.get(class_name)
        if not knowledge_record or not knowledge_record.get("knowledge"):
            continue

        records.append(
            {
                "image_path": image_key,
                "class_name": class_name,
                "caption": caption_record["caption"],
                "knowledge": knowledge_record["knowledge"],
            }
        )

    return records


def generate_qa_batch(
    generator,
    batch: list[dict[str, str]],
    num_qa_pairs: int,
    max_new_tokens: int,
) -> list[list[dict[str, str]] | None]:
    """Generate QA pairs for one batch."""
    prompts = [
        [
            {"role": "system", "content": QA_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": QA_USER_TEMPLATE.format(
                    class_name=record["class_name"],
                    knowledge=record["knowledge"],
                    caption=record["caption"],
                    num_qa_pairs=num_qa_pairs,
                ),
            },
        ]
        for record in batch
    ]

    results = generator(
        prompts,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    return [parse_qa_pairs(extract_generated_text(result)) for result in results]


def run_qa_generation_stage(
    image_root: Path,
    captions_path: Path,
    knowledge_path: Path,
    output_path: Path,
    model_id: str,
    num_qa_pairs: int,
    max_new_tokens: int,
    batch_size: int,
    load_in_4bit: bool,
    load_in_8bit: bool,
    resume: bool,
) -> dict[str, int | str]:
    """Run Stage III and persist the synthesized QA dataset."""
    records = build_qa_records(
        image_root=image_root,
        captions_path=captions_path,
        knowledge_path=knowledge_path,
    )
    existing = load_jsonl_index(output_path, "image_path") if resume else {}
    pending_records = [record for record in records if record["image_path"] not in existing]

    if not pending_records:
        return {
            "stage": "qa",
            "total_images": len(records),
            "processed": 0,
            "skipped_existing": len(records),
            "output_path": str(output_path),
        }

    print(f"[Stage III] Loading QA synthesis model: {model_id}")
    generator = load_generation_pipeline(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    parse_failures = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as handle:
        for start_index in range(0, len(pending_records), batch_size):
            batch = pending_records[start_index:start_index + batch_size]
            try:
                batch_outputs = generate_qa_batch(
                    generator=generator,
                    batch=batch,
                    num_qa_pairs=num_qa_pairs,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[Stage III][WARN] Batch starting at {start_index} failed: {exc}")
                batch_outputs = [None] * len(batch)

            for record, qa_pairs in zip(batch, batch_outputs):
                if qa_pairs is None:
                    parse_failures += 1
                    qa_pairs = []

                output_record = {
                    "image_path": record["image_path"],
                    "class_name": record["class_name"],
                    "caption": record["caption"],
                    "knowledge": record["knowledge"],
                    "qa_pairs": qa_pairs,
                }
                handle.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                handle.flush()

            tqdm.write(
                f"[Stage III] Processed {min(start_index + len(batch), len(pending_records))}/"
                f"{len(pending_records)} pending images"
            )

    return {
        "stage": "qa",
        "total_images": len(records),
        "processed": len(pending_records),
        "skipped_existing": len(records) - len(pending_records),
        "parse_failures": parse_failures,
        "output_path": str(output_path),
    }
