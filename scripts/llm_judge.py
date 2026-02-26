"""
llm_judge.py
============
An LLM-as-a-Judge evaluation script that uses Qwen3-30B (via vLLM) to score
model outputs against ground-truth reference answers.

Overview
--------
Given a model's predictions (JSONL) and a reference/ground-truth file (JSONL),
this script:
  1. Matches samples between the two files using a shared image path key.
  2. Constructs an evaluation prompt per sample (MCQ or open-ended).
  3. Runs the judge model in batches using vLLM for fast inference.
  4. Saves scored outputs to a results JSONL file, with checkpoint support
     for resuming interrupted runs.

Evaluation Modes
----------------
- MCQ mode    : Detected automatically when the reference filename contains
                "MCQs". Scores are binary (0 = wrong, 1 = correct).
- Open-ended  : All other files. Scores are 1–4 (poor → excellent).

Usage
-----
Run from the command line:

    python llm_judge.py \\
        --model_output  path/to/model_predictions.jsonl \\
        --reference     path/to/ground_truth.jsonl \\
        [--output       path/to/results.jsonl] \\
        [--chunk_size   500] \\
        [--judge_model  Qwen/Qwen3-30B-A3B-Instruct-2507] \\
        [--tensor_parallel_size 1] \\
        [--max_model_len 32768]

Arguments
---------
--model_output          (required) Path to the model predictions JSONL file.
--reference             (required) Path to the ground-truth reference JSONL file.
--output                (optional) Path for the scored output JSONL.
                        Defaults to "judged_<model_output_basename>".
--chunk_size            (optional) Number of prompts per vLLM batch. Default: 500.
--judge_model           (optional) HuggingFace model ID for the judge LLM.
                        Default: "Qwen/Qwen3-30B-A3B-Instruct-2507".
--tensor_parallel_size  (optional) Number of GPUs for tensor parallelism. Default: 1.
--max_model_len         (optional) Max token context length. Default: 32768.

Expected Data Format
--------------------
Both --model_output and --reference must be JSONL files where each line is a
valid JSON object with the following structure:

    {
        "images": ["path/to/image.jpg"],   // used as the unique sample key
        "messages": [
            {"role": "user",      "content": "What disease is shown?"},
            {"role": "assistant", "content": "Leaf blight caused by ..."}
        ]
    }

  - "images"   : A list with at least one image path string. The first entry
                 is used as the unique identifier to match samples across files.
  - "messages" : A list of alternating user/assistant turns. Each "content"
                 field can be either a plain string or a list of content blocks
                 (e.g., [{"type": "text", "text": "..."}]).

Folder Structure Example
------------------------
    project/
    ├── llm_judge.py
    ├── model_predictions.jsonl      ← --model_output
    ├── ground_truth.jsonl           ← --reference
    └── judged_model_predictions.jsonl   ← output (auto-generated)

Output Format
-------------
The output JSONL mirrors the model_output file, with each assistant message
enriched with an "evaluation" field:

    {
        "images": [...],
        "image_path": "path/to/image.jpg",
        "messages": [
            {"role": "user", "content": "..."},
            {
                "role": "assistant",
                "content": "...",
                "evaluation": {
                    "score": 3,
                    "justification": "Correct answer but slightly verbose."
                }
            }
        ]
    }

Checkpointing
-------------
A checkpoint file ("checkpoint_<output_basename>") is written incrementally
after each batch. If the script is interrupted, re-running with the same
arguments will skip already-evaluated samples and resume from where it left off.
The checkpoint file can be safely deleted once evaluation is complete.

Dependencies
------------
    pip install transformers vllm
"""

import argparse
import gc
import json
import math
import os
import re
import sys

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Argument Parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-Judge: score model outputs against ground-truth using vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_output", required=True,
        help="Path to the model predictions JSONL file.",
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to the ground-truth reference JSONL file.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path for the scored output JSONL. Defaults to judged_<model_output_basename>.",
    )
    parser.add_argument(
        "--chunk_size", type=int, default=500,
        help="Number of prompts per vLLM inference batch.",
    )
    parser.add_argument(
        "--judge_model", default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        help="HuggingFace model ID for the judge LLM.",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--max_model_len", type=int, default=32768,
        help="Maximum token context length for the judge model.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_judge_model(model_id: str, tensor_parallel_size: int, max_model_len: int):
    """Load the judge tokenizer and vLLM engine."""
    print(f"Loading judge model: {model_id} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load tokenizer: {e}")

    try:
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            enforce_eager=False,
            max_model_len=max_model_len,
        )
    except Exception as e:
        sys.exit(f"[ERROR] Failed to load vLLM engine: {e}")

    sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)

    print("Judge model loaded successfully.")
    return tokenizer, llm, sampling_params


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_jsonl(filepath: str) -> tuple[dict, list]:
    """
    Load a JSONL file and index records by their image path key.

    Returns:
        data_by_key  : dict mapping image_path -> record dict
        ordered_keys : list of keys in file order
    """
    data_by_key: dict = {}
    ordered_keys: list = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = _get_image_key(record)
            if key:
                data_by_key[key] = record
                ordered_keys.append(key)

    return data_by_key, ordered_keys


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load previously computed results from a checkpoint file."""
    cache: dict = {}
    if not os.path.exists(checkpoint_path):
        return cache

    print(f"Checkpoint found at '{checkpoint_path}'. Resuming...")
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                if "_index_map" in record:
                    cache[tuple(record["_index_map"])] = record["result"]
            except json.JSONDecodeError:
                pass

    print(f"  Loaded {len(cache)} previously evaluated samples.")
    return cache


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def create_open_ended_prompt(question: str, ground_truth: str, model_output: str) -> str:
    return f"""You are an expert evaluator assessing an AI model's response. Evaluate systematically and objectively.

**QUESTION**: {question}

**GROUND TRUTH (Correct Answer)**:
{ground_truth}

**MODEL OUTPUT (To Evaluate)**:
{model_output}

---

**EVALUATION CRITERIA**
1. **Correctness**: Does the output contain the correct information from the Ground Truth?
2. **Completeness**: Does the output include all important information from Ground Truth?
3. **Clarity**: Is the output clear, well-organized, and easy to understand?
4. **Conciseness**: Is the output appropriately concise without unnecessary content?

---

**SCORING RUBRIC** (1–4 scale)
- **Score 1 (Poor)**: Major deficiencies. Factually incorrect or missing multiple key facts.
- **Score 2 (Fair)**: Significant issues. Missing 1–2 important facts or minor inaccuracies.
- **Score 3 (Good)**: Solid with minor issues only. Factually accurate but slightly verbose or misses tiny details.
- **Score 4 (Excellent)**: Outstanding quality. Perfectly accurate, complete, clear, and concise.

---

**OUTPUT**: Provide evaluation in this EXACT JSON format:
{{
  "score": <integer 1-4>,
  "justification": "1-2 sentence summary"
}}

Begin evaluation:"""


def create_mcq_prompt(question: str, ground_truth: str, model_output: str) -> str:
    return f"""You are an expert evaluator assessing an AI model's answer to a multiple-choice question (MCQ).

**QUESTION**: {question}

**CORRECT ANSWER**:
{ground_truth}

**MODEL OUTPUT (To Evaluate)**:
{model_output}

---

**TASK**
Determine whether the model's output selects or clearly indicates the correct answer.
- Award **Score 1** if the model's final answer matches the correct answer (even if reasoning contains minor errors or extra text).
- Award **Score 0** if the model's final answer does not match, is missing, or is ambiguous.

Focus only on whether the final selected answer is correct. Treat "A", "Option A", and "A." as equivalent.

---

**OUTPUT**: Provide evaluation in this EXACT JSON format:
{{
  "score": <integer 0 or 1>,
  "justification": "1-2 sentence summary"
}}

Begin evaluation:"""


# ---------------------------------------------------------------------------
# Output Parsing
# ---------------------------------------------------------------------------

def parse_judge_output(output_text: str) -> dict:
    """Extract the JSON evaluation result from the judge model's raw output."""
    try:
        text = output_text.replace("```json", "").replace("```", "")
        matches = re.findall(r'\{.*?"score":\s*\d+.*?\}', text, re.DOTALL)
        if matches:
            return json.loads(matches[-1])
    except Exception:
        pass
    return {"score": 0, "justification": "Error: could not parse judge output."}


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _get_image_key(record: dict) -> str | None:
    """Extract the normalized image path as the unique sample identifier."""
    images = record.get("images", [])
    if isinstance(images, list) and images:
        return images[0].replace("\\", "/").strip()
    return None


def _extract_text(content) -> str:
    """Safely extract plain text from a message content field."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(item["text"])
            for item in content
            if isinstance(item, dict) and item.get("type") == "text" and item.get("text")
        ]
        return " ".join(parts)
    return ""


def _is_mcq_file(filepath: str) -> bool:
    """Infer MCQ evaluation mode from the reference filename."""
    return "MCQs" in os.path.basename(filepath)


# ---------------------------------------------------------------------------
# Core Evaluation Pipeline
# ---------------------------------------------------------------------------

def build_tasks(
    ref_data: dict,
    cand_data: dict,
    matched_keys: set,
    tokenizer,
    mcq_mode: bool,
    results_cache: dict,
) -> tuple[list, list, list]:
    """
    Build the list of evaluation prompts and metadata for all un-cached samples.

    Returns:
        prompts       : list of formatted prompt strings for vLLM
        task_keys     : list of (image_key, message_idx) cache keys
        task_metadata : list of dicts with 'image_path' and 'question'
    """
    prompts, task_keys, task_metadata = [], [], []

    for img_key in matched_keys:
        ref_messages = ref_data[img_key].get("messages", [])
        cand_messages = cand_data[img_key].get("messages", [])
        current_question = ""

        for idx, (ref_msg, cand_msg) in enumerate(zip(ref_messages, cand_messages)):
            role = cand_msg.get("role")
            cand_text = _extract_text(cand_msg.get("content"))

            if role == "user":
                current_question = cand_text
            elif role == "assistant":
                cache_key = (img_key, idx)
                if cache_key in results_cache:
                    continue

                ref_text = _extract_text(ref_msg.get("content"))
                raw_prompt = (
                    create_mcq_prompt(current_question, ref_text, cand_text)
                    if mcq_mode
                    else create_open_ended_prompt(current_question, ref_text, cand_text)
                )

                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": raw_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prompts.append(formatted)
                task_keys.append(cache_key)
                task_metadata.append({"image_path": img_key, "question": current_question})

    return prompts, task_keys, task_metadata


def run_inference(
    prompts: list,
    task_keys: list,
    task_metadata: list,
    llm: LLM,
    sampling_params: SamplingParams,
    chunk_size: int,
    checkpoint_path: str,
    results_cache: dict,
) -> None:
    """Run vLLM inference in chunks, updating the cache and checkpoint file."""
    total = len(prompts)
    num_chunks = math.ceil(total / chunk_size)
    print(f"Running inference on {total} samples in {num_chunks} batch(es)...")

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = start + chunk_size

        chunk_prompts = prompts[start:end]
        chunk_keys = task_keys[start:end]
        chunk_meta = task_metadata[start:end]

        print(f"  Batch {chunk_idx + 1}/{num_chunks} ({len(chunk_prompts)} prompts)...")
        outputs = llm.generate(chunk_prompts, sampling_params)

        checkpoint_lines = []
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            result = parse_judge_output(generated_text)
            key = chunk_keys[j]
            meta = chunk_meta[j]

            results_cache[key] = result
            checkpoint_lines.append(json.dumps({
                "id": meta["image_path"],
                "_index_map": list(key),
                "question": meta["question"],
                "result": result,
            }))

        with open(checkpoint_path, "a", encoding="utf-8") as f:
            f.write("\n".join(checkpoint_lines) + "\n")

        del outputs
        gc.collect()


def save_results(
    output_path: str,
    cand_data: dict,
    ordered_keys: list,
    results_cache: dict,
) -> None:
    """Write the final annotated JSONL output file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for img_key in ordered_keys:
            record = cand_data[img_key]
            for idx, msg in enumerate(record.get("messages", [])):
                cache_key = (img_key, idx)
                if cache_key in results_cache:
                    msg["evaluation"] = results_cache[cache_key]
            if "image_path" not in record:
                record["image_path"] = img_key
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve output and checkpoint paths
    base_name = os.path.basename(args.model_output)
    output_path = args.output or f"judged_{base_name}"
    checkpoint_path = f"checkpoint_{os.path.basename(output_path)}"

    print(f"\n{'='*60}")
    print(f"  Model Output : {args.model_output}")
    print(f"  Reference    : {args.reference}")
    print(f"  Output       : {output_path}")
    print(f"  Judge Model  : {args.judge_model}")
    print(f"{'='*60}\n")

    # Validate inputs
    for path, label in [(args.model_output, "model_output"), (args.reference, "reference")]:
        if not os.path.exists(path):
            sys.exit(f"[ERROR] {label} file not found: {path}")

    # Determine evaluation mode
    mcq_mode = _is_mcq_file(args.reference)
    mode_label = "MCQ (0/1)" if mcq_mode else "Open-ended (1–4)"
    print(f"Evaluation mode: {mode_label}\n")

    # Load data
    print(f"Loading reference file: {args.reference}")
    ref_data, _ = load_jsonl(args.reference)
    print(f"  Loaded {len(ref_data)} reference samples.")

    print(f"Loading model output file: {args.model_output}")
    cand_data, ordered_keys = load_jsonl(args.model_output)
    print(f"  Loaded {len(cand_data)} candidate samples.")

    # Match samples
    matched_keys = set(ref_data.keys()) & set(cand_data.keys())
    print(f"\nMatched samples: {len(matched_keys)}")
    if not matched_keys:
        sys.exit("[ERROR] No matching samples found between the two files. "
                 "Ensure both files share the same image paths in the 'images' field.")

    # Load checkpoint
    results_cache = load_checkpoint(checkpoint_path)

    # Load judge model
    tokenizer, llm, sampling_params = load_judge_model(
        args.judge_model, args.tensor_parallel_size, args.max_model_len
    )

    # Build prompts
    print("\nBuilding evaluation prompts...")
    prompts, task_keys, task_metadata = build_tasks(
        ref_data, cand_data, matched_keys, tokenizer, mcq_mode, results_cache
    )
    print(f"  {len(prompts)} new prompts to evaluate "
          f"({len(results_cache)} already cached).")

    # Run inference
    if prompts:
        run_inference(
            prompts, task_keys, task_metadata,
            llm, sampling_params, args.chunk_size,
            checkpoint_path, results_cache,
        )

    # Save results
    print(f"\nSaving results to: {output_path}")
    save_results(output_path, cand_data, ordered_keys, results_cache)

    print(f"\n[DONE] Evaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
