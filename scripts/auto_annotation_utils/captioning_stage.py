"""
Stage I: visual caption generation.

This module captions agricultural images with Gemma 3 and writes a resumable
JSONL artifact that can be consumed by the downstream stages.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .common import collect_images, infer_class_name, load_jsonl_index

CAPTION_PROMPT = (
    "Write a concise but information-dense agricultural image caption in 3 to 5 sentences.\n"
    "Describe only what is visually grounded in the image.\n"
    "If visible, include crop identity, growth stage, planting density, camera perspective, "
    "environment, and obvious health signals.\n"
    "If a detail cannot be inferred reliably, say unknown instead of speculating."
)


def load_captioning_components(model_id: str):
    """Load the Gemma 3 model and processor lazily."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
    return model, processor


def generate_caption(image_path: Path, model, processor, max_new_tokens: int) -> str:
    """Generate one grounded caption for *image_path*."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": CAPTION_PROMPT},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    return processor.decode(generated[0][prompt_length:], skip_special_tokens=True).strip()


def run_captioning_stage(
    image_root: Path,
    output_path: Path,
    model_id: str,
    max_new_tokens: int,
    resume: bool,
) -> dict[str, int | str]:
    """Run Stage I and persist captions to *output_path*."""
    images = collect_images(image_root=image_root, recursive=True)
    existing = load_jsonl_index(output_path, "image_path") if resume else {}
    pending_images = [image for image in images if str(image) not in existing]

    if not pending_images:
        return {
            "stage": "captions",
            "total_images": len(images),
            "processed": 0,
            "skipped_existing": len(images),
            "output_path": str(output_path),
        }

    print(f"[Stage I] Loading captioning model: {model_id}")
    model, processor = load_captioning_components(model_id=model_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as handle:
        for image_path in tqdm(pending_images, desc="Stage I - captioning", unit="image"):
            try:
                caption = generate_caption(
                    image_path=image_path,
                    model=model,
                    processor=processor,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[Stage I][WARN] Failed on {image_path.name}: {exc}")
                caption = ""

            record = {
                "image_path": str(image_path),
                "class_name": infer_class_name(image_root=image_root, image_path=image_path),
                "caption": caption,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    return {
        "stage": "captions",
        "total_images": len(images),
        "processed": len(pending_images),
        "skipped_existing": len(images) - len(pending_images),
        "output_path": str(output_path),
    }
