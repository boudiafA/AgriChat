"""
inference_AgriChat_lora.py
================================
Inference script for a LoRA fine-tuned LLaVA-OneVision (Qwen2 + SigLIP) model.
--------------------------------------------------------------------------------

OVERVIEW
--------
Loads the base LLaVA-OneVision model together with a trained LoRA adapter,
then runs single-image inference given an image path and a text prompt.

USAGE
-----
  Command line:
    python inference_AgriChat_lora.py \\
        --image  path/to/image.jpg \\
        --prompt "What disease is affecting this crop?" \\
        --adapter ./finetune_output/final_adapter

  In code:
    from inference_AgriChat_lora import load_model, run_inference

    model, processor = load_model(adapter_path="./finetune_output/final_adapter")
    response = run_inference(model, processor, image_path="img.jpg", prompt="Describe this image.")
    print(response)

ARGUMENTS (CLI)
---------------
  --image       Path to the input image (any format supported by Pillow).
  --prompt      Text prompt / question to ask about the image.
  --adapter     Path to the saved LoRA adapter directory (default: ./finetune_output/final_adapter).
  --base-model  HuggingFace model ID for the base model (default: llava-hf/llava-onevision-qwen2-7b-ov-hf).
  --max-tokens  Maximum number of new tokens to generate (default: 512).
  --device      Device to run on: "cuda", "cpu", or "auto" (default: auto).

REQUIREMENTS
------------
  Python       >= 3.10
  PyTorch      >= 2.1
  transformers >= 4.45
  peft         >= 0.12
  Pillow
"""

import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from peft import PeftModel


# ============================================================
# CONFIG DEFAULTS
# ============================================================

DEFAULT_BASE_MODEL  = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_ADAPTER_DIR = "./models/AgriChat_128_32"
DEFAULT_MAX_TOKENS  = 512


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(
    adapter_path: str = DEFAULT_ADAPTER_DIR,
    base_model_id: str = DEFAULT_BASE_MODEL,
    device: str = "auto",
) -> tuple:
    """
    Load the base model, merge the LoRA adapter, and return (model, processor).

    Args:
        adapter_path  : Path to the saved PEFT adapter directory.
        base_model_id : HuggingFace model ID for the base model.
        device        : "auto", "cuda", or "cpu".

    Returns:
        (model, processor) ready for inference.
    """
    print(f"Loading processor from: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)

    print(f"Loading base model from: {base_model_id}")
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    print("✓ Model ready.\n")
    return model, processor


# ============================================================
# INFERENCE
# ============================================================

def run_inference(
    model,
    processor,
    image_path: str,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """
    Run inference on a single image and text prompt.

    Args:
        model          : Loaded (and LoRA-merged) LLaVA-OneVision model.
        processor      : Corresponding AutoProcessor.
        image_path     : Path to the input image file.
        prompt         : Text question or instruction about the image.
        max_new_tokens : Maximum number of tokens to generate.

    Returns:
        The model's response as a plain string.
    """
    # ---- Load image ----
    image = Image.open(image_path).convert("RGB")

    # ---- Build conversation ----
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # ---- Apply chat template ----
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)

    # ---- Tokenize ----
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )

    # Move tensors to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # ---- Generate ----
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # ---- Decode (strip the input prompt tokens) ----
    input_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, input_len:]
    response = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return response.strip()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a LoRA fine-tuned LLaVA-OneVision model."
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt or question about the image.",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER_DIR,
        help=f"Path to the LoRA adapter directory (default: {DEFAULT_ADAPTER_DIR}).",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help=f"HuggingFace base model ID (default: {DEFAULT_BASE_MODEL}).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Maximum number of new tokens to generate (default: {DEFAULT_MAX_TOKENS}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    model, processor = load_model(
        adapter_path=args.adapter,
        base_model_id=args.base_model,
        device=args.device,
    )

    print(f"Image : {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 60)

    response = run_inference(
        model=model,
        processor=processor,
        image_path=args.image,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
    )

    print("Response:")
    print(response)


if __name__ == "__main__":
    main()
