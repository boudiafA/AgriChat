"""
chatbot_AgriChat_lora.py
================================
Gradio Chatbot UI for a LoRA Fine-Tuned LLaVA-OneVision (Qwen2 + SigLIP) Model
--------------------------------------------------------------------------------

OVERVIEW
--------
Launches an interactive web-based chatbot that loads the LoRA fine-tuned
LLaVA-OneVision model and allows multi-turn conversations with optional
image uploads. Designed for agricultural / plant disease diagnostic use cases.

The model is loaded once at startup in 4-bit quantization (NF4) to minimise
GPU memory usage while preserving response quality.

USAGE
-----
  python chatbot_AgriChat_lora.py

  Then open the URL printed in the terminal (default: http://localhost:7860).

CONFIGURATION (edit constants below)
--------------------------------------
  BASE_MODEL_ID   : HuggingFace model ID for the base model.
  AGRICHAT_WEIGHTS_PATH
                  : Path to the released AgriChat LoRA weights directory.
  SERVER_PORT     : Port to serve the Gradio app on (default: 7860).
  SHARE           : Set to True to generate a public Gradio share link.

GENERATION PARAMETERS (adjustable in the UI)
---------------------------------------------
  Temperature     : Controls randomness (higher → more creative).
  Top P           : Nucleus sampling threshold.
  Top K           : Limits the token pool to the top-K candidates.
  Repetition Penalty : Penalises repeated tokens (higher → less repetition).

REQUIREMENTS
------------
  Python       >= 3.10
  PyTorch      >= 2.1
  transformers >= 4.45
  peft         >= 0.12
  bitsandbytes >= 0.43
  gradio       >= 4.0
  Pillow
"""

import torch
import gradio as gr
from PIL import Image
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

BASE_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
AGRICHAT_WEIGHTS_PATH = "./weights/AgriChat"
MAX_NEW_TOKENS = 512
SERVER_PORT    = 7860
SHARE          = False


# ============================================================
# MODEL LOADING
# ============================================================

def _load_model_and_processor():
    """
    Load the processor, base model (4-bit quantized), and LoRA adapter.

    4-bit NF4 quantization is used to reduce GPU memory requirements
    without significantly degrading output quality.

    Returns:
        (model, processor)
    """
    print("=" * 60)
    print("  Loading LLaVA-OneVision + LoRA Adapter")
    print("=" * 60)

    # ---- Processor ----
    print(f"\n[1/3] Loading processor from: {BASE_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # ---- 4-bit quantization config ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---- Base model ----
    print(f"[2/3] Loading base model: {BASE_MODEL_ID}")
    base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # ---- LoRA adapter ----
    print(f"[3/3] Loading AgriChat weights from: {AGRICHAT_WEIGHTS_PATH}")
    model = PeftModel.from_pretrained(base_model, AGRICHAT_WEIGHTS_PATH)
    model.eval()

    print("\n✓ Model ready.\n")
    return model, processor


# Load once at startup — shared across all Gradio sessions
model, processor = _load_model_and_processor()


# ============================================================
# CHAT LOGIC
# ============================================================

def _parse_history(history: list) -> list:
    """
    Normalise Gradio chat history into a list of {role, content} dicts.

    Gradio may pass history as either:
      - A list of [user_msg, bot_msg] pairs (older Gradio format), or
      - A list of {role, content} dicts (newer Gradio format).
    """
    conversation = []
    for turn in history:
        if isinstance(turn, (list, tuple)):
            user_msg, bot_msg = turn
            if user_msg:
                conversation.append({
                    "role": "user",
                    "content": [{"type": "text", "text": str(user_msg)}],
                })
            if bot_msg:
                conversation.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(bot_msg)}],
                })
        elif isinstance(turn, dict):
            role    = turn.get("role")
            content = turn.get("content")
            if role and content:
                conversation.append({
                    "role": role,
                    "content": _normalize_content_blocks(content),
                })
    return conversation


def _normalize_content_blocks(content) -> list:
    """Convert Gradio history content into chat-template blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]

    if isinstance(content, dict):
        blocks = []
        text = content.get("text", "")
        files = content.get("files", [])
        for file_path in files:
            try:
                Image.open(file_path).close()
                blocks.append({"type": "image"})
            except Exception:
                continue
        if text:
            blocks.append({"type": "text", "text": str(text)})
        return blocks or [{"type": "text", "text": str(content)}]

    if isinstance(content, list):
        blocks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "image"}:
                if item["type"] == "text":
                    blocks.append({"type": "text", "text": str(item.get("text", ""))})
                else:
                    blocks.append({"type": "image"})
            elif isinstance(item, str):
                blocks.append({"type": "text", "text": item})
        return blocks or [{"type": "text", "text": str(content)}]

    return [{"type": "text", "text": str(content)}]


def bot(
    message: dict,
    history: list,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> str:
    """
    Core chat handler called by Gradio on every user submission.

    Args:
        message           : Gradio multimodal dict with keys "text" and "files".
        history           : Previous turns in the conversation.
        temperature       : Sampling temperature.
        top_p             : Nucleus sampling probability.
        top_k             : Top-K token filtering.
        repetition_penalty: Penalty for repeated tokens.

    Returns:
        The model's response as a plain string.
    """
    text_input = message.get("text", "").strip()
    files      = message.get("files", [])

    if not text_input and not files:
        return "Please enter a message or upload an image."

    # ---- Build conversation from history ----
    conversation = _parse_history(history)

    # ---- Build current user turn ----
    current_content: list = []
    input_images:    list = []

    for file_path in files:
        image = Image.open(file_path).convert("RGB")
        input_images.append(image)
        current_content.append({"type": "image"})

    if text_input:
        current_content.append({"type": "text", "text": text_input})

    conversation.append({"role": "user", "content": current_content})

    # ---- Tokenize ----
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(
        text=[prompt],
        images=input_images if input_images else None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    # ---- Generate ----
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=int(top_k),
            repetition_penalty=repetition_penalty,
            pad_token_id=processor.tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    response  = processor.tokenizer.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()

    return response


# ============================================================
# GRADIO UI
# ============================================================

def _build_ui() -> gr.ChatInterface:
    """Construct and return the Gradio ChatInterface."""

    generation_controls = [
        gr.Slider(minimum=0.1, maximum=2.0,  value=0.7,  step=0.05,  label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0,  value=0.9,  step=0.05,  label="Top P"),
        gr.Slider(minimum=1,   maximum=100,  value=40,   step=1,     label="Top K"),
        gr.Slider(minimum=1.0, maximum=2.0,  value=1.2,  step=0.05,  label="Repetition Penalty"),
    ]

    return gr.ChatInterface(
        fn=bot,
        title="🌿 Plant Disease Diagnostic Assistant",
        description=(
            "Upload an image of a plant and ask a question to receive an AI-powered diagnosis. "
            "Multi-turn conversations are supported — you can follow up with additional questions."
        ),
        multimodal=True,
        textbox=gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Upload a plant image and ask a question, e.g. 'What disease is affecting this crop?'",
        ),
        additional_inputs=generation_controls,
        additional_inputs_accordion=gr.Accordion(label="⚙️ Generation Parameters", open=False),
        examples=[
            [{"text": "What disease is affecting this plant?",    "files": []}],
            [{"text": "Is this plant healthy or diseased?",        "files": []}],
            [{"text": "What treatment would you recommend?",       "files": []}],
        ],
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    demo = _build_ui()
    demo.launch(
        server_port=SERVER_PORT,
        share=SHARE,
    )
