"""
finetune_AgriChat_lora.py
=========================
LoRA fine-tuning script for AgriChat (LLaVA-OneVision + Qwen2 + SigLIP).

This clean script is derived from `abilitation_study_codes/run2_vision_llm_lora.py`.
It applies LoRA to both the Qwen2 language model and the SigLIP
vision encoder, using separate ranks/alphas for each branch. Training samples
are expected in JSONL format with:

{
    "images": ["relative/path/to/image.jpg"],
    "messages": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]},
        {"role": "assistant", "content": [{"type": "text", "text": "..."}]}
    ]
}

Only assistant turns contribute to the loss. User turns are masked with -100.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


DEFAULT_BASE_MODEL = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
DEFAULT_OUTPUT_DIR = "./finetune_output"
DEFAULT_AGRICHAT_WEIGHTS_DIR = "./weights/AgriChat"
DEFAULT_LOG_FILE = "training_metrics.jsonl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(output_dir: str) -> str | None:
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=os.path.getmtime,
    )
    return checkpoints[-1] if checkpoints else None


def normalize_image_path(image_root: str, relative_path: str) -> str:
    return os.path.join(image_root, relative_path.replace("\\", "/"))


def ensure_first_turn_has_image(messages: list[dict]) -> list[dict]:
    normalized = copy.deepcopy(messages)
    if not normalized:
        return normalized

    first_turn = normalized[0]
    if first_turn.get("role") != "user":
        return normalized

    content = first_turn.get("content")
    if isinstance(content, str):
        if "<image>" not in content:
            first_turn["content"] = "<image>\n" + content
        return normalized

    if isinstance(content, list):
        has_image = any(
            isinstance(item, dict) and item.get("type") == "image"
            for item in content
        )
        if not has_image:
            first_turn["content"] = [{"type": "image"}] + content
    return normalized


class JsonlMetricLogger(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        payload = {
            "step": state.global_step,
            "epoch": state.epoch,
            **{k: v for k, v in logs.items() if isinstance(v, (int, float, str))},
        }
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


class AgriConversationDataset(Dataset):
    def __init__(self, jsonl_path: str, image_root: str, processor):
        self.samples: list[dict] = []
        self.image_root = image_root
        self.processor = processor
        self.im_start_token_id = 151644
        self.im_end_token_id = 151645

        with open(jsonl_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "images" in record and "messages" in record:
                    self.samples.append(record)

        print(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        image_paths = item.get("images", [])
        if not image_paths:
            return self._dummy_entry()

        image_path = normalize_image_path(self.image_root, image_paths[0])
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: missing image {image_path}")
            return self._dummy_entry()

        conversation = ensure_first_turn_has_image(item["messages"])
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
        )

        batch = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids = batch["input_ids"][0]
        labels = self._build_labels(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"][0],
            "pixel_values": batch["pixel_values"][0],
            "image_sizes": batch["image_sizes"][0],
            "labels": labels,
        }

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        labels = torch.full_like(input_ids, fill_value=-100)
        start_positions = (input_ids == self.im_start_token_id).nonzero(as_tuple=True)[0]

        for start_idx in start_positions:
            role_slice = input_ids[start_idx + 1 : start_idx + 4]
            role_text = self.processor.tokenizer.decode(role_slice, skip_special_tokens=True)
            if "assistant" not in role_text.lower():
                continue

            end_matches = (input_ids[start_idx:] == self.im_end_token_id).nonzero(as_tuple=True)[0]
            if len(end_matches) == 0:
                continue

            end_idx = start_idx + end_matches[0] + 1
            labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        return labels

    def _dummy_entry(self) -> dict[str, torch.Tensor]:
        dummy_image = Image.new("RGB", (336, 336), (0, 0, 0))
        dummy_text = "<|im_start|>user\n<image>\nignore<|im_end|>"
        batch = self.processor(
            text=[dummy_text],
            images=[dummy_image],
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_ids": batch["input_ids"][0],
            "attention_mask": batch["attention_mask"][0],
            "pixel_values": batch["pixel_values"][0],
            "image_sizes": batch["image_sizes"][0],
            "labels": torch.full_like(batch["input_ids"][0], fill_value=-100),
        }


@dataclass
class AgriDataCollator:
    processor: object

    def __call__(self, examples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = [entry["input_ids"] for entry in examples]
        attention_mask = [entry["attention_mask"] for entry in examples]
        labels = [entry["labels"] for entry in examples]
        pixel_values = [entry["pixel_values"] for entry in examples]
        image_sizes = [entry["image_sizes"] for entry in examples]

        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_sizes": torch.stack(image_sizes),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune AgriChat with LoRA on both the vision encoder and LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-jsonl", required=True, help="Path to the training JSONL file.")
    parser.add_argument("--eval-jsonl", help="Optional path to the evaluation JSONL file.")
    parser.add_argument("--image-root", required=True, help="Root directory for image paths in the JSONL files.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for training checkpoints and logs.")
    parser.add_argument(
        "--agrichat-weights-dir",
        default=DEFAULT_AGRICHAT_WEIGHTS_DIR,
        help="Directory where the final AgriChat PEFT weights will be exported.",
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base Hugging Face model id.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate.")
    parser.add_argument("--num-epochs", type=float, default=1.0, help="Training epochs.")
    parser.add_argument("--save-steps", type=int, default=10, help="Checkpoint save frequency.")
    parser.add_argument("--logging-steps", type=int, default=1, help="Trainer logging frequency.")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--lora-rank", type=int, default=128, help="LoRA rank for the LLM.")
    parser.add_argument("--lora-alpha", type=int, default=256, help="LoRA alpha for the LLM.")
    parser.add_argument("--vision-lora-rank", type=int, default=32, help="LoRA rank for the vision encoder.")
    parser.add_argument("--vision-lora-alpha", type=int, default=64, help="LoRA alpha for the vision encoder.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--resume-from", default=None, help="Resume from a specific checkpoint path.")
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention implementation passed to the base model loader.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    print("=" * 70)
    print("AgriChat LoRA Fine-Tuning")
    print(f"Base model        : {args.base_model}")
    print(f"Train JSONL       : {args.train_jsonl}")
    print(f"Eval JSONL        : {args.eval_jsonl or 'disabled'}")
    print(f"Image root        : {args.image_root}")
    print(f"Output dir        : {args.output_dir}")
    print(f"AgriChat weights  : {args.agrichat_weights_dir}")
    print(f"LLM LoRA rank     : {args.lora_rank}")
    print(f"Vision LoRA rank  : {args.vision_lora_rank}")
    print("=" * 70)

    processor = AutoProcessor.from_pretrained(args.base_model)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset = AgriConversationDataset(args.train_jsonl, args.image_root, processor)
    eval_dataset = (
        AgriConversationDataset(args.eval_jsonl, args.image_root, processor)
        if args.eval_jsonl
        else None
    )

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "out_proj",
            "fc1",
            "fc2",
        ],
        rank_pattern={".*vision_tower.*": args.vision_lora_rank},
        alpha_pattern={".*vision_tower.*": args.vision_lora_alpha},
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        tf32=True,
        dataloader_num_workers=args.num_workers,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        eval_strategy="no",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AgriDataCollator(processor),
        callbacks=[JsonlMetricLogger(os.path.join(args.output_dir, DEFAULT_LOG_FILE))],
    )

    resume_path = args.resume_from or find_latest_checkpoint(args.output_dir)
    if resume_path:
        print(f"Resuming from checkpoint: {resume_path}")

    trainer.train(resume_from_checkpoint=resume_path)

    os.makedirs(args.agrichat_weights_dir, exist_ok=True)
    trainer.model.save_pretrained(args.agrichat_weights_dir)
    print(f"Saved AgriChat weights to {args.agrichat_weights_dir}")


if __name__ == "__main__":
    main()
