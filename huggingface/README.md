---
base_model: llava-hf/llava-onevision-qwen2-7b-ov-hf
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - agriculture
  - multimodal
  - vision-language
  - llava-onevision
  - qwen2
  - peft
  - lora
---

# AgriChat

AgriChat is a domain-specialized multimodal large language model for agricultural image understanding. It is built on top of **LLaVA-OneVision / Qwen-2-7B** and adapted with **LoRA** for fine-grained plant species identification, plant disease diagnosis, and crop counting.

## Model Summary

- **Base model:** `llava-hf/llava-onevision-qwen2-7b-ov-hf`
- **Adaptation:** LoRA on both the SigLIP vision encoder and the Qwen2 language model
- **Domain:** Agriculture
- **Main use cases:** species recognition, disease reasoning, cultivation-related visual QA, crop counting

## Training Data

AgriChat is fine-tuned on **AgriMM**, a large multi-source agricultural multimodal instruction dataset covering:

- fine-grained plant identification
- disease classification and diagnosis
- crop counting and grounded visual reasoning

The AgriMM data generation pipeline combines:

1. image-grounded captioning with Gemma 3 (12B)
2. verified knowledge retrieval with Gemini 3 Pro and Google Search grounding
3. QA synthesis with LLaMA 3.1-8B-Instruct

## Intended Use

AgriChat is intended for:

- agricultural visual question answering
- plant disease analysis support
- crop and species recognition research
- multimodal benchmarking in agriculture

It is not intended to replace expert agronomists, plant pathologists, or field inspection in high-stakes decision making.

## Quickstart

```python
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

BASE_MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
AGRICHAT_WEIGHTS = "boudiafA/AgriChat"  # placeholder repo id

processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model = PeftModel.from_pretrained(base_model, AGRICHAT_WEIGHTS)
model.eval()

image = Image.open("path/to/image.jpg").convert("RGB")
prompt = "What is shown in this agricultural image?"

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ],
    }
]

text = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
device = next(model.parameters()).device
inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

input_len = inputs["input_ids"].shape[1]
response = processor.tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
print(response.strip())
```

## Performance Snapshot

AgriChat outperforms strong open-source generalist baselines on multiple agriculture benchmarks.

| Benchmark | AgriChat |
|-----------|----------|
| AgriMM | 66.70 METEOR / 77.43 LLM Judge |
| PlantVillageVQA | 19.52 METEOR / 74.26 LLM Judge |
| CDDM | 39.59 METEOR / 69.94 LLM Judge |
| AGMMU | 63.87 accuracy |

## Limitations

- Performance depends on image quality and coverage of the training data.
- The model can still make confident but incorrect statements.
- Outputs should be reviewed carefully before use in real agricultural decision workflows.

## Citation

```bibtex
@article{boudiaf2026agrichat,
  title     = {AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding},
  author    = {Boudiaf, Abderrahmene and Hussain, Irfan and Javed, Sajid},
  journal   = {Submitted to Computers and Electronics in Agriculture},
  year      = {2026}
}
```
