# AgriChat Weights

Place the released AgriChat PEFT weights in:

```text
weights/AgriChat/
├── adapter_config.json
└── adapter_model.safetensors
```

The repository scripts default to this location:

- `scripts/inference_AgriChat_lora.py`
- `scripts/chatbot_AgriChat_lora.py`
- `scripts/finetune_AgriChat_lora.py --agrichat-weights-dir`

This GitHub repository intentionally omits `adapter_model.safetensors` because the file is too large for a normal source-code push.

Model weights link: `TBD`
