# <img src="imgs/icon2.png" alt="AgriChat Icon" width="40" style="vertical-align: bottom;"> AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding

<p align="center">
  <strong>Abderrahmene Boudiaf, Irfan Hussain, Sajid Javed</strong><br>
  Department of Computer Science, Khalifa University of Science and Technology, Abu Dhabi, UAE
</p>

<p align="center">
  A specialized Multimodal Large Language Model (MLLM) for fine-grained plant species identification, plant disease diagnosis, and crop counting.
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-AgriMM-87CEEB" alt="Dataset"></a>
  <a href="#"><img src="https://img.shields.io/badge/Model-AgriChat--7B-green" alt="Model"></a>
</p>

---

## 📢 Latest Updates

- **[2026-02-26]**: AgriChat paper submitted to *Computers and Electronics in Agriculture*
- **[2026-02-26]**: Repository, dataset, and code released

---

## 🌟 Overview

<p align="center">
  <img src="imgs/LLavaOneVision_Chatbot.png" alt="AgriChat Conversational Examples" width="100%">
</p>

**AgriChat** is a domain-specialized Multimodal Large Language Model (MLLM) designed for interactive agricultural diagnostics. Built on the [LLaVA-OneVision](https://github.com/LLaVA-VL/LLaVA-NeXT) architecture, AgriChat employs an **adaptive resolution (AnyRes) strategy** to preserve native pixel information up to **1344×1344** resolution — critical for resolving fine-grained visual features such as early-onset lesions, subtle phenotypic traits, and individual crop units. The model uses a **SigLIP-SO400M** vision encoder and a **Qwen-2-7B** language decoder, adapted to agriculture via parameter-efficient **LoRA** fine-tuning on our proposed AgriMM dataset.

### Why AgriChat?

General-purpose MLLMs lack the verified domain expertise to reason reliably across diverse plant taxonomies. AgriChat addresses this by:

- Training on **607,125 VQA pairs** grounded in verified phytopathological literature (not hallucinated by frozen LLMs)
- Covering **3,000+ agricultural classes** — the widest taxonomic diversity of any agricultural MLLM to date
- Running in **~2.3 seconds** on consumer-grade hardware (RTX 3090), enabling real-time field deployment

---

## 🏆 Key Contributions

1. **AgriMM Dataset**: The largest publicly available agricultural VQA benchmark — **121,425 images** and **607,125 QA pairs** spanning **3,099 classes** across **63 source datasets**, covering fine-grained species identification, disease diagnosis, and crop counting.

2. **Vision-to-Verified-Knowledge (V2VK) Pipeline**: A novel 3-stage data synthesis framework that integrates visual captioning with web-augmented scientific retrieval, eliminating biological hallucinations by grounding training data in verified literature.

3. **AgriChat Model**: The first agricultural MLLM fine-tuned on such a broad and diverse corpus, achieving state-of-the-art performance on four agriculture benchmarks and demonstrating superior zero-shot generalization over larger open-source generalist models.

---

## 📂 AgriMM Dataset

<p align="center">
  <img src="imgs/data_pipeline.png" alt="V2VK Synthesis Pipeline" width="80%">
</p>

**AgriMM** consolidates **63 source datasets** into a unified benchmark of **121,425 images** and **607,125 instruction-following QA pairs**. It is the first publicly available, multi-source agricultural VQA benchmark integrating fine-grained taxonomy, counting, and web-verified knowledge.

### Dataset Components

| Component | Images | Classes | Description |
|-----------|--------|---------|-------------|
| Fine-Grained Identification | 48,580 | 2,956 species | Sourced from iNatAg; 9 taxonomic categories |
| Disease Classification | 49,348 | 110 diseases | 29 datasets across 33 major crops |
| Crop Counting & Detection | 23,497 | 33 crops | 33 detection datasets with bounding-box-derived counts |

### Vision-to-Verified-Knowledge (V2VK) Pipeline

Our pipeline ensures scientific accuracy through three stages:

1. **Stage I — Visual Grounding**: Gemma 3 (8B) generates structured image captions conditioned on ground-truth labels, extracting growth stage, planting density, and environmental context.
2. **Stage II — Knowledge Retrieval**: Gemini 3 Pro with web-search retrieves verified botanical descriptions, disease etiology, and management protocols from authoritative sources.
3. **Stage III — Instruction Synthesis**: LLaMA 3.1-8B-Instruct synthesizes both the visual caption and retrieved knowledge into 5 diverse QA pairs per image (Identification, Visual Reasoning, Health Condition, Cultivation Knowledge, Quantification).

### 📥 Download

| Resource | Link |
|----------|------|
| AgriMM Dataset | **Coming Soon** |
| Source Dataset List | See [Appendix A](/) in the paper |

---

## 🧠 Model

AgriChat is built on LLaVA-OneVision (7B) and adapted to agriculture via LoRA fine-tuning of both the vision encoder and language decoder.

### Architecture

| Component | Model | Details |
|-----------|-------|---------|
| Vision Encoder | SigLIP-SO400M | 384×384 native tile resolution, LoRA rank=32, α=64 |
| Cross-Modal Projector | 2-layer MLP | Projects d_v=1152 → d_llm=3584 |
| Language Decoder | Qwen-2-7B | LoRA rank=128, α=256 |
| Max Resolution | 1344×1344 | Via adaptive grid (up to 12 tiles) |

### Performance Summary

AgriChat vs. state-of-the-art generalist baselines (METEOR / LLM Judge scores):

| Benchmark | LLaVA-OneVision (7B) | Llama-3.2 (11B) | Qwen-2.5 (7B) | **AgriChat (7B)** |
|-----------|---------------------|-----------------|---------------|-------------------|
| AgriMM | 37.89 / 55.12 | 32.43 / 57.18 | 29.11 / 65.77 | **66.70 / 77.43** |
| PlantVillageVQA | 17.25 / 57.41 | 6.72 / 54.44 | 3.43 / 53.21 | **19.52 / 74.26** |
| CDDM (Diagnosis) | 17.17 / 55.53 | 18.63 / 53.03 | 18.11 / 59.51 | **39.59 / 69.94** |
| AGMMU (MCQs) | 8.88 / — | 28.06 / — | 31.70 / — | **63.87 / —** |

### Inference

| Metric | Value |
|--------|-------|
| Avg. Inference Time | 2.315 seconds |
| Memory (4-bit quantized) | 10.71–12.32 GB |
| Throughput | ~1,555 queries/hour |
| Hardware | NVIDIA RTX 3090 (24GB) |

### 📥 Download

| Model | Base | Link |
|-------|------|------|
| AgriChat-7B | LLaVA-OneVision / Qwen-2-7B | **Coming Soon** (released upon publication) |

---

## 🔧 Installation

### Step 1: Create Conda Environment
```bash
conda create -n agrichat python=3.10 -y
conda activate agrichat
```

### Step 2: Install PyTorch with CUDA Support
```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8, use cu118 instead
```

### Step 3: Install Dependencies
```bash
pip install --upgrade transformers accelerate bitsandbytes pillow requests protobuf peft
```

### Step 4: Clone Repository
```bash
git clone https://github.com/boudiafA/AgriChat.git
cd AgriChat
export PYTHONPATH="./:$PYTHONPATH"
```

---

## ⚙️ V2VK Data Generation Pipeline

Scripts to reproduce the Vision-to-Verified-Knowledge pipeline:

```bash
# Stage I: Visual Grounding (Image Captioning via Gemma 3)
python scripts/stage1_captioning.py --data_dir ./AgriMM/images --output ./captions.json

# Stage II: Knowledge Retrieval (Web-RAG via Gemini 3 Pro)
python scripts/stage2_knowledge.py --classes ./AgriMM/classes.json --output ./knowledge.json

# Stage III: Instruction Synthesis (QA Generation via LLaMA 3.1)
python scripts/stage3_qa_generation.py --captions ./captions.json --knowledge ./knowledge.json --output ./AgriMM_QA.json
```

> See [Appendix B](/) in the paper for the exact prompt templates used in each stage.

---

## 🏋️ Training

AgriChat is fine-tuned in a **single stage** on the full multimodal AgriMM dataset using LoRA adapters on both the vision encoder and language decoder.

```bash
python train.py \
  --model_name_or_path llava-onevision-7b \
  --data_path ./AgriMM/ \
  --output_dir ./checkpoints/agrichat-7b \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --bf16 True \
  --lora_r 128 \
  --lora_alpha 256 \
  --vision_lora_r 32 \
  --vision_lora_alpha 64
```

**Training Details:**
- **Hardware**: Single NVIDIA RTX 3090 (24GB VRAM)
- **Precision**: bfloat16 mixed precision
- **Effective batch size**: 16 (1 × 16 gradient accumulation)
- **Epochs**: 1
- **Strategy**: Single-stage (outperforms two-stage curriculum; see ablation in paper)

---

## 📊 Evaluation

We employ a multi-faceted evaluation framework combining lexical, semantic, and LLM-based metrics:

| Category | Metrics |
|----------|---------|
| Lexical | BLEU-4, ROUGE-2, METEOR |
| Semantic | BERTScore, LongCLIP, T5 Cosine, SBERT |
| LLM-as-a-Judge | Qwen3-30B-A3B-Instruct (4-point Likert scale) |

```bash
# Run evaluation on a benchmark
python evaluate.py \
  --model_path ./checkpoints/agrichat-7b \
  --benchmark agrimm \
  --output_dir ./results/
```

---

## 📚 Qualitative Examples

AgriChat demonstrates expert-level agricultural reasoning across diverse tasks:

<p align="center">
  <img src="imgs/qualitative_1.png" alt="Disease Diagnosis Example" width="45%">
  <img src="imgs/qualitative_2.png" alt="Crop Counting Example" width="45%">
</p>

**Left**: Zero-shot disease diagnosis — AgriChat correctly identifies Tomato Yellow Leaf Curl Virus while generalist models misdiagnose or refuse. **Right**: Precise crop counting — AgriChat returns the exact count (61 wheat heads) while baselines give vague or incorrect answers.

---

## 📜 Citation

If you find our work useful, please cite:
```bibtex
@article{boudiaf2026agrichat,
  title     = {AgriChat: A Multimodal Large Language Model for Agriculture Image Understanding},
  author    = {Boudiaf, Abderrahmene and Hussain, Irfan and Javed, Sajid},
  journal   = {Submitted to Computers and Electronics in Agriculture},
  year      = {2026}
}
```

---

## 📄 License

This project is released under the [Apache 2.0 License](LICENSE).

The AgriMM dataset is released for **research purposes only**. Individual source datasets retain their original licenses — see [Appendix A](/) for the complete source list.

## 🙏 Acknowledgments

This work was supported by Khalifa University of Science and Technology, Abu Dhabi, UAE. We thank the creators of the 63 source datasets that make AgriMM possible.

## 📧 Contact

For questions or collaborations, please contact: **100058322@ku.ac.ae** (Abderrahmene Boudiaf)
