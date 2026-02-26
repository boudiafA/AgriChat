# <img src="imgs/icon2.png" alt="Agri-OneVision Icon" width="40" style="vertical-align: bottom;"> Agri-OneVision: High-Resolution Agricultural Vision-Language Model

<p align="center">
  <strong>Abderrahmene Boudiaf, Irfan Hussain, Sajid Javed</strong>
</p>

<p align="center">
  A specialized Multimodal Large Language Model (MLLM) for fine-grained plant pathology, species identification, and crop counting.
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Online-Demo-red" alt="Demo"></a>
  <a href="#"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg" alt="Paper"></a>
  <a href="#"><img src="https://img.shields.io/badge/Dataset-AgriVQA--600K-87CEEB" alt="Dataset"></a>
</p>

---

## 📢 Latest Updates

- **[Date]**: Agri-OneVision paper released on arXiv
- **[Date]**: Code, AgriVQA-600K dataset, and pretrained checkpoints are now available!

---

## 🌟 Overview

<p align="center">
  <img src="imgs/LLavaOneVision_Chatbot.png" alt="Agri-OneVision Architecture" width="100%">
</p>

**Agri-OneVision** is a specialized LLaVA-OneVision framework designed for agricultural visual diagnostics. Unlike general-purpose models that downsample images (destroying high-frequency details needed for disease detection), our model employs an **AnyRes strategy** to preserve native pixel information up to **1344×1344** resolution. It is powered by a **SigLIP-SO400M** vision encoder and a **Qwen-2-7B** language decoder, fine-tuned via QLoRA.

---

## 🏆 Key Contributions

1. **AgriVQA-600K Dataset**: We introduce the most taxonomically diverse agricultural VQA benchmark, aggregating 63 datasets to cover **3,000+ classes**, including fine-grained species ID, crop counting, and disease classification.

2. **Vision-to-Verified-Knowledge Pipeline**: We introduce a novel 3-stage synthesis pipeline using RAG (Retrieval-Augmented Generation). By grounding visual captions in web-retrieved scientific knowledge, we eliminate biological hallucinations in training data.

3. **Agri-OneVision Architecture**: We propose a high-resolution architecture capable of resolving millimeter-scale features (e.g., early fungal infections or pest eggs) that are typically lost by standard VLM preprocessing.

---

## 📂 AgriVQA-600K Dataset

<p align="center">
  <img src="imgs/data_pipeline.png" alt="Synthesis Pipeline" width="80%">
</p>

We constructed **AgriVQA-600K**, comprising **121,425 images** and **607,125 instruction-following pairs**. The dataset is built systematically across three categories:

### Dataset Categories

- **Fine-Grained Species Classification**: Derived from iNatAg, covering 2,959 plant species with rich morphological descriptions.

- **Crop Counting**: Aggregated from 33 detection datasets (e.g., wheat heads, apples, pods) to enable precise quantitative reasoning.

- **Disease Classification**: The largest component, covering 110 distinct pathological classes across 29 datasets, including "Healthy" baselines often omitted in other benchmarks.

### Vision-to-Verified-Knowledge Pipeline

Our pipeline ensures factuality through three stages:

1. **Visual Grounding**: Image captioning via Gemini 3 8B
2. **Knowledge Retrieval**: Web-search via Gemini 3 Pro to fetch up-to-date botanical data
3. **Instruction Generation**: LLaMA 3.1 8B creates diverse QA pairs grounded in retrieved knowledge

### 📥 Download

Access AgriVQA-600K: **Coming Soon**

---

## 🧠 Model Zoo

We release the following checkpoints based on the Qwen-2-7B architecture, trained with our specialized high-resolution strategy.

| Model Name | Base Model | Resolution Strategy | Link |
|------------|------------|---------------------|------|
| Agri-OneVision-7B | Qwen-2-7B | AnyRes (Grid) | **Coming Soon** |

---

## 🔧 Installation

We recommend setting up a clean conda environment to run Agri-OneVision.

### Step 1: Create and Activate the Conda Environment
```bash
conda create -n llava_env python=3.10 -y
conda activate llava_env
```

### Step 2: Install PyTorch with CUDA Support

For most modern GPUs (CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> **Note**: If you are on an older system with CUDA 11.8, use `cu118` instead of `cu121`

### Step 3: Install Required Libraries
```bash
pip install --upgrade transformers accelerate bitsandbytes pillow requests protobuf
```

**For Windows Users**: If `bitsandbytes` fails to import, install the Windows-specific version:
```bash
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
```

### Step 4: Clone Repository
```bash
git clone https://github.com/username/Agri-OneVision
cd Agri-OneVision
export PYTHONPATH="./:$PYTHONPATH"
```

---

## ⚙️ Data Generation Pipeline

Scripts to reproduce the Vision-to-Verified-Knowledge pipeline (Visual Captioning → Web RAG → QA Generation).
```bash
# Coming soon
```

---

## 🏋️ Training

We utilize a 3-stage curriculum learning pipeline (Alignment → Foundational Knowledge → Domain Specialization) using QLoRA on a single NVIDIA H200 GPU.
```bash
# Example training command
sh scripts/train_agri_onevision.sh --data_path ./AgriVQA-600K --output_dir ./checkpoints
```

---

## 📊 Evaluation and Metrics

Evaluating agricultural diagnostics requires more than lexical overlap. We employ a multi-faceted evaluation framework:

- **Lexical Metrics**: BLEU-4, ROUGE-L, CIDEr
- **Semantic Metrics**: BERTScore, SBERT, and Long-CLIP similarity to measure embedding alignment
- **LLM-as-a-Judge**: We use DeepSeek-R1-Distill-Llama-70B with a strict prompt to penalize verbosity and "length-hacking," ensuring the model is rewarded for accuracy and diagnostic reasoning rather than fluff

<p align="center">
  <img src="imgs/evaluation_results.png" alt="Evaluation Results" width="80%">
</p>

---

## 📚 Qualitative Examples

AgriChat demonstrates superior performance in identifying subtle disease features and performing accurate crop counts compared to general-purpose VLMs.

<p align="center">
  <img src="imgs/qualitative_1.png" alt="Disease Diagnosis Example" width="45%">
  <img src="imgs/qualitative_2.png" alt="Crop Counting Example" width="45%">
</p>

---

## 📜 Citation

If you find our work useful, please cite:
```bibtex
@article{AgriOneVision2025,
  title   = {Agri-OneVision: High-Resolution Vision-Language Model for Agricultural Diagnostics},
  author  = {[Author List Placeholder]},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

[Add acknowledgments here]

## 📧 Contact

For questions or collaborations, please contact: [your-email@example.com]
