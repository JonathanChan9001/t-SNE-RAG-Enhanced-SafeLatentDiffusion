# RAG-Enhanced Safe Latent Diffusion with t-SNE Analysis (RAG-SLD)

**FIT5230 – Malicious AI** | Monash University

A safety pipeline that combines **Retrieval-Augmented Generation (RAG)** with **Safe Latent Diffusion (SLD)** to defend text-to-image generation against adversarial and harmful prompts. This repository also contains the work of the **opposition team (chickenrice)**, who perform adversarial video generation attacks using FLIRT on Open-Sora.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Datasets & CSV Files](#datasets--csv-files)
- [Opposition Team – chickenrice](#opposition-team--chickenrice)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [t-SNE Visualisation](#t-sne-visualisation)

---

## Overview

<p align="center">
  <a href="https://huggingface.co/AIML-TUDA/stable-diffusion-safe"><img src="https://img.shields.io/badge/🤗_HuggingFace-FFD21E?style=for-the-badge&logoColor=black" alt="HuggingFace"></a>
  <a href="https://www.trychroma.com/"><img src="https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge&logo=databricks&logoColor=white" alt="ChromaDB"></a>
  <a href="https://huggingface.co/AIML-TUDA/stable-diffusion-safe"><img src="https://img.shields.io/badge/Safe_Latent_Diffusion-6B4FBB?style=for-the-badge&logo=artstation&logoColor=white" alt="Safe Latent Diffusion"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="https://developer.nvidia.com/cuda-toolkit"><img src="https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="NVIDIA CUDA"></a>
</p>

The RAG-SLD pipeline intercepts prompts before they reach a diffusion model. It embeds each prompt and compares it against a **ChromaDB vector store** of known harmful adversarial prompts sourced from the [I2P dataset](https://huggingface.co/datasets/AIML-TUDA/i2p) and the [Ring-A-Bell Violence category](https://arxiv.org/abs/2310.10012). Based on the cosine similarity score, it makes one of three decisions before allowing (or blocking) image generation through [AIML-TUDA/stable-diffusion-safe](https://huggingface.co/AIML-TUDA/stable-diffusion-safe). The pipeline is further extended with **t-SNE visualisation** to project semantic embeddings of adversarial, obfuscated, and safe prompts into 2D space, providing interpretable insight into why the retrieval-based safety filter remains robust even under obfuscation attacks.

---

## How It Works

```
User Prompt
     │
     ▼
[SentenceTransformer Embedder]  ← all-MiniLM-L6-v2
     │
     ▼
[ChromaDB Vector Store Query]   ← ~1,355 harmful prompts (I2P + Ring-A-Bell)
     │
     ▼
 Cosine Similarity Score
     │
     ├── score ≥ 0.6         →  🚫 BLOCK       (generation refused)
     ├── score ∈ [0.4, 0.6)  →  ⚠️  SOFT_ALLOW  (maximum SLD safety settings)
     └── score < 0.4         →  ✅ ALLOW        (standard SLD safety settings)
                                      │
                                      ▼
                           [StableDiffusionPipelineSafe]
                                      │
                                      ▼
                                Generated Image
```

**Vector Store Contents:**
- **I2P dataset** — 680 violence-category prompts stored (90% split), 76 held out for testing
- **Ring-A-Bell Violence** — 675 prompts stored (3 CSV files × 250 records, 90% split), 75 held out

---

## Project Structure

```
MAI_Assignment/
│
├── RAG_SLD.ipynb                                # Main RAG + SLD pipeline notebook
├── RAG_SLD_tSNE.ipynb                           # Extended: obfuscation robustness + t-SNE
├── obs.txt                                      # Obfuscation utility (AdvancedObfuscator)
├── pyproject.toml                               # Python project dependencies
├── MANIFEST.in
├── uv.lock
│
├── chickenrice/                                 # Opposition team – adversarial video attacks
│   └── Copy_of_Dark_Sleepy_v2.ipynb             # FLIRT attack on Open-Sora v2
│
├── ReLaion-COCO_200/                            # Safe prompt evaluation set
│   └── ReLaion-COCO_safe_prompts_200.csv        # 200 safe prompts from ReLaion-COCO
│
├── eval_images/                                 # Generated evaluation images
│
├── eval_test_prompts.csv                        # 151 harmful test prompts (10% held-out)
├── blocked_prompts.csv                          # Prompts blocked by the pipeline (115 samples)
├── rag_evaluation_results.csv                   # Evaluation on 151 harmful prompts
├── rag_evaluation_200_relaion_coco_results.csv  # Evaluation on 200 safe prompts
└── text_prompts_200.csv                         # 200 text prompts used in evaluation
```

---

## Notebooks

### `RAG_SLD.ipynb` — Main Pipeline

The primary notebook implementing the full RAG + SLD safety pipeline.

| Section | Description |
|---|---|
| Pre-Setup | Install dependencies (`chromadb`, `diffusers`, `sentence-transformers`, etc.) |
| Vector Store Creation | Initialise ChromaDB persistent store (`adv_violence_prompts` collection) |
| Process Datasets | GPU-accelerated batch ingestion of I2P and Ring-A-Bell Violence datasets |
| RAG + SLD Pipeline | `RAGSLDPipeline` class: embedding query → similarity lookup → 3-tier decision → SLD generation |
| Example Usage | Run the pipeline on a single prompt |
| Evaluation | `RAGSLDEvaluator`: confusion matrix, precision/recall/F1, decision distribution, image generation |

> **Quick start:** Run all cells up to and including the **"Example of Using RAG + SLD Pipeline"** section to use the pipeline interactively. The Evaluation section is optional.

---

### `RAG_SLD_tSNE.ipynb` — Extended Analysis

Builds on `RAG_SLD.ipynb` with two additional capabilities:

**1. Obfuscation Robustness Testing**
- `AdvancedObfuscator` inserts Unicode zero-width characters and paraphrase injections at intensity levels `[0.0, 0.3, 0.5, 0.8, 1.0]`
- Augments the vector store with obfuscated variants to harden the pipeline
- Results: **100% block rate** at intensity ≤ 0.5; **97.4% block rate** at intensities 0.8 and 1.0

**2. t-SNE Embedding Visualisation**
- Embeds 50 test prompts, 200 original adversarial prompts, and 200 obfuscated variants
- Applies `sklearn.manifold.TSNE` (2D, perplexity=30) to project embeddings
- Saves visualisation as `rag_embedding_visualization.pdf`

---

## Datasets & CSV Files

| File | Description | Rows |
|---|---|---|
| `eval_test_prompts.csv` | Held-out harmful prompts (10% split from I2P + Ring-A-Bell) | 151 |
| `blocked_prompts.csv` | Test prompts that received a BLOCK decision | 115 |
| `rag_evaluation_results.csv` | RAG evaluation on harmful prompts — columns: `prompt`, `rag_decision`, `similarity_score`, `reason`, `ground_truth`, `top_match`, `seed` | 151 |
| `rag_evaluation_200_relaion_coco_results.csv` | RAG evaluation on 200 **safe** prompts (false-positive analysis) | 200 |
| `ReLaion-COCO_200/ReLaion-COCO_safe_prompts_200.csv` | Source safe prompts for the above evaluation | 200 |
| `text_prompts_200.csv` | 200 prompts used in image-generation evaluation | 200 |

**Evaluation columns explained:**

| Column | Description |
|---|---|
| `rag_decision` | `BLOCK`, `SOFT_ALLOW`, or `ALLOW` |
| `similarity_score` | Cosine similarity to the closest vector in the store |
| `ground_truth` | `1` = harmful, `0` = safe |
| `top_match` | The closest matching adversarial prompt from the store |

---

## Opposition Team – chickenrice

**Team members:** Nadya Tan (31878458), Edith Kabade (35405007)

The `chickenrice/` directory contains the opposition team's adversarial attack notebook, targeting **video generative AI** rather than image generation.

### `chickenrice/Copy_of_Dark_Sleepy_v2.ipynb`

| Component | Description |
|---|---|
| **Target model** | [Open-Sora v2](https://github.com/hpcaitech/Open-Sora) (`hpcai-tech/Open-Sora-v2`) |
| **Attack method** | [FLIRT](https://github.com/amazon-science/FLIRT) — greedy prompt replacement fuzzing |
| **Safety evaluator** | [NudeNet](https://github.com/notAI-tech/NudeNet) `NudeClassifier` — per-frame `mean_unsafe` score |
| **Attack loop** | Iteratively mutates prompts, scores generated video frames, keeps mutations that maximise the unsafe score |
| **Infrastructure** | Google Colab + Google Drive (`/content/drive/MyDrive/Dark.Sleepy`) |

**Key components:**

- `run_opensora_video(prompt, cfg)` — calls Open-Sora inference via `torchrun`
- `VideoSafetyEvaluator.score_video(video_path)` — samples frames every `frame_step` steps, returns `max_unsafe` and `mean_unsafe`
- `flirt_run(seed_text, queue_text, ...)` — FLIRT attack loop with greedy replacement strategy
- Results saved to `summary.json` in the configured output directory

---

## Setup & Installation

### Option 1 – `uv` (recommended)

```bash
pip install uv
uv sync
```

### Option 2 – pip

```bash
pip install .
```

### Key dependencies

| Package | Version |
|---|---|
| `torch` | ≥ 2.8.0 |
| `diffusers` | ≥ 0.35.1 |
| `chromadb` | ≥ 1.0.20 |
| `sentence-transformers` | ≥ 5.1.0 |
| `transformers` | ≥ 4.56.0 |
| `scikit-learn` | for evaluation plots and t-SNE |
| `matplotlib` / `seaborn` | for evaluation plots |

### Hugging Face Authentication

A Hugging Face token with **read** access is required to load the SLD model.

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Read** role
3. Set it as an environment variable before running:

```bash
export HF_TOKEN="hf_YOUR_TOKEN_HERE"
```

### Downloading Required Datasets

```bash
# I2P benchmark dataset
huggingface-cli download AIML-TUDA/i2p --repo-type dataset --local-dir ./i2p

# Ring-A-Bell Violence category
gdown --folder 'https://drive.google.com/drive/folders/1XRIWJvUjAii077w416K8GE6oVnrYSPY4?usp=drive_link'
```

---

## Usage

### Running the Pipeline on a Single Prompt

```python
# In RAG_SLD.ipynb or RAG_SLD_tSNE.ipynb
prompt = "your prompt here"
seed = 42

image, analysis = rag_sld_pipeline.generate_safe_image(
    prompt,
    seed=seed,
    return_analysis=True
)
image  # displays the generated image (or None if blocked)
```

### Decision Thresholds

| Parameter | Default | Meaning |
|---|---|---|
| `tau_block` | `0.6` | Similarity ≥ this → BLOCK |
| `tau_soft` | `0.4` | Similarity ∈ [tau_soft, tau_block) → SOFT_ALLOW |

### Device Configuration

```python
device = "cuda"   # NVIDIA GPU (recommended)
# device = "cpu"  # CPU only
# device = "mps"  # Apple Silicon (M1/M2/M3)
```

---

## Evaluation Results

### Harmful Prompts (151 test prompts from I2P + Ring-A-Bell)

| Decision | Count | Rate |
|---|---|---|
| BLOCK | 115 | ~76% |
| SOFT_ALLOW | ~30 | ~20% |
| ALLOW | ~6 | ~4% |

### Safe Prompts – False Positive Analysis (200 ReLaion-COCO prompts)

All 200 safe prompts received an **ALLOW** decision, confirming negligible over-censorship on benign content.

### Obfuscation Robustness (tested on 115 blocked prompts)

| Obfuscation Level | Blocked | Bypassed | Avg Similarity |
|---|---|---|---|
| 0.0 (original) | 115 / 115 | 0 | 0.730 |
| 0.3 | 115 / 115 | 0 | 0.730 |
| 0.5 | 115 / 115 | 0 | 0.729 |
| 0.8 | 112 / 115 | 3 | 0.723 |
| 1.0 | 112 / 115 | 3 | 0.725 |

The pipeline maintains a **97–100% block rate** even against Unicode zero-width character injections and paraphrase-based obfuscation.

---

## t-SNE Visualisation

The `RAG_SLD_tSNE.ipynb` notebook generates a 2D t-SNE scatter plot (`rag_embedding_visualization.pdf`) showing:

- **Original adversarial prompts** in the vector store
- **Obfuscated variants** (zero-width unicode + paraphrase injections)
- **Test query prompts**

Even after obfuscation, harmful prompt embeddings cluster closely with their originals in the semantic embedding space — explaining the pipeline's high block rate across all obfuscation intensities.

---

## Project Context

This project was developed as part of **FIT5230 – Malicious AI** at Monash University.
The **defender team** built the RAG-SLD safety filter for text-to-image generation.
The **opposition team (chickenrice)** independently developed FLIRT-based adversarial attacks against video generation to probe the boundaries of AI safety systems.
