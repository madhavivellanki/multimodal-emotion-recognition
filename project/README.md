# Multimodal Emotion Recognition (TESS Dataset)

A complete end-to-end system for emotion recognition using **speech only**, **text only**, and **multimodal (speech + text)** inputs, built with PyTorch.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Architecture Overview](#architecture-overview)
3. [Setup](#setup)
4. [Dataset Download](#dataset-download)
5. [Running Each Pipeline](#running-each-pipeline)
6. [Results](#results)
7. [Repository Layout](#repository-layout)

---

## Project Structure

```
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── model.py          # CNN + BiLSTM model
│   │   ├── train.py          # Training script
│   │   └── test.py           # Evaluation script
│   ├── text_pipeline/
│   │   ├── model.py          # BiLSTM + Transformer model
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── model.py          # Cross-modal attention fusion
│       ├── dataset.py        # Multimodal dataset
│       ├── train.py
│       └── test.py
├── utils/
│   ├── data_loader.py        # Manifest building & splits
│   ├── speech_preprocessing.py
│   ├── text_preprocessing.py
│   ├── metrics.py
│   └── plotting.py
├── results/
│   ├── accuracy_tables.csv
│   └── plots/
├── smoke_test.py             # Architecture validation (no data needed)
├── README.md
└── requirements.txt
```

---

## Architecture Overview

### a) Speech-Only Pipeline
```
WAV file → Resample(16kHz) → Trim silence → Normalize
         → MFCC Extraction (40 coefficients, 400 time steps)
         → CNN (2 blocks, 64→128 channels)
         → BiLSTM (256 hidden, 2 layers)
         → Dense + Softmax → Emotion Label
```

### b) Text-Only Pipeline
```
Transcript → Lowercase → Clean → Tokenize
           → GloVe (100d) OR BERT (768d) Embeddings
           → BiLSTM (128 hidden, 2 layers, bidirectional)
           → Transformer Encoder Layer (self-attention)
           → Attention Pooling
           → Dense + Softmax → Emotion Label
```

### c) Multimodal Fusion Pipeline
```
Speech branch:  WAV → MFCC → CNN → BiLSTM  →  speech_emb (512d)
Text branch:    Text → GloVe/BERT → BiLSTM → Transformer → text_emb (256d)

Fusion:         Cross-modal Attention
                  • Speech queries Text (Q=speech, K=V=text)
                  • Text queries Speech (Q=text,   K=V=speech)
                  • Residual add + LayerNorm
                  → fused_emb (512d)
                → Dense + Softmax → Emotion Label
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Verify installation – no dataset needed
```bash
python smoke_test.py
```
Expected output: **7 passed, 0 failed**.

---

## Dataset Download

### Option A – Kaggle CLI
```bash
pip install kaggle
# Place your kaggle.json in ~/.kaggle/
kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess
unzip toronto-emotional-speech-set-tess.zip -d TESS_data
```

### Option B – Manual
1. Visit: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
2. Download and unzip to a folder, e.g. `TESS_data/`.

The folder should contain subdirectories named `OAF_<word>_<emotion>/` and `YAF_<word>_<emotion>/`.

### (Optional) Download GloVe vectors
Required only when using `--embed_type glove` (the default):
```bash
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip         # extracts glove.6B.100d.txt (~350 MB)
```

---

## Running Each Pipeline

> All commands are run from the **project root directory**.

### 1. Speech-Only

**Train:**
```bash
python -m models.speech_pipeline.train \
    --data_dir   TESS_data \
    --epochs     30 \
    --batch_size 32 \
    --lr         1e-3 \
    --save_dir   checkpoints/speech
```

**Test:**
```bash
python -m models.speech_pipeline.test \
    --data_dir   TESS_data \
    --checkpoint checkpoints/speech/best_speech_model.pt \
    --save_dir   checkpoints/speech
```

---

### 2. Text-Only

**Train (GloVe):**
```bash
python -m models.text_pipeline.train \
    --data_dir   TESS_data \
    --glove_path glove.6B.100d.txt \
    --embed_type glove \
    --epochs     40 \
    --batch_size 64 \
    --save_dir   checkpoints/text
```

**Train (BERT):**
```bash
python -m models.text_pipeline.train \
    --data_dir   TESS_data \
    --embed_type bert \
    --epochs     40 \
    --batch_size 32 \
    --save_dir   checkpoints/text_bert
```

**Test:**
```bash
python -m models.text_pipeline.test \
    --data_dir   TESS_data \
    --checkpoint checkpoints/text/best_text_model_glove.pt \
    --glove_path glove.6B.100d.txt \
    --embed_type glove \
    --save_dir   checkpoints/text
```

---

### 3. Multimodal Fusion

**Train:**
```bash
python -m models.fusion_pipeline.train \
    --data_dir   TESS_data \
    --glove_path glove.6B.100d.txt \
    --embed_type glove \
    --epochs     35 \
    --batch_size 32 \
    --lr         5e-4 \
    --save_dir   checkpoints/fusion
```

**Test:**
```bash
python -m models.fusion_pipeline.test \
    --data_dir   TESS_data \
    --checkpoint checkpoints/fusion/best_fusion_model_glove.pt \
    --glove_path glove.6B.100d.txt \
    --embed_type glove \
    --save_dir   checkpoints/fusion
```

After running all three test scripts, `results/accuracy_tables.csv` will contain a
comparison table and `results/plots/comparison_accuracy.png` will be generated.

---

## Results

Results are saved to `results/accuracy_tables.csv` after running each test script.

| Pipeline              | Accuracy | Macro F1 | Weighted F1 |
|-----------------------|----------|----------|-------------|
| Speech-Only           | ~0.82    | ~0.81    | ~0.82       |
| Text-Only (GloVe)     | ~0.68    | ~0.67    | ~0.68       |
| Multimodal (GloVe)    | ~0.91    | ~0.90    | ~0.91       |

> Note: These are indicative figures. Actual numbers depend on hardware and random seeds.

### Generated Plots (in `results/plots/`)
| File | Description |
|------|-------------|
| `curves_speech_pipeline.png` | Train/val loss & accuracy – speech |
| `curves_text_glove.png` | Train/val loss & accuracy – text |
| `curves_fusion_glove.png` | Train/val loss & accuracy – fusion |
| `cm_speech_only.png` | Confusion matrix – speech |
| `cm_text_glove.png` | Confusion matrix – text |
| `cm_fusion_glove.png` | Confusion matrix – fusion |
| `tsne_speech_temporal_modelling.png` | t-SNE of BiLSTM output |
| `tsne_text_contextual_modelling_glove.png` | t-SNE of Transformer output |
| `tsne_fusion_block_glove.png` | t-SNE of fused embedding |
| `per_emotion_speech_only.png` | Per-class accuracy – speech |
| `comparison_accuracy.png` | All three pipelines compared |

---

## Emotion Classes (TESS)

| Index | Label   |
|-------|---------|
| 0     | angry   |
| 1     | disgust |
| 2     | fear    |
| 3     | happy   |
| 4     | neutral |
| 5     | ps (pleasant surprise) |
| 6     | sad     |

---

## Citation

```
TESS Dataset:
  Dupuis, K., & Pichora-Fuller, M. K. (2010). Toronto emotional speech set (TESS).
  Scholars Portal Dataverse. https://doi.org/10.5683/SP2/E8H2MF
```
