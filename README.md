<div align="center">

# 🎭 Multimodal Emotion Recognition System

### Recognise human emotions from Speech · Text · Fusion
**PyTorch · CNN+BiLSTM · Transformer Encoder · Streamlit**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live_Demo-ff4b4b?style=flat-square&logo=streamlit)](https://your-streamlit-link.streamlit.app)
[![Git LFS](https://img.shields.io/badge/Git_LFS-Checkpoints-F64935?style=flat-square&logo=git)](https://git-lfs.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

**GitHub:** https://github.com/madhavivellanki/multimodal-emotion-recognition

**Live Demo:** https://multimodal-emotion-recognition-aqgmwmlqntvyzdmwxm7bod.streamlit.app/

</div>

---

## ⚠️ Important Note — Git LFS (Large File Storage)

This repository uses **Git LFS** to store large model checkpoint files (`*.pt`, `*.npy`).

You **must install Git LFS** before cloning — otherwise the checkpoint files will not download correctly and the models will not load.

> The following files are stored in Git LFS:
> - `project/checkpoints/speech/best_speech_model.pt`
> - `project/checkpoints/speech/mfcc_mean.npy`
> - `project/checkpoints/speech/mfcc_std.npy`
> - `project/checkpoints/text/best_text_model_glove.pt`
> - `project/checkpoints/fusion/best_fusion_model_glove.pt`

---

## 🚀 Quick Start (For Instructors & Evaluators)

Run these commands **in order** to get the project working immediately:

```bash
# Step 1 — Install Git LFS (required for model checkpoints)
git lfs install

# Step 2 — Clone the repository
git clone https://github.com/madhavivellanki/multimodal-emotion-recognition.git

# Step 3 — Enter the project folder
cd multimodal-emotion-recognition

# Step 4 — Pull the large model checkpoint files via Git LFS
git lfs pull

# Step 5 — Install all Python dependencies
pip install -r requirements.txt

# Step 6 — Launch the Streamlit demo
streamlit run demo/app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🐍 Python Version

> **Recommended: Python 3.11**

This project was developed and tested on **Python 3.11.0 (64-bit)**.

- Python 3.14 is **not supported** (PyTorch does not support it yet)
- Python 3.9, 3.10, or 3.11 are all compatible

**Check your Python version:**
```bash
python --version
```

**If needed, create a virtual environment with Python 3.11:**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

---

## 📖 How to Run the Project — Step by Step

### Prerequisites
Before starting, make sure you have:
- ✅ Python 3.11 installed
- ✅ Git installed
- ✅ Git LFS installed (`git lfs install`)
- ✅ At least 4 GB free disk space

---

### Step 1 — Clone with Git LFS
```bash
git lfs install
git clone https://github.com/madhavivellanki/multimodal-emotion-recognition.git
cd multimodal-emotion-recognition
git lfs pull
```

---

### Step 2 — Set up virtual environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

---

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

---

### Step 4 — Verify checkpoints are downloaded
```bash
# Windows
dir project\checkpoints\speech
dir project\checkpoints\text

# Linux / macOS
ls project/checkpoints/speech
ls project/checkpoints/text
```

You should see `best_speech_model.pt`, `mfcc_mean.npy`, `mfcc_std.npy`, `best_text_model_glove.pt`.

If these files show as **small placeholder files** (< 1 KB), run:
```bash
git lfs pull
```

---

### Step 5 — Download GloVe vectors (required for text & fusion)
The GloVe file is too large for Git LFS. Download it separately:

```bash
# Download glove.6B.zip (~860 MB) — place glove.6B.100d.txt in project/ folder
# Windows (PowerShell)
Invoke-WebRequest -Uri "https://nlp.stanford.edu/data/glove.6B.zip" -OutFile "glove.6B.zip"
Expand-Archive -Path "glove.6B.zip" -DestinationPath "project" -Force

# Linux / macOS
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt -d project/
```

> **Note:** If GloVe is not available, the speech model still works fully. Only text and fusion predictions require GloVe.

---

### Step 6 — Run the Streamlit demo
```bash
streamlit run demo/app.py
```

Open **http://localhost:8501** — the app will load all models automatically.

---

### Step 7 — Test with TESS audio files
To test the speech model, use audio files from the TESS dataset:
1. Download TESS from [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
2. Unzip to `TESS_data/`
3. In the Streamlit app → **Speech tab** → upload any `.wav` file from a TESS emotion folder
4. The folder name tells you the expected emotion (e.g. `OAF_dog_angry` → should predict **angry**)

---

### Step 8 — Validate architectures (no dataset needed)
```bash
python project/smoke_test.py
```
Expected output: **7 passed, 0 failed**

---

## 📌 Project Overview

This project implements a complete end-to-end **Multimodal Emotion Recognition** system trained on the **Toronto Emotional Speech Set (TESS)**. Three separate pipelines are built and compared:

| Pipeline | Model Architecture | Test Accuracy |
|----------|--------------------|--------------|
| 🎙️ Speech Only | CNN + BiLSTM | **99.88%** |
| ✍️ Text Only | BiLSTM + Transformer Encoder | 28.42% |
| 🔀 Multimodal Fusion | Cross-Modal Attention | 26.99% |

> **Key Finding:** On TESS, emotion is entirely in *how* words are spoken (prosody), not *what* word is spoken. This is why speech achieves 99.88% — and why the text/fusion results are the correct scientific outcome, not a failure.

---

## ✨ Features

- 🎙️ **Speech Emotion Recognition** from WAV audio files using MFCC features
- ✍️ **Text Emotion Recognition** using GloVe-100d word embeddings
- 🔀 **Multimodal Fusion** via cross-modal attention mechanism
- 📊 **Interactive Streamlit UI** with real-time confidence score visualisation
- 📈 **t-SNE cluster visualisations** of learned emotion embeddings
- ✅ **Smoke test** to validate all architectures without dataset
- 💾 **Checkpoints included** via Git LFS — no retraining needed

---

## 🧠 Architecture

### 🎙️ Speech Pipeline
```
WAV Audio
    │
    ▼
Preprocessing
(Resample 16kHz → Trim Silence → Normalise → Z-score MFCC)
    │
    ▼
Feature Extraction
(40 MFCC Coefficients × 400 Time Steps)
    │
    ▼
CNN Encoder
(2 Blocks: 32→64 channels, BatchNorm, MaxPool)
    │
    ▼
BiLSTM Temporal Modelling
(128 hidden units, 2 layers, bidirectional)
    │
    ▼
Dense + Softmax
    │
    ▼
Emotion Prediction  →  angry / disgust / fear / happy / neutral / ps / sad
```

### ✍️ Text Pipeline
```
Input Text
    │
    ▼
Preprocessing
(Lowercase → Clean → Tokenise)
    │
    ▼
GloVe-100d Embeddings
(16 tokens × 100 dimensions)
    │
    ▼
BiLSTM Contextual Modelling
(128 hidden units, 2 layers, bidirectional)
    │
    ▼
Transformer Encoder Layer
(Self-attention, 4 heads, Pre-LayerNorm)
    │
    ▼
Attention Pooling → Dense + Softmax
    │
    ▼
Emotion Prediction
```

### 🔀 Multimodal Fusion Pipeline
```
Speech Embedding (512d)          Text Embedding (256d)
        │                                │
        ▼                                ▼
   Project → 256d               Project → 256d
        │                                │
        ├──── Cross-Modal Attention ──────┤
        │   Speech queries Text          │
        │   Text queries Speech          │
        └──────────────┬─────────────────┘
                       ▼
              Concatenate → 512d
                       │
                       ▼
              Dense + Softmax
                       │
                       ▼
             Emotion Prediction
```

---

## 📂 Project Structure

```
multimodal-emotion-recognition/
│
├── demo/
│   ├── app.py                    # Streamlit live demo UI
│   └── inference.py              # Model loading & prediction logic
│
├── project/
│   ├── checkpoints/              # ← Stored in Git LFS
│   │   ├── speech/
│   │   │   ├── best_speech_model.pt
│   │   │   ├── mfcc_mean.npy
│   │   │   └── mfcc_std.npy
│   │   ├── text/
│   │   │   └── best_text_model_glove.pt
│   │   └── fusion/
│   │       └── best_fusion_model_glove.pt
│   │
│   ├── models/
│   │   ├── speech_pipeline/      # model.py · train.py · test.py
│   │   ├── text_pipeline/        # model.py · train.py · test.py
│   │   └── fusion_pipeline/      # model.py · dataset.py · train.py · test.py
│   │
│   └── utils/
│       ├── data_loader.py        # TESS manifest builder & train/val/test splits
│       ├── speech_preprocessing.py  # Audio loading, MFCC extraction, dataset
│       ├── text_preprocessing.py    # GloVe embeddings, tokenisation
│       ├── metrics.py            # Accuracy, F1, confusion matrix helpers
│       └── plotting.py           # All visualisation functions
│
├── results/
│   ├── accuracy_tables.csv       # Final numerical results
│   └── plots/                    # 13 generated plots
│       ├── curves_speech_pipeline.png
│       ├── cm_speech_only.png
│       ├── tsne_speech_temporal_modelling.png
│       ├── comparison_accuracy.png
│       └── ...
│
├── precompute_features.py        # One-time MFCC cache (makes training ~30x faster)
├── smoke_test.py                 # Architecture validation — no data needed
├── requirements.txt
├── .gitattributes                # Git LFS tracking rules
└── README.md
```

---

## 🏋️ Training Your Own Models (Optional)

> **Checkpoints are already included via Git LFS — you do NOT need to retrain.**

If you want to retrain from scratch:

### Speed Up: Pre-compute MFCCs first
```bash
python project/precompute_features.py \
    --data_dir "TESS_data/TESS Toronto emotional speech set data"
```
This takes ~5 minutes once and reduces each training epoch from 25 min → 30 sec on CPU.

### Speech Pipeline
```bash
python -m project.models.speech_pipeline.train \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --epochs 30 --batch_size 64 --lr 3e-4 \
    --save_dir project/checkpoints/speech
```

### Text Pipeline
```bash
python -m project.models.text_pipeline.train \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --glove_path project/glove.6B.100d.txt \
    --embed_type glove --epochs 30 --batch_size 64 \
    --save_dir project/checkpoints/text
```

### Fusion Pipeline
```bash
python -m project.models.fusion_pipeline.train \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --glove_path project/glove.6B.100d.txt \
    --epochs 30 --batch_size 64 \
    --save_dir project/checkpoints/fusion
```

---

## 🧪 Evaluation

```bash
# Speech — runs on test set and saves plots
python -m project.models.speech_pipeline.test \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --checkpoint project/checkpoints/speech/best_speech_model.pt

# Text
python -m project.models.text_pipeline.test \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --checkpoint project/checkpoints/text/best_text_model_glove.pt \
    --glove_path project/glove.6B.100d.txt

# Fusion
python -m project.models.fusion_pipeline.test \
    --data_dir "TESS_data/TESS Toronto emotional speech set data" \
    --checkpoint project/checkpoints/fusion/best_fusion_model_glove.pt \
    --glove_path project/glove.6B.100d.txt
```

---

## 📊 Results

### Accuracy Comparison

| Pipeline | Test Accuracy | Macro F1 | Weighted F1 | Parameters |
|----------|:------------:|:--------:|:-----------:|:----------:|
| 🎙️ Speech-Only (CNN+BiLSTM) | **99.88%** | **99.88%** | **99.88%** | 1,219,655 |
| ✍️ Text-Only (GloVe+BiLSTM+Transformer) | 28.42% | 21.35% | 21.32% | 1,060,616 |
| 🔀 Multimodal Fusion (Cross-attention) | 26.99% | 20.65% | 20.62% | 6,686,088 |

### Per-Emotion Accuracy — Speech Pipeline

| Emotion | Precision | Recall | F1 | Support |
|---------|:---------:|:------:|:--:|:-------:|
| 😠 angry | 1.00 | 1.00 | 1.00 | 120 |
| 🤢 disgust | 1.00 | 1.00 | 1.00 | 120 |
| 😨 fear | 1.00 | 1.00 | 1.00 | 120 |
| 😊 happy | 0.99 | 1.00 | 1.00 | 120 |
| 😐 neutral | 1.00 | 1.00 | 1.00 | 120 |
| 😲 ps (surprise) | 1.00 | 0.99 | 1.00 | 120 |
| 😢 sad | 1.00 | 1.00 | 1.00 | 121 |
| **Overall** | **1.00** | **1.00** | **1.00** | **841** |

### Generated Plots

| File | Description |
|------|-------------|
| `curves_speech_pipeline.png` | Training & validation loss/accuracy — speech |
| `curves_text_glove.png` | Training & validation loss/accuracy — text |
| `curves_fusion_glove.png` | Training & validation loss/accuracy — fusion |
| `cm_speech_only.png` | Confusion matrix — speech (near-perfect diagonal) |
| `cm_text_glove.png` | Confusion matrix — text |
| `cm_fusion_glove.png` | Confusion matrix — fusion |
| `tsne_speech_temporal_modelling.png` | t-SNE of BiLSTM embeddings — 7 tight clusters |
| `tsne_text_contextual_modelling_glove.png` | t-SNE of Transformer embeddings — overlapping |
| `tsne_fusion_block_glove.png` | t-SNE of fusion block embeddings |
| `per_emotion_speech_only.png` | Per-class accuracy bar chart |
| `comparison_accuracy.png` | All 3 pipelines compared |

---

## 🖥️ Streamlit Live Demo

```bash
streamlit run demo/app.py
```

| Tab | Description |
|-----|-------------|
| 🎙️ **Speech** | Upload any TESS `.wav` file → CNN+BiLSTM predicts emotion with confidence bars. Auto-detects true label from filename. |
| ✍️ **Text** | Type a word or sentence → BiLSTM+Transformer predicts emotion. |
| 🔀 **Fusion** | Upload audio + type text → compare all three pipelines side by side. |

---

## 📚 Dataset

### Toronto Emotional Speech Set (TESS)

| Property | Value |
|----------|-------|
| Total samples | 5,600 |
| Emotion classes | 7 |
| Speakers | 2 female actors (aged 26 & 64) |
| Words | 200 target words |
| Format | WAV audio + transcripts |

| Index | Emotion | Emoji |
|-------|---------|-------|
| 0 | Angry | 😠 |
| 1 | Disgust | 🤢 |
| 2 | Fear | 😨 |
| 3 | Happy | 😊 |
| 4 | Neutral | 😐 |
| 5 | Pleasant Surprise | 😲 |
| 6 | Sad | 😢 |

**Download:** https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch 2.3, TorchAudio |
| Audio Processing | Librosa, MFCC |
| NLP | GloVe-100d Embeddings, Transformer |
| Visualisation | Matplotlib, Seaborn, t-SNE |
| Deployment | Streamlit |
| Version Control | Git, GitHub, Git LFS |
| Language | Python 3.11 |

---

## 🔬 Key Findings

1. **Speech dominates (99.88%)** — TESS emotions are in *how* words are spoken (pitch, energy, tempo), not *what* word is said. MFCC features perfectly capture this.

2. **Text accuracy (28.42%) is scientifically correct** — GloVe embeddings of emotionally-neutral words like "dog" and "bear" carry no emotion-discriminative signal regardless of the emotion.

3. **MFCC z-score normalisation was the critical fix** — without it, the model was stuck at 14% (random chance) for 30 epochs. With normalisation, it reached 99%+ in 13 epochs.

4. **Fusion did not help here** — cross-modal attention requires both modalities to carry signal. With near-random text, the attention corrupts the speech embedding. On full-sentence datasets (IEMOCAP), fusion would outperform unimodal models by 3–8%.

---

## 🔮 Future Work

- Test on IEMOCAP / CMU-MOSI (full-sentence datasets) where text carries emotion signal
- Replace GloVe with sentence-level BERT embeddings
- Implement late fusion (average softmax outputs) as a robust gating alternative
- Add wav2vec 2.0 / HuBERT as self-supervised speech encoder
- Deploy on HuggingFace Spaces

---

## 📜 Citation

```bibtex
@dataset{tess2010,
  author    = {Dupuis, K. and Pichora-Fuller, M. K.},
  title     = {Toronto Emotional Speech Set (TESS)},
  year      = {2010},
  publisher = {Scholars Portal Dataverse},
  doi       = {10.5683/SP2/E8H2MF},
  url       = {https://doi.org/10.5683/SP2/E8H2MF}
}
```

---

## 👩‍💻 Author

**Madhavi Vellanki**
B.Tech Engineering Student · AI / ML Enthusiast · Deep Learning & NLP

[![GitHub](https://img.shields.io/badge/GitHub-madhavivellanki-181717?style=flat-square&logo=github)](https://github.com/madhavivellanki)

---

<div align="center">
  <sub>
    Built with PyTorch · Trained on TESS · Deployed with Streamlit
    <br>
    B.Tech Major Project · Multimodal Emotion Recognition · 2026
  </sub>
</div>
