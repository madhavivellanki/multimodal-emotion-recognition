"""
utils/plotting.py
Plotting utilities used across all three pipelines:
  - Training curves (loss & accuracy)
  - Confusion matrix heatmap
  - t-SNE emotion cluster scatter
  - Per-emotion accuracy bar chart
  - Accuracy comparison table / bar chart
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils.data_loader import IDX_TO_EMOTION, NUM_CLASSES

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

EMOTION_NAMES = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
PALETTE       = sns.color_palette("tab10", NUM_CLASSES)


# ── Training curves ───────────────────────────────────────────────────────────
def plot_training_curves(train_losses, val_losses,
                         train_accs,  val_accs,
                         title: str = "training", save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, label="Train Loss",   color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",     color="tomato")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Loss – {title}"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_accs, label="Train Acc",  color="steelblue")
    axes[1].plot(epochs, val_accs,   label="Val Acc",    color="tomato")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"Accuracy – {title}"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"curves_{title.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── Confusion matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str = "confusion",
                          save: bool = True):
    fig, ax = plt.subplots(figsize=(8, 7))
    # Normalise rows for readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=EMOTION_NAMES, yticklabels=EMOTION_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {title}")
    fig.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"cm_{title.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── t-SNE emotion cluster visualisation ──────────────────────────────────────
def plot_tsne(embeddings: np.ndarray, labels,
              title: str = "tsne", save: bool = True):
    """
    Reduce *embeddings* to 2-D with t-SNE and scatter-plot by emotion.
    embeddings: (N, D)
    labels:     (N,) integer class indices
    """
    print(f"[t-SNE] Fitting on {embeddings.shape[0]} samples …")
    tsne = TSNE(n_components=2, random_state=42,
                perplexity=min(30, len(labels) // 4),
                max_iter=1000)
    coords = tsne.fit_transform(embeddings)   # (N, 2)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls_idx in range(NUM_CLASSES):
        mask = np.array(labels) == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=EMOTION_NAMES[cls_idx],
                   color=PALETTE[cls_idx], alpha=0.7, s=25)
    ax.legend(markerscale=1.5, fontsize=9)
    ax.set_title(f"t-SNE Emotion Clusters – {title}")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    fig.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"tsne_{title.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── Per-emotion accuracy bar chart ────────────────────────────────────────────
def plot_per_emotion_accuracy(all_labels, all_preds,
                              title: str = "per_emotion", save: bool = True):
    from sklearn.metrics import confusion_matrix
    cm  = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    per = cm.diagonal() / (cm.sum(axis=1) + 1e-8)

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(EMOTION_NAMES, per, color=PALETTE)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class Accuracy")
    ax.set_title(f"Per-Emotion Accuracy – {title}")
    for bar, v in zip(bars, per):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, f"per_emotion_{title.replace(' ', '_')}.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved → {path}")
    plt.close(fig)


# ── Accuracy comparison bar chart ────────────────────────────────────────────
def plot_comparison(results: dict, save: bool = True):
    """
    results = { "Speech-only": 0.82, "Text-only": 0.75, "Multimodal": 0.91 }
    """
    names  = list(results.keys())
    values = [results[k] for k in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, values, color=["steelblue", "darkorange", "seagreen"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy Comparison: Speech vs Text vs Multimodal")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "comparison_accuracy.png")
        fig.savefig(path, dpi=150)
        print(f"[plot] Saved → {path}")
    plt.close(fig)
