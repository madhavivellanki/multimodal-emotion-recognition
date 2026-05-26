"""
utils/metrics.py
Evaluation helpers: accuracy, F1, confusion matrix, and pretty printing.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)
from utils.data_loader import IDX_TO_EMOTION, NUM_CLASSES


def evaluate(model, dataloader, device, return_embeddings: bool = False):
    """
    Run *model* on *dataloader* and collect predictions, labels,
    and optionally intermediate embeddings (for t-SNE / UMAP plots).

    Returns
    -------
    metrics : dict  – accuracy, macro_f1, weighted_f1
    all_preds  : list[int]
    all_labels : list[int]
    all_embeds : list[np.ndarray] | None
    """
    model.eval()
    all_preds, all_labels, all_embeds = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if return_embeddings:
                # Model must accept return_embedding=True kwarg
                *inputs, labels = batch
                logits, embed = model(*[x.to(device) for x in inputs],
                                      return_embedding=True)
                all_embeds.extend(embed.cpu().numpy())
            else:
                *inputs, labels = batch
                logits = model(*[x.to(device) for x in inputs])

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc  = accuracy_score(all_labels, all_preds)
    mf1  = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    wf1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    metrics = {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1}

    if return_embeddings:
        return metrics, all_preds, all_labels, np.array(all_embeds)
    return metrics, all_preds, all_labels, None


def print_report(all_labels, all_preds, title: str = ""):
    """Print sklearn classification report with emotion names."""
    label_names = [IDX_TO_EMOTION[i] for i in range(NUM_CLASSES)]
    print(f"\n{'='*60}")
    if title:
        print(f"  {title}")
        print(f"{'='*60}")
    print(classification_report(all_labels, all_preds,
                                 target_names=label_names, zero_division=0))


def get_confusion_matrix(all_labels, all_preds) -> np.ndarray:
    return confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
