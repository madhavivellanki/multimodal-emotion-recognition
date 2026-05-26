"""
models/text_pipeline/test.py
Evaluate the best Text (BiLSTM + Transformer) checkpoint on held-out test set.

Usage
-----
  python -m models.text_pipeline.test \
         --data_dir   /path/to/TESS_data \
         --checkpoint checkpoints/text/best_text_model_glove.pt \
         --glove_path /path/to/glove.6B.100d.txt \
         --embed_type glove \
         --save_dir   checkpoints/text
"""

import os
import sys
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader          import build_manifest, split_manifest
from utils.text_preprocessing   import (GloVeEmbeddings, BERTEmbedder,
                                         GloVeTextDataset, BERTTextDataset)
from utils.metrics              import evaluate, print_report, get_confusion_matrix
from utils.plotting             import (plot_confusion_matrix,
                                        plot_per_emotion_accuracy,
                                        plot_tsne)
from models.text_pipeline.model import build_text_model_glove, build_text_model_bert


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}  |  Embedding: {args.embed_type.upper()}")

    # ── Load test manifest ─────────────────────────────────────────────────
    manifest_path = os.path.join(args.save_dir, "test_manifest.csv")
    if os.path.exists(manifest_path):
        test_df = pd.read_csv(manifest_path)
        print(f"[test] Loaded test manifest ({len(test_df)} samples)")
    else:
        print("[test] Rebuilding split …")
        manifest = build_manifest(args.data_dir)
        _, _, test_df = split_manifest(manifest)

    # ── Embeddings & dataset ───────────────────────────────────────────────
    if args.embed_type == "glove":
        embedder  = GloVeEmbeddings(args.glove_path)
        test_set  = GloVeTextDataset(test_df, embedder)
        model     = build_text_model_glove(device)
    else:
        embedder  = BERTEmbedder(device=device)
        test_set  = BERTTextDataset(test_df, embedder)
        model     = build_text_model_bert(device)

    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    # ── Load checkpoint ────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[test] Loaded checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # ── Evaluate ───────────────────────────────────────────────────────────
    metrics, preds, labels, embeddings = evaluate(
        model, test_loader, device, return_embeddings=True
    )

    print(f"\n[test] Test Accuracy : {metrics['accuracy']:.4f}")
    print(f"[test] Macro  F1     : {metrics['macro_f1']:.4f}")
    print(f"[test] Weighted F1   : {metrics['weighted_f1']:.4f}")
    print_report(labels, preds, title=f"Text-Only ({args.embed_type.upper()}) Pipeline")

    # ── Plots ──────────────────────────────────────────────────────────────
    tag = f"text_{args.embed_type}"
    cm = get_confusion_matrix(labels, preds)
    plot_confusion_matrix(cm,        title=tag)
    plot_per_emotion_accuracy(labels, preds, title=tag)
    plot_tsne(embeddings, labels,    title=f"text_contextual_modelling_{args.embed_type}")

    # ── Save results ───────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    pipeline_name = f"Text-Only ({args.embed_type.upper()})"
    result_row = {
        "pipeline":    pipeline_name,
        "accuracy":    round(metrics["accuracy"],    4),
        "macro_f1":    round(metrics["macro_f1"],    4),
        "weighted_f1": round(metrics["weighted_f1"], 4),
    }
    result_path = "results/accuracy_tables.csv"
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        df = df[df["pipeline"] != pipeline_name]
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    df.to_csv(result_path, index=False)
    print(f"\n[test] Results saved → {result_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Text Emotion Model")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--checkpoint", default="checkpoints/text/best_text_model_glove.pt")
    parser.add_argument("--glove_path", default="glove.6B.100d.txt")
    parser.add_argument("--embed_type", choices=["glove", "bert"], default="glove")
    parser.add_argument("--save_dir",   default="checkpoints/text")
    main(parser.parse_args())
