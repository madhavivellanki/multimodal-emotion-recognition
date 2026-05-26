"""
models/text_pipeline/train.py
Train the BiLSTM + Transformer text emotion model on TESS.

Usage
-----
  # Using GloVe embeddings (default, faster):
  python -m models.text_pipeline.train \
         --data_dir   /path/to/TESS_data \
         --glove_path /path/to/glove.6B.100d.txt \
         --embed_type glove \
         --epochs 40 \
         --save_dir checkpoints/text

  # Using BERT embeddings (slower, usually more accurate):
  python -m models.text_pipeline.train \
         --data_dir   /path/to/TESS_data \
         --embed_type bert \
         --epochs 40 \
         --save_dir checkpoints/text_bert
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader         import build_manifest, split_manifest
from utils.text_preprocessing  import (GloVeEmbeddings, BERTEmbedder,
                                        GloVeTextDataset, BERTTextDataset)
from utils.plotting            import plot_training_curves
from models.text_pipeline.model import build_text_model_glove, build_text_model_bert


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for emb, labels in tqdm(loader, desc="  train", leave=False):
        emb, labels = emb.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(emb)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for emb, labels in loader:
        emb, labels = emb.to(device), labels.to(device)
        logits = model(emb)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=-1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}  |  Embedding: {args.embed_type.upper()}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ───────────────────────────────────────────────────────────────
    manifest               = build_manifest(args.data_dir)
    train_df, val_df, test_df = split_manifest(manifest)

    # Persist test split
    test_df.to_csv(os.path.join(args.save_dir, "test_manifest.csv"), index=False)

    # Choose embedding backend
    if args.embed_type == "glove":
        embedder     = GloVeEmbeddings(args.glove_path)
        train_set    = GloVeTextDataset(train_df, embedder)
        val_set      = GloVeTextDataset(val_df,   embedder)
        model        = build_text_model_glove(device)
        ckpt_name    = "best_text_model_glove.pt"
        curve_title  = "text_glove"
    else:   # bert
        embedder     = BERTEmbedder(device=device)
        train_set    = BERTTextDataset(train_df, embedder)
        val_set      = BERTTextDataset(val_df,   embedder)
        model        = build_text_model_bert(device)
        ckpt_name    = "best_text_model_bert.pt"
        curve_title  = "text_bert"

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0)

    # ── Training setup ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6, verbose=True
    )

    best_val_acc              = 0.0
    train_losses, val_losses  = [], []
    train_accs,   val_accs    = [], []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate_epoch(model,  val_loader,   criterion, device)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)

        scheduler.step(vl_acc)

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt_path    = os.path.join(args.save_dir, ckpt_name)
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "val_acc":    vl_acc,
                "embed_type": args.embed_type,
                "args":       vars(args),
            }, ckpt_path)
            print(f"  ✓ New best val_acc={vl_acc:.4f} → saved to {ckpt_path}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs,
                         title=curve_title)

    print(f"\n[train] Done.  Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Text Emotion Model")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--glove_path", default="glove.6B.100d.txt",
                        help="Path to GloVe .txt file (only needed for --embed_type glove)")
    parser.add_argument("--embed_type", choices=["glove", "bert"], default="glove")
    parser.add_argument("--epochs",     type=int,   default=40)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--save_dir",   default="checkpoints/text")
    main(parser.parse_args())
