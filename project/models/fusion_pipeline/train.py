import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader             import build_manifest, split_manifest
from utils.text_preprocessing      import GloVeEmbeddings, BERTEmbedder
from utils.plotting                import plot_training_curves
from models.fusion_pipeline.dataset import MultimodalDataset
from models.fusion_pipeline.model   import build_fusion_model_glove, build_fusion_model_bert


def train_one_epoch(model, loader, opt, criterion, sched, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for mfcc, text_emb, labels in tqdm(loader, desc="  train", leave=False):
        mfcc, text_emb, labels = mfcc.to(device), text_emb.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(mfcc, text_emb)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        loss_sum += loss.item() * len(labels)
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += len(labels)
    return loss_sum/total, correct/total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for mfcc, text_emb, labels in loader:
        mfcc, text_emb, labels = mfcc.to(device), text_emb.to(device), labels.to(device)
        logits = model(mfcc, text_emb)
        loss   = criterion(logits, labels)
        loss_sum += loss.item() * len(labels)
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += len(labels)
    return loss_sum/total, correct/total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}  |  Embedding: {args.embed_type.upper()}")
    os.makedirs(args.save_dir, exist_ok=True)

    manifest = build_manifest(args.data_dir)
    train_df, val_df, test_df = split_manifest(manifest)
    test_df.to_csv(os.path.join(args.save_dir, "test_manifest.csv"), index=False)

    if args.embed_type == "glove":
        embedder  = GloVeEmbeddings(args.glove_path)
        model     = build_fusion_model_glove(device)
        ckpt_name = "best_fusion_model_glove.pt"
        title     = "fusion_glove"
    else:
        embedder  = BERTEmbedder(device=device)
        model     = build_fusion_model_bert(device)
        ckpt_name = "best_fusion_model_bert.pt"
        title     = "fusion_bert"

    train_set = MultimodalDataset(train_df, embedder, fit=True)
    mean, std = train_set.mean, train_set.std
    np.save(os.path.join(args.save_dir, "mfcc_mean.npy"), mean)
    np.save(os.path.join(args.save_dir, "mfcc_std.npy"),  std)
    val_set = MultimodalDataset(val_df, embedder, mean=mean, std=std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1
    )

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion, device)
        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);    val_accs.append(vl_acc)
        print(f"Epoch {epoch:03d}/{args.epochs}  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}")
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            ckpt = os.path.join(args.save_dir, ckpt_name)
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": vl_acc, "embed_type": args.embed_type,
                        "args": vars(args)}, ckpt)
            print(f"  ? New best val_acc={vl_acc:.4f} ? {ckpt}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, title=title)
    print(f"\n[train] Done.  Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--glove_path", default="glove.6B.100d.txt")
    parser.add_argument("--embed_type", choices=["glove","bert"], default="glove")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save_dir",   default="checkpoints/fusion")
    main(parser.parse_args())
