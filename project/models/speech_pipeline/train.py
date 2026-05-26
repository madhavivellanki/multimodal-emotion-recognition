"""
models/speech_pipeline/train.py - v2 with MFCC normalisation + OneCycleLR
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader          import build_manifest, split_manifest
from utils.speech_preprocessing import load_and_preprocess, extract_mfcc, MAX_TIME_STEPS, N_MFCC
from utils.plotting             import plot_training_curves


class NormSpeechDataset(Dataset):
    def __init__(self, manifest_df, mean=None, std=None, fit=False):
        self.df = manifest_df.reset_index(drop=True)
        self.npy_paths = []
        for _, row in self.df.iterrows():
            npy = row["wav_path"].replace(".wav", "_mfcc.npy")
            if not os.path.exists(npy):
                audio = load_and_preprocess(row["wav_path"])
                np.save(npy, extract_mfcc(audio))
            self.npy_paths.append(npy)
        if fit:
            print("[Dataset] Computing MFCC mean/std on training set ...")
            all_mfcc = np.stack([np.load(p) for p in self.npy_paths])
            self.mean = all_mfcc.mean(axis=(0,1), keepdims=True)
            self.std  = all_mfcc.std(axis=(0,1),  keepdims=True) + 1e-8
            print(f"[Dataset] mean={self.mean.mean():.3f}  std={self.std.mean():.3f}")
        else:
            self.mean = mean
            self.std  = std

    def __len__(self): return len(self.npy_paths)

    def __getitem__(self, idx):
        mfcc  = np.load(self.npy_paths[idx]).astype(np.float32)  # (T, F)
        norm  = (mfcc - self.mean.squeeze()) / self.std.squeeze() # both now (F,)
        label = int(self.df.iloc[idx]["label"])
        return torch.tensor(norm, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SpeechEmotionModelV2(nn.Module):
    def __init__(self, n_mfcc=N_MFCC, num_classes=7, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2), nn.Dropout2d(0.1),
        )
        lstm_in = 64 * (n_mfcc // 4)
        self.lstm = nn.LSTM(lstm_in, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(256, 64), nn.ReLU(True), nn.Linear(64, num_classes)
        )

    def forward(self, x, return_embedding=False):
        B, T, F = x.shape
        x = self.cnn(x.unsqueeze(1))
        C, T2, F2 = x.shape[1], x.shape[2], x.shape[3]
        x, _ = self.lstm(x.permute(0,2,1,3).reshape(B, T2, C*F2))
        emb = x[:, -1, :]
        logits = self.classifier(emb)
        return (logits, emb) if return_embedding else logits


def train_one_epoch(model, loader, opt, criterion, sched, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for mfcc, labels in tqdm(loader, desc="  train", leave=False):
        mfcc, labels = mfcc.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(mfcc)
        loss = criterion(logits, labels)
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
    for mfcc, labels in loader:
        mfcc, labels = mfcc.to(device), labels.to(device)
        logits = model(mfcc)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * len(labels)
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += len(labels)
    return loss_sum/total, correct/total


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    manifest = build_manifest(args.data_dir)
    train_df, val_df, test_df = split_manifest(manifest)
    test_df.to_csv(os.path.join(args.save_dir, "test_manifest.csv"), index=False)

    train_set = NormSpeechDataset(train_df, fit=True)
    mean, std = train_set.mean, train_set.std
    np.save(os.path.join(args.save_dir, "mfcc_mean.npy"), mean)
    np.save(os.path.join(args.save_dir, "mfcc_std.npy"),  std)
    val_set = NormSpeechDataset(val_df, mean=mean, std=std)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpeechEmotionModelV2().to(device)
    print(f"[model] Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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
            ckpt = os.path.join(args.save_dir, "best_speech_model.pt")
            torch.save({"epoch": epoch, "state_dict": model.state_dict(), "val_acc": vl_acc, "args": vars(args)}, ckpt)
            print(f"  ✓ New best val_acc={vl_acc:.4f} → {ckpt}")

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, title="speech_pipeline")
    print(f"\n[train] Done.  Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--save_dir",   default="checkpoints/speech")
    main(parser.parse_args())
