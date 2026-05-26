"""
models/speech_pipeline/test.py
Evaluate the best Speech checkpoint on the held-out test set.
Uses the same SpeechEmotionModelV2 and NormSpeechDataset as train.py.
"""
import os, sys, argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader           import build_manifest, split_manifest, NUM_CLASSES
from utils.speech_preprocessing  import load_and_preprocess, extract_mfcc, N_MFCC
from utils.metrics               import print_report, get_confusion_matrix
from utils.plotting              import (plot_confusion_matrix,
                                         plot_per_emotion_accuracy, plot_tsne)


# ── Same dataset class as train.py ────────────────────────────────────────────
class NormSpeechDataset(Dataset):
    def __init__(self, manifest_df, mean, std):
        self.df = manifest_df.reset_index(drop=True)
        self.npy_paths = []
        for _, row in self.df.iterrows():
            npy = row["wav_path"].replace(".wav", "_mfcc.npy")
            if not os.path.exists(npy):
                audio = load_and_preprocess(row["wav_path"])
                np.save(npy, extract_mfcc(audio))
            self.npy_paths.append(npy)
        self.mean = mean
        self.std  = std

    def __len__(self): return len(self.npy_paths)

    def __getitem__(self, idx):
        mfcc  = np.load(self.npy_paths[idx]).astype(np.float32)
        norm  = (mfcc - self.mean.squeeze()) / self.std.squeeze()
        label = int(self.df.iloc[idx]["label"])
        return torch.tensor(norm, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ── Same model class as train.py ──────────────────────────────────────────────
import torch.nn as nn

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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}")

    # ── Load test manifest ────────────────────────────────────────────────
    manifest_path = os.path.join(args.save_dir, "test_manifest.csv")
    if os.path.exists(manifest_path):
        test_df = pd.read_csv(manifest_path)
        print(f"[test] Loaded test manifest ({len(test_df)} samples)")
    else:
        manifest = build_manifest(args.data_dir)
        _, _, test_df = split_manifest(manifest)

    # ── Load normalisation stats saved during training ────────────────────
    mean_path = os.path.join(args.save_dir, "mfcc_mean.npy")
    std_path  = os.path.join(args.save_dir, "mfcc_std.npy")
    if not os.path.exists(mean_path):
        raise FileNotFoundError(
            f"mfcc_mean.npy not found in {args.save_dir}. "
            "Re-run train.py first to generate normalisation stats."
        )
    mean = np.load(mean_path)
    std  = np.load(std_path)
    print(f"[test] Loaded MFCC stats: mean={mean.mean():.3f}, std={std.mean():.3f}")

    # ── Dataset & loader ──────────────────────────────────────────────────
    test_set    = NormSpeechDataset(test_df, mean, std)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    # ── Load model ────────────────────────────────────────────────────────
    model = SpeechEmotionModelV2().to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[test] Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    # ── Evaluate ──────────────────────────────────────────────────────────
    model.eval()
    all_preds, all_labels, all_embeds = [], [], []
    with torch.no_grad():
        for mfcc, labels in test_loader:
            mfcc = mfcc.to(device)
            logits, emb = model(mfcc, return_embedding=True)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())
            all_embeds.extend(emb.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n[test] Test Accuracy  : {acc:.4f}")
    print(f"[test] Macro F1       : {mf1:.4f}")
    print(f"[test] Weighted F1    : {wf1:.4f}")
    print_report(all_labels, all_preds, title="Speech-Only Pipeline")

    # ── Plots ─────────────────────────────────────────────────────────────
    cm = get_confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm,              title="speech_only")
    plot_per_emotion_accuracy(all_labels, all_preds, title="speech_only")
    plot_tsne(np.array(all_embeds), all_labels, title="speech_temporal_modelling")

    # ── Save results CSV ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    result_row = {"pipeline": "Speech-Only",
                  "accuracy": round(acc, 4),
                  "macro_f1": round(mf1, 4),
                  "weighted_f1": round(wf1, 4)}
    result_path = "results/accuracy_tables.csv"
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        df = df[df["pipeline"] != "Speech-Only"]
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    df.to_csv(result_path, index=False)
    print(f"\n[test] Results saved → {result_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--checkpoint", default="checkpoints/speech/best_speech_model.pt")
    parser.add_argument("--save_dir",   default="checkpoints/speech")
    main(parser.parse_args())
