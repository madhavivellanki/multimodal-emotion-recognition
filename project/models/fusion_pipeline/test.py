import os, sys, argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from utils.data_loader             import build_manifest, split_manifest
from utils.text_preprocessing      import GloVeEmbeddings, BERTEmbedder
from utils.metrics                 import print_report, get_confusion_matrix
from utils.plotting                import (plot_confusion_matrix,
                                           plot_per_emotion_accuracy,
                                           plot_tsne, plot_comparison)
from models.fusion_pipeline.dataset import MultimodalDataset
from models.fusion_pipeline.model   import build_fusion_model_glove, build_fusion_model_bert


def evaluate_fusion(model, loader, device, return_embeddings=False):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    all_preds, all_labels, all_embeds = [], [], []
    with torch.no_grad():
        for mfcc, text_emb, labels in loader:
            mfcc, text_emb = mfcc.to(device), text_emb.to(device)
            if return_embeddings:
                logits, embed = model(mfcc, text_emb, return_embedding=True)
                all_embeds.extend(embed.cpu().numpy())
            else:
                logits = model(mfcc, text_emb)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labels.tolist())
    acc = accuracy_score(all_labels, all_preds)
    mf1 = f1_score(all_labels, all_preds, average="macro",    zero_division=0)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    embeds = np.array(all_embeds) if return_embeddings else None
    return {"accuracy": acc, "macro_f1": mf1, "weighted_f1": wf1}, all_preds, all_labels, embeds


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[test] Device: {device}  |  Embedding: {args.embed_type.upper()}")

    manifest_path = os.path.join(args.save_dir, "test_manifest.csv")
    if os.path.exists(manifest_path):
        test_df = pd.read_csv(manifest_path)
        print(f"[test] Loaded test manifest ({len(test_df)} samples)")
    else:
        manifest = build_manifest(args.data_dir)
        _, _, test_df = split_manifest(manifest)

    mean_path = os.path.join(args.save_dir, "mfcc_mean.npy")
    std_path  = os.path.join(args.save_dir, "mfcc_std.npy")
    mean = np.load(mean_path)
    std  = np.load(std_path)

    if args.embed_type == "glove":
        embedder = GloVeEmbeddings(args.glove_path)
        model    = build_fusion_model_glove(device)
    else:
        embedder = BERTEmbedder(device=device)
        model    = build_fusion_model_bert(device)

    test_set    = MultimodalDataset(test_df, embedder, mean=mean, std=std)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[test] Loaded checkpoint  epoch={ckpt['epoch']}  val_acc={ckpt['val_acc']:.4f}")

    metrics, preds, labels, embeddings = evaluate_fusion(model, test_loader, device, return_embeddings=True)

    print(f"\n[test] Test Accuracy  : {metrics['accuracy']:.4f}")
    print(f"[test] Macro F1       : {metrics['macro_f1']:.4f}")
    print(f"[test] Weighted F1    : {metrics['weighted_f1']:.4f}")
    print_report(labels, preds, title=f"Multimodal Fusion (GLOVE)")

    tag = f"fusion_{args.embed_type}"
    cm = get_confusion_matrix(labels, preds)
    plot_confusion_matrix(cm,              title=tag)
    plot_per_emotion_accuracy(labels, preds, title=tag)
    plot_tsne(embeddings, labels,          title=f"fusion_block_{args.embed_type}")

    os.makedirs("results", exist_ok=True)
    pipeline_name = f"Multimodal ({args.embed_type.upper()})"
    result_row = {"pipeline": pipeline_name,
                  "accuracy": round(metrics["accuracy"], 4),
                  "macro_f1": round(metrics["macro_f1"], 4),
                  "weighted_f1": round(metrics["weighted_f1"], 4)}
    result_path = "results/accuracy_tables.csv"
    if os.path.exists(result_path):
        df = pd.read_csv(result_path)
        df = df[df["pipeline"] != pipeline_name]
        df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        df = pd.DataFrame([result_row])
    df.to_csv(result_path, index=False)

    print(f"\n[test] Final Results Table:")
    print(df.to_string(index=False))

    if len(df) >= 3:
        comparison = {row["pipeline"]: row["accuracy"] for _, row in df.iterrows()}
        plot_comparison(comparison)
        print("[test] Comparison chart saved ? results/plots/comparison_accuracy.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--checkpoint", default="checkpoints/fusion/best_fusion_model_glove.pt")
    parser.add_argument("--glove_path", default="glove.6B.100d.txt")
    parser.add_argument("--embed_type", choices=["glove","bert"], default="glove")
    parser.add_argument("--save_dir",   default="checkpoints/fusion")
    main(parser.parse_args())
