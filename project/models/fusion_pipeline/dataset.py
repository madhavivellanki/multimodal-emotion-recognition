import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.speech_preprocessing import load_and_preprocess, extract_mfcc, N_MFCC
from utils.text_preprocessing   import GloVeEmbeddings, BERTEmbedder, MAX_TEXT_LEN

class MultimodalDataset(Dataset):
    def __init__(self, manifest_df, embedder, mean=None, std=None, fit=False):
        self.df = manifest_df.reset_index(drop=True)
        self.embedder = embedder
        self.is_bert  = isinstance(embedder, BERTEmbedder)
        self.npy_paths = []
        for _, row in self.df.iterrows():
            npy = row["wav_path"].replace(".wav", "_mfcc.npy")
            if not os.path.exists(npy):
                audio = load_and_preprocess(row["wav_path"])
                np.save(npy, extract_mfcc(audio))
            self.npy_paths.append(npy)
        if fit:
            print("[MultimodalDataset] Computing MFCC mean/std ...")
            all_mfcc  = np.stack([np.load(p) for p in self.npy_paths])
            self.mean = all_mfcc.mean(axis=(0,1), keepdims=True)
            self.std  = all_mfcc.std(axis=(0,1),  keepdims=True) + 1e-8
        else:
            self.mean = mean
            self.std  = std

    def __len__(self): return len(self.npy_paths)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        mfcc = np.load(self.npy_paths[idx]).astype(np.float32)
        mfcc = (mfcc - self.mean.squeeze()) / self.std.squeeze()
        if self.is_bert:
            text_emb = self.embedder.embed(row["transcript"])
        else:
            text_emb = self.embedder.sentence_matrix(row["transcript"], max_tokens=MAX_TEXT_LEN)
        label = int(row["label"])
        return (torch.tensor(mfcc, dtype=torch.float32),
                torch.tensor(text_emb, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))
