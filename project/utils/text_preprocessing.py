"""
utils/text_preprocessing.py
Text preprocessing, tokenisation, and embedding lookup utilities.

Strategy
--------
TESS transcripts are *single words* (e.g. "dog", "bear"), so we use a
simple character-level + word-level pipeline.  Two embedding backends are
supported:
  1. GloVe (default)  – 100-d vectors from glove.6B, fast & CPU-friendly.
  2. BERT             – contextual embeddings via HuggingFace transformers.

The DataLoader transparently switches between both.
"""

import re
import os
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Text cleaning ─────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── GloVe embedding loader ────────────────────────────────────────────────────
class GloVeEmbeddings:
    """
    Loads GloVe vectors from a text file.
    Falls back to random 100-d vectors for OOV words.
    """
    DIM = 100

    def __init__(self, glove_path: str):
        if not os.path.isfile(glove_path):
            raise FileNotFoundError(
                f"GloVe file not found at {glove_path}.\n"
                "Download with:\n"
                "  wget http://nlp.stanford.edu/data/glove.6B.zip\n"
                "  unzip glove.6B.zip"
            )
        self.word2vec = {}
        print(f"[GloVe] Loading vectors from {glove_path} …")
        with open(glove_path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                word  = parts[0]
                vec   = np.array(parts[1:], dtype=np.float32)
                self.word2vec[word] = vec
        print(f"[GloVe] Loaded {len(self.word2vec):,} word vectors (dim={self.DIM}).")

    def embed(self, word: str) -> np.ndarray:
        """Return 100-d vector for *word*, or a zero vector if OOV."""
        return self.word2vec.get(word.lower(), np.zeros(self.DIM, dtype=np.float32))

    def sentence_matrix(self, text: str, max_tokens: int = 16) -> np.ndarray:
        """
        Tokenise *text* and stack per-word vectors into (max_tokens, DIM).
        Pads with zeros for short sentences, truncates long ones.
        """
        words  = clean_text(text).split()[:max_tokens]
        matrix = np.zeros((max_tokens, self.DIM), dtype=np.float32)
        for i, w in enumerate(words):
            matrix[i] = self.embed(w)
        return matrix    # (max_tokens, DIM)


# ── BERT tokeniser / embedding ────────────────────────────────────────────────
class BERTEmbedder:
    """
    Uses a pre-trained BERT model to produce a (seq_len, 768) embedding matrix.
    Only the last hidden state is used.
    """
    MODEL_NAME  = "bert-base-uncased"
    MAX_LEN     = 16
    HIDDEN_DIM  = 768

    def __init__(self, device: str = "cpu"):
        from transformers import BertTokenizer, BertModel
        self.device    = device
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = BertModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()
        print(f"[BERT] Model loaded on {device}.")

    @torch.no_grad()
    def embed(self, text: str) -> np.ndarray:
        """Return (MAX_LEN, HIDDEN_DIM) numpy array."""
        text = clean_text(text)
        enc  = self.tokenizer(
            text,
            max_length=self.MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out  = self.model(**enc)
        # last_hidden_state: (1, seq, hidden)
        return out.last_hidden_state.squeeze(0).cpu().numpy()


# ── Simple word → integer vocabulary ─────────────────────────────────────────
class Vocabulary:
    """
    Builds a word→int lookup from a list of texts.
    Used internally by GloVeTextDataset.
    """
    PAD = 0
    UNK = 1

    def __init__(self):
        self.word2idx = {"<PAD>": self.PAD, "<UNK>": self.UNK}
        self.idx2word = {self.PAD: "<PAD>", self.UNK: "<UNK>"}

    def build(self, texts):
        for text in texts:
            for w in clean_text(text).split():
                if w not in self.word2idx:
                    idx = len(self.word2idx)
                    self.word2idx[w] = idx
                    self.idx2word[idx] = w
        print(f"[Vocab] {len(self.word2idx)} unique tokens.")

    def encode(self, text: str, max_len: int = 16) -> np.ndarray:
        words = clean_text(text).split()[:max_len]
        ids   = [self.word2idx.get(w, self.UNK) for w in words]
        # Pad to max_len
        ids  += [self.PAD] * (max_len - len(ids))
        return np.array(ids, dtype=np.int64)

    def __len__(self):
        return len(self.word2idx)


# ── PyTorch Datasets ──────────────────────────────────────────────────────────
MAX_TEXT_LEN = 16   # TESS words are always 1 token; keep room for generality


class GloVeTextDataset(Dataset):
    """
    Returns (embedding_matrix, label).
    embedding_matrix shape: (MAX_TEXT_LEN, GloVeEmbeddings.DIM) = (16, 100)
    """
    def __init__(self, manifest_df, glove: GloVeEmbeddings):
        self.df    = manifest_df.reset_index(drop=True)
        self.glove = glove

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        mat   = self.glove.sentence_matrix(row["transcript"], max_tokens=MAX_TEXT_LEN)
        label = int(row["label"])
        return torch.tensor(mat), torch.tensor(label, dtype=torch.long)


class BERTTextDataset(Dataset):
    """
    Returns (bert_embedding, label).
    bert_embedding shape: (MAX_TEXT_LEN, BERTEmbedder.HIDDEN_DIM)
    NOTE: embeddings are computed on-the-fly; cache them for speed if needed.
    """
    def __init__(self, manifest_df, bert: BERTEmbedder):
        self.df   = manifest_df.reset_index(drop=True)
        self.bert = bert

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        emb   = self.bert.embed(row["transcript"])  # (seq, 768)
        label = int(row["label"])
        return torch.tensor(emb), torch.tensor(label, dtype=torch.long)
