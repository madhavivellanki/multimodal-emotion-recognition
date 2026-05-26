"""
models/fusion_pipeline/model.py
Multimodal (Speech + Text) Emotion Recognition with Attention-based Fusion.

Architecture Overview
---------------------

  Speech branch
    ┌──────────────┐   ┌───────────────────┐
    │ CNN (2-block)│ → │ BiLSTM (temporal) │ → speech_embed (B, 512)
    └──────────────┘   └───────────────────┘

  Text branch
    ┌──────────────────────────┐   ┌──────────────────────────────┐
    │ BiLSTM (2-layer)         │ → │ Transformer encoder (1-layer) │ → text_embed (B, 256)
    └──────────────────────────┘   └──────────────────────────────┘

  Fusion
    ┌───────────────────────────────────────────────────────┐
    │  Cross-modal attention  (speech queries, text keys)   │
    │  + self-attention on concatenated embed               │
    │  → fused_embed  (B, 512)                              │
    └───────────────────────────────────────────────────────┘

  Classifier
    Dense(512→256) → ReLU → Dropout → Dense(256→NUM_CLASSES) → Softmax

Fusion rationale
----------------
Simple concatenation ignores inter-modal alignment.  Cross-modal attention
lets the speech representation query the text representation (and vice versa),
learning *which* text tokens are most consistent with each acoustic segment.
A residual add after attention stabilises gradients.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.speech_preprocessing import MAX_TIME_STEPS, N_MFCC
from utils.data_loader          import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────
# Sub-module: Speech Encoder (CNN + BiLSTM)
# ─────────────────────────────────────────────────────────────────────────────
class SpeechEncoder(nn.Module):
    """
    Encodes MFCC sequence → fixed-length speech embedding.
    Input : (B, T, N_MFCC)
    Output: (B, lstm_hidden*2)
    """
    def __init__(self, n_mfcc=N_MFCC,
                 cnn_channels=(64, 128),
                 lstm_hidden=256,
                 dropout=0.4):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels[0], 3, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(cnn_channels[0], cnn_channels[1], 3, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.t_out = MAX_TIME_STEPS // 4
        self.f_out = n_mfcc // 4
        self.lstm_in = cnn_channels[-1] * self.f_out

        self.lstm = nn.LSTM(
            input_size=self.lstm_in,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.out_dim = lstm_hidden * 2

    def forward(self, x):
        B, T, F = x.shape
        x = x.unsqueeze(1)
        x = self.cnn(x)
        C, T2, F2 = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 2, 1, 3).reshape(B, T2, C * F2)
        x, _ = self.lstm(x)
        return x[:, -1, :]   # (B, 2*hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-module: Text Encoder (BiLSTM + Transformer)
# ─────────────────────────────────────────────────────────────────────────────
class TextEncoder(nn.Module):
    """
    Encodes token embedding sequence → fixed-length text embedding.
    Input : (B, seq_len, embed_dim)
    Output: (B, bilstm_hidden*2)
    """
    def __init__(self, embed_dim=100,
                 bilstm_hidden=128,
                 num_heads=4,
                 ff_dim=256,
                 dropout=0.4):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=bilstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        bilstm_out = bilstm_hidden * 2

        enc_layer = nn.TransformerEncoderLayer(
            d_model=bilstm_out,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=1)
        self.attn_pool   = nn.Linear(bilstm_out, 1)
        self.out_dim     = bilstm_out

    def forward(self, x):
        x, _ = self.bilstm(x)                         # (B, seq, 2*hidden)
        x     = self.transformer(x)                   # (B, seq, 2*hidden)
        w     = torch.softmax(self.attn_pool(x), dim=1)  # (B, seq, 1)
        return (x * w).sum(dim=1)                     # (B, 2*hidden)


# ─────────────────────────────────────────────────────────────────────────────
# Sub-module: Cross-modal Attention Fusion
# ─────────────────────────────────────────────────────────────────────────────
class CrossModalFusion(nn.Module):
    """
    Given speech_emb (B, ds) and text_emb (B, dt):
      1. Project both to a common dim d_fuse.
      2. Speech queries text (cross-attention: Q=speech, K=V=text).
      3. Text queries speech (cross-attention: Q=text,   K=V=speech).
      4. Concatenate both attended outputs → (B, 2*d_fuse).
      5. Add residual + LayerNorm.
    Output: (B, fused_dim)
    """
    def __init__(self, speech_dim, text_dim, d_fuse=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.d_fuse = d_fuse

        # Project each modality into the shared fusion space
        self.proj_speech = nn.Linear(speech_dim, d_fuse)
        self.proj_text   = nn.Linear(text_dim,   d_fuse)

        # Cross-attention: speech → text
        self.cross_s2t = nn.MultiheadAttention(
            embed_dim=d_fuse, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        # Cross-attention: text → speech
        self.cross_t2s = nn.MultiheadAttention(
            embed_dim=d_fuse, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_fuse)
        self.norm2 = nn.LayerNorm(d_fuse)

        self.out_dim = d_fuse * 2

    def forward(self, speech_emb, text_emb):
        # Project: (B, d_fuse) → unsqueeze to (B, 1, d_fuse) for MHA
        s = self.proj_speech(speech_emb).unsqueeze(1)   # (B, 1, d_fuse)
        t = self.proj_text(text_emb).unsqueeze(1)       # (B, 1, d_fuse)

        # Speech queries text context
        s_ctx, _ = self.cross_s2t(query=s, key=t, value=t)   # (B, 1, d_fuse)
        s_ctx     = self.norm1(s + s_ctx).squeeze(1)          # residual

        # Text queries speech context
        t_ctx, _ = self.cross_t2s(query=t, key=s, value=s)   # (B, 1, d_fuse)
        t_ctx     = self.norm2(t + t_ctx).squeeze(1)          # residual

        # Concatenate both attended representations
        fused = torch.cat([s_ctx, t_ctx], dim=-1)   # (B, 2*d_fuse)
        return fused


# ─────────────────────────────────────────────────────────────────────────────
# Top-level: Multimodal Fusion Model
# ─────────────────────────────────────────────────────────────────────────────
class MultimodalEmotionModel(nn.Module):
    """
    Inputs : speech (B, T, N_MFCC)  +  text (B, seq_len, embed_dim)
    Output : (B, NUM_CLASSES)
    """

    def __init__(self,
                 embed_dim:    int = 100,    # GloVe=100, BERT=768
                 num_classes:  int = NUM_CLASSES,
                 d_fuse:       int = 256,
                 dropout:      float = 0.4):
        super().__init__()

        # Unimodal encoders
        self.speech_encoder = SpeechEncoder(dropout=dropout)
        self.text_encoder   = TextEncoder(embed_dim=embed_dim, dropout=dropout)

        # Fusion module
        self.fusion = CrossModalFusion(
            speech_dim=self.speech_encoder.out_dim,
            text_dim=self.text_encoder.out_dim,
            d_fuse=d_fuse,
            dropout=dropout,
        )

        fused_dim = self.fusion.out_dim   # 2 * d_fuse = 512

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fused_dim, 256),
            nn.ReLU(True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, speech, text, return_embedding: bool = False):
        """
        speech : (B, T, N_MFCC)
        text   : (B, seq_len, embed_dim)
        """
        speech_emb = self.speech_encoder(speech)   # (B, 512)
        text_emb   = self.text_encoder(text)       # (B, 256)

        fused_emb  = self.fusion(speech_emb, text_emb)  # (B, 512)
        logits     = self.classifier(fused_emb)          # (B, num_classes)

        if return_embedding:
            return logits, fused_emb
        return logits


# ── Convenience constructors ──────────────────────────────────────────────────
def build_fusion_model_glove(device: str = "cpu") -> MultimodalEmotionModel:
    model = MultimodalEmotionModel(embed_dim=100)
    model.to(device)
    print(f"[FusionModel-GloVe] Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def build_fusion_model_bert(device: str = "cpu") -> MultimodalEmotionModel:
    model = MultimodalEmotionModel(embed_dim=768, d_fuse=512)
    model.to(device)
    print(f"[FusionModel-BERT] Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model
