"""
models/text_pipeline/model.py
BiLSTM + self-attention (Transformer encoder layer) text emotion model.

Architecture rationale
----------------------
BiLSTM       – captures sequential context in both forward and backward
               directions, crucial for understanding emotional polarity.
Self-attention – a single Transformer encoder layer re-weights token
               representations by global token-to-token similarity,
               allowing the model to focus on emotionally salient words
               regardless of position.
Dense + Softmax – maps the aggregated contextual embedding to class logits.

Supports two input modes (configured at init time):
  • GloVe  – input: (B, seq_len, 100)
  • BERT   – input: (B, seq_len, 768)
"""

import torch
import torch.nn as nn
from project.utils.data_loader import NUM_CLASSES


class TextEmotionModel(nn.Module):
    """
    Input : (batch, seq_len, embed_dim)
    Output: (batch, NUM_CLASSES)
    """

    def __init__(self,
                 embed_dim:    int = 100,   # 100 for GloVe, 768 for BERT
                 bilstm_hidden: int = 128,
                 num_heads:    int = 4,
                 ff_dim:       int = 256,
                 num_classes:  int = NUM_CLASSES,
                 dropout:      float = 0.4):
        super().__init__()

        # ── 1. BiLSTM – contextual token modelling ─────────────────────────
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=bilstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        bilstm_out_dim = bilstm_hidden * 2   # forward + backward

        # ── 2. Transformer encoder layer – self-attention ──────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bilstm_out_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ── 3. Attention pooling (weighted mean over seq) ──────────────────
        self.attn_pool = nn.Linear(bilstm_out_dim, 1)

        # ── 4. Classifier ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(bilstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, return_embedding: bool = False):
        """
        x : (B, seq_len, embed_dim)
        """
        # BiLSTM – (B, seq, 2*hidden)
        x, _ = self.bilstm(x)

        # Transformer encoder layer – refine with self-attention
        x = self.transformer(x)        # (B, seq, 2*hidden)

        # Attention pooling over sequence dimension
        attn_weights = torch.softmax(self.attn_pool(x), dim=1)  # (B, seq, 1)
        embedding    = (x * attn_weights).sum(dim=1)            # (B, 2*hidden)

        logits = self.classifier(embedding)    # (B, num_classes)

        if return_embedding:
            return logits, embedding
        return logits


# ── Convenience constructors ──────────────────────────────────────────────────
def build_text_model_glove(device: str = "cpu") -> TextEmotionModel:
    model = TextEmotionModel(embed_dim=100)
    model.to(device)
    print(f"[TextModel-GloVe] Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def build_text_model_bert(device: str = "cpu") -> TextEmotionModel:
    model = TextEmotionModel(embed_dim=768, bilstm_hidden=256, num_heads=8, ff_dim=512)
    model.to(device)
    print(f"[TextModel-BERT] Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model
