"""
models/speech_pipeline/model.py
CNN + LSTM speech emotion recognition model.

Architecture rationale
----------------------
CNN   – extracts local spectro-temporal patterns (e.g., formants, pitch
        contours) from the MFCC sequence.  Convolutional kernels act as
        learned filter banks that are translation-invariant over time.
LSTM  – models long-range temporal dependencies in the emotional prosody
        after local features have been compressed by the CNN.
Dense + Softmax – maps the final hidden state to emotion class logits.
"""

import torch
import torch.nn as nn
from project.utils.speech_preprocessing import MAX_TIME_STEPS, N_MFCC
from project.utils.data_loader import NUM_CLASSES


class SpeechEmotionModel(nn.Module):
    """
    Input : (batch, time_steps, n_mfcc)   – e.g. (B, 400, 40)
    Output: (batch, NUM_CLASSES)           – raw logits
    """

    def __init__(self,
                 n_mfcc:        int = N_MFCC,
                 cnn_channels:  tuple = (64, 128),
                 lstm_hidden:   int = 256,
                 lstm_layers:   int = 2,
                 num_classes:   int = NUM_CLASSES,
                 dropout:       float = 0.4):
        super().__init__()

        # ── 1. CNN feature extractor ───────────────────────────────────────
        # Input  : (B, 1, T, n_mfcc)   treat as 2-D "image" (time × features)
        # Output : (B, cnn_channels[-1], T', n_mfcc')
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, cnn_channels[0], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),   # T/2, F/2
            nn.Dropout2d(p=0.2),

            # Block 2
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),   # T/4, F/4
            nn.Dropout2d(p=0.2),
        )

        # Compute the flattened feature size after CNN (along feature axis)
        # T_out = MAX_TIME_STEPS // 4 ; F_out = n_mfcc // 4
        self.t_out = MAX_TIME_STEPS // 4
        self.f_out = n_mfcc // 4
        self.lstm_in_size = cnn_channels[-1] * self.f_out

        # ── 2. Temporal modelling (LSTM) ──────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.lstm_in_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,     # BiLSTM for richer context
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── 3. Classifier ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_hidden * 2, 128),  # ×2 for bidirectional
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_embedding: bool = False):
        """
        x : (B, T, n_mfcc)
        """
        B, T, F = x.shape

        # CNN expects (B, 1, T, F)
        x = x.unsqueeze(1)              # (B, 1, T, F)
        x = self.cnn(x)                 # (B, C, T', F')

        # Reshape for LSTM: (B, T', C*F')
        C, T2, F2 = x.shape[1], x.shape[2], x.shape[3]
        x = x.permute(0, 2, 1, 3)      # (B, T', C, F')
        x = x.reshape(B, T2, C * F2)   # (B, T', lstm_in_size)

        # LSTM – take the last time-step output
        x, _ = self.lstm(x)            # (B, T', 2*hidden)
        embedding = x[:, -1, :]        # (B, 2*hidden) — temporal summary

        logits = self.classifier(embedding)   # (B, num_classes)

        if return_embedding:
            return logits, embedding
        return logits


# ── Convenience constructor ───────────────────────────────────────────────────
def build_speech_model(device: str = "cpu") -> SpeechEmotionModel:
    model = SpeechEmotionModel()
    model.to(device)
    print(f"[SpeechModel] Parameters: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model
