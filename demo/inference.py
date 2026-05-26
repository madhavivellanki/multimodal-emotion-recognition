import os
import sys
import tempfile
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn

# ---------------- PATH FIX ----------------

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(DEMO_DIR)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PROJECT_DIR = os.path.join(ROOT_DIR, "project")

print("ROOT_DIR:", ROOT_DIR)
print("PROJECT_DIR:", PROJECT_DIR)

# ---------------- IMPORTS ----------------

from project.utils.speech_preprocessing import (
    load_and_preprocess,
    extract_mfcc,
    N_MFCC,
    MAX_TIME_STEPS
)

from project.utils.text_preprocessing import (
    GloVeEmbeddings,
    clean_text,
    MAX_TEXT_LEN
)

from project.utils.data_loader import (
    IDX_TO_EMOTION,
    NUM_CLASSES
)

DEVICE = "cpu"

EMOTION_EMOJIS = {
    "angry":"😠",
    "disgust":"🤢",
    "fear":"😨",
    "happy":"😊",
    "neutral":"😐",
    "ps":"😌",
    "sad":"😢"
}

EMOTION_COLORS = {
    "angry":"#E24B4A",
    "disgust":"#EF9F27",
    "fear":"#7F77DD",
    "happy":"#97C459",
    "neutral":"#888780",
    "ps":"#1D9E75",
    "sad":"#378ADD"
}

# ---------------- SPEECH MODEL ----------------

class SpeechEmotionModelV2(nn.Module):

    def __init__(self, n_mfcc=N_MFCC, num_classes=7, dropout=0.3):

        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)
        )

        lstm_in = 64 * (n_mfcc // 4)

        self.lstm = nn.LSTM(
            lstm_in,
            128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256,64),
            nn.ReLU(True),
            nn.Linear(64,num_classes)
        )

    def forward(self,x):

        B,T,F = x.shape

        x = self.cnn(x.unsqueeze(1))

        C,T2,F2 = x.shape[1],x.shape[2],x.shape[3]

        x = x.permute(0,2,1,3).reshape(B,T2,C*F2)

        x,_ = self.lstm(x)

        return self.classifier(x[:,-1,:])

# ---------------- TEXT MODEL ----------------

class TextEmotionModel(nn.Module):

    def __init__(
        self,
        embed_dim=100,
        bilstm_hidden=128,
        num_heads=4,
        ff_dim=256,
        num_classes=7,
        dropout=0.4
    ):

        super().__init__()

        self.bilstm = nn.LSTM(
            embed_dim,
            bilstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        bilstm_out = bilstm_hidden * 2

        enc = nn.TransformerEncoderLayer(
            d_model=bilstm_out,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            enc,
            num_layers=1
        )

        self.attn_pool = nn.Linear(bilstm_out,1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bilstm_out,128),
            nn.ReLU(True),
            nn.Dropout(dropout/2),
            nn.Linear(128,num_classes)
        )

    def forward(self,x):

        x,_ = self.bilstm(x)

        x = self.transformer(x)

        w = torch.softmax(self.attn_pool(x),dim=1)

        x = (x*w).sum(dim=1)

        return self.classifier(x)

# ---------------- MODEL MANAGER ----------------

class ModelManager:

    def __init__(self):

        self.speech_model = None
        self.text_model = None

        self.glove = None

        self.mfcc_mean = None
        self.mfcc_std = None

        self.errors = []

    def load_all(self, project_dir):

        print(f"[inference] project_dir = {project_dir}")

        # ---------------- MFCC ----------------

        mp = os.path.join(
            project_dir,
            "checkpoints",
            "speech",
            "mfcc_mean.npy"
        )

        sp = os.path.join(
            project_dir,
            "checkpoints",
            "speech",
            "mfcc_std.npy"
        )

        if os.path.exists(mp) and os.path.exists(sp):

            self.mfcc_mean = np.load(mp)

            self.mfcc_std = np.load(sp)

            print("MFCC stats loaded")

        else:

            self.errors.append("MFCC stats missing")

        # ---------------- SPEECH MODEL ----------------

        sc = os.path.join(
            project_dir,
            "checkpoints",
            "speech",
            "best_speech_model.pt"
        )

        if os.path.exists(sc) and self.mfcc_mean is not None:

            try:

                self.speech_model = SpeechEmotionModelV2().to(DEVICE)

                ck = torch.load(sc, map_location=DEVICE)

                self.speech_model.load_state_dict(
                    ck["state_dict"]
                )

                self.speech_model.eval()

                print("Speech model loaded")

            except Exception as e:

                self.errors.append(f"Speech load error: {e}")

        else:

            self.errors.append("Speech checkpoint missing")

        # ---------------- GLOVE ----------------

        GLOVE_PATH = os.path.join(
            project_dir,
            "glove.6B.100d.txt"
        )

        if not os.path.exists(GLOVE_PATH):

            try:

                print("Downloading GloVe embeddings...")

                zip_path = os.path.join(
                    ROOT_DIR,
                    "glove.6B.zip"
                )

                urllib.request.urlretrieve(
                    "https://nlp.stanford.edu/data/glove.6B.zip",
                    zip_path
                )

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:

                    zip_ref.extract(
                        "glove.6B.100d.txt",
                        project_dir
                    )

                print("GloVe downloaded successfully")

            except Exception as e:

                self.errors.append(
                    f"GloVe download failed: {e}"
                )

        # ---------------- LOAD GLOVE ----------------

        try:

            self.glove = GloVeEmbeddings(GLOVE_PATH)

            print("GloVe loaded")

        except Exception as e:

            self.errors.append(f"GloVe load error: {e}")

        # ---------------- TEXT MODEL ----------------

        tc = os.path.join(
            project_dir,
            "checkpoints",
            "text",
            "best_text_model_glove.pt"
        )

        if os.path.exists(tc) and self.glove is not None:

            try:

                self.text_model = TextEmotionModel().to(DEVICE)

                ck = torch.load(tc, map_location=DEVICE)

                if "state_dict" in ck:

                    self.text_model.load_state_dict(
                        ck["state_dict"]
                    )

                else:

                    self.text_model.load_state_dict(ck)

                self.text_model.eval()

                print("Text model loaded")

            except Exception as e:

                self.errors.append(f"Text load error: {e}")

        else:

            self.errors.append("Text checkpoint missing")

        print("Errors:", self.errors)

    # ---------------- SPEECH PREDICTION ----------------

    def predict_speech(self, audio_bytes):

        if self.speech_model is None:

            return {"error":"Speech model not loaded"}

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:

            f.write(audio_bytes)

            tmp = f.name

        try:

            audio = load_and_preprocess(tmp)

            mfcc = extract_mfcc(audio).astype(np.float32)

            mfcc = (
                mfcc - self.mfcc_mean.squeeze()
            ) / self.mfcc_std.squeeze()

            x = torch.tensor(
                mfcc,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                probs = torch.softmax(
                    self.speech_model(x),
                    dim=-1
                ).squeeze().numpy()

            return self._result(probs)

        except Exception as e:

            return {
                "error":f"Speech prediction error: {e}"
            }

        finally:

            os.unlink(tmp)

    # ---------------- TEXT PREDICTION ----------------

    def predict_text(self, text):

        if self.text_model is None:

            return {"error":"Text model not loaded"}

        try:

            mat = self.glove.sentence_matrix(
                clean_text(text),
                max_tokens=MAX_TEXT_LEN
            )

            x = torch.tensor(
                mat,
                dtype=torch.float32
            ).unsqueeze(0)

            with torch.no_grad():

                probs = torch.softmax(
                    self.text_model(x),
                    dim=-1
                ).squeeze().numpy()

            return self._result(probs)

        except Exception as e:

            return {
                "error":f"Text prediction error: {e}"
            }

    # ---------------- FUSION ----------------

    def predict_fusion(self, audio_bytes, text):

        sr = self.predict_speech(audio_bytes) if audio_bytes else None

        tr = self.predict_text(text) if text else None

        results = {
            "speech":sr,
            "text":tr
        }

        valid = []

        if sr and "all_probs" in sr:

            valid.append([
                sr["all_probs"][IDX_TO_EMOTION[i]] / 100
                for i in range(NUM_CLASSES)
            ])

        if tr and "all_probs" in tr:

            valid.append([
                tr["all_probs"][IDX_TO_EMOTION[i]] / 100
                for i in range(NUM_CLASSES)
            ])

        if valid:

            results["fusion"] = self._result(
                np.mean(valid,axis=0)
            )

        else:

            results["fusion"] = {
                "error":"No valid predictions"
            }

        return results

    # ---------------- RESULT FORMAT ----------------

    def _result(self, probs):

        idx = int(np.argmax(probs))

        em = IDX_TO_EMOTION[idx]

        return {

            "emotion": em,

            "emoji": EMOTION_EMOJIS[em],

            "color": EMOTION_COLORS[em],

            "confidence": round(
                float(probs[idx]) * 100,
                1
            ),

            "all_probs": {

                IDX_TO_EMOTION[i]:
                round(float(probs[i]) * 100,1)

                for i in range(NUM_CLASSES)
            }
        }

    @property
    def speech_ready(self):
        return self.speech_model is not None

    @property
    def text_ready(self):
        return self.text_model is not None

    @property
    def both_ready(self):
        return self.speech_ready and self.text_ready

# ---------------- SINGLETON ----------------

_manager = None

def get_manager(project_dir):

    global _manager

    if _manager is None:

        _manager = ModelManager()

        _manager.load_all(project_dir)

    return _manager