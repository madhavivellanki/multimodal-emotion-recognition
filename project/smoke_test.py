"""
smoke_test.py
Validates that all three model architectures forward-pass correctly
using randomly generated tensors (no real dataset needed).

Run from the project root:
  python smoke_test.py

All assertions should pass with zero errors.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

from utils.data_loader           import NUM_CLASSES
from utils.speech_preprocessing  import MAX_TIME_STEPS, N_MFCC
from utils.text_preprocessing    import MAX_TEXT_LEN

from models.speech_pipeline.model  import build_speech_model
from models.text_pipeline.model    import build_text_model_glove, build_text_model_bert
from models.fusion_pipeline.model  import build_fusion_model_glove, build_fusion_model_bert

BATCH = 4
DEVICE = "cpu"

def green(s): return f"\033[92m{s}\033[0m"
def red(s):   return f"\033[91m{s}\033[0m"

def test_speech():
    print("\n── Speech Pipeline (CNN + BiLSTM) ──")
    model = build_speech_model(DEVICE)
    x     = torch.randn(BATCH, MAX_TIME_STEPS, N_MFCC)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH},{NUM_CLASSES}), got {logits.shape}"
    assert emb.shape[0] == BATCH
    print(green(f"  ✓ logits={tuple(logits.shape)}, embedding={tuple(emb.shape)}"))


def test_text_glove():
    print("\n── Text Pipeline (BiLSTM + Transformer) – GloVe ──")
    model = build_text_model_glove(DEVICE)
    x     = torch.randn(BATCH, MAX_TEXT_LEN, 100)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (BATCH, NUM_CLASSES)
    print(green(f"  ✓ logits={tuple(logits.shape)}, embedding={tuple(emb.shape)}"))


def test_text_bert():
    print("\n── Text Pipeline (BiLSTM + Transformer) – BERT ──")
    model = build_text_model_bert(DEVICE)
    x     = torch.randn(BATCH, MAX_TEXT_LEN, 768)
    logits, emb = model(x, return_embedding=True)
    assert logits.shape == (BATCH, NUM_CLASSES)
    print(green(f"  ✓ logits={tuple(logits.shape)}, embedding={tuple(emb.shape)}"))


def test_fusion_glove():
    print("\n── Fusion Pipeline (Cross-modal Attention) – GloVe ──")
    model  = build_fusion_model_glove(DEVICE)
    speech = torch.randn(BATCH, MAX_TIME_STEPS, N_MFCC)
    text   = torch.randn(BATCH, MAX_TEXT_LEN, 100)
    logits, emb = model(speech, text, return_embedding=True)
    assert logits.shape == (BATCH, NUM_CLASSES)
    print(green(f"  ✓ logits={tuple(logits.shape)}, embedding={tuple(emb.shape)}"))


def test_fusion_bert():
    print("\n── Fusion Pipeline (Cross-modal Attention) – BERT ──")
    model  = build_fusion_model_bert(DEVICE)
    speech = torch.randn(BATCH, MAX_TIME_STEPS, N_MFCC)
    text   = torch.randn(BATCH, MAX_TEXT_LEN, 768)
    logits, emb = model(speech, text, return_embedding=True)
    assert logits.shape == (BATCH, NUM_CLASSES)
    print(green(f"  ✓ logits={tuple(logits.shape)}, embedding={tuple(emb.shape)}"))


def test_backward_speech():
    print("\n── Backward Pass – Speech ──")
    model     = build_speech_model(DEVICE)
    x         = torch.randn(BATCH, MAX_TIME_STEPS, N_MFCC)
    labels    = torch.randint(0, NUM_CLASSES, (BATCH,))
    criterion = torch.nn.CrossEntropyLoss()
    logits    = model(x)
    loss      = criterion(logits, labels)
    loss.backward()
    print(green(f"  ✓ loss={loss.item():.4f}"))


def test_backward_fusion():
    print("\n── Backward Pass – Fusion ──")
    model     = build_fusion_model_glove(DEVICE)
    speech    = torch.randn(BATCH, MAX_TIME_STEPS, N_MFCC)
    text      = torch.randn(BATCH, MAX_TEXT_LEN, 100)
    labels    = torch.randint(0, NUM_CLASSES, (BATCH,))
    criterion = torch.nn.CrossEntropyLoss()
    logits    = model(speech, text)
    loss      = criterion(logits, labels)
    loss.backward()
    print(green(f"  ✓ loss={loss.item():.4f}"))


if __name__ == "__main__":
    print("=" * 55)
    print("  Multimodal Emotion Recognition – Smoke Test")
    print("=" * 55)
    tests = [
        test_speech,
        test_text_glove,
        test_text_bert,
        test_fusion_glove,
        test_fusion_bert,
        test_backward_speech,
        test_backward_fusion,
    ]
    passed, failed = 0, 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(red(f"  ✗ {fn.__name__} FAILED: {e}"))
            failed += 1

    print(f"\n{'='*55}")
    print(f"  Results: {green(str(passed)+' passed')}  "
          f"{red(str(failed)+' failed') if failed else ''}")
    print("=" * 55)
    sys.exit(0 if failed == 0 else 1)
