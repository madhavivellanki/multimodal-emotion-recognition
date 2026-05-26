"""
precompute_features.py
One-time script that precomputes ALL MFCC features and saves them as
.npy files next to each .wav file.  After running this once, training
loads pre-saved arrays instead of recomputing librosa MFCCs every epoch,
making each epoch ~20-30x faster on CPU.

Usage (run from project/ folder):
  python precompute_features.py --data_dir "..\TESS_data\TESS Toronto emotional speech set data"
"""

import os
import sys
import argparse
import glob
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from utils.speech_preprocessing import load_and_preprocess, extract_mfcc


def precompute(data_dir: str):
    wav_files = sorted(glob.glob(
        os.path.join(data_dir, "**", "*.wav"), recursive=True
    ))
    print(f"Found {len(wav_files)} wav files. Pre-computing MFCCs …")

    skipped = 0
    for wav_path in tqdm(wav_files, desc="Extracting MFCCs"):
        npy_path = wav_path.replace(".wav", "_mfcc.npy")
        if os.path.exists(npy_path):          # already done
            skipped += 1
            continue
        try:
            audio = load_and_preprocess(wav_path)
            mfcc  = extract_mfcc(audio)       # (400, 40)
            np.save(npy_path, mfcc)
        except Exception as e:
            print(f"\n[warn] Failed on {wav_path}: {e}")

    print(f"\nDone. {len(wav_files) - skipped} files processed, "
          f"{skipped} already cached.")
    print("You can now run training — it will be much faster!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    precompute(args.data_dir)
