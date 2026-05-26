"""
utils/data_loader.py
Shared dataset loading, preprocessing, and splitting utilities for the
Toronto Emotional Speech Set (TESS).

Handles ALL known Kaggle TESS folder/filename layouts:
  Layout A – emotion in filename last token:
    OAF_dog_angry/OAF_dog_angry.wav
  Layout B – emotion in folder name last token, word in filename middle:
    OAF_angry/OAF_dog_angry.wav
  Layout C – 'pleasantsurprise' or 'pleasant_surprise' variants
"""

import os
import glob
import pandas as pd


# ── Emotion label mapping ──────────────────────────────────────────────────────
# Map every known TESS emotion string → integer index
EMOTION_LABELS = {
    "angry":            0,
    "disgust":          1,
    "fear":             2,
    "happy":            3,
    "neutral":          4,
    "ps":               5,   # pleasant surprise (short form)
    "pleasant_surprise":5,
    "pleasantsurprise": 5,
    "surprised":        5,
    "sad":              6,
}
# Canonical names for display / plotting (7 classes)
CANONICAL = {0:"angry", 1:"disgust", 2:"fear", 3:"happy",
             4:"neutral", 5:"ps", 6:"sad"}
IDX_TO_EMOTION = CANONICAL
NUM_CLASSES    = 7          # angry disgust fear happy neutral ps sad


def _emotion_from_string(s: str):
    """Return canonical emotion key or None if not recognised."""
    s = s.lower().strip()
    return EMOTION_LABELS.get(s)


def build_manifest(tess_root: str) -> pd.DataFrame:
    """
    Walk *tess_root* recursively, find every .wav file, and infer
    emotion + transcript from folder/filename.

    Strategy (tried in order for each file):
      1. Parse the parent folder name  – last '_'-separated token
      2. Parse the wav filename itself – last '_'-separated token
      3. Scan all tokens for any known emotion keyword

    Returns DataFrame with columns: wav_path, transcript, emotion, label
    """
    # Sanity-check the root exists
    if not os.path.isdir(tess_root):
        raise FileNotFoundError(
            f"TESS data directory not found: '{tess_root}'\n"
            f"Current working directory: {os.getcwd()}\n"
            "Make sure you ran:  kaggle datasets download ... and unzipped into TESS_data/"
        )

    wav_files = sorted(glob.glob(
        os.path.join(tess_root, "**", "*.wav"), recursive=True
    ))

    if not wav_files:
        raise FileNotFoundError(
            f"No .wav files found under '{tess_root}'.\n"
            "Please check the dataset was unzipped correctly."
        )

    print(f"[data_loader] Scanning {len(wav_files)} wav files under '{tess_root}' …")

    records = []
    skipped = 0

    for wav_path in wav_files:
        folder   = os.path.basename(os.path.dirname(wav_path))
        basename = os.path.splitext(os.path.basename(wav_path))[0]

        emotion_idx = None
        word        = "unknown"

        # ── Strategy 1: folder name last token ──────────────────────────
        folder_parts = folder.split("_")
        emo = _emotion_from_string(folder_parts[-1])
        if emo is not None:
            emotion_idx = emo
            # word = middle token(s) of folder name, skip first (speaker)
            if len(folder_parts) >= 3:
                word = "_".join(folder_parts[1:-1])
            elif len(folder_parts) == 2:
                word = folder_parts[0]

        # ── Strategy 2: filename last token ─────────────────────────────
        if emotion_idx is None:
            file_parts = basename.split("_")
            emo = _emotion_from_string(file_parts[-1])
            if emo is not None:
                emotion_idx = emo
                if len(file_parts) >= 3:
                    word = "_".join(file_parts[1:-1])

        # ── Strategy 3: scan all tokens for any emotion keyword ──────────
        if emotion_idx is None:
            all_tokens = folder.lower().split("_") + basename.lower().split("_")
            for tok in all_tokens:
                emo = _emotion_from_string(tok)
                if emo is not None:
                    emotion_idx = emo
                    break

        if emotion_idx is None:
            skipped += 1
            continue

        records.append({
            "wav_path":   wav_path,
            "transcript": word.lower(),
            "emotion":    CANONICAL[emotion_idx],
            "label":      emotion_idx,
        })

    if skipped:
        print(f"[data_loader] Skipped {skipped} files (emotion not recognised).")

    if not records:
        # Print a sample of what we found to help debug
        sample = wav_files[:5]
        print("[data_loader] Sample paths found:")
        for p in sample:
            print(f"  folder='{os.path.basename(os.path.dirname(p))}'  "
                  f"file='{os.path.basename(p)}'")
        raise ValueError(
            "Could not extract any emotion labels from the dataset.\n"
            "Please paste the output of:  dir TESS_data  in your terminal\n"
            "so we can see the exact folder structure."
        )

    df = pd.DataFrame(records)
    print(f"[data_loader] Found {len(df)} labelled samples across "
          f"{df['emotion'].nunique()} emotion classes.")
    print(f"[data_loader] Class distribution:\n{df['emotion'].value_counts().to_string()}")
    return df


# ── Train / val / test split ───────────────────────────────────────────────────
def split_manifest(df: pd.DataFrame,
                   train_ratio: float = 0.70,
                   val_ratio:   float = 0.15,
                   seed: int = 42) -> tuple:
    """
    Stratified split so every emotion class is represented in every split.
    Returns (train_df, val_df, test_df).
    """
    from sklearn.model_selection import train_test_split

    # First cut: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=(1 - train_ratio),
        stratify=df["label"], random_state=seed
    )
    # Second cut: val vs test from the remainder
    relative_val = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val),
        stratify=temp_df["label"], random_state=seed
    )

    print(f"[data_loader] Split → train={len(train_df)}, "
          f"val={len(val_df)}, test={len(test_df)}")
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True), \
           test_df.reset_index(drop=True) 