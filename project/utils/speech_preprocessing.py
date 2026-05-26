"""
utils/speech_preprocessing.py
All speech-specific preprocessing: resampling, silence trimming,
length normalisation, and MFCC feature extraction.
"""

import os
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import librosa


# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SR       = 16_000   # resample everything to 16 kHz
MAX_DURATION_S  = 4.0      # clip / pad to this many seconds
MAX_SAMPLES     = int(TARGET_SR * MAX_DURATION_S)

N_MFCC          = 40       # number of MFCC coefficients
N_FFT           = 512
HOP_LENGTH      = 160      # 10 ms at 16 kHz
N_MELS          = 64
MAX_TIME_STEPS  = int(np.ceil(MAX_SAMPLES / HOP_LENGTH))   # ≈ 400


# ── Audio loading & normalisation ─────────────────────────────────────────────
def load_and_preprocess(wav_path: str) -> np.ndarray:
    """
    Load a WAV file, resample to TARGET_SR, trim leading/trailing silence,
    and pad/clip to MAX_SAMPLES.  Returns 1-D float32 array.
    """
    # Load with torchaudio (fast, handles various formats)
    audio, sr = librosa.load(
    wav_path,
    sr=TARGET_SR,
    mono=True
)

    waveform = torch.tensor(audio)
    # Resample if necessary
    if sr != TARGET_SR:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform  = resampler(waveform)

    audio = waveform.numpy().astype(np.float32)

    # Trim silence (librosa uses energy threshold)
    audio, _ = librosa.effects.trim(audio, top_db=20)

    # Normalise amplitude to [-1, 1]
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    # Pad or truncate to MAX_SAMPLES
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
    else:
        audio = audio[:MAX_SAMPLES]

    return audio   # shape: (MAX_SAMPLES,)


# ── MFCC feature extraction ───────────────────────────────────────────────────
def extract_mfcc(audio: np.ndarray) -> np.ndarray:
    """
    Extract MFCC features from a 1-D audio array.
    Returns array of shape (MAX_TIME_STEPS, N_MFCC).
    """
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=TARGET_SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )                                   # (N_MFCC, time)
    mfcc = mfcc.T                       # (time, N_MFCC)

    # Pad or truncate time axis
    if mfcc.shape[0] < MAX_TIME_STEPS:
        pad_len = MAX_TIME_STEPS - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_len), (0, 0)))
    else:
        mfcc = mfcc[:MAX_TIME_STEPS]

    return mfcc.astype(np.float32)     # (MAX_TIME_STEPS, N_MFCC)


# ── Spectrogram feature extraction (alternative / additional) ─────────────────
def extract_log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Extract log-Mel spectrogram. Returns (MAX_TIME_STEPS, N_MELS).
    """
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).T  # (time, N_MELS)

    if log_mel.shape[0] < MAX_TIME_STEPS:
        log_mel = np.pad(log_mel, ((0, MAX_TIME_STEPS - log_mel.shape[0]), (0, 0)))
    else:
        log_mel = log_mel[:MAX_TIME_STEPS]

    return log_mel.astype(np.float32)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────
class SpeechDataset(torch.utils.data.Dataset):
    """
    Returns (mfcc_tensor, label) pairs.
    mfcc_tensor shape: (MAX_TIME_STEPS, N_MFCC)

    Fast path: if a pre-computed _mfcc.npy cache file exists beside the wav,
    it is loaded directly (avoids librosa recomputation every epoch — ~30x faster).
    Generate the cache once with:
        python precompute_features.py --data_dir <path>
    """

    def __init__(self, manifest_df, augment: bool = False):
        self.df      = manifest_df.reset_index(drop=True)
        self.augment = augment

        # Detect whether MFCC cache files exist
        first_npy = self.df.iloc[0]["wav_path"].replace(".wav", "_mfcc.npy")
        self.use_cache = os.path.exists(first_npy)
        if self.use_cache:
            print("[SpeechDataset] Using pre-computed MFCC cache (fast mode).")
        else:
            print("[SpeechDataset] No cache — computing MFCCs on-the-fly (slow).")
            print("  Run: python precompute_features.py --data_dir <path>  to speed up.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.use_cache:
            # Fast: load pre-saved numpy array (~0.2 ms per sample)
            npy_path = row["wav_path"].replace(".wav", "_mfcc.npy")
            mfcc = np.load(npy_path)
        else:
            # Slow: recompute from raw audio (~80-120 ms per sample)
            audio = load_and_preprocess(row["wav_path"])
            if self.augment and np.random.rand() < 0.3:
                shift = np.random.randint(0, TARGET_SR // 4)
                audio = np.roll(audio, shift)
            mfcc = extract_mfcc(audio)

        label = int(row["label"])
        return torch.tensor(mfcc), torch.tensor(label, dtype=torch.long)