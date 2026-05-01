from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


class UrbanSoundSirenDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        augment: bool = False,
        sample_rate: int = 22050,
        n_mels: int = 64,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_frames: int = 173,
    ):
        self.df = pd.read_csv(manifest_path)
        self.augment = augment
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_frames = max_frames

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: Path) -> np.ndarray:
        y, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        if self.augment:
            noise = np.random.randn(len(y)) * 0.001
            y = y + noise
        return y

    def _apply_spec_augment(self, spec: np.ndarray) -> np.ndarray:
        spec_aug = spec.copy()
        # Frequency mask
        freq_mask = np.random.randint(0, max(1, self.n_mels // 8))
        freq_start = np.random.randint(0, max(1, self.n_mels - freq_mask))
        spec_aug[freq_start:freq_start + freq_mask, :] = spec_aug.mean()
        # Time mask
        time_mask = np.random.randint(0, max(1, self.max_frames // 8))
        time_start = np.random.randint(0, max(1, self.max_frames - time_mask))
        spec_aug[:, time_start:time_start + time_mask] = spec_aug.mean()
        return spec_aug

    def _to_log_mel(self, y: np.ndarray) -> np.ndarray:
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < self.max_frames:
            pad = self.max_frames - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
        else:
            mel_db = mel_db[:, : self.max_frames]
        if self.augment:
            mel_db = self._apply_spec_augment(mel_db)
        return mel_db

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row["path"])
        label = int(row["label"])
        y = self._load_audio(path)
        mel = self._to_log_mel(y)
        tensor = torch.tensor(mel).unsqueeze(0).float()
        return tensor, torch.tensor(label).float(), str(path)


def prepare_audio_manifest(
    base_dir: Path,
    output_dir: Path,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    train_manifest = output_dir / "train_manifest.csv"
    val_manifest = output_dir / "val_manifest.csv"

    if train_manifest.exists() and val_manifest.exists():
        logger.info("Using existing audio manifests")
        return manifest_path, train_manifest, val_manifest

    meta_path = base_dir / "metadata" / "UrbanSound8K.csv"
    audio_root = base_dir / "audio"
    df = pd.read_csv(meta_path)
    df["label"] = (df["class"] == "siren").astype(int)
    df["path"] = df.apply(
        lambda r: str(audio_root / f"fold{int(r['fold'])}" / r["slice_file_name"]), axis=1
    )
    df = df[["path", "label"]]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)

    train_df, val_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    train_df.to_csv(train_manifest, index=False)
    val_df.to_csv(val_manifest, index=False)
    logger.info("Prepared audio manifest with %d train and %d val samples", len(train_df), len(val_df))
    return manifest_path, train_manifest, val_manifest


def build_audio_loaders(
    train_manifest: Path,
    val_manifest: Path,
    batch_size: int = 16,
    num_workers: int = 0,
    augment: bool = False,
    balance: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = UrbanSoundSirenDataset(train_manifest, augment=augment)
    val_ds = UrbanSoundSirenDataset(val_manifest, augment=False)
    sampler = None
    if balance:
        df = pd.read_csv(train_manifest)
        counts = df["label"].value_counts()
        weights = df["label"].map(lambda x: 1.0 / max(1, counts.get(x, 1)))
        sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=not balance,
        num_workers=num_workers,
        sampler=sampler,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
