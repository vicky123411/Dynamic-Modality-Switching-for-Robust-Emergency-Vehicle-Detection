from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .audio_model import AudioCNN
from .metrics import classification_metrics, save_confusion_matrix

logger = logging.getLogger(__name__)


def train_audio_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    weights_out: Path | None = None,
    patience: int = 5,
    min_delta: float = 1e-4,
    batch_size: int | None = None,
    pos_weight: torch.Tensor | None = None,
) -> Tuple[dict, List[dict]]:
    model = AudioCNN().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_f1 = 0.0
    epochs_no_improve = 0
    history: List[dict] = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Audio Train {epoch + 1}/{epochs}", leave=False):
            x, y, _ = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        avg_loss = running_loss / max(1, len(train_loader.dataset))
        val_metrics, _ = evaluate_audio_model(model, val_loader, device)
        history.append({"epoch": epoch + 1, "loss": avg_loss, **val_metrics})
        scheduler.step()

        if val_metrics["f1"] > best_f1 + min_delta:
            best_f1 = val_metrics["f1"]
            epochs_no_improve = 0
            if weights_out:
                weights_out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), weights_out)
                logger.info("Saved improved audio weights to %s", weights_out)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info("Early stopping audio training after %d epochs without improvement", patience)
                break
        logger.info(
            "Audio epoch %d loss %.4f precision %.3f recall %.3f f1 %.3f acc %.3f",
            epoch + 1,
            avg_loss,
            val_metrics["precision"],
            val_metrics["recall"],
            val_metrics["f1"],
            val_metrics["accuracy"],
        )

    return history[-1] if history else {}, history


def evaluate_audio_model(
    model: AudioCNN,
    loader: DataLoader,
    device: torch.device,
    outputs_dir: Path | None = None,
) -> Tuple[dict, List[dict]]:
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    preds: List[dict] = []
    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(prob.tolist())
            all_labels.extend(y.numpy().astype(int).tolist())
            preds.extend(
                [
                    {"path": path, "gt": int(label), "prob": float(p), "pred": int(p >= 0.5)}
                    for path, label, p in zip(paths, y.numpy(), prob)
                ]
            )
    metrics = classification_metrics(all_labels, all_probs)
    if outputs_dir:
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pred_path = outputs_dir / "audio_predictions.csv"
        pd.DataFrame(preds).to_csv(pred_path, index=False)
        save_confusion_matrix(metrics["confusion_matrix"], outputs_dir / "audio_confusion_matrix.csv")
        logger.info("Saved audio predictions to %s", pred_path)
    return metrics, preds


def load_audio_model(weights_path: Path, device: torch.device) -> AudioCNN:
    model = AudioCNN().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
