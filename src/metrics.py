from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(y_true: Iterable[int], y_scores: Iterable[float], threshold: float = 0.5) -> dict:
    y_true = np.array(list(y_true)).astype(int)
    y_scores = np.array(list(y_scores)).astype(float)
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(acc),
        "confusion_matrix": cm,
    }


def save_confusion_matrix(cm: np.ndarray, out_path: Path, labels: List[str] | None = None) -> None:
    df = pd.DataFrame(cm, index=labels or [0, 1], columns=labels or [0, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=True)
