from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .metrics import classification_metrics, save_confusion_matrix


@dataclass
class AudioPred:
    path: str
    gt: int
    prob: float
    pred: int


def compute_weights(context: Dict[str, str | float]) -> Dict[str, float]:
    lighting = context.get("lighting", "day")
    weather = context.get("weather", "clear")
    traffic = context.get("traffic_density", "low")

    if lighting == "night":
        audio_weight, vision_weight = 0.7, 0.3
    elif weather in ["rain", "fog"]:
        audio_weight, vision_weight = 0.6, 0.4
    elif traffic == "high":
        audio_weight, vision_weight = 0.65, 0.35
    else:
        audio_weight, vision_weight = 0.3, 0.7
    return {"audio": audio_weight, "vision": vision_weight}


def fuse_predictions(
    vision_preds,
    audio_preds: List[AudioPred],
    outputs_dir: Path,
) -> tuple[dict, List[dict]]:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fused_rows: List[dict] = []
    y_true: List[int] = []
    y_scores: List[float] = []

    if not audio_preds:
        audio_preds = [AudioPred(path="", gt=0, prob=0.0, pred=0)] * len(vision_preds)

    for idx, vp in enumerate(vision_preds):
        ap = audio_preds[idx % len(audio_preds)]
        ctx = vp.context or {}
        weights = compute_weights(ctx)
        score = vp.confidence * weights["vision"] + ap.prob * weights["audio"]
        target = 1 if (vp.gt == 1 or ap.gt == 1) else 0
        y_true.append(target)
        y_scores.append(score)
        fused_rows.append(
            {
                "image_path": str(vp.image_path),
                "audio_path": ap.path,
                "vision_conf": vp.confidence,
                "audio_prob": ap.prob,
                "fusion_score": score,
                "fusion_pred": int(score >= 0.5),
                "gt": target,
                "lighting": ctx.get("lighting"),
                "weather": ctx.get("weather"),
                "traffic_density": ctx.get("traffic_density"),
            }
        )

    metrics = classification_metrics(y_true, y_scores)
    pd.DataFrame(fused_rows).to_csv(outputs_dir / "fusion_predictions.csv", index=False)
    save_confusion_matrix(metrics["confusion_matrix"], outputs_dir / "fusion_confusion_matrix.csv")
    return metrics, fused_rows
