from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .metrics import classification_metrics, save_confusion_matrix
from . import fusion_old

logger = logging.getLogger(__name__)

USE_NEW_FUSION = True


@dataclass
class AudioPred:
    path: str
    gt: int
    prob: float
    pred: int


def _clamp01(val: float) -> float:
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, fval))


def _calibrate_conf(val: float, temperature: float = 1.0) -> float:
    val = _clamp01(val)
    if temperature != 1.0:
        scaled = (val - 0.5) / max(1e-6, temperature)
        val = 1.0 / (1.0 + math.exp(-scaled))
    return _clamp01(val)


def _dynamic_threshold(context: Dict[str, str | float]) -> float:
    threshold = 0.35
    if context.get("lighting") == "night":
        threshold = 0.3
    elif context.get("weather") in ["rain", "fog"]:
        threshold = 0.32
    elif context.get("traffic_density") == "high":
        threshold = 0.33
    return threshold


def compute_weights(context: Dict[str, str | float], vision_conf: float = 0.0, audio_conf: float = 0.0) -> Dict[str, float]:
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

    if vision_conf < 0.4:
        audio_weight += 0.2
    if audio_conf < 0.4:
        vision_weight += 0.2

    total = audio_weight + vision_weight
    if total <= 0:
        audio_weight = vision_weight = 0.5
        total = 1.0
    audio_weight /= total
    vision_weight /= total
    return {"audio": audio_weight, "vision": vision_weight}


def _write_outputs(fused_rows: List[dict], metrics: dict, outputs_dir: Path, name_suffix: str = "") -> None:
    outputs_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{name_suffix}" if name_suffix else ""
    pd.DataFrame(fused_rows).to_csv(outputs_dir / f"fusion_predictions{suffix}.csv", index=False)
    cm_path = outputs_dir / f"fusion_confusion_matrix{suffix}.csv"
    save_confusion_matrix(metrics.get("confusion_matrix"), cm_path)
    _plot_confusion_matrix(metrics.get("confusion_matrix"), outputs_dir / f"fusion_confusion_matrix{suffix}.png")


def _plot_confusion_matrix(cm, out_path: Path) -> None:
    if cm is None:
        return
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def fuse_predictions_new(
    vision_preds,
    audio_preds: List[AudioPred],
    outputs_dir: Path,
    name_suffix: str = "",
    threshold_override: float | None = None,
    weight_scale: Tuple[float, float] = (1.0, 1.0),
    smoothing_temp: float = 1.0,
    verbose: bool = False,
) -> tuple[dict, List[dict]]:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fused_rows: List[dict] = []
    y_true: List[int] = []
    y_scores: List[float] = []
    y_preds: List[int] = []

    if not audio_preds:
        audio_preds = [AudioPred(path="", gt=0, prob=0.0, pred=0)] * len(vision_preds)

    for idx, vp in enumerate(vision_preds):
        ap = audio_preds[idx % len(audio_preds)]
        ctx = vp.context or {}
        vision_conf = _calibrate_conf(vp.confidence, temperature=smoothing_temp)
        audio_conf = _calibrate_conf(ap.prob, temperature=smoothing_temp)
        weights = compute_weights(ctx, vision_conf=vision_conf, audio_conf=audio_conf)
        audio_weight = weights["audio"] * weight_scale[0]
        vision_weight = weights["vision"] * weight_scale[1]
        total_w = audio_weight + vision_weight or 1.0
        weights = {"audio": audio_weight / total_w, "vision": vision_weight / total_w}

        dyn_threshold = threshold_override if threshold_override is not None else _dynamic_threshold(ctx)

        fusion_score = vision_conf * weights["vision"] + audio_conf * weights["audio"]
        vision_detected = vision_conf >= 0.5
        audio_detected = audio_conf >= 0.5

        emergency = fusion_score >= dyn_threshold
        if audio_detected and vision_conf < 0.3:
            emergency = True
        if vision_detected and audio_conf < 0.3:
            emergency = True

        target = 1 if (vp.gt == 1 or ap.gt == 1) else 0
        y_true.append(target)
        y_scores.append(fusion_score)
        y_preds.append(int(emergency))

        fused_rows.append(
            {
                "image_path": str(vp.image_path),
                "audio_path": ap.path,
                "vision_conf": vision_conf,
                "audio_prob": audio_conf,
                "fusion_score": fusion_score,
                "fusion_pred": int(emergency),
                "gt": target,
                "lighting": ctx.get("lighting"),
                "weather": ctx.get("weather"),
                "traffic_density": ctx.get("traffic_density"),
                "vision_weight": weights["vision"],
                "audio_weight": weights["audio"],
                "threshold": dyn_threshold,
                "vision_detected": int(vision_detected),
                "audio_detected": int(audio_detected),
            }
        )

        if verbose:
            print("Context:", ctx)
            print("Weights:", weights["vision"], weights["audio"])
            print("Fusion score:", fusion_score)
            print("Threshold:", dyn_threshold)

    metrics = classification_metrics(y_true, y_preds)
    _write_outputs(fused_rows, metrics, outputs_dir, name_suffix=name_suffix)
    return metrics, fused_rows


def fuse_predictions(
    vision_preds,
    audio_preds: List[AudioPred],
    outputs_dir: Path,
) -> tuple[dict, List[dict]]:
    if USE_NEW_FUSION:
        return fuse_predictions_new(vision_preds, audio_preds, outputs_dir)
    return fusion_old.fuse_predictions(vision_preds, audio_preds, outputs_dir)


def tune_fusion_parameters(
    vision_preds,
    audio_preds: List[AudioPred],
    outputs_dir: Path,
    thresholds: List[float] | None = None,
    weight_scales: List[Tuple[float, float]] | None = None,
    smoothing_temps: List[float] | None = None,
) -> dict:
    thresholds = thresholds or [0.3, 0.35, 0.4, 0.45]
    weight_scales = weight_scales or [(1.0, 1.0), (0.9, 1.1), (1.1, 0.9)]
    smoothing_temps = smoothing_temps or [1.0]

    results: List[dict] = []
    best = {"f1": -1.0}
    for th in thresholds:
        for ws in weight_scales:
            for temp in smoothing_temps:
                metrics, _ = fuse_predictions_new(
                    vision_preds,
                    audio_preds,
                    outputs_dir,
                    name_suffix=f"tune_th{th}_w{ws[0]:.2f}_{ws[1]:.2f}_t{temp:.2f}",
                    threshold_override=th,
                    weight_scale=ws,
                    smoothing_temp=temp,
                    verbose=False,
                )
                entry = {"threshold": th, "audio_weight_scale": ws[0], "vision_weight_scale": ws[1], "temp": temp, **metrics}
                results.append(entry)
                if metrics.get("f1", 0) > best.get("f1", -1):
                    best = entry

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    tuning_path = outputs_dir / "fusion_tuning_results.txt"
    lines = [
        "=== FUSION TUNING RESULTS ===",
        *(f"th={r['threshold']:.2f} aw={r['audio_weight_scale']:.2f} vw={r['vision_weight_scale']:.2f} temp={r['temp']:.2f} F1={r['f1']:.3f} Recall={r['recall']:.3f}" for r in results),
        "",
        f"Best -> th={best.get('threshold'):.2f} aw={best.get('audio_weight_scale'):.2f} vw={best.get('vision_weight_scale'):.2f} temp={best.get('temp'):.2f} F1={best.get('f1', 0):.3f}",
    ]
    tuning_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved fusion tuning results to %s", tuning_path)
    return best


def compare_fusions(
    vision_preds,
    audio_preds: List[AudioPred],
    outputs_dir: Path,
) -> Tuple[dict, List[dict], dict, List[dict]]:
    outputs_dir = Path(outputs_dir)
    old_metrics, old_rows = fusion_old.fuse_predictions(vision_preds, audio_preds, outputs_dir / "old_fusion")
    new_metrics, new_rows = fuse_predictions_new(vision_preds, audio_preds, outputs_dir / "new_fusion", name_suffix="new")

    comparison_path = outputs_dir / "fusion_comparison.txt"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(
        "\n".join(
            [
                "=== FUSION COMPARISON ===",
                f"Old F1: {old_metrics.get('f1', 0):.3f}",
                f"New F1: {new_metrics.get('f1', 0):.3f}",
                f"Old Recall: {old_metrics.get('recall', 0):.3f}",
                f"New Recall: {new_metrics.get('recall', 0):.3f}",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Saved fusion comparison to %s", comparison_path)
    return old_metrics, old_rows, new_metrics, new_rows
