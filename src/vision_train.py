from __future__ import annotations

import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from ultralytics import YOLO

from .metrics import classification_metrics, save_confusion_matrix

logger = logging.getLogger(__name__)


@dataclass
class VisionPrediction:
    image_path: Path
    gt: int
    pred: int
    confidence: float
    context: dict | None = None


def _extract_box_metrics(results) -> Tuple[float, float, float]:
    box = results.box
    precision = float(box.mp)
    recall = float(box.mr)
    map50 = float(box.map50)
    return precision, recall, map50


def train_yolo(
    data_yaml: Path,
    weights_out: Path,
    epochs: int = 30,
    imgsz: int = 512,
    batch: int | str = "auto",
    device: str | None = None,
    model_size: str = "yolov8n",
    freeze_backbone: bool = False,
    enable_aug: bool = True,
    mosaic: float = 1.0,
    fliplr: float = 0.5,
    scale: float = 0.5,
) -> dict:
    weights_name = f"{model_size}.pt"
    model = YOLO(weights_name)
    aug_kwargs = {}
    if enable_aug:
        aug_kwargs = {
            "mosaic": mosaic,
            "fliplr": fliplr,
            "scale": scale,
        }
    freeze_layers = 10 if freeze_backbone else 0
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/detect",
        name="emergency",
        exist_ok=True,
        pretrained=True,
        freeze=freeze_layers,
        **aug_kwargs,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_path, weights_out)
    logger.info("YOLO training complete. Best weights at %s", weights_out)
    metrics, _ = evaluate_yolo(
        weights_out,
        data_yaml,
        [],
        outputs_dir=weights_out.parent,
        imgsz=imgsz,
        skip_predictions=True,
    )
    return metrics


def _predict_single(model: YOLO, image_path: Path, imgsz: int, conf: float) -> float:
    res = model.predict(source=str(image_path), imgsz=imgsz, conf=conf, verbose=False)[0]
    score = 0.0
    if res.boxes is None or len(res.boxes) == 0:
        logger.debug("No detections for %s", image_path)
        return score
    for box in res.boxes:
        cls_id = int(box.cls[0])
        if cls_id == 0:
            score = float(box.conf[0])
            break
    return score


def evaluate_yolo(
    weights_path: Path,
    data_yaml: Path,
    val_records: Iterable,
    outputs_dir: Path,
    imgsz: int = 512,
    conf: float = 0.25,
    skip_predictions: bool = False,
) -> dict:
    model = YOLO(str(weights_path))
    val_results = model.val(data=str(data_yaml), imgsz=imgsz, verbose=False)
    det_precision, det_recall, map50 = _extract_box_metrics(val_results)
    det_f1 = (2 * det_precision * det_recall / (det_precision + det_recall + 1e-8)) if det_precision + det_recall > 0 else 0.0
    cls_metrics = {
        "precision": det_precision,
        "recall": det_recall,
        "f1": det_f1,
        "accuracy": 0.0,
        "map50": map50,
    }

    predictions: List[VisionPrediction] = []
    if not skip_predictions:
        for rec in val_records:
            score = _predict_single(model, rec.image_path, imgsz=imgsz, conf=conf)
            pred_label = 1 if score > 0 else 0
            predictions.append(VisionPrediction(rec.image_path, rec.label, pred_label, score))
        cls_metrics = classification_metrics([p.gt for p in predictions], [p.confidence for p in predictions])
        cls_metrics["map50"] = map50
        cls_metrics["det_precision"] = det_precision
        cls_metrics["det_recall"] = det_recall
        cls_metrics["det_f1"] = det_f1

        outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pred_path = outputs_dir / "vision_predictions.csv"
        df = pd.DataFrame(
            {
                "image_path": [str(p.image_path) for p in predictions],
                "gt": [p.gt for p in predictions],
                "pred": [p.pred for p in predictions],
                "confidence": [p.confidence for p in predictions],
            }
        )
        df.to_csv(pred_path, index=False)
        save_confusion_matrix(cls_metrics["confusion_matrix"], outputs_dir / "vision_confusion_matrix.csv")
        logger.info("Saved vision predictions to %s", pred_path)

    logger.info(
        "Vision metrics - Precision: %.3f Recall: %.3f F1: %.3f Acc: %.3f mAP@0.5: %.3f",
        cls_metrics.get("precision", 0),
        cls_metrics.get("recall", 0),
        cls_metrics.get("f1", 0),
        cls_metrics.get("accuracy", 0),
        cls_metrics.get("map50", 0),
    )
    return cls_metrics, predictions
