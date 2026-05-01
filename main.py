from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
import torch

from src.audio_data import build_audio_loaders, prepare_audio_manifest
from src.audio_train import evaluate_audio_model, load_audio_model, train_audio_model
from src.context import detect_context
from src.fusion import AudioPred, fuse_predictions, compare_fusions, tune_fusion_parameters, USE_NEW_FUSION
from src.metrics import classification_metrics
from src.vision_data import prepare_vision_dataset, validate_dataset, write_vision_yaml
from src.vision_train import VisionPrediction, evaluate_yolo, train_yolo

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Emergency vehicle multimodal pipeline")
    parser.add_argument("--train", action="store_true", help="Run training for both modalities")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation for both modalities")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size for YOLO")
    parser.add_argument("--epochs-vision", type=int, default=80, help="YOLO training epochs")
    parser.add_argument("--epochs-audio", type=int, default=30, help="Audio training epochs")
    parser.add_argument("--batch", type=float, default=0.0, help="YOLO batch size (0 for auto)")
    parser.add_argument("--audio-batch", type=int, default=32, help="Audio batch size")
    parser.add_argument("--device", default=None, help="Device for YOLO (e.g., 0 for GPU)")
    parser.add_argument("--model-size", default="yolov8n", choices=["yolov8n", "yolov8s", "yolov8m"], help="YOLO model scale")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze YOLO backbone and fine-tune head")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split for datasets")
    parser.add_argument("--keep-vision-negatives", action="store_true", help="Keep negative vision samples (backgrounds)")
    parser.add_argument("--vision-conf", type=float, default=0.05, help="Confidence threshold for vision evaluation")
    parser.add_argument("--audio-lr", type=float, default=3e-3, help="Learning rate for audio model")
    parser.add_argument("--no-audio-balance", action="store_true", help="Disable class balancing for audio training")
    return parser.parse_args()


def add_context(preds: List[VisionPrediction]) -> List[VisionPrediction]:
    for p in preds:
        p.context = detect_context(p.image_path)
    return preds


def condition_breakdown(rows: List[dict], key: str, value: str) -> dict | None:
    subset = [r for r in rows if r.get(key) == value]
    if not subset:
        return None
    metrics = classification_metrics([r["gt"] for r in subset], [r["fusion_score"] for r in subset])
    return metrics


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    vision_root = root / "Emergency_Vehicles"
    audio_root = root / "UrbanSound8K"
    data_vision = root / "data" / "vision"
    data_audio = root / "data" / "audio"
    outputs_dir = root / "outputs"
    weights_dir = root / "weights"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    training_log = outputs_dir / "training_log.txt"
    evaluation_log = outputs_dir / "evaluation_results.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device %s", device)

    # Prepare datasets
    val_records = prepare_vision_dataset(
        vision_root,
        data_vision,
        val_split=args.val_split,
        drop_negative=not args.keep_vision_negatives,
    )
    validation_warnings = validate_dataset(data_vision)
    if validation_warnings:
        logger.warning("Dataset validation reported %d warning(s)", len(validation_warnings))
    vision_yaml = root / "vision.yaml"
    write_vision_yaml(data_vision, vision_yaml)

    _, train_manifest, val_manifest = prepare_audio_manifest(audio_root, data_audio)
    train_df = pd.read_csv(train_manifest)
    label_counts = train_df["label"].value_counts()
    pos_count = int(label_counts.get(1, 0))
    neg_count = int(label_counts.get(0, 0))
    pos_weight = torch.tensor(neg_count / max(1, pos_count), device=device) if pos_count > 0 else None
    train_loader, val_loader = build_audio_loaders(
        train_manifest,
        val_manifest,
        batch_size=args.audio_batch,
        num_workers=0,
        augment=True,
        balance=not args.no_audio_balance,
    )

    yolov8_weights = weights_dir / "yolov8n_emergency.pt"
    audio_weights = weights_dir / "audio_cnn.pt"

    if args.train:
        yolo_batch = int(args.batch) if args.batch > 0 else 0
        train_yolo(
            vision_yaml,
            yolov8_weights,
            epochs=args.epochs_vision,
            imgsz=args.imgsz,
            batch=yolo_batch,
            device=args.device,
            model_size=args.model_size,
            freeze_backbone=args.freeze_backbone,
        )
        train_audio_model(
            train_loader,
            val_loader,
            device=device,
            epochs=args.epochs_audio,
            weights_out=audio_weights,
            batch_size=args.audio_batch,
            lr=args.audio_lr,
            pos_weight=pos_weight,
        )
        with training_log.open("a", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "=== TRAINING RUN ===",
                        f"Vision epochs={args.epochs_vision} imgsz={args.imgsz} batch={yolo_batch} model={args.model_size} freeze_backbone={args.freeze_backbone}",
                        f"Audio epochs={args.epochs_audio} batch={args.audio_batch}",
                        "",
                    ]
                )
            )

    if args.evaluate:
        if not yolov8_weights.exists():
            logger.error("Missing YOLO weights at %s. Run with --train first.", yolov8_weights)
            return
        vision_metrics, vision_preds = evaluate_yolo(
            yolov8_weights,
            vision_yaml,
            val_records,
            outputs_dir=outputs_dir,
            imgsz=args.imgsz,
            conf=args.vision_conf,
        )
        vision_preds = add_context(vision_preds)

        if not audio_weights.exists():
            logger.error("Missing audio weights at %s. Run with --train first.", audio_weights)
            return
        audio_model = load_audio_model(audio_weights, device=device)
        audio_metrics, audio_pred_dicts = evaluate_audio_model(audio_model, val_loader, device=device, outputs_dir=outputs_dir)
        audio_preds = [AudioPred(**{k: d[k] for k in ["path", "gt", "prob", "pred"]}) for d in audio_pred_dicts]
        fusion_metrics, fused_rows = fuse_predictions(vision_preds, audio_preds, outputs_dir=outputs_dir)
        old_metrics, old_rows, new_metrics, new_rows = compare_fusions(vision_preds, audio_preds, outputs_dir=outputs_dir)
        fused_rows = new_rows

        best_fusion = tune_fusion_parameters(
            vision_preds,
            audio_preds,
            outputs_dir=outputs_dir,
        )

        # Bonus: per-condition summaries
        day_metrics = condition_breakdown(fused_rows, "lighting", "day")
        night_metrics = condition_breakdown(fused_rows, "lighting", "night")
        rain_metrics = condition_breakdown(fused_rows, "weather", "rain")

        print("=== VISION METRICS ===")
        print(f"Precision: {vision_metrics.get('precision', 0):.3f}")
        print(f"Recall: {vision_metrics.get('recall', 0):.3f}")
        print(f"F1-score: {vision_metrics.get('f1', 0):.3f}")
        print(f"Accuracy: {vision_metrics.get('accuracy', 0):.3f}")
        print(f"mAP@0.5: {vision_metrics.get('map50', 0):.3f}")

        print("\n=== AUDIO METRICS ===")
        print(f"Precision: {audio_metrics.get('precision', 0):.3f}")
        print(f"Recall: {audio_metrics.get('recall', 0):.3f}")
        print(f"F1-score: {audio_metrics.get('f1', 0):.3f}")
        print(f"Accuracy: {audio_metrics.get('accuracy', 0):.3f}")

        print("\n=== FUSION METRICS (MAIN) ===")
        print(f"Precision: {fusion_metrics.get('precision', 0):.3f}")
        print(f"Recall: {fusion_metrics.get('recall', 0):.3f}")
        print(f"F1-score: {fusion_metrics.get('f1', 0):.3f}")
        print(f"Accuracy: {fusion_metrics.get('accuracy', 0):.3f}")

        print("\n=== FUSION TUNING (BEST) ===")
        print(
            f"Threshold: {best_fusion.get('threshold', 0):.2f} | Audio scale: {best_fusion.get('audio_weight_scale', 1.0):.2f} | Vision scale: {best_fusion.get('vision_weight_scale', 1.0):.2f} | F1: {best_fusion.get('f1', 0):.3f}"
        )

        print("\n=== FUSION COMPARISON ===")
        print(f"Old F1: {old_metrics.get('f1', 0):.3f}")
        print(f"New F1: {new_metrics.get('f1', 0):.3f}")
        print(f"Old Recall: {old_metrics.get('recall', 0):.3f}")
        print(f"New Recall: {new_metrics.get('recall', 0):.3f}")

        if day_metrics:
            print("\n=== CONDITION: DAY ===")
            print(f"Accuracy: {day_metrics.get('accuracy', 0):.3f} | F1: {day_metrics.get('f1', 0):.3f}")
        if night_metrics:
            print("=== CONDITION: NIGHT ===")
            print(f"Accuracy: {night_metrics.get('accuracy', 0):.3f} | F1: {night_metrics.get('f1', 0):.3f}")
        if rain_metrics:
            print("=== CONDITION: RAIN ===")
            print(f"Accuracy: {rain_metrics.get('accuracy', 0):.3f} | F1: {rain_metrics.get('f1', 0):.3f}")

        with evaluation_log.open("a", encoding="utf-8") as f:
            f.write(
                "\n".join(
                    [
                        "=== EVALUATION RUN ===",
                        f"Vision P={vision_metrics.get('precision', 0):.3f} R={vision_metrics.get('recall', 0):.3f} F1={vision_metrics.get('f1', 0):.3f} Acc={vision_metrics.get('accuracy', 0):.3f} mAP50={vision_metrics.get('map50', 0):.3f}",
                        f"Audio P={audio_metrics.get('precision', 0):.3f} R={audio_metrics.get('recall', 0):.3f} F1={audio_metrics.get('f1', 0):.3f} Acc={audio_metrics.get('accuracy', 0):.3f}",
                        f"Fusion P={fusion_metrics.get('precision', 0):.3f} R={fusion_metrics.get('recall', 0):.3f} F1={fusion_metrics.get('f1', 0):.3f} Acc={fusion_metrics.get('accuracy', 0):.3f}",
                        f"Fusion tuned th={best_fusion.get('threshold')} aw={best_fusion.get('audio_weight_scale')} vw={best_fusion.get('vision_weight_scale')} F1={best_fusion.get('f1')} R={best_fusion.get('recall')}",
                        "",
                    ]
                )
            )


if __name__ == "__main__":
    main()
