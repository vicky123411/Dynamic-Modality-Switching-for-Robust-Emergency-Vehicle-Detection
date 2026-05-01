from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass
class VisionRecord:
    image_path: Path
    label: int


def _ensure_dirs(root: Path) -> dict:
    paths = {
        "images_train": root / "images" / "train",
        "images_val": root / "images" / "val",
        "labels_train": root / "labels" / "train",
        "labels_val": root / "labels" / "val",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def _write_label(label_path: Path, label: int, bbox: Tuple[float, float, float, float]) -> None:
    label_path.parent.mkdir(parents=True, exist_ok=True)
    if label == 1:
        content = f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
        label_path.write_text(content, encoding="utf-8")
    else:
        label_path.write_text("", encoding="utf-8")


def prepare_vision_dataset(
    base_dir: Path,
    output_dir: Path,
    val_split: float = 0.2,
    seed: int = 42,
    bbox_default: Tuple[float, float, float, float] = (0.5, 0.5, 0.6, 0.6),
    drop_negative: bool = False,
) -> List[VisionRecord]:
    output_dir = Path(output_dir)
    base_dir = Path(base_dir)
    metadata_path = output_dir / "val_metadata.csv"
    if drop_negative:
        # Clean stale splits so we can rebuild without background-only samples.
        shutil.rmtree(output_dir / "images", ignore_errors=True)
        shutil.rmtree(output_dir / "labels", ignore_errors=True)
        metadata_path.unlink(missing_ok=True)

    if metadata_path.exists() and not drop_negative:
        logger.info("Using existing vision val metadata at %s", metadata_path)
        df = pd.read_csv(metadata_path)
        return [VisionRecord(Path(p), int(l)) for p, l in zip(df.image_path, df.label)]

    paths = _ensure_dirs(output_dir)
    train_csv = base_dir / "train.csv"
    images_root = base_dir / "train"
    df = pd.read_csv(train_csv)
    df = df.rename(columns={df.columns[0]: "image", df.columns[1]: "label"})
    if drop_negative:
        df = df[df["label"] == 1]
        if df.empty:
            raise ValueError("No positive samples found after dropping negatives.")
        stratify = None
    else:
        stratify = df["label"]

    train_df, val_df = train_test_split(
        df, test_size=val_split, random_state=seed, stratify=stratify
    )

    def _process_split(split_df: pd.DataFrame, image_dest: Path, label_dest: Path):
        records: List[VisionRecord] = []
        for row in split_df.itertuples(index=False):
            src = images_root / str(row.image)
            dst = image_dest / str(row.image)
            if not dst.exists():
                shutil.copy2(src, dst)
            label_path = label_dest / (dst.stem + ".txt")
            _write_label(label_path, int(row.label), bbox_default)
            records.append(VisionRecord(dst, int(row.label)))
        return records

    train_records = _process_split(train_df, paths["images_train"], paths["labels_train"])
    val_records = _process_split(val_df, paths["images_val"], paths["labels_val"])

    val_meta = pd.DataFrame({
        "image_path": [str(r.image_path) for r in val_records],
        "label": [r.label for r in val_records],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    val_meta.to_csv(metadata_path, index=False)
    logger.info("Prepared vision dataset with %d train and %d val samples", len(train_records), len(val_records))
    return val_records


def write_vision_yaml(root: Path, yaml_path: Path) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    content = """path: data/vision
train: images/train
val: images/val

names:
  0: emergency_vehicle
"""
    yaml_path.write_text(content, encoding="utf-8")
    logger.info("vision.yaml written to %s", yaml_path)


def validate_dataset(dataset_root: Path) -> list[str]:
    """Lightweight validation to flag missing/invalid labels without stopping execution."""
    dataset_root = Path(dataset_root)
    warnings: list[str] = []
    images_dir = dataset_root / "images"
    labels_dir = dataset_root / "labels"
    exts = {".jpg", ".jpeg", ".png"}

    def _warn(msg: str):
        warnings.append(msg)
        logger.warning(msg)

    for split in ["train", "val"]:
        split_images = []
        split_root = images_dir / split
        for ext in exts:
            split_images.extend(split_root.glob(f"**/*{ext}"))

        for img_path in split_images:
            lbl_path = labels_dir / split / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                _warn(f"Missing label for {img_path}")
                continue
            content = lbl_path.read_text(encoding="utf-8").strip()
            if not content:
                _warn(f"Empty label file: {lbl_path}")
                continue
            for line in content.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    _warn(f"Invalid label format in {lbl_path}: {line}")
                    continue
                try:
                    cls_id, cx, cy, w, h = map(float, parts)
                except ValueError:
                    _warn(f"Non-numeric label in {lbl_path}: {line}")
                    continue
                if cls_id not in (0.0, 0):
                    _warn(f"Unexpected class id {cls_id} in {lbl_path}")
                for name, val in {"cx": cx, "cy": cy, "w": w, "h": h}.items():
                    if not (0.0 <= val <= 1.0):
                        _warn(f"Bounding box {name} out of range in {lbl_path}: {val}")
    if not warnings:
        logger.info("Dataset validation passed: no blocking issues found")
    return warnings
