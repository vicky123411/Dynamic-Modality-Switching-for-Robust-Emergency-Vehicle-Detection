from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def parse_latest_evaluation_metrics(evaluation_file: Path) -> dict[str, dict[str, float]]:
    text = evaluation_file.read_text(encoding="utf-8")

    # Split by runs and keep the last non-empty run block
    runs = [block.strip() for block in text.split("=== EVALUATION RUN ===") if block.strip()]
    if not runs:
        raise ValueError("No evaluation runs found in evaluation_results.txt")

    latest = runs[-1]

    patterns = {
        "Vision": r"Vision\s+P=(?P<P>[0-9]*\.?[0-9]+)\s+R=(?P<R>[0-9]*\.?[0-9]+)\s+F1=(?P<F1>[0-9]*\.?[0-9]+)",
        "Audio": r"Audio\s+P=(?P<P>[0-9]*\.?[0-9]+)\s+R=(?P<R>[0-9]*\.?[0-9]+)\s+F1=(?P<F1>[0-9]*\.?[0-9]+)",
        "Fusion": r"Fusion\s+P=(?P<P>[0-9]*\.?[0-9]+)\s+R=(?P<R>[0-9]*\.?[0-9]+)\s+F1=(?P<F1>[0-9]*\.?[0-9]+)",
    }

    metrics: dict[str, dict[str, float]] = {}
    for model_name, pattern in patterns.items():
        match = re.search(pattern, latest)
        if not match:
            raise ValueError(f"Could not find {model_name} metrics in latest evaluation run")
        metrics[model_name] = {
            "Precision": float(match.group("P")),
            "Recall": float(match.group("R")),
            "F1-score": float(match.group("F1")),
        }

    return metrics


def plot_single_comparison_graph(metrics: dict[str, dict[str, float]], save_path: Path) -> None:
    models = ["Vision", "Audio", "Fusion"]
    precision = [metrics[m]["Precision"] for m in models]
    recall = [metrics[m]["Recall"] for m in models]
    f1_score = [metrics[m]["F1-score"] for m in models]

    x = np.arange(len(models))
    width = 0.24

    plt.figure(figsize=(10, 6))
    bars1 = plt.bar(x - width, precision, width, label="Precision")
    bars2 = plt.bar(x, recall, width, label="Recall")
    bars3 = plt.bar(x + width, f1_score, width, label="F1-score")

    plt.ylim(0, 1.05)
    plt.xticks(x, models)
    plt.ylabel("Score")
    plt.xlabel("Model")
    plt.title("Fig. 4. Comparison of Precision, Recall, and F1-score across Vision, Audio, and Fusion models")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend()

    # Label bars with values for readability
    for bars in (bars1, bars2, bars3):
        for bar in bars:
            h = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.015,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parent
    evaluation_file = root / "outputs" / "evaluation_results.txt"
    save_path = root / "outputs" / "performance_comparison.png"

    if not evaluation_file.exists():
        raise FileNotFoundError(f"Missing file: {evaluation_file}")

    metrics = parse_latest_evaluation_metrics(evaluation_file)
    plot_single_comparison_graph(metrics, save_path)
    print(f"Graph generated: {save_path}")


if __name__ == "__main__":
    main()
