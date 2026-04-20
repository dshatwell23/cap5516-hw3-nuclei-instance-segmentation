from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = ("dice", "aji", "pq")


def abbreviate_class_name(name: str) -> str:
    if name.startswith("human "):
        return f"h. {name[len('human '):]}"
    if name.startswith("mouse "):
        return f"m. {name[len('mouse '):]}"
    return name


def load_class_metric_means(metrics_csv: Path) -> List[Dict[str, float | str]]:
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {metric: [] for metric in METRICS}
    )
    with metrics_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            organ = row["organ"]
            for metric in METRICS:
                grouped[organ][metric].append(float(row[metric]))

    summary_rows: List[Dict[str, float | str]] = []
    for organ, metric_values in grouped.items():
        summary_rows.append(
            {
                "organ": organ,
                "display_name": abbreviate_class_name(organ),
                "dice_mean": float(np.mean(metric_values["dice"])),
                "aji_mean": float(np.mean(metric_values["aji"])),
                "pq_mean": float(np.mean(metric_values["pq"])),
            }
        )
    return summary_rows


def save_summary_csv(summary_rows: List[Dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not summary_rows:
        return
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_metric(summary_rows: List[Dict[str, float | str]], metric_key: str, output_dir: Path) -> None:
    sorted_rows = sorted(summary_rows, key=lambda row: float(row[metric_key]), reverse=True)
    labels = [str(row["display_name"]) for row in sorted_rows]
    values = [float(row[metric_key]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(24, 8))
    bars = ax.bar(labels, values, color="#2E6F95")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric_key.replace("_", " ").upper())
    ax.set_title(f"{metric_key.replace('_', ' ').upper()} by Class")
    ax.tick_params(axis="x", rotation=90)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(value + 0.015, 0.995),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    fig.savefig(output_dir / f"{metric_key}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


"""
python scripts/plot_nuinsseg_class_performance.py \
    --run-dir /media/dshatwell/SharedData/courses/cap5516_medical_imaging_computing/cap5516-hw3-nuclei-instance-segmentation/runs/nuinsseg_mobilesam_lora_final
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot NuInsSeg cross-validation test performance across classes."
    )
    parser.add_argument("--run-dir", required=True, help="Run directory containing aggregate_test_metrics.csv")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional override for the aggregate test metrics CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for the figure output directory.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_csv = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv
        else run_dir / "aggregate_test_metrics.csv"
    )
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Aggregate metrics CSV not found: {metrics_csv}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / "class_performance_figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = load_class_metric_means(metrics_csv)
    save_summary_csv(summary_rows, output_dir / "class_metric_summary.csv")

    for metric_key in ("dice_mean", "aji_mean", "pq_mean"):
        plot_metric(summary_rows, metric_key=metric_key, output_dir=output_dir)

    print(f"Saved class performance figures to {output_dir}")


if __name__ == "__main__":
    main()
