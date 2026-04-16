from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries

from .runtime import PROJECT_ROOT


def _overlay_boundaries(image_rgb: np.ndarray, instances: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = image_rgb.copy()
    boundaries = find_boundaries(instances > 0 if instances.dtype == bool else instances, mode="outer")
    output[boundaries] = np.array(color, dtype=np.uint8)
    return output


def save_prediction_panel(
    image_path: str,
    gt_binary: np.ndarray,
    pred_binary: np.ndarray,
    gt_instances: np.ndarray,
    pred_instances: np.ndarray,
    metrics: Dict[str, float],
    save_path: Path,
) -> None:
    raw_image = np.array(Image.open(PROJECT_ROOT / image_path).convert("RGB"))
    gt_overlay = _overlay_boundaries(raw_image, gt_instances, (255, 255, 0))
    pred_overlay = _overlay_boundaries(raw_image, pred_instances, (0, 255, 255))

    figure, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(raw_image)
    axes[0].set_title("Image")
    axes[1].imshow(gt_binary, cmap="gray")
    axes[1].set_title("GT Binary")
    axes[2].imshow(pred_binary, cmap="gray")
    axes[2].set_title("Pred Binary")
    axes[3].imshow(gt_overlay)
    axes[3].set_title("GT Overlay")
    axes[4].imshow(pred_overlay)
    axes[4].set_title(
        f"Pred Overlay\nDice {metrics['dice']:.3f} | AJI {metrics['aji']:.3f} | PQ {metrics['pq']:.3f}"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def generate_qualitative_figures(
    aggregate_metrics_csv: Path, output_dir: Path, max_examples_per_organ: int
) -> None:
    with aggregate_metrics_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["organ"], []).append(row)

    for organ in grouped:
        grouped[organ].sort(key=lambda row: row["sample_id"])

    for organ, organ_rows in grouped.items():
        for row in organ_rows[:max_examples_per_organ]:
            gt_binary = np.array(Image.open(PROJECT_ROOT / row["binary_mask_path"]).convert("L")) > 0
            pred_binary = np.array(Image.open(PROJECT_ROOT / row["prediction_binary_path"]).convert("L")) > 0
            gt_instances = np.array(Image.open(PROJECT_ROOT / row["instance_mask_path"]))
            pred_instances = np.array(Image.open(PROJECT_ROOT / row["prediction_instance_path"]))
            save_prediction_panel(
                image_path=row["image_path"],
                gt_binary=gt_binary,
                pred_binary=pred_binary,
                gt_instances=gt_instances,
                pred_instances=pred_instances,
                metrics={
                    "dice": float(row["dice"]),
                    "aji": float(row["aji"]),
                    "pq": float(row["pq"]),
                },
                save_path=output_dir / organ / f"{row['sample_id']}.png",
            )
