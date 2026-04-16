from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .runtime import PROJECT_ROOT, project_relative


MANIFEST_FIELDS = [
    "sample_id",
    "organ",
    "image_path",
    "instance_mask_path",
    "binary_mask_path",
    "ignore_mask_path",
]


def _sorted_organ_dirs(dataset_root: Path) -> List[Path]:
    return sorted([path for path in dataset_root.iterdir() if path.is_dir()])


def _resolve_required_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def scan_dataset(dataset_root: Path) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for organ_dir in _sorted_organ_dirs(dataset_root):
        organ = organ_dir.name
        image_dir = organ_dir / "tissue images"
        instance_dir = organ_dir / "label masks modify"
        binary_dir = organ_dir / "mask binary"
        ignore_dir = organ_dir / "vague areas" / "mask binary"
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            instance_path = _resolve_required_file(
                instance_dir / f"{stem}.tif", "instance mask"
            )
            binary_path = _resolve_required_file(binary_dir / f"{stem}.png", "binary mask")
            ignore_path = _resolve_required_file(ignore_dir / f"{stem}.png", "ignore mask")
            records.append(
                {
                    "sample_id": stem,
                    "organ": organ,
                    "image_path": project_relative(image_path),
                    "instance_mask_path": project_relative(instance_path),
                    "binary_mask_path": project_relative(binary_path),
                    "ignore_mask_path": project_relative(ignore_path),
                }
            )
    records.sort(key=lambda row: (row["organ"], row["sample_id"]))
    return records


def write_csv(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _kfold_indices(n_samples: int, n_splits: int, random_state: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    indices = np.arange(n_samples)
    rng = np.random.RandomState(random_state)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        splits.append((train_indices, test_indices))
        current = stop
    return splits


def _make_inner_validation_split(
    train_rows: Sequence[Dict[str, str]], val_fraction: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in train_rows:
        grouped[row["organ"]].append(row)

    rng = np.random.RandomState(seed)
    train_split: List[Dict[str, str]] = []
    val_split: List[Dict[str, str]] = []
    for organ in sorted(grouped):
        group = list(grouped[organ])
        group.sort(key=lambda row: row["sample_id"])
        shuffled_indices = np.arange(len(group))
        rng.shuffle(shuffled_indices)
        desired = int(round(len(group) * val_fraction))
        if len(group) == 1:
            desired = 0
        else:
            desired = min(max(desired, 1), len(group) - 1)
        val_indices = set(shuffled_indices[:desired].tolist())
        for index, row in enumerate(group):
            if index in val_indices:
                val_split.append(row)
            else:
                train_split.append(row)

    train_split.sort(key=lambda row: (row["organ"], row["sample_id"]))
    val_split.sort(key=lambda row: (row["organ"], row["sample_id"]))
    return train_split, val_split


def prepare_splits(
    dataset_root: Path,
    split_root: Path,
    n_splits: int = 5,
    outer_seed: int = 19,
    val_fraction: float = 0.125,
) -> Dict[str, object]:
    records = scan_dataset(dataset_root)
    split_root.mkdir(parents=True, exist_ok=True)
    manifest_path = split_root / "manifest.csv"
    write_csv(records, manifest_path)

    outer_splits = _kfold_indices(len(records), n_splits=n_splits, random_state=outer_seed)
    fold_summaries: List[Dict[str, object]] = []
    for fold_index, (train_indices, test_indices) in enumerate(outer_splits):
        outer_train_rows = [records[index] for index in train_indices]
        test_rows = [records[index] for index in test_indices]
        train_rows, val_rows = _make_inner_validation_split(
            outer_train_rows, val_fraction=val_fraction, seed=outer_seed + fold_index
        )
        fold_dir = split_root / f"fold_{fold_index}"
        write_csv(train_rows, fold_dir / "train.csv")
        write_csv(val_rows, fold_dir / "val.csv")
        write_csv(test_rows, fold_dir / "test.csv")
        fold_summary = {
            "fold": fold_index,
            "train_size": len(train_rows),
            "val_size": len(val_rows),
            "test_size": len(test_rows),
        }
        fold_summaries.append(fold_summary)

    organs = sorted({row["organ"] for row in records})
    summary = {
        "dataset_root": project_relative(dataset_root),
        "manifest_path": project_relative(manifest_path),
        "sample_count": len(records),
        "organ_count": len(organs),
        "organs": organs,
        "n_splits": n_splits,
        "outer_seed": outer_seed,
        "val_fraction": val_fraction,
        "folds": fold_summaries,
    }
    (split_root / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def manifest_summary(manifest_rows: Sequence[Dict[str, str]]) -> Dict[str, int]:
    return {
        "sample_count": len(manifest_rows),
        "organ_count": len({row["organ"] for row in manifest_rows}),
    }
