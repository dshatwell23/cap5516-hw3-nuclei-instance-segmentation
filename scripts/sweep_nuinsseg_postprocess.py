from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.config import ExperimentConfig, str2bool
from nuinsseg_sam.modeling import build_model
from nuinsseg_sam.trainer import (
    _load_best_checkpoint,
    build_eval_dataloader,
    evaluate_loader,
    save_metric_rows,
)


POSTPROCESS_FIELDS = (
    "probability_threshold",
    "min_object_size",
    "peak_min_distance",
    "peak_threshold_abs",
)


def _load_config(run_dir: Path) -> ExperimentConfig:
    config_dict = json.loads((run_dir / "config.json").read_text())
    return ExperimentConfig(**config_dict)


def _normalize_grid_dict(raw_grid: Dict[str, object]) -> Dict[str, List[float | int]]:
    normalized: Dict[str, List[float | int]] = {}
    for key, values in raw_grid.items():
        if key not in POSTPROCESS_FIELDS:
            raise ValueError(
                f"Unsupported grid key '{key}'. Supported keys: {', '.join(POSTPROCESS_FIELDS)}"
            )
        if not isinstance(values, list) or not values:
            raise ValueError(f"Grid values for '{key}' must be a non-empty list.")
        normalized[key] = list(values)
    if not normalized:
        raise ValueError("Parameter grid is empty.")
    return normalized


def _build_grid_from_args(args: argparse.Namespace) -> Dict[str, List[float | int]]:
    if args.grid_json and args.grid_path:
        raise ValueError("Use only one of --grid-json or --grid-path.")

    if args.grid_json:
        return _normalize_grid_dict(json.loads(args.grid_json))
    if args.grid_path:
        return _normalize_grid_dict(json.loads(Path(args.grid_path).read_text()))

    grid: Dict[str, List[float | int]] = {}
    if args.probability_thresholds:
        grid["probability_threshold"] = args.probability_thresholds
    if args.min_object_sizes:
        grid["min_object_size"] = args.min_object_sizes
    if args.peak_min_distances:
        grid["peak_min_distance"] = args.peak_min_distances
    if args.peak_threshold_abs_values:
        grid["peak_threshold_abs"] = args.peak_threshold_abs_values
    return _normalize_grid_dict(grid)


def _iter_grid(grid: Dict[str, List[float | int]]) -> Iterable[Dict[str, float | int]]:
    keys = list(grid.keys())
    for combination in itertools.product(*(grid[key] for key in keys)):
        yield dict(zip(keys, combination))


def _sample_random_configs(
    grid: Dict[str, List[float | int]], num_trials: int, random_seed: int
) -> List[Dict[str, float | int]]:
    rng = random.Random(random_seed)
    sampled_configs: List[Dict[str, float | int]] = []
    for _ in range(num_trials):
        sampled_config: Dict[str, float | int] = {}
        for key, values in grid.items():
            lower = min(values)
            upper = max(values)
            if key in {"min_object_size", "peak_min_distance"}:
                sampled_config[key] = rng.randint(int(lower), int(upper))
            else:
                sampled_config[key] = rng.uniform(float(lower), float(upper))
        sampled_configs.append(sampled_config)
    return sampled_configs


def _write_csv(csv_path: Path, rows: List[Dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep NuInsSeg postprocessing parameters for an existing trained run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--split-name", default="val")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--metric", default="aji_mean", choices=("dice_mean", "aji_mean", "pq_mean"))
    parser.add_argument("--search-mode", default="grid", choices=("grid", "random"))
    parser.add_argument("--num-trials", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-predictions", type=str2bool, default=False)
    parser.add_argument("--use-autocast", type=str2bool, default=True)
    parser.add_argument("--mobile-sam-ckpt", default=None)
    parser.add_argument("--grid-json", default=None)
    parser.add_argument("--grid-path", default=None)
    parser.add_argument("--probability-thresholds", nargs="*", type=float, default=None)
    parser.add_argument("--min-object-sizes", nargs="*", type=int, default=None)
    parser.add_argument("--peak-min-distances", nargs="*", type=int, default=None)
    parser.add_argument("--peak-threshold-abs-values", nargs="*", type=float, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else run_dir / f"postprocess_sweep_{args.split_name}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _load_config(run_dir)
    if args.mobile_sam_ckpt:
        config.mobile_sam_ckpt = args.mobile_sam_ckpt

    checkpoint_path = (
        Path(args.checkpoint_path).resolve() if args.checkpoint_path else run_dir / "checkpoint_best.pth"
    )
    split_csv = (
        Path(args.split_csv).resolve()
        if args.split_csv
        else config.resolved_split_root() / f"fold_{config.fold}" / f"{args.split_name}.csv"
    )
    grid = _build_grid_from_args(args)
    if args.search_mode == "random" and args.num_trials <= 0:
        raise ValueError("--num-trials must be positive when --search-mode random is used.")

    combinations = (
        list(_iter_grid(grid))
        if args.search_mode == "grid"
        else _sample_random_configs(grid, num_trials=args.num_trials, random_seed=args.random_seed)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = build_eval_dataloader(
        config,
        split_csv=split_csv,
        split_name=args.split_name,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    model, parameter_summary = build_model(config, device=device)
    del parameter_summary
    _load_best_checkpoint(model, checkpoint_path, device)

    sweep_rows: List[Dict[str, object]] = []
    for combo_index, combo in enumerate(combinations, start=1):
        sweep_config = replace(config, **combo)
        combo_name = f"combo_{combo_index:03d}"
        combo_output_dir = output_dir / combo_name
        summary, rows = evaluate_loader(
            model=model,
            dataloader=dataloader,
            config=sweep_config,
            device=device,
            output_dir=combo_output_dir,
            split_name=args.split_name,
            save_predictions=args.save_predictions,
            use_autocast=args.use_autocast,
        )
        save_metric_rows(combo_output_dir / f"{args.split_name}_metrics.csv", rows)
        (combo_output_dir / f"{args.split_name}_summary.json").write_text(json.dumps(summary, indent=2))
        row: Dict[str, object] = {
            "combo_id": combo_name,
            "split_name": args.split_name,
            "metric": args.metric,
            "search_mode": args.search_mode,
            "output_dir": str(combo_output_dir),
            "probability_threshold": sweep_config.probability_threshold,
            "min_object_size": sweep_config.min_object_size,
            "peak_min_distance": sweep_config.peak_min_distance,
            "peak_threshold_abs": sweep_config.peak_threshold_abs,
            "loss_mean": float(summary["loss_mean"]),
            "dice_mean": float(summary["dice_mean"]),
            "aji_mean": float(summary["aji_mean"]),
            "pq_mean": float(summary["pq_mean"]),
        }
        sweep_rows.append(row)
        print(json.dumps(row))

    sweep_rows.sort(key=lambda row: float(row[args.metric]), reverse=True)
    best_row = sweep_rows[0]

    _write_csv(output_dir / "sweep_results.csv", sweep_rows)
    (output_dir / "sweep_results.json").write_text(json.dumps(sweep_rows, indent=2))
    (output_dir / "best_result.json").write_text(json.dumps(best_row, indent=2))
    (output_dir / "grid.json").write_text(json.dumps(grid, indent=2))
    (output_dir / "search_config.json").write_text(
        json.dumps(
            {
                "search_mode": args.search_mode,
                "metric": args.metric,
                "num_trials": args.num_trials if args.search_mode == "random" else len(combinations),
                "random_seed": args.random_seed if args.search_mode == "random" else None,
                "eval_batch_size": args.eval_batch_size if args.eval_batch_size is not None else config.eval_batch_size,
                "num_workers": args.num_workers if args.num_workers is not None else config.num_workers,
                "use_autocast": args.use_autocast,
            },
            indent=2,
        )
    )

    print(json.dumps({"best_metric": args.metric, "best_result": best_row}, indent=2))


if __name__ == "__main__":
    main()
