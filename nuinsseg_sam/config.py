from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from .runtime import PROJECT_ROOT


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class ExperimentConfig:
    dataset_root: str = "archive"
    output_root: str = "runs/nuinsseg_mobilesam_lora"
    split_root: Optional[str] = None
    mobile_sam_ckpt: str = ""
    arch: str = "vit_t"
    random_seed: int = 19
    n_splits: int = 5
    val_fraction: float = 0.125
    image_size: int = 1024
    batch_size: int = 4
    eval_batch_size: int = 1
    num_workers: int = 4
    max_steps: int = 4000
    eval_every: int = 500
    log_every: int = 20
    save_every: int = 500
    grad_accum_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    amp: bool = True
    early_stop_patience: int = 0
    fold: int = 0
    folds: List[int] = field(default_factory=list)
    lora_rank: int = 4
    enable_encoder_lora: bool = True
    enable_decoder_lora: bool = True
    encoder_lora_layers: List[int] = field(default_factory=list)
    probability_threshold: float = 0.5
    min_object_size: int = 10
    peak_min_distance: int = 5
    peak_threshold_abs: float = 0.1
    save_test_predictions: bool = True
    max_examples_per_organ: int = 5

    def resolved_dataset_root(self) -> Path:
        return (PROJECT_ROOT / self.dataset_root).resolve()

    def resolved_output_root(self) -> Path:
        return (PROJECT_ROOT / self.output_root).resolve()

    def resolved_split_root(self) -> Path:
        if self.split_root:
            return (PROJECT_ROOT / self.split_root).resolve()
        return self.resolved_output_root() / "splits"

    def resolved_ckpt(self) -> Path:
        return (PROJECT_ROOT / self.mobile_sam_ckpt).resolve()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", default="archive")
    parser.add_argument("--output-root", default="runs/nuinsseg_mobilesam_lora")
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--mobile-sam-ckpt", default="")
    parser.add_argument("--arch", default="vit_t")
    parser.add_argument("--random-seed", type=int, default=19)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.125)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--amp", type=str2bool, default=True)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", nargs="*", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--enable-encoder-lora", type=str2bool, default=True)
    parser.add_argument("--enable-decoder-lora", type=str2bool, default=True)
    parser.add_argument("--encoder-lora-layers", nargs="*", type=int, default=None)
    parser.add_argument("--probability-threshold", type=float, default=0.5)
    parser.add_argument("--min-object-size", type=int, default=10)
    parser.add_argument("--peak-min-distance", type=int, default=5)
    parser.add_argument("--peak-threshold-abs", type=float, default=0.1)
    parser.add_argument("--save-test-predictions", type=str2bool, default=True)
    parser.add_argument("--max-examples-per-organ", type=int, default=5)


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    folds = args.folds if args.folds is not None else []
    encoder_lora_layers = (
        args.encoder_lora_layers if args.encoder_lora_layers is not None else []
    )
    return ExperimentConfig(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        split_root=args.split_root,
        mobile_sam_ckpt=args.mobile_sam_ckpt,
        arch=args.arch,
        random_seed=args.random_seed,
        n_splits=args.n_splits,
        val_fraction=args.val_fraction,
        image_size=args.image_size,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        max_steps=args.max_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        save_every=args.save_every,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        amp=args.amp,
        early_stop_patience=args.early_stop_patience,
        fold=args.fold,
        folds=folds,
        lora_rank=args.lora_rank,
        enable_encoder_lora=args.enable_encoder_lora,
        enable_decoder_lora=args.enable_decoder_lora,
        encoder_lora_layers=encoder_lora_layers,
        probability_threshold=args.probability_threshold,
        min_object_size=args.min_object_size,
        peak_min_distance=args.peak_min_distance,
        peak_threshold_abs=args.peak_threshold_abs,
        save_test_predictions=args.save_test_predictions,
        max_examples_per_organ=args.max_examples_per_organ,
    )


def save_config(config: ExperimentConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dict = config.to_dict()
    json_path = output_dir / "config.json"
    yaml_path = output_dir / "config.yaml"
    json_path.write_text(json.dumps(config_dict, indent=2))
    yaml_path.write_text(yaml.safe_dump(config_dict, sort_keys=False))


def parse_folds(config: ExperimentConfig) -> List[int]:
    if config.folds:
        return sorted(set(config.folds))
    return list(range(config.n_splits))


def summarize_namespace(parser: argparse.ArgumentParser, argv: Optional[Iterable[str]] = None) -> ExperimentConfig:
    args = parser.parse_args(argv)
    return config_from_args(args)
