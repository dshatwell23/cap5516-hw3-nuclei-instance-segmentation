from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.config import ExperimentConfig
from nuinsseg_sam.trainer import evaluate_checkpoint


def _load_config(run_dir: Path) -> ExperimentConfig:
    config_dict = json.loads((run_dir / "config.json").read_text())
    return ExperimentConfig(**config_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an existing NuInsSeg checkpoint.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--split-csv", default=None)
    parser.add_argument("--split-name", default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-predictions", default="true")
    parser.add_argument("--mobile-sam-ckpt", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    config = _load_config(run_dir)
    if args.mobile_sam_ckpt:
        config.mobile_sam_ckpt = args.mobile_sam_ckpt

    checkpoint_path = Path(args.checkpoint_path).resolve() if args.checkpoint_path else run_dir / "checkpoint_best.pth"
    split_csv = (
        Path(args.split_csv).resolve()
        if args.split_csv
        else config.resolved_split_root() / f"fold_{config.fold}" / f"{args.split_name}.csv"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / f"reeval_{args.split_name}"
    result = evaluate_checkpoint(
        config=config,
        split_csv=split_csv,
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        split_name=args.split_name,
        save_predictions=str(args.save_predictions).lower() in {"1", "true", "yes", "y"},
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
