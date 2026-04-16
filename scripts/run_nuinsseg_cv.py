from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.config import add_common_arguments, config_from_args, parse_folds, save_config
from nuinsseg_sam.data import prepare_splits
from nuinsseg_sam.trainer import run_cross_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full 5-fold NuInsSeg MobileSAM LoRA pipeline.")
    add_common_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    output_root = config.resolved_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    save_config(config, output_root)
    prepare_splits(
        dataset_root=config.resolved_dataset_root(),
        split_root=config.resolved_split_root(),
        n_splits=config.n_splits,
        outer_seed=config.random_seed,
        val_fraction=config.val_fraction,
    )
    result = run_cross_validation(config, folds=parse_folds(config))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
