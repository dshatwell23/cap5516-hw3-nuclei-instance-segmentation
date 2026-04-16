from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.config import add_common_arguments, config_from_args
from nuinsseg_sam.data import prepare_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic NuInsSeg manifest and folds.")
    add_common_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    summary = prepare_splits(
        dataset_root=config.resolved_dataset_root(),
        split_root=config.resolved_split_root(),
        n_splits=config.n_splits,
        outer_seed=config.random_seed,
        val_fraction=config.val_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
