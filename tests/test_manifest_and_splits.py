from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.data import prepare_splits, read_csv_rows, scan_dataset


class ManifestAndSplitTests(unittest.TestCase):
    def test_manifest_counts_match_dataset(self) -> None:
        rows = scan_dataset(ROOT / "archive")
        self.assertEqual(len(rows), 665)
        self.assertEqual(len({row["organ"] for row in rows}), 31)

    def test_prepare_splits_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_root = Path(first_dir)
            second_root = Path(second_dir)
            first_summary = prepare_splits(ROOT / "archive", first_root, n_splits=5, outer_seed=19, val_fraction=0.125)
            second_summary = prepare_splits(ROOT / "archive", second_root, n_splits=5, outer_seed=19, val_fraction=0.125)

            self.assertEqual(first_summary["sample_count"], 665)
            self.assertEqual(first_summary["organ_count"], 31)
            first_summary_json = json.loads((first_root / "summary.json").read_text())
            second_summary_json = json.loads((second_root / "summary.json").read_text())
            self.assertEqual(first_summary_json["sample_count"], second_summary_json["sample_count"])
            self.assertEqual(first_summary_json["organ_count"], second_summary_json["organ_count"])
            self.assertEqual(first_summary_json["organs"], second_summary_json["organs"])
            self.assertEqual(first_summary_json["folds"], second_summary_json["folds"])

            for fold in range(5):
                for split_name in ("train", "val", "test"):
                    first_rows = read_csv_rows(first_root / f"fold_{fold}" / f"{split_name}.csv")
                    second_rows = read_csv_rows(second_root / f"fold_{fold}" / f"{split_name}.csv")
                    self.assertEqual(first_rows, second_rows)


if __name__ == "__main__":
    unittest.main()
