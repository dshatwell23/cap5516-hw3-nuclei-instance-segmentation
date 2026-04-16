from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if importlib.util.find_spec("torch") is not None:
    from nuinsseg_sam.data import prepare_splits
    from nuinsseg_sam.dataset import NuInsSegDataset
else:
    prepare_splits = None
    NuInsSegDataset = None


@unittest.skipIf(prepare_splits is None or NuInsSegDataset is None, "torch is not installed")
class DatasetLoaderTests(unittest.TestCase):
    def test_dataset_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            split_root = Path(tmp_dir)
            prepare_splits(ROOT / "archive", split_root, n_splits=5, outer_seed=19, val_fraction=0.125)
            dataset = NuInsSegDataset(split_root / "fold_0" / "train.csv", image_size=1024, mode="train")
            sample = dataset[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 1024, 1024))
            self.assertEqual(tuple(sample["binary_mask"].shape), (1, 512, 512))
            self.assertEqual(tuple(sample["ignore_mask"].shape), (1, 512, 512))
            self.assertEqual(tuple(sample["instance_mask"].shape), (512, 512))


if __name__ == "__main__":
    unittest.main()
