from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.metrics import aggregate_jaccard_index, dice_score, panoptic_quality


class MetricTests(unittest.TestCase):
    def test_perfect_match_scores_one(self) -> None:
        gt = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        pred = gt.copy()
        ignore = np.zeros_like(gt, dtype=bool)
        self.assertEqual(dice_score(pred > 0, gt > 0, ignore), 1.0)
        self.assertEqual(aggregate_jaccard_index(gt, pred, ignore), 1.0)
        self.assertEqual(panoptic_quality(gt, pred, ignore), 1.0)

    def test_ignore_mask_removes_penalty(self) -> None:
        gt = np.array([[1, 1], [0, 0]], dtype=np.int32)
        pred = np.array([[1, 0], [0, 0]], dtype=np.int32)
        ignore = np.array([[0, 1], [0, 0]], dtype=bool)
        self.assertEqual(dice_score(pred > 0, gt > 0, ignore), 1.0)
        self.assertEqual(aggregate_jaccard_index(gt, pred, ignore), 1.0)
        self.assertEqual(panoptic_quality(gt, pred, ignore), 1.0)

    def test_split_prediction_degrades_instance_metrics(self) -> None:
        gt = np.array(
            [
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
            ],
            dtype=np.int32,
        )
        pred = np.array(
            [
                [0, 1, 2, 0],
                [0, 1, 2, 0],
                [0, 3, 4, 0],
                [0, 3, 4, 0],
            ],
            dtype=np.int32,
        )
        ignore = np.zeros_like(gt, dtype=bool)
        self.assertLess(aggregate_jaccard_index(gt, pred, ignore), 1.0)
        self.assertLess(panoptic_quality(gt, pred, ignore), 1.0)


if __name__ == "__main__":
    unittest.main()
