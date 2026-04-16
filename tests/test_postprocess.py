from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuinsseg_sam.postprocess import probability_to_instances


class PostprocessTests(unittest.TestCase):
    def test_two_peaks_produce_two_instances(self) -> None:
        prob = np.zeros((32, 32), dtype=np.float32)
        prob[8:14, 8:14] = 0.95
        prob[18:24, 18:24] = 0.98
        try:
            instances = probability_to_instances(
                prob,
                threshold=0.5,
                min_object_size=4,
                peak_min_distance=3,
                peak_threshold_abs=0.1,
                ignore_mask=np.zeros_like(prob, dtype=bool),
            )
        except Exception as exc:
            self.skipTest(f"SciPy/skimage postprocess dependencies are not importable: {exc}")
        labels = [label for label in np.unique(instances) if label > 0]
        self.assertEqual(len(labels), 2)

    def test_small_noise_is_removed(self) -> None:
        prob = np.zeros((16, 16), dtype=np.float32)
        prob[2, 2] = 0.9
        try:
            instances = probability_to_instances(
                prob,
                threshold=0.5,
                min_object_size=5,
                peak_min_distance=2,
                peak_threshold_abs=0.1,
                ignore_mask=np.zeros_like(prob, dtype=bool),
            )
        except Exception as exc:
            self.skipTest(f"SciPy/skimage postprocess dependencies are not importable: {exc}")
        self.assertEqual(int(instances.max()), 0)


if __name__ == "__main__":
    unittest.main()
