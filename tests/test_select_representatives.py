from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

try:
    import cv2
except ImportError:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.imread = lambda _path: np.zeros((2, 2, 3), dtype=np.uint8)
    sys.modules["cv2"] = fake_cv2
    cv2 = fake_cv2
if not hasattr(cv2, "imread"):
    cv2.imread = lambda _path: np.zeros((2, 2, 3), dtype=np.uint8)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import select_representatives as selector


class RepresentativeSelectionTests(unittest.TestCase):
    def test_invalid_parameters_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                selector.select_representatives(directory, interval=0)
            with self.assertRaises(ValueError):
                selector.select_representatives(directory, dedup_threshold=101)
            with self.assertRaises(ValueError):
                selector.select_representatives(directory, w_pix=2)

    def test_empty_directory_returns_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual([], selector.select_representatives(directory))

    def test_first_and_last_keyframes_are_hard_anchors(self):
        keyframes = [(0.0, "first.jpg"), (10.0, "middle.jpg"), (20.0, "last.jpg")]
        images = {
            "first.jpg": np.zeros((2, 2, 3), dtype=np.uint8),
            "middle.jpg": np.ones((2, 2, 3), dtype=np.uint8),
            "last.jpg": np.full((2, 2, 3), 2, dtype=np.uint8),
        }
        with mock.patch.object(selector, "load_keyframes", return_value=keyframes), \
             mock.patch.object(selector.cv2, "imread", side_effect=lambda path: images[path]), \
             mock.patch.object(selector, "_pixel_diff", side_effect=lambda a, b: float(np.mean(np.abs(a - b)))):
            selected = selector.select_representatives("unused", interval=15)
        self.assertEqual(0.0, selected[0]["t"])
        self.assertEqual(20.0, selected[-1]["t"])
        self.assertIn(selected[0]["reason"], {"first", "bucket"})
        self.assertEqual("last", selected[-1]["reason"])


if __name__ == "__main__":
    unittest.main()
