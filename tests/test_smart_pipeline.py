from pathlib import Path
import sys
import types
import unittest

try:
    import cv2  # noqa: F401
except ImportError:
    sys.modules["cv2"] = types.ModuleType("cv2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smart_pipeline import SmartPipeline


class SmartPipelineStateTests(unittest.TestCase):
    def test_invalid_runtime_parameters_fail_early(self):
        with self.assertRaises(ValueError):
            SmartPipeline({"fast_scale": 0})
        with self.assertRaises(ValueError):
            SmartPipeline({"keyframe_interval_hz": 0})
        with self.assertRaises(ValueError):
            SmartPipeline({"motion_window": 1})

    def test_event_history_is_bounded_but_counts_are_exact(self):
        pipe = SmartPipeline({"event_history_limit": 2})
        pipe._record_events([
            {"type": "motion_start"},
            {"type": "motion"},
            {"type": "motion_end"},
        ])
        self.assertEqual(2, len(pipe.events))
        self.assertEqual(3, sum(pipe.event_counts.values()))

    def test_keyframe_checks_do_not_repeat_every_frame(self):
        pipe = SmartPipeline({"keyframe_interval_hz": 1.0})
        self.assertTrue(pipe._should_check_keyframe(0.0))
        pipe.last_keyframe_t = 0.0
        self.assertFalse(pipe._should_check_keyframe(1.0))
        self.assertTrue(pipe._should_check_keyframe(3.0))
        self.assertFalse(pipe._should_check_keyframe(3.1))
        self.assertTrue(pipe._should_check_keyframe(4.0))

    def test_finalize_closes_active_segment_once(self):
        pipe = SmartPipeline()
        pipe.motion_active = True
        pipe.current_segment = {"start": 1.0, "end": 2.0, "max_ratio": 0.2}
        events = pipe.finalize(2.5)
        self.assertEqual("motion_end", events[0]["type"])
        self.assertEqual(1.0, events[0]["segment_start"])
        self.assertEqual(1, len(pipe.motion_segments))
        self.assertEqual([], pipe.finalize(3.0))

    def test_alignment_uses_half_open_boundaries(self):
        pipe = SmartPipeline()
        pipe.motion_segments = [
            {"start": 0.5, "end": 1.0, "max_ratio": 0.2},
            {"start": 1.0, "end": 1.5, "max_ratio": 0.3},
        ]
        pipe.keyframes = [
            {"t": 0.0, "file": "keyframes/first.jpg"},
            {"t": 1.0, "file": "keyframes/second.jpg"},
        ]
        aligned = pipe.align_asr_streaming([
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 2.0, "text": "b"},
        ])
        self.assertEqual(1, aligned[0]["linked_keyframes"])
        self.assertEqual(1, aligned[0]["linked_motion_segments"])
        self.assertEqual(1, aligned[1]["linked_motion_segments"])
        self.assertEqual("keyframes/second.jpg", aligned[1]["keyframes"][0]["file"])


if __name__ == "__main__":
    unittest.main()
