from pathlib import Path
import json
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

try:
    import cv2  # noqa: F401
except ImportError:
    fake_cv2 = types.ModuleType("cv2")
    sys.modules["cv2"] = fake_cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import integrated_pipeline
from asr_sherpa import ASRConfigurationError
from integrated_pipeline import StreamingConsumer, _attach_keyframe_file


class FakeCapture:
    def __init__(self, _path):
        self.frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        values = {
            integrated_pipeline.cv2.CAP_PROP_FPS: 10.0,
            integrated_pipeline.cv2.CAP_PROP_FRAME_COUNT: 1,
            integrated_pipeline.cv2.CAP_PROP_FRAME_WIDTH: 4,
            integrated_pipeline.cv2.CAP_PROP_FRAME_HEIGHT: 4,
        }
        return values[prop]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        return None


class FakePipeline:
    def __init__(self, _config=None):
        self.event_history_limit = 2
        self.frame_count = 0
        self.keyframes = []

    def process_frame(self, _frame, timestamp, fps=None):
        record = {"t": timestamp, "frame_idx": self.frame_count, "reason": "first_frame"}
        self.keyframes.append(record)
        self.frame_count += 1
        return [{"type": "keyframe", **record}]

    def finalize(self, timestamp):
        return [{"type": "motion_end", "t": timestamp, "segment_start": 0.0, "duration": timestamp}]

    def get_summary(self):
        return {"total_frames": self.frame_count, "keyframes": len(self.keyframes)}

    def save_results(self, path):
        payload = {"schema_version": 1, "keyframes": self.keyframes}
        Path(path).write_text(json.dumps(payload), encoding="utf-8")
        return payload


class IntegratedPipelineHelpersTests(unittest.TestCase):
    def setUp(self):
        constants = {
            "CAP_PROP_FPS": 1,
            "CAP_PROP_FRAME_COUNT": 2,
            "CAP_PROP_FRAME_WIDTH": 3,
            "CAP_PROP_FRAME_HEIGHT": 4,
            "IMWRITE_JPEG_QUALITY": 95,
        }
        for name, value in constants.items():
            setattr(integrated_pipeline.cv2, name, value)

    def test_consumer_uses_explicit_segment_start_and_bounded_history(self):
        consumer = StreamingConsumer(history_limit=1)
        consumer.consume({"type": "motion_start", "t": 1.0})
        consumer.consume({"type": "motion_end", "t": 2.0, "segment_start": 1.0, "duration": 1.0})
        snapshot = consumer.snapshot()
        self.assertEqual(2, snapshot["event_count"])
        self.assertEqual(1, snapshot["retained_event_history"])
        self.assertEqual(1.0, snapshot["motion_segments"][0]["start"])

    def test_keyframe_file_is_attached_to_event_and_metadata(self):
        pipe = types.SimpleNamespace(keyframes=[{"frame_idx": 7, "t": 0.5}])
        event = {"type": "keyframe", "frame_idx": 7, "t": 0.5}
        _attach_keyframe_file(pipe, event, "keyframes/kf.jpg")
        self.assertEqual("keyframes/kf.jpg", event["file"])
        self.assertEqual("keyframes/kf.jpg", pipe.keyframes[0]["file"])

    def test_visual_only_run_writes_traceable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with mock.patch.object(integrated_pipeline.cv2, "VideoCapture", FakeCapture, create=True), \
                 mock.patch.object(integrated_pipeline.cv2, "imwrite", return_value=True, create=True), \
                 mock.patch.object(integrated_pipeline, "SmartPipeline", FakePipeline):
                pipe, aligned, asr_segments = integrated_pipeline.run_realtime_pipeline(
                    "input.mp4", str(output), visual_only=True
                )

            self.assertEqual([], aligned)
            self.assertEqual([], asr_segments)
            self.assertEqual(1, pipe.frame_count)
            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("succeeded", manifest["status"])
            self.assertEqual("disabled", manifest["asr"]["status"])
            self.assertFalse((output / "aligned_output.json").exists())
            self.assertEqual(2, len((output / "events.jsonl").read_text(encoding="utf-8").splitlines()))

    def test_asr_preflight_failure_is_recorded_and_not_downgraded(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            with mock.patch.object(integrated_pipeline.cv2, "VideoCapture", FakeCapture, create=True), \
                 mock.patch.object(integrated_pipeline, "require_ffmpeg", side_effect=ASRConfigurationError("missing")):
                with self.assertRaises(ASRConfigurationError):
                    integrated_pipeline.run_realtime_pipeline("input.mp4", str(output))

            manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual("failed", manifest["status"])
            self.assertEqual("failed", manifest["asr"]["status"])
            self.assertFalse((output / "aligned_output.json").exists())


if __name__ == "__main__":
    unittest.main()
