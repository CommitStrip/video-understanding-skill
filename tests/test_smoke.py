"""端到端冒烟测试：合成小视频 -> run_realtime_pipeline -> 产物落盘。

CI 无相机/无模型时也必须通过：ASR 在 sherpa-onnx 缺失时走 mock，
ffmpeg 缺失时 extract_audio 返回 None、声音链自动跳过。
"""
import json
import cv2
import numpy as np
import pytest

from vus.integrated_pipeline import run_realtime_pipeline


@pytest.fixture(scope="module")
def smoke_video(tmp_path_factory):
    """3 秒 320x240@15fps：静止 -> 中段移动色块 -> 静止收尾（让运动段闭合）。"""
    path = tmp_path_factory.mktemp("video") / "smoke.mp4"
    fps, w, h, dur = 15, 320, 240, 3.0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "VideoWriter 打开失败"
    n = int(fps * dur)
    for i in range(n):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        if n // 3 <= i < 2 * n // 3:  # 中段 15 帧移动，其余静止
            cx = 30 + (i - n // 3) * 6
            cv2.circle(frame, (cx, 120), 18, (0, 200, 120), -1)
        vw.write(frame)
    vw.release()
    return str(path)


def test_smoke_pipeline_produces_artifacts(smoke_video, tmp_path):
    out_dir = tmp_path / "out"
    pipe, aligned, asr_segments = run_realtime_pipeline(
        smoke_video, str(out_dir), save_keyframes=True,
        config={"fast_scale": 0.25, "keyframe_interval_hz": 1.5},
    )

    assert pipe is not None
    assert pipe.get_summary()["total_frames"] == 45

    results_file = out_dir / "pipeline_results.json"
    aligned_file = out_dir / "aligned_output.json"
    assert results_file.exists(), "缺 pipeline_results.json"
    assert aligned_file.exists(), "缺 aligned_output.json"

    results = json.loads(results_file.read_text(encoding="utf-8"))
    assert results["summary"]["keyframes"] >= 1
    assert results["summary"]["total_frames"] == 45

    aligned_doc = json.loads(aligned_file.read_text(encoding="utf-8"))
    assert "pipeline_summary" in aligned_doc
    assert aligned_doc["pipeline_summary"]["total_frames"] == 45

    # 有运动 => 至少一个运动段被闭合
    assert len(results["motion_segments"]) >= 1


def test_smoke_event_timestamps_monotonic(smoke_video, tmp_path):
    pipe, _, _ = run_realtime_pipeline(
        smoke_video, str(tmp_path / "out2"), save_keyframes=False,
        config={"fast_scale": 0.25},
    )
    ts = [e["t"] for e in pipe.events]
    assert ts == sorted(ts), "事件时间戳必须单调不减"
