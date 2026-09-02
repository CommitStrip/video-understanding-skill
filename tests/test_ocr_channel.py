"""OCR 第三通道测试（W3）。

两层验证：
  1) mock rapidocr 引擎（monkeypatch 注入 sys.modules）：事件结构
     {t, type:"ocr", text, conf}、空文本过滤、空帧安全、未安装显式报错、
     run_realtime_pipeline 的 --ocr 接入（事件进 consumer 并落到
     aligned_output.json 的 ocr_events 键，现有键不受影响）。
  2) 真实引擎（已安装 rapidocr 时才跑）：合成文字帧能识别出文本。
"""
import json
import sys
import types

import cv2
import numpy as np
import pytest

from vus.integrated_pipeline import run_realtime_pipeline
from vus.ocr_channel import OcrChannel


class _FakeRapidOCR:
    """伪造 RapidOCR 引擎：任意帧返回两条固定识别结果（其中一条空文本）。"""

    def __init__(self):
        self.calls = 0

    def __call__(self, frame):
        self.calls += 1
        result = [
            [[[0, 0], [100, 0], [100, 20], [0, 20]], "ALERT 123", 0.92],
            [[[0, 30], [100, 30], [100, 50], [0, 50]], "   ", 0.80],  # 空文本应被过滤
        ]
        return result, 0.05


@pytest.fixture
def fake_rapidocr(monkeypatch):
    engine = _FakeRapidOCR()
    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = lambda: engine
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    return engine


def text_frame(text="HELLO"):
    img = np.full((240, 480, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img


# ==================== OcrChannel（mock 引擎） ====================

def test_ocr_event_structure(fake_rapidocr):
    ch = OcrChannel()
    events = ch.process(text_frame(), 12.3456)
    assert len(events) == 1, "空文本识别结果必须被过滤"
    ev = events[0]
    assert set(ev) == {"t", "type", "text", "conf"}
    assert ev["type"] == "ocr"
    assert ev["t"] == pytest.approx(12.346, abs=1e-3)  # round(12.3456, 3)
    assert isinstance(ev["t"], float)
    assert ev["text"] == "ALERT 123"
    assert ev["conf"] == pytest.approx(0.92, abs=1e-3)


def test_ocr_process_counts_engine_calls(fake_rapidocr):
    ch = OcrChannel()
    ch.process(text_frame(), 0.0)
    ch.process(text_frame(), 1.0)
    assert fake_rapidocr.calls == 2, "每关键帧恰好调用一次引擎"


def test_ocr_empty_frame_returns_empty_list(fake_rapidocr):
    ch = OcrChannel()
    assert ch.process(None, 0.0) == []
    assert ch.process(np.zeros((0, 0, 3), dtype=np.uint8), 0.0) == []
    assert fake_rapidocr.calls == 0, "空帧不应触发引擎调用"


def test_ocr_missing_package_raises_with_hint(monkeypatch):
    """rapidocr 未安装（模拟为 sys.modules 置 None -> ImportError）时显式报错。"""
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(RuntimeError) as ei:
        OcrChannel()
    msg = str(ei.value)
    assert "rapidocr" in msg and "pip install" in msg


# ==================== integrated_pipeline --ocr 接入 ====================

@pytest.fixture(scope="module")
def tiny_video(tmp_path_factory):
    """3 秒 320x240@15fps 静止视频（必然产出关键帧）。"""
    path = tmp_path_factory.mktemp("ocrvid") / "ocr.mp4"
    fps, w, h, dur = 15, 320, 240, 3.0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(int(fps * dur)):
        vw.write(np.full((h, w, 3), 200, dtype=np.uint8))
    vw.release()
    return str(path)


def test_pipeline_ocr_events_into_aligned_output(tiny_video, tmp_path, fake_rapidocr):
    out_dir = tmp_path / "out"
    run_realtime_pipeline(tiny_video, str(out_dir), save_keyframes=True,
                          config={"fast_scale": 0.25, "keyframe_interval_hz": 1.5},
                          ocr=True)

    doc = json.loads((out_dir / "aligned_output.json").read_text(encoding="utf-8"))
    # 新键存在且结构正确
    assert isinstance(doc["ocr_events"], list)
    assert len(doc["ocr_events"]) >= 1
    assert doc["ocr_events"][0]["type"] == "ocr"
    assert set(doc["ocr_events"][0]) == {"t", "type", "text", "conf"}
    # 按时间序
    ts = [e["t"] for e in doc["ocr_events"]]
    assert ts == sorted(ts)
    # 现有键不受影响
    for key in ("live", "aligned_segments", "asr_segments",
                "pipeline_summary", "stream_event_count"):
        assert key in doc


def test_pipeline_without_ocr_flag_keeps_zero_events(tiny_video, tmp_path):
    """默认不开 --ocr：ocr_events 键存在但为空，且不触发引擎导入。"""
    out_dir = tmp_path / "out"
    run_realtime_pipeline(tiny_video, str(out_dir), save_keyframes=False,
                          config={"fast_scale": 0.25})
    doc = json.loads((out_dir / "aligned_output.json").read_text(encoding="utf-8"))
    assert doc["ocr_events"] == []


def test_pipeline_ocr_missing_package_explicit_error(tiny_video, tmp_path, monkeypatch):
    """开 --ocr 但 rapidocr 未安装：显式报错（不静默降级为无文字事件）。"""
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", None)
    with pytest.raises(RuntimeError, match=r"rapidocr"):
        run_realtime_pipeline(tiny_video, str(tmp_path / "out"),
                              save_keyframes=False,
                              config={"fast_scale": 0.25}, ocr=True)


# ==================== 真实引擎（已安装才跑） ====================

def test_ocr_real_engine_recognizes_synthetic_text():
    pytest.importorskip("rapidocr_onnxruntime",
                        reason="rapidocr-onnxruntime 未安装: pip install -e \".[ocr]\"")
    ch = OcrChannel()
    events = ch.process(text_frame("FIRE ALARM 2026"), 1.0)
    assert isinstance(events, list)
    assert len(events) >= 1, "大号清晰黑字白底应至少识别出一条"
    for ev in events:
        assert set(ev) == {"t", "type", "text", "conf"}
        assert 0.0 <= ev["conf"] <= 1.0
