"""W2 实时源测试：FrameSource 契约 + File/Camera/RTSP 三实现 + 管线接入。

不连真实 RTSP/摄像头：用 ScriptedCapture 伪造底层 VideoCapture
（子类覆写 RTSPSource._open_capture / monkeypatch cv2.VideoCapture），
验证背压丢帧、断流重连、恢复续读与 Ctrl-C 优雅落盘。
"""
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from vus.integrated_pipeline import run_realtime_pipeline
from vus.source import (
    FrameSource, FileSource, CameraSource, RTSPSource,
)

STATS_KEYS = {"frames_read", "frames_dropped", "reconnects", "source"}


# ---------------- fixtures / 伪造捕获 ----------------

@pytest.fixture(scope="module")
def video_1s(tmp_path_factory):
    """1 秒 64x48@30fps 合成视频（30 帧，中段有移动色块）。"""
    path = tmp_path_factory.mktemp("src_video") / "src30.mp4"
    fps, w, h, dur = 30, 64, 48, 1.0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "VideoWriter 打开失败"
    n = int(fps * dur)
    for i in range(n):
        frame = np.full((h, w, 3), 40, dtype=np.uint8)
        if n // 3 <= i < 2 * n // 3:
            cx = 8 + (i - n // 3) * 3
            cv2.circle(frame, (cx, h // 2), 5, (0, 200, 120), -1)
        vw.write(frame)
    vw.release()
    return str(path)


def make_frame(value):
    """恒定像素值帧（用像素值标识帧的来源阶段）。"""
    return np.full((16, 16, 3), value, dtype=np.uint8)


class ScriptedCapture:
    """脚本化伪造 cv2.VideoCapture。

    script: 元素为 ndarray（正常帧）或 "fail"（grab 返回 False，模拟断流）。
    脚本耗尽后 grab 持续返回 False（模拟流彻底结束）。
    """

    def __init__(self, script):
        self.script = list(script)
        self.pos = 0
        self.released = False
        self.last_frame = None
        self.props = {}

    def isOpened(self):
        return not self.released

    def grab(self):
        if self.released or self.pos >= len(self.script):
            return False
        item = self.script[self.pos]
        self.pos += 1
        if isinstance(item, str):  # "fail"
            return False
        self.last_frame = item
        return True

    def retrieve(self):
        if self.last_frame is None:
            return False, None
        return True, self.last_frame

    def read(self):
        if self.grab():
            return self.retrieve()
        return False, None

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def release(self):
        self.released = True


class ScriptedRTSP(RTSPSource):
    """用 ScriptedCapture 替换底层 VideoCapture 的 RTSPSource。

    scripts: 每次重连依次取一个脚本；耗尽后 _open_capture 返回 None
    （模拟重连也连不上）。
    """

    def __init__(self, scripts, **kw):
        super().__init__("rtsp://fake/test", **kw)
        self._scripts = [list(s) for s in scripts]

    def _open_capture(self):
        if not self._scripts:
            return None
        return ScriptedCapture(self._scripts.pop(0))


def read_all(src, max_iters=1000, pause=0.0):
    """持续 read 直到断流/结束，返回成功帧列表。"""
    frames = []
    for _ in range(max_iters):
        ok, frame, _ = src.read()
        if not ok:
            break
        frames.append(frame)
        if pause:
            time.sleep(pause)
    return frames


# ---------------- FrameSource 契约 ----------------

def test_frame_source_is_abstract():
    with pytest.raises(TypeError):
        FrameSource()  # 抽象类不可实例化


def test_file_source_contract_and_open_failure(tmp_path):
    src = FileSource(str(tmp_path / "missing.mp4"))
    assert isinstance(src, FrameSource)
    assert src.open() is False  # 打不开不抛异常
    assert STATS_KEYS <= set(src.stats.keys())
    assert src.stats["last_error"]  # 错误信息进 stats
    ok, frame, ts = src.read()
    assert ok is False and frame is None
    src.close()  # 可安全重复 close
    src.close()


def test_camera_source_contract(monkeypatch):
    fake = ScriptedCapture([make_frame(1), make_frame(2)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda index: fake)
    src = CameraSource(camera_index=0, width=320, height=240)
    assert isinstance(src, FrameSource)
    assert src.open() is True
    assert STATS_KEYS <= set(src.stats.keys())
    ok, frame, ts = src.read()
    assert ok is True and frame is not None
    assert ts > 0  # 单调时钟时间戳
    ok2, frame2, ts2 = src.read()
    assert ok2 and ts2 >= ts
    ok3, _, _ = src.read()
    assert ok3 is False  # 脚本耗尽
    src.close()
    assert src.stats["frames_read"] == 2
    assert src.stats["source"] == "camera:0"


# ---------------- FileSource ----------------

def test_file_source_sequential_read(video_1s):
    src = FileSource(video_1s, realtime=False)
    assert src.open() is True
    assert src.fps == pytest.approx(30.0, abs=0.5)
    assert src.total_frames == 30
    frames = read_all(src)
    assert len(frames) == 30, "必须读完整段视频"
    src.close()
    assert src.stats["frames_read"] == 30
    assert src.stats["frames_dropped"] == 0


def test_file_source_timestamps_monotonic(video_1s):
    src = FileSource(video_1s)
    assert src.open() is True
    ts = []
    while True:
        ok, _, t = src.read()
        if not ok:
            break
        ts.append(t)
    src.close()
    assert ts == sorted(ts), "时间戳必须单调不减"
    assert len(set(ts)) == len(ts), "文件源时间戳应严格递增"
    assert ts[0] == pytest.approx(0.0, abs=1e-6)
    assert ts[-1] == pytest.approx(29 / 30.0, abs=0.01)  # frame_idx/fps


def test_file_source_realtime_throttle(video_1s):
    src = FileSource(video_1s, realtime=True, speed=1.0)
    assert src.open() is True
    t0 = time.monotonic()
    frames = read_all(src)
    elapsed = time.monotonic() - t0
    src.close()
    assert len(frames) == 30, "节流喂帧不能丢帧"
    assert elapsed >= 0.5, "30fps 1 秒视频按节拍读入应至少耗时 0.5s"
    assert elapsed < 5.0, "节流不应失控"


# ---------------- RTSPSource: 背压与重连 ----------------

def test_rtsp_backpressure_drops_old_frames():
    # 20 帧被后台线程快速 grab，消费端故意读得慢 -> 旧帧被丢弃
    src = ScriptedRTSP([[make_frame(i) for i in range(20)]],
                       reconnect_delay=0.01, max_reconnects=1)
    assert src.open() is True
    read_all(src, pause=0.05, max_iters=10)  # 慢消费：每帧间隔 50ms
    src.close()
    st = src.stats
    assert st["frames_read"] >= 1
    assert st["frames_dropped"] >= 1, "消费慢时必须统计到旧帧丢弃"
    # 账目守恒：成功 grab 的帧 = 读掉 + 丢弃
    assert st["frames_read"] + st["frames_dropped"] == 20
    assert st["source"] == "rtsp:rtsp://fake/test"


def test_rtsp_reconnect_and_recovery():
    # 阶段1(值1)吐 3 帧后断流 -> 重连 -> 阶段2(值2)吐 4 帧后结束
    src = ScriptedRTSP(
        [[make_frame(1)] * 3 + ["fail"], [make_frame(2)] * 4],
        reconnect_delay=0.02, max_reconnects=2)
    assert src.open() is True
    frames = read_all(src, pause=0.01)
    src.close()

    values = [int(f[0, 0, 0]) for f in frames]
    assert values, "重连后必须还能读到帧"
    assert values == sorted(values), "帧序必须保持（只能跳帧不能乱序）"
    assert values[0] == 1, "断流前读到阶段1的帧"
    assert values[-1] == 2, "恢复后必须能继续读到阶段2的帧"
    st = src.stats
    assert st["reconnects"] >= 1, "必须统计到重连"
    assert st["frames_read"] + st["frames_dropped"] == 7  # 3 + 4 帧守恒


def test_rtsp_reconnect_exhaustion():
    # 一连上就断，且重连永远失败 -> 最多重连 N 次后放弃，read 返回 False
    src = ScriptedRTSP([["fail"]], reconnect_delay=0.03, max_reconnects=2)
    assert src.open() is True
    t0 = time.monotonic()
    ok, frame, ts = src.read()  # 阻塞直到重连耗尽
    elapsed = time.monotonic() - t0
    assert ok is False and frame is None and ts > 0
    assert elapsed < 5.0, "重连耗尽后必须能退出，不能无限阻塞"
    st = src.stats
    assert st["reconnects"] == 2, "max_reconnects=2 应恰好重连 2 次"
    assert "耗尽" in (st["last_error"] or "")
    src.close()


def test_rtsp_open_failure_graceful():
    # _open_capture 失败（等价连不上 RTSP 服务器）：open 返回 False，不抛异常
    src = ScriptedRTSP([], reconnect_delay=0.01, max_reconnects=0)
    assert src.open() is False
    st = src.stats
    assert STATS_KEYS <= set(st.keys())
    assert st["last_error"], "失败原因必须进 stats"
    ok, frame, ts = src.read()
    assert ok is False and frame is None and ts > 0
    src.close()


# ---------------- integrated_pipeline 接入 ----------------

def test_pipeline_accepts_file_source(video_1s, tmp_path):
    out = tmp_path / "out"
    src = FileSource(video_1s)
    pipe, aligned, asr_segments = run_realtime_pipeline(
        video_path=None, output_dir=str(out), save_keyframes=False,
        config={"fast_scale": 0.25}, source=src)
    assert pipe is not None
    assert pipe.get_summary()["total_frames"] == 30

    results_file = out / "pipeline_results.json"
    aligned_file = out / "aligned_output.json"
    assert results_file.exists()
    assert aligned_file.exists()

    doc = json.loads(aligned_file.read_text(encoding="utf-8"))
    assert "pipeline_summary" in doc
    # W2: source.stats 并入 pipeline_summary
    sst = doc["pipeline_summary"]["source"]
    assert sst["frames_read"] == 30
    assert sst["source"].startswith("file:")
    # 文件回放事件时间戳仍是 frame_idx/fps（首帧 t=0）
    ts = [e["t"] for e in pipe.events]
    assert ts == sorted(ts)


def test_keyboard_interrupt_saves_partial_results(video_1s, tmp_path,
                                                  monkeypatch, capsys):
    out = tmp_path / "out"
    calls = {"n": 0}
    orig_read = FileSource.read

    def interrupted_read(self):
        calls["n"] += 1
        if calls["n"] >= 5:
            raise KeyboardInterrupt
        return orig_read(self)

    monkeypatch.setattr(FileSource, "read", interrupted_read)
    pipe, _, _ = run_realtime_pipeline(
        video_path=video_1s, output_dir=str(out), save_keyframes=False,
        config={"fast_scale": 0.25})
    assert pipe is not None, "中断后必须走保存路径并正常返回"
    assert calls["n"] == 5, "KeyboardInterrupt 应立即终止取帧循环"
    assert (out / "pipeline_results.json").exists(), "中断后必须落盘部分结果"
    assert (out / "aligned_output.json").exists()
    assert "收到中断" in capsys.readouterr().out
