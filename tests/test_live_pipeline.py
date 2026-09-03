"""W8c E2E 测试：file-as-live 全链路 + SSE 服务 + CLI。

全链路：合成视频（移动方块）按实时节拍回放 → 快/慢系统 → 毫秒标签 →
MockVLM 触发式理解 → 滚动落盘。SSE：/state /healthz /events 真实 HTTP 往返。

HTTP 请求边界：仅允许访问本测试进程启动的 http://127.0.0.1 服务
（_local_get 内做协议/host 显式校验后再请求，防 SSRF）。
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from vus.live import EventBus, LiveServer, MockVLM, SessionState, build_source, run_live

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ALLOWED_HOST = "127.0.0.1"


def _local_get(base, path, timeout=5.0):
    """测试专用 GET：校验目标是本进程启动的 127.0.0.1 http 服务后才请求。"""
    parsed = urlparse(base)
    assert parsed.scheme == "http", f"仅允许 http 本地服务: {base!r}"
    assert parsed.hostname == _ALLOWED_HOST, f"仅允许访问 {_ALLOWED_HOST}: {base!r}"
    return urllib.request.urlopen(
        f"http://{_ALLOWED_HOST}:{parsed.port}{path}", timeout=timeout)


def _make_video(path, seconds=6.0, fps=10.0, w=320, h=240):
    """合成带移动方块的测试视频（快系统可检出运动）。"""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    n = int(seconds * fps)
    for i in range(n):
        frame = np.full((h, w, 3), (30, 30, 30), dtype=np.uint8)
        x = int((i / n) * (w - 60))
        cv2.rectangle(frame, (x, 90), (x + 60, 150), (0, 220, 60), -1)
        vw.write(frame)
    vw.release()
    return n


def test_file_as_live_full_chain(tmp_path):
    """6s 合成视频仿真实时：T0/T0.5/T2 全链路产出 + 四件产物落盘。"""
    video = tmp_path / "in.mp4"
    _make_video(video)
    out_dir = tmp_path / "out"

    source = build_source("file", video_path=str(video), realtime=True)
    ret = run_live(source, output_dir=str(out_dir),
                   vlm=MockVLM(), labeler=True,
                   live_cfg={"min_call_interval": 0.0},
                   audio="auto", quiet=True)
    assert ret is not None
    assert ret["proc_fps"] > 0

    # 四件产物
    assert (out_dir / "live_state.json").exists()
    assert (out_dir / "live_context.md").exists()
    assert (out_dir / "pipeline_results.json").exists()
    assert (out_dir / "aligned_output.json").exists()
    kfs = list((out_dir / "keyframes").glob("kf_*.jpg"))
    assert kfs, "实时链必须落关键帧"

    doc = json.loads((out_dir / "live_state.json").read_text(encoding="utf-8"))
    assert doc["session"]["frame_count"] >= 60           # 6s × 10fps 全部处理
    assert doc["telemetry"]["t05_labels"] >= 1           # T0.5 标签在跑
    assert doc["telemetry"]["t2_calls"] >= 1             # T2 至少触发一次
    assert doc["t2"]["now"]                              # MockVLM 回复进入滚动摘要
    assert doc["telemetry"]["lag"]["t2_s"] is not None
    assert doc["t2"]["timeline"], "时间线应有条目"
    # 无音轨合成视频：声音链安全跳过，对齐段为空
    aligned = json.loads((out_dir / "aligned_output.json").read_text(encoding="utf-8"))
    assert aligned["live"] is True
    assert aligned["aligned_segments"] == []
    # Markdown 上下文包含滚动摘要
    md = (out_dir / "live_context.md").read_text(encoding="utf-8")
    assert "实时视频理解上下文" in md


def test_vlm_off_pure_local_mode(tmp_path):
    """--vlm off：纯本地免费模式（T0+T0.5），无 T2 调用、无时间线，链路不报错。"""
    video = tmp_path / "in.mp4"
    _make_video(video, seconds=3.0)
    out_dir = tmp_path / "out"
    source = build_source("file", video_path=str(video), realtime=True)
    ret = run_live(source, output_dir=str(out_dir), vlm=None, labeler=True,
                   audio="off", quiet=True)
    assert ret is not None
    doc = json.loads((out_dir / "live_state.json").read_text(encoding="utf-8"))
    assert doc["telemetry"]["t2_calls"] == 0
    assert doc["telemetry"]["t05_labels"] >= 1
    assert doc["t2"]["now"] == ""
    aligned = json.loads((out_dir / "aligned_output.json").read_text(encoding="utf-8"))
    assert aligned["aligned_segments"] == []


def test_live_server_state_healthz_events():
    """SSE 服务三端点真实 HTTP 往返。"""
    bus, state = EventBus(), SessionState()
    srv = LiveServer(bus, state, port=0)
    srv.start()
    try:
        with _local_get(srv.url, "/state") as r:
            doc = json.loads(r.read().decode("utf-8"))
        assert doc["session"]["t_now"] == 0.0

        with _local_get(srv.url, "/healthz") as r:
            assert json.loads(r.read().decode("utf-8"))["status"] == "ok"

        resp = _local_get(srv.url, "/events")
        data = resp.read(256)  # 首包：retry + state 快照
        assert b"event: state" in data or b"retry:" in data
        bus.publish({"type": "motion", "t": 1.0})
        deadline = time.time() + 5
        while b"motion" not in data and time.time() < deadline:
            chunk = resp.read(256)
            if not chunk:
                break
            data += chunk
        assert b"motion" in data
        resp.close()
    finally:
        srv.stop()


def test_cli_help_smoke():
    """python -m vus.live --help 可运行（入口装配无误）。

    子进程显式设 OPENBLAS_NUM_THREADS=1（SKILL.md 环境注意事项）：
    pytest 父进程内存压力下 OpenBLAS 多线程预分配可能失败，与被测代码无关。
    """
    import os
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-m", "vus.live", "--help"],
                       cwd=str(_REPO_ROOT), capture_output=True, timeout=60,
                       text=True, encoding="utf-8", errors="replace", env=env)
    assert r.returncode == 0
    assert "--vlm" in r.stdout and "--serve" in r.stdout
