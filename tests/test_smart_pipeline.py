"""SmartPipeline 单元测试。

用 numpy 合成视频帧（静止背景 + 移动色块）驱动流式管线，
覆盖：运动事件发射/闭合、关键帧触发、get_summary 数值、align_asr_streaming 结构。
全部输入确定性生成，路径仅用内存对象，Windows/Linux 均可运行。
"""
import numpy as np
import cv2
import pytest

from vus.smart_pipeline import SmartPipeline

FPS = 30.0
W, H = 320, 240
BLOCK_COLOR = (0, 200, 120)


def make_frame(block_center=None, radius=20, bg=40):
    """合成一帧：平坦背景 + 可选圆形色块。"""
    frame = np.full((H, W, 3), bg, dtype=np.uint8)
    if block_center is not None:
        cv2.circle(frame, block_center, radius, BLOCK_COLOR, -1)
    return frame


def make_pattern_frame(kind):
    """合成整幅纹理帧（用于触发慢系统关键帧的场景切换）。

    kind='ramp': 水平灰度渐变（20-60）；kind='noise': 固定种子噪声。
    两者像素差分、直方图、pHash 都差异显著，保证关键帧触发是确定性的。
    """
    if kind == "ramp":
        base = np.tile(np.linspace(20, 60, W, dtype=np.float32), (H, 1))
    elif kind == "noise":
        rng = np.random.RandomState(42)
        base = rng.uniform(0, 255, size=(H, W)).astype(np.float32)
    else:
        raise ValueError(f"unknown pattern kind: {kind}")
    ch = base.astype(np.uint8)
    return np.stack([ch, ch, ch], axis=-1)


def feed(pipe, frames):
    """按 30fps 时间戳逐帧喂入，收集全部流式事件。"""
    events = []
    for i, frame in enumerate(frames):
        events.extend(pipe.process_frame(frame, i / FPS))
    return events


# ==================== 运动事件（快系统） ====================

def _moving_scenario_frames():
    """5 静止帧 + 30 帧色块匀速横移(1s) + 30 静止帧(1s)。"""
    frames = [make_frame() for _ in range(5)]
    frames += [make_frame((40 + 8 * i, 120)) for i in range(30)]
    frames += [make_frame() for _ in range(30)]
    return frames


def test_motion_events_emitted_and_closed():
    pipe = SmartPipeline({"fast_scale": 0.25, "keyframe_interval_hz": 1.5})
    events = feed(pipe, _moving_scenario_frames())
    types = [e["type"] for e in events]

    assert "motion_start" in types, f"应发射 motion_start, 实际事件: {types}"
    assert "motion_end" in types, "运动结束后应发射 motion_end"

    # 事件时间戳单调不减
    ts = [e["t"] for e in events]
    assert ts == sorted(ts)

    # 运动段闭合：时长 >= 0.5s 才计入 motion_segments
    end_ev = [e for e in events if e["type"] == "motion_end"][-1]
    assert end_ev["duration"] >= 0.5
    assert len(pipe.motion_segments) >= 1
    seg = pipe.motion_segments[0]
    assert seg["start"] < seg["end"]
    assert round(seg["end"] - seg["start"], 3) == pytest.approx(end_ev["duration"], abs=0.05)

    # 序列结尾静止：运动状态应复位
    assert pipe.motion_active is False
    assert pipe.current_segment is None


def test_no_motion_for_static_frames():
    pipe = SmartPipeline()
    events = feed(pipe, [make_frame() for _ in range(20)])
    motion = [e for e in events if e["type"].startswith("motion")]
    assert motion == []
    assert pipe.get_summary()["motion_segments"] == 0


def test_motion_event_payload_structure():
    pipe = SmartPipeline()
    events = feed(pipe, _moving_scenario_frames())
    starts = [e for e in events if e["type"] == "motion_start"]
    assert starts, "缺少 motion_start"
    ev = starts[0]
    assert set(ev) >= {"type", "t", "boxes", "motion_ratio"}
    assert isinstance(ev["boxes"], list) and ev["boxes"]
    box = ev["boxes"][0]
    assert set(box) == {"bbox", "centroid", "area"}
    x, y, w, h = box["bbox"]
    cx, cy = box["centroid"]
    assert x <= cx <= x + w and y <= cy <= y + h
    assert 0.0 < ev["motion_ratio"] < 1.0


# ==================== 关键帧（慢系统） ====================

def test_first_frame_is_keyframe():
    pipe = SmartPipeline()
    events = feed(pipe, [make_frame() for _ in range(3)])
    kf = [e for e in events if e["type"] == "keyframe"]
    assert len(kf) == 1
    assert kf[0]["reason"] == "first_frame"
    assert kf[0]["t"] == 0.0
    assert pipe.keyframes[0]["frame_idx"] == 0


def test_keyframe_triggered_by_scene_change():
    """首帧渐变纹理 -> 31 帧不变（不触发）-> 噪声场景（应触发新关键帧）。"""
    pipe = SmartPipeline({"fast_scale": 0.25, "keyframe_interval_hz": 1.5})
    frames = [make_pattern_frame("ramp")] * 32 + [make_pattern_frame("noise")] * 6
    events = feed(pipe, frames)

    kfs = [e for e in events if e["type"] == "keyframe"]
    # 首帧关键帧 + 场景切换关键帧
    assert len(pipe.keyframes) == 2, f"期望 2 个关键帧, 实际: {pipe.keyframes}"
    assert kfs[0]["reason"] == "first_frame"
    # 第二个关键帧带变化打分信息，且发生在场景切换之后（t >= 31/30）
    second = kfs[1]
    assert second["t"] >= 31 / FPS - 1e-6
    assert second["score"] > 10.0
    assert "phash_dist" in second and "hist_corr" in second


def test_no_duplicate_keyframe_for_identical_frames():
    """画面完全不变时，除首帧外不应再触发关键帧。"""
    pipe = SmartPipeline()
    events = feed(pipe, [make_frame()] * 40)
    kfs = [e for e in events if e["type"] == "keyframe"]
    assert len(kfs) == 1 and kfs[0]["reason"] == "first_frame"


# ==================== 摘要 ====================

def test_get_summary_values_are_sane():
    pipe = SmartPipeline({"fast_scale": 0.25})
    frames = _moving_scenario_frames()
    events = feed(pipe, frames)
    summary = pipe.get_summary()

    assert summary["total_frames"] == len(frames)
    expected_motion_events = len([e for e in events if e["type"].startswith("motion")])
    assert summary["motion_events"] == expected_motion_events
    assert summary["motion_events"] >= 2  # 至少 start + end
    assert summary["motion_segments"] == len(pipe.motion_segments)
    assert summary["keyframes"] == len(pipe.keyframes)
    assert summary["total_events"] == len(pipe.events) == len(events)
    assert summary["avg_process_time_ms"] >= 0
    assert summary["avg_fps"] > 0
    assert summary["fast_scale"] == 0.25


def test_save_results_writes_json(tmp_path):
    pipe = SmartPipeline()
    feed(pipe, _moving_scenario_frames())
    out_file = tmp_path / "results.json"
    results = pipe.save_results(str(out_file))
    assert out_file.exists()
    loaded = results  # 返回值即写盘内容
    assert loaded["summary"]["total_frames"] == 65
    assert set(loaded) == {"summary", "motion_segments", "keyframes", "events_summary"}


# ==================== 对齐层 ====================

def test_align_asr_streaming_structure():
    pipe = SmartPipeline()
    events = feed(pipe, _moving_scenario_frames())

    segments = [
        {"t": 0.0, "text": "hello"},
        {"t": 1.0, "text": "world"},
        {"t": 3.0, "text": "!"},
    ]
    aligned = pipe.align_asr_streaming(segments)

    assert len(aligned) == 3
    for a in aligned:
        assert set(a) == {"start", "end", "text", "linked_motion_events",
                          "linked_keyframes", "motion_types"}
        assert a["start"] < a["end"]
        assert isinstance(a["linked_motion_events"], int)
        assert isinstance(a["linked_keyframes"], int)
        assert isinstance(a["motion_types"], list)
        assert all(mt.startswith("motion") for mt in a["motion_types"])

    # 相邻段首尾相接：end_i == t_{i+1}；最后一段补 2s
    assert aligned[0]["start"] == 0.0
    assert aligned[0]["end"] == 1.0
    assert aligned[1]["end"] == 3.0
    assert aligned[2]["end"] == pytest.approx(5.0, abs=1e-6)

    # 运动事件应被关联进对应时间窗
    linked_total = sum(a["linked_motion_events"] for a in aligned)
    motion_event_count = len([e for e in events if e["type"].startswith("motion")])
    assert linked_total == motion_event_count


def test_align_asr_streaming_empty_events():
    pipe = SmartPipeline()
    aligned = pipe.align_asr_streaming([{"t": 2.0, "text": "x"}])
    assert len(aligned) == 1
    assert aligned[0]["linked_motion_events"] == 0
    assert aligned[0]["linked_keyframes"] == 0
    assert aligned[0]["motion_types"] == []


# ==================== W1: events 有界化（防长直播内存泄漏） ====================

def _long_moving_frames(n=200):
    """n 帧持续运动（色块横移、越界回绕），保证几乎每帧都产生运动事件。"""
    return [make_frame((40 + (i * 8) % 200, 120)) for i in range(n)]


def test_max_events_bounded_drops_oldest():
    """max_events=50 喂 200 帧运动视频：events 不超上限且丢弃计数 > 0。"""
    pipe = SmartPipeline({"max_events": 50, "fast_scale": 0.25})
    feed(pipe, _long_moving_frames(200))

    assert len(pipe.events) <= 50
    summary = pipe.get_summary()
    assert summary["max_events"] == 50
    assert summary["events_dropped"] > 0
    assert summary["events_dropped"] == pipe.events_dropped
    # 保留的是最新事件：时间戳仍单调不减
    ts = [e["t"] for e in pipe.events]
    assert ts == sorted(ts)
    assert ts[-1] > 0  # 确实丢弃了早期事件，尾部事件时间靠后


def test_max_events_zero_keeps_unbounded():
    """max_events=0 = 无界（旧行为）：事件全保留、丢弃数为 0。"""
    pipe = SmartPipeline({"max_events": 0, "fast_scale": 0.25})
    events = feed(pipe, _long_moving_frames(200))

    assert len(pipe.events) == len(events)
    assert len(events) > 50, "前提检查：200 帧运动视频应产出超过 50 个事件"
    assert pipe.events_dropped == 0
    assert pipe.get_summary()["events_dropped"] == 0


def test_max_events_default_is_bounded():
    """安全默认：不配置时也有界（100000），防止长视频 list 无限增长。"""
    pipe = SmartPipeline()
    assert pipe.max_events == 100000
    from collections import deque
    assert isinstance(pipe.events, deque)


def test_streaming_consumer_bounded():
    """StreamingConsumer 同样有界：超限丢最旧，snapshot 携带丢弃计数。"""
    from vus.integrated_pipeline import StreamingConsumer

    c = StreamingConsumer(max_events=10)
    for i in range(25):
        c.consume({"type": "motion", "t": round(i / FPS, 3)})

    assert len(c.events) == 10
    assert c.events_dropped == 15
    snap = c.snapshot()
    assert snap["event_count"] == 10
    assert snap["events_dropped"] == 15
    assert snap["max_events"] == 10
    # 保留的是最新事件（首条 t=15/30, 末条 t=24/30）
    assert snap["events"][0]["t"] == pytest.approx(15 / FPS)
    assert snap["events"][-1]["t"] == pytest.approx(24 / FPS)


def test_streaming_consumer_unbounded_and_default():
    """StreamingConsumer：max_events=0 无界全保留；默认有界 100000。"""
    from collections import deque
    from vus.integrated_pipeline import StreamingConsumer

    c0 = StreamingConsumer(max_events=0)
    assert isinstance(c0.events, list)
    for i in range(25):
        c0.consume({"type": "motion", "t": round(i / FPS, 3)})
    assert c0.snapshot()["event_count"] == 25
    assert c0.events_dropped == 0

    assert StreamingConsumer().max_events == 100000
    assert isinstance(StreamingConsumer().events, deque)


def test_align_asr_streaming_uses_end_t_for_last_segment():
    """对齐层可选消费 ASR 段的 end_t：末段无下一段边界时用 end_t 而非 t+2.0。"""
    pipe = SmartPipeline()
    aligned = pipe.align_asr_streaming([
        {"t": 0.0, "text": "a", "end_t": 2.0},
        {"t": 2.0, "text": "b", "end_t": 2.9},
    ])
    assert aligned[0]["end"] == pytest.approx(2.0)   # 仍取下一段起点
    assert aligned[1]["end"] == pytest.approx(2.9)   # 末段用 end_t
    # 兼容旧结构：无 end_t 时末段仍回退 t+2.0
    aligned_legacy = pipe.align_asr_streaming([{"t": 1.0, "text": "x"}])
    assert aligned_legacy[0]["end"] == pytest.approx(3.0)
