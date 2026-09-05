"""W8b UnderstandingWorker 单元测试：触发策略 / 地板间隔 / 单飞合并 / 时间线压缩 / 退避重试 / 滞后遥测。

时间用注入时钟（fake clock）保证确定性；VLM 用 MockVLM（latency 模拟 API 延迟）。
"""
import time

import cv2
import numpy as np

from vus.live import EventBus, MockVLM, SessionState, UnderstandingWorker


def _make_kf(tmp_path, name="kf.jpg"):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((48, 64, 3), 200, dtype=np.uint8))
    return str(p)


def _kf_event(t, path, reason="scene_change"):
    return {"type": "keyframe", "t": t, "reason": reason, "path": path}


def _start(bus, state, vlm, cfg=None, clock_fn=time.monotonic):
    w = UnderstandingWorker(bus, state, vlm, config=cfg, clock_fn=clock_fn)
    w.start()
    return w


def test_keyframe_trigger_calls_vlm_and_updates_state(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM(scripted=[{"now": "有人出现", "segment": {"summary": "入场"},
                             "entities": {"人": "访客"}}])
    out_sub = bus.subscribe()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        bus.publish(_kf_event(1.0, _make_kf(tmp_path)))
        assert w.wait_idle(timeout=5.0)
        snap = state.snapshot()
        assert snap["t2"]["now"] == "有人出现"
        assert snap["t2"]["timeline"][0]["summary"] == "入场"
        assert snap["t2"]["entities"] == {"人": "访客"}
        assert snap["t2"]["model"] == "mock"
        assert vlm.calls[0]["n_frames"] == 1
        # 理解结果回流总线（SSE 订阅者可见）
        evs = out_sub.drain(timeout=0.1)
        assert any(e["type"] == "understanding" and e["now"] == "有人出现" for e in evs)
    finally:
        w.stop()


def test_first_frame_keyframe_does_not_trigger(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        bus.publish(_kf_event(0.0, _make_kf(tmp_path), reason="first_frame"))
        assert w.wait_idle(timeout=2.0)
        time.sleep(0.1)
        assert vlm.calls == []  # 首帧不算语义事件
    finally:
        w.stop()


def test_min_call_interval_floor_with_fake_clock(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    clock = {"now": 1000.0}
    w = _start(bus, state, vlm, cfg={"min_call_interval": 10.0},
               clock_fn=lambda: clock["now"])
    try:
        p = _make_kf(tmp_path)
        bus.publish(_kf_event(1.0, p))
        assert w.wait_idle(timeout=5.0)
        assert len(vlm.calls) == 1

        bus.publish(_kf_event(2.0, p))          # 距上次调用 0s < 地板 10s
        assert w.wait_idle(timeout=0.3) is False  # 持续触发但被地板压住
        assert len(vlm.calls) == 1

        clock["now"] = 1011.0                    # 时钟越过地板
        assert w.wait_idle(timeout=5.0)
        assert len(vlm.calls) == 2
    finally:
        w.stop()


def test_single_flight_coalescing_no_call_explosion(tmp_path):
    """VLM 在飞时新触发只累积不排队：3 次触发 → 恰好 2 次调用、素材不丢。"""
    bus, state = EventBus(), SessionState()
    vlm = MockVLM(latency=0.3)  # 模拟 API 延迟，制造在飞窗口
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        p = _make_kf(tmp_path)
        bus.publish(_kf_event(1.0, p))
        time.sleep(0.05)  # 确保第一窗已发出、调用在飞
        bus.publish(_kf_event(2.0, p))
        bus.publish(_kf_event(3.0, p))
        assert w.wait_idle(timeout=5.0)
        assert len(vlm.calls) == 2
        assert sum(c["n_frames"] for c in vlm.calls) == 3  # 3 张关键帧全被消费
    finally:
        w.stop()


def _eventually(fn, want, timeout=5.0):
    """轮询断言：状态由后台线程推进，给写入可见性一个截止窗口。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if fn() == want:
                return True
        except (IndexError, KeyError, TypeError):
            pass
        time.sleep(0.02)
    return False


def test_asr_trigger_prompt_material_and_aligned(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        bus.publish({"type": "motion_start", "t": 0.5})
        bus.publish({"type": "motion_end", "t": 1.5, "segment_start": 0.5,
                     "duration": 1.0})
        bus.publish({"type": "asr_final", "t": 2.0, "end_t": 4.0, "text": "大家好"})
        bus.publish({"type": "asr_final", "t": 4.0, "end_t": 6.0, "text": "今天讲无人机"})
        assert w.wait_idle(timeout=5.0)
        prompt = vlm.calls[0]["prompt"]
        assert "大家好" in prompt and "今天讲无人机" in prompt
        assert "运动事件" in prompt
        # 滚动对齐：段 2 无后续边界，保持 pending（与批式"最后一段 flush 闭窗"一致）
        assert len(w.aligned_segments()) == 1
        assert w.aligned_segments()[0] == {
            "start": 2.0, "end": 4.0, "text": "大家好",
            "linked_motion_events": 0, "linked_keyframes": 0, "motion_types": []}
        # 流结束 flush：最后一段用 end_t 兜底闭窗
        assert w.flush_aligner()[1]["end"] == 6.0
    finally:
        w.stop()


def test_timeline_compaction_into_chapters(tmp_path):
    bus, state = EventBus(), SessionState(max_timeline=3)
    vlm = MockVLM()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        p = _make_kf(tmp_path)
        for i in range(6):
            bus.publish(_kf_event(float(i), p))
            assert w.wait_idle(timeout=5.0)
        snap = state.snapshot()
        assert len(snap["t2"]["timeline"]) == 3   # 上限不破
        assert len(snap["t2"]["chapters"]) == 3   # 溢出合并成章节
    finally:
        w.stop()


def test_vlm_failure_backoff_then_requeue_succeeds(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    vlm.fail_next = 1
    w = _start(bus, state, vlm,
               cfg={"min_call_interval": 0.0, "backoff_base": 0.01,
                    "backoff_max": 0.05})
    try:
        bus.publish(_kf_event(1.0, _make_kf(tmp_path)))
        assert w.wait_idle(timeout=5.0)
        assert len(vlm.calls) == 2            # 失败 1 次 + 重试成功
        assert state.snapshot()["telemetry"]["t2_errors"] == 1
        assert state.snapshot()["t2"]["now"]  # 重试后状态正常更新
    finally:
        w.stop()


def test_understanding_lag_telemetry(tmp_path):
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        bus.publish(_kf_event(5.0, _make_kf(tmp_path)))
        assert w.wait_idle(timeout=5.0)
        bus.publish({"type": "motion", "t": 9.0})  # 时钟推进到 9s
        # 状态由采集线程异步写入，轮询等待可见
        assert _eventually(
            lambda: state.snapshot()["telemetry"]["lag"]["t2_s"], 4.0)  # 9 - 5
    finally:
        w.stop()


def test_tag_event_updates_t05_state():
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    w = _start(bus, state, vlm, cfg={"min_call_interval": 0.0})
    try:
        bus.publish({"type": "tag", "t": 1.0,
                     "labels": [{"label": "有人脸", "score": 0.8}],
                     "source": "basic", "ms": 8.0})
        assert w.wait_idle(timeout=2.0)
        snap = state.snapshot()
        assert snap["t05"]["labels"][-1]["labels"][0]["label"] == "有人脸"
        assert snap["telemetry"]["t05_labels"] == 1
    finally:
        w.stop()


def test_stop_is_idempotent(tmp_path):
    bus, state = EventBus(), SessionState()
    w = _start(bus, state, MockVLM())
    w.stop()
    w.stop()  # 二次 stop 不抛错


# ---------- W9 运动框裁剪喂 VLM（motion_crop） ----------

import base64  # noqa: E402

import cv2  # noqa: E402

from vus.live.understanding import _Window  # noqa: E402


def _decode_size(b64):
    buf = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[:2]


def _motion_event(t, bbox=(2, 2, 8, 6), scale=0.25, area=48):
    return {"type": "motion", "t": t, "boxes": [
        {"bbox": list(bbox), "centroid": [2, 2], "area": area}],
        "motion_ratio": 0.01, "scale": scale}


def _worker_with_material(tmp_path, cfg=None, with_motion=True, n_kf=2):
    """同步喂数（不经线程），直接对 _win 断言 _build_prompt 结果。"""
    bus, state = EventBus(), SessionState()
    vlm = MockVLM()
    w = UnderstandingWorker(bus, state, vlm, config=cfg, clock_fn=time.monotonic)
    if with_motion:
        w._on_event(_motion_event(0.5))
    for i in range(n_kf):
        w._on_event(_kf_event(1.0 + i, _make_kf(tmp_path, f"kf{i}.jpg")))
    return w, vlm


def test_motion_crop_replaces_last_frame_with_region(tmp_path):
    # _make_kf 图为 64x48（h=48）：整帧编码 (48,64)；裁剪图 (2,2,8,6)@0.25
    # → (8,8,32,24) 外扩 25% → (0,2,48,36) → (36,48)——尺寸可区分末张来源
    w, _vlm = _worker_with_material(tmp_path, n_kf=2)
    prompt, frames = w._build_prompt(w._win)
    assert len(frames) == 2
    assert "运动区域放大图" in prompt
    assert _decode_size(frames[-1]) == (36, 48)   # 末张=裁剪图
    assert _decode_size(frames[0]) == (48, 64)    # 首张=整帧锚定


def test_motion_crop_single_kf_also_crops(tmp_path):
    w, _vlm = _worker_with_material(tmp_path, n_kf=1)
    prompt, frames = w._build_prompt(w._win)
    assert len(frames) == 1
    assert "运动区域放大图" in prompt
    assert _decode_size(frames[-1]) == (36, 48)


def test_motion_crop_disabled_keeps_full_frames(tmp_path):
    w, _vlm = _worker_with_material(tmp_path, cfg={"motion_crop": False}, n_kf=2)
    prompt, frames = w._build_prompt(w._win)
    assert len(frames) == 2
    assert "运动区域放大图" not in prompt
    assert all(_decode_size(b) == (48, 64) for b in frames)


def test_no_motion_events_keeps_full_frames(tmp_path):
    w, _vlm = _worker_with_material(tmp_path, with_motion=False, n_kf=2)
    prompt, frames = w._build_prompt(w._win)
    assert len(frames) == 2
    assert "运动区域放大图" not in prompt


def test_window_merge_keeps_recent_motion_boxes():
    a, b = _Window(), _Window()
    for i in range(10):
        a.motion_boxes.append({"t": float(i), "bbox": [i, 0, 1, 1], "scale": 0.25})
    b.motion_boxes.append({"t": 99.0, "bbox": [9, 9, 1, 1], "scale": 0.25})
    a.merge(b)
    assert len(a.motion_boxes) == 8               # 保尾截断
    assert a.motion_boxes[-1]["t"] == 99.0        # 最新框在尾部（裁剪取尾部）
