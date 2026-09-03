"""W8b SessionState 单元测试：四层写入 / 环上限 / 时间线压缩 / 快照与滚动落盘。"""
import json

from vus.live import SessionState, StateWriter


def test_frame_events_drive_clock_and_motion_stats():
    st = SessionState()
    st.apply_frame_event({"type": "motion_start", "t": 1.0})
    st.apply_frame_event({"type": "keyframe", "t": 2.0})
    st.apply_frame_event({"type": "motion_end", "t": 3.5,
                          "segment_start": 1.0, "duration": 2.5})
    snap = st.snapshot()
    assert snap["session"]["t_now"] == 3.5
    assert snap["session"]["frame_count"] == 3
    assert snap["t0"]["motion_segments"] == 1
    assert snap["t0"]["last_motion"]["duration"] == 2.5


def test_asr_ring_cap():
    st = SessionState()
    for i in range(35):
        st.apply_asr({"t": float(i), "end_t": i + 1.0, "text": f"s{i}"})
    ring = st.snapshot()["asr"]
    assert len(ring) == 30
    assert ring[-1]["text"] == "s34"  # 丢的是最旧


def test_label_ring_and_avg_ms():
    st = SessionState()
    for i in range(55):
        st.apply_label({"t": float(i), "labels": [{"label": "静态画面", "score": 0.7}],
                        "source": "basic", "ms": 10.0})
    snap = st.snapshot()
    assert len(snap["t05"]["labels"]) == 50
    assert snap["telemetry"]["t05_labels"] == 55
    assert snap["telemetry"]["t05_avg_ms"] == 10.0


def test_understanding_merge_and_lag():
    st = SessionState()
    st.apply_frame_event({"type": "motion", "t": 9.0})       # t_now = 9.0
    st.apply_understanding(
        {"now": "有人进门", "segment": {"summary": "入口有人"},
         "entities": {"人": "疑似访客"}},
        window_t0=5.0, window_t1=7.0, model="mock")
    snap = st.snapshot()
    assert snap["t2"]["now"] == "有人进门"
    assert snap["t2"]["timeline"] == [{"start": 5.0, "end": 7.0, "summary": "入口有人"}]
    assert snap["t2"]["entities"] == {"人": "疑似访客"}
    assert snap["t2"]["understood_t"] == 7.0
    assert snap["telemetry"]["lag"]["t2_s"] == 2.0   # 9.0 - 7.0
    assert snap["t2"]["model"] == "mock"


def test_understanding_partial_json_degrades_safely():
    st = SessionState()
    st.apply_understanding({"now": "只有 now"}, 0.0, 2.0, model="mock")
    snap = st.snapshot()
    assert snap["t2"]["now"] == "只有 now"
    assert snap["t2"]["timeline"][0]["summary"] == "只有 now"  # 缺 segment 用 now 兜底
    assert snap["t2"]["entities"] == {}


def test_compact_timeline_merges_overflow_into_chapters():
    st = SessionState(max_timeline=5)
    for i in range(8):
        st.apply_understanding({"now": f"n{i}", "segment": {"summary": f"s{i}"}},
                               float(i), float(i + 1), model="mock")
    merged = st.compact_timeline()
    snap = st.snapshot()
    assert merged == 3
    assert len(snap["t2"]["timeline"]) == 5
    assert len(snap["t2"]["chapters"]) == 1
    ch = snap["t2"]["chapters"][0]
    assert ch["start"] == 0.0 and ch["end"] == 3.0
    assert "s0；s1；s2" == ch["summary"]
    assert st.compact_timeline() == 0  # 未超限不再合并


def test_error_and_drop_telemetry():
    st = SessionState()
    st.note_call_latency(1.23)
    st.note_call_error("超时")
    st.note_dropped_window()
    tel = st.snapshot()["telemetry"]
    assert tel["t2_last_latency_s"] == 1.23
    assert tel["t2_errors"] == 1 and "超时" in tel["t2_last_error"]
    assert tel["t2_dropped_windows"] == 1


def test_write_outputs_atomic_roundtrip(tmp_path):
    st = SessionState(source_desc="rtsp://cam/1")
    st.apply_understanding({"now": "画面静止"}, 0.0, 1.0, model="mock")
    st.write_outputs(str(tmp_path))
    doc = json.loads((tmp_path / "live_state.json").read_text(encoding="utf-8"))
    assert doc["session"]["source"] == "rtsp://cam/1"
    assert doc["t2"]["now"] == "画面静止"
    md = (tmp_path / "live_context.md").read_text(encoding="utf-8")
    assert "实时视频理解上下文" in md
    assert "画面静止" in md
    assert not list(tmp_path.glob("*.tmp"))  # 原子替换不残留临时文件


def test_state_writer_periodic_dump(tmp_path):
    import time
    st = SessionState()
    writer = StateWriter(st, output_dir=str(tmp_path), interval=0.05)
    writer.start()
    time.sleep(0.2)
    writer.stop()
    assert (tmp_path / "live_state.json").exists()
    assert (tmp_path / "live_context.md").exists()


def test_state_writer_without_dir_is_noop():
    st = SessionState()
    writer = StateWriter(st, output_dir=None)
    writer.start()  # 不应起线程也不应抛错
    writer.stop()
