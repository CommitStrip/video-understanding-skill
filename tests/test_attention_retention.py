"""W9 v0.6 测试：注意力回链 + 关键帧磁盘 retention。"""
import json

import pytest

from vus.asr_clean import detect_attention_windows
from vus.integrated_pipeline import prune_keyframes
from vus.select_representatives import force_include_attention_windows


# ==================== 注意力窗口检测 ====================

def _seg(t, text, end=None):
    s = {"t": t, "text": text}
    if end is not None:
        s["end_t"] = end
    return s


def test_detect_attention_cues_hit():
    segs = [_seg(10.0, "大家注意看右下角的数字"), _seg(20.0, "这个不重要跳过")]
    windows = detect_attention_windows(segs)
    assert len(windows) == 1
    assert windows[0]["t"] == 10.0
    assert "注意" in windows[0]["cue"]


def test_detect_merges_adjacent_hits():
    segs = [_seg(10.0, "注意看这里", end=12.0), _seg(14.0, "重点来了", end=16.0)]
    windows = detect_attention_windows(segs, merge_gap_s=5.0)
    assert len(windows) == 1
    assert windows[0]["end"] >= 16.0


def test_detect_hallucination_skipped():
    segs = [{"t": 10.0, "text": "SIL", "hallucination": True},
            _seg(20.0, "look at the board")]
    windows = detect_attention_windows(segs)
    # 幻觉段跳过；正常英文线索命中
    assert all(w["t"] != 10.0 for w in windows)
    assert any("look at" in w["cue"] for w in windows)


def test_detect_no_cues_empty():
    assert detect_attention_windows([_seg(5.0, "普通的叙述句子")]) == []


# ==================== 注意力回链强制纳入 ====================

def _kfs(tmp_path, ts, color=90):
    files = []
    for i, t in enumerate(ts):
        import cv2
        import numpy as np
        p = tmp_path / f"kf_{i:04d}_t{t:.1f}s.jpg"
        cv2.imwrite(str(p), np.full((48, 64, 3), color, dtype=np.uint8))
        files.append((float(t), str(p)))
    return files


def test_force_include_pulls_nearest_kf(tmp_path):
    kfs = _kfs(tmp_path, [100.0, 140.0, 300.0])
    reps = [{"t": 100.0, "path": kfs[0][1]}]  # 只有首帧被选
    windows = [{"t": 295.0, "end": 305.0, "cue": "注意"}]
    out = force_include_attention_windows(reps, kfs, windows)
    assert len(out) == 2
    assert out[-1]["t"] == 300.0


def test_force_include_noop_when_already_covered(tmp_path):
    kfs = _kfs(tmp_path, [100.0, 300.0])
    reps = [{"t": 300.0, "path": kfs[1][1]}]
    windows = [{"t": 295.0, "end": 305.0}]
    out = force_include_attention_windows(reps, kfs, windows)
    assert len(out) == 1


def test_force_include_no_kf_in_window(tmp_path):
    kfs = _kfs(tmp_path, [100.0])
    reps = [{"t": 100.0, "path": kfs[0][1]}]
    windows = [{"t": 500.0, "end": 510.0}]
    out = force_include_attention_windows(reps, kfs, windows)
    assert len(out) == 1


def test_force_include_sorted_output(tmp_path):
    kfs = _kfs(tmp_path, [100.0, 200.0, 300.0])
    reps = [{"t": 300.0, "path": kfs[2][1]}]
    windows = [{"t": 100.0, "end": 110.0}, {"t": 195.0, "end": 205.0}]
    out = force_include_attention_windows(reps, kfs, windows)
    assert [r["t"] for r in out] == [100.0, 200.0, 300.0]


# ==================== 磁盘 retention ====================

def test_prune_keyframes_under_quota(tmp_path):
    d = tmp_path / "keyframes"
    d.mkdir()
    for i in range(5):
        (d / f"kf_{i:04d}_t{i*10}s.jpg").write_bytes(b"x" * 100)
    assert prune_keyframes(str(d), max_mb=1) == 0  # 配额宽裕不动


def test_prune_keyframes_over_quota_removes_oldest(tmp_path):
    d = tmp_path / "keyframes"
    d.mkdir()
    for i in range(10):
        (d / f"kf_{i:04d}_t{i*10}s.jpg").write_bytes(b"x" * 1000)
    removed = prune_keyframes(str(d), max_mb=0.004)  # 配额 4KB，共 10KB
    remaining = sorted(p.name for p in d.glob("kf_*.jpg"))
    assert removed > 0
    assert len(remaining) == 10 - removed
    assert remaining[0] > "kf_0000"  # 最旧的先被删


def test_prune_keyframes_empty_dir(tmp_path):
    d = tmp_path / "keyframes"
    d.mkdir()
    assert prune_keyframes(str(d), max_mb=1) == 0
