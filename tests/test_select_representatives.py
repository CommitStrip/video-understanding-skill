"""select_representatives 单元测试。

在 tmp_path 写合成 jpg 关键帧（文件名匹配 `_t{秒}s.jpg` 约定），
覆盖：文件名解析排序、时间分桶（桶锚点时间 >= interval）、
去重阈值行为、pixel_diff 纯函数性质。
"""
import cv2
import numpy as np
import pytest

from vus.select_representatives import (
    load_keyframes,
    select_representatives,
    summarize,
    _pixel_diff,
)


def write_kf(kf_dir, t, color):
    """写一张纯色关键帧，文件名符合 load_keyframes 的解析约定。"""
    img = np.full((64, 64, 3), color, dtype=np.uint8)
    path = kf_dir / f"kf_{int(t*10):05d}_t{t:.1f}s.jpg"
    ok = cv2.imwrite(str(path), img)
    assert ok, f"cv2.imwrite 失败: {path}"
    return str(path)


@pytest.fixture
def kf_dir(tmp_path):
    return tmp_path / "keyframes"


# ==================== load_keyframes ====================

def test_load_keyframes_sorted_by_time(kf_dir):
    kf_dir.mkdir()
    write_kf(kf_dir, 5.0, 30)
    write_kf(kf_dir, 1.0, 30)
    write_kf(kf_dir, 3.0, 30)
    items = load_keyframes(str(kf_dir))
    assert [t for t, _ in items] == [1.0, 3.0, 5.0]
    # 忽略不匹配命名的文件
    (kf_dir / "readme.txt").write_text("x")
    assert len(load_keyframes(str(kf_dir))) == 3


def test_load_keyframes_empty_dir(kf_dir):
    kf_dir.mkdir()
    assert load_keyframes(str(kf_dir)) == []
    assert summarize(str(kf_dir)) == {"count": 0}


# ==================== pixel_diff ====================

def test_pixel_diff_identical_is_zero():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    assert _pixel_diff(img, img) == pytest.approx(0.0)


def test_pixel_diff_black_vs_white():
    black = np.zeros((100, 100, 3), dtype=np.uint8)
    white = np.full((100, 100, 3), 255, dtype=np.uint8)
    assert _pixel_diff(black, white) == pytest.approx(100.0)


def test_pixel_diff_range_and_monotonic():
    black = np.zeros((50, 50, 3), dtype=np.uint8)
    d20 = _pixel_diff(black, np.full((50, 50, 3), 20, dtype=np.uint8))
    d100 = _pixel_diff(black, np.full((50, 50, 3), 100, dtype=np.uint8))
    d200 = _pixel_diff(black, np.full((50, 50, 3), 200, dtype=np.uint8))
    assert 0 <= d20 < d100 < d200 <= 100


# ==================== 时间分桶 ====================

def test_bucket_anchor_interval_guarantee(kf_dir):
    """每 interval 秒一个桶：桶锚点时间间隔 >= interval（时间完整性硬保证）。"""
    kf_dir.mkdir()
    interval = 10.0
    # 0..45s 每 1s 一帧，纯色不同色阶（差异小但不为零）
    ts = [float(i) for i in range(46)]
    for i, t in enumerate(ts):
        write_kf(kf_dir, t, 10 + (i % 5))
    reps = select_representatives(str(kf_dir), interval=interval)

    assert len(reps) >= 5  # 46s / 10s 至少 5 个桶
    times = [r["t"] for r in reps]
    for a, b in zip(times, times[1:]):
        assert b - a >= interval - 0.5  # 允许帧间隔粒度误差


def test_representatives_cover_first_and_last_bucket(kf_dir):
    """首帧必选；最后一帧所在桶必有一个锚点。"""
    kf_dir.mkdir()
    write_kf(kf_dir, 0.0, 40)
    for i in range(1, 30):
        write_kf(kf_dir, float(i), 40)
    write_kf(kf_dir, 120.0, 200)  # 远端稀疏区一帧
    reps = select_representatives(str(kf_dir), interval=60.0)

    times = [r["t"] for r in reps]
    assert times[0] == 0.0            # 首帧是第一个桶锚点
    assert times[-1] == 120.0         # 稀疏区不被漏掉


def test_bucket_picks_most_different_frame(kf_dir):
    """桶内应选与桶首差异最大的帧作锚点。"""
    kf_dir.mkdir()
    # 桶 [0,60)：0s 灰 40；10s/20s 灰 42（差异小）；50s 白 255（差异最大）
    write_kf(kf_dir, 0.0, 40)
    write_kf(kf_dir, 10.0, 42)
    write_kf(kf_dir, 20.0, 42)
    write_kf(kf_dir, 50.0, 255)
    reps = select_representatives(str(kf_dir), interval=60.0)

    assert len(reps) == 1
    assert reps[0]["t"] == 50.0


# ==================== 去重阈值 ====================

def test_dedup_merges_similar_anchors(kf_dir):
    kf_dir.mkdir()
    interval = 10.0
    # 三个桶，锚点内容几乎一致（同灰度）-> 开去重后合并；末帧不同 -> 保留
    for i in range(3):
        base = float(i * interval)
        write_kf(kf_dir, base, 60)
        write_kf(kf_dir, base + interval - 1.0, 60)
    write_kf(kf_dir, 3 * interval, 250)

    all_reps = select_representatives(str(kf_dir), interval=interval, dedup_threshold=0.0)
    deduped = select_representatives(str(kf_dir), interval=interval, dedup_threshold=8.0)

    assert len(deduped) < len(all_reps)
    assert deduped[-1]["t"] == pytest.approx(3 * interval)  # 差异大的锚点保留


def test_return_structure(kf_dir):
    kf_dir.mkdir()
    write_kf(kf_dir, 0.0, 30)
    write_kf(kf_dir, 5.0, 240)
    reps = select_representatives(str(kf_dir), interval=60.0)
    assert all(set(r) == {"t", "path"} for r in reps)
    assert all(isinstance(r["t"], float) for r in reps)
