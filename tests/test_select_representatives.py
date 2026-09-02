"""select_representatives 单元测试。

在 tmp_path 写合成 jpg 关键帧（文件名匹配 `_t{秒}s.jpg` 约定），
覆盖：文件名解析排序、时间分桶（桶锚点时间 >= interval）、
去重阈值行为、pixel_diff 纯函数性质，
以及 W1 新增：桶内多样性 top-k（最远点采样）与自适应桶宽。
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


def write_kf_noise(kf_dir, t, seed=1):
    """写一张固定种子噪声关键帧（与任意纯色帧差异都大且确定）。"""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
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


# ==================== W1: 桶内多样性 top-k（最远点采样） ====================

@pytest.fixture
def diverse_bucket_dir(kf_dir):
    """单个桶（interval=60）内 5 张差异分明的帧：
    0s 灰40(桶首) / 10s 灰42(近似桶首) / 20s 黑 / 30s 白(与桶首差异最大) / 45s 噪声。"""
    kf_dir.mkdir()
    write_kf(kf_dir, 0.0, 40)
    write_kf(kf_dir, 10.0, 42)
    write_kf(kf_dir, 20.0, 0)
    write_kf(kf_dir, 30.0, 255)
    write_kf_noise(kf_dir, 45.0, seed=1)
    return str(kf_dir)


def test_k3_returns_three_distinct_sorted_frames(diverse_bucket_dir):
    """k=3：桶内返回 3 帧、互不相同、按时间有序，且带 score 调试字段。"""
    reps = select_representatives(diverse_bucket_dir, interval=60.0, k=3)

    assert len(reps) == 3
    times = [r["t"] for r in reps]
    assert times == sorted(times), "输出必须按时间排序"
    assert len(set(times)) == 3, "3 帧必须互不相同"
    assert len({r["path"] for r in reps}) == 3
    # 全部来自同一桶（0-60s）
    assert all(0.0 <= t < 60.0 for t in times)
    for r in reps:
        assert set(r) == {"t", "path", "score"}
        assert isinstance(r["score"], float) and 0.0 <= r["score"] <= 1.0
    # 第一选中帧必是与桶首差异最大的白帧(30s)；桶首灰40(0s)不该入选
    assert 30.0 in times
    assert 0.0 not in times


def test_k1_output_matches_legacy(diverse_bucket_dir):
    """k=1（默认）：输出与旧逻辑完全一致——与桶首差异最大 1 帧，无 score 字段。"""
    legacy = select_representatives(diverse_bucket_dir, interval=60.0)
    k1 = select_representatives(diverse_bucket_dir, interval=60.0, k=1)

    assert k1 == legacy
    assert len(k1) == 1
    assert k1[0]["t"] == 30.0            # 白帧与桶首灰40差异最大
    assert set(k1[0]) == {"t", "path"}   # 旧结构契约不变


def test_k_exceeds_bucket_size_clamps(diverse_bucket_dir):
    """k 大于桶内帧数时返回全部帧（不报错、不重复）。"""
    reps = select_representatives(diverse_bucket_dir, interval=60.0, k=10)
    assert len(reps) == 5
    times = [r["t"] for r in reps]
    assert times == sorted(times) and len(set(times)) == 5


def test_k3_multi_bucket_still_guarantees_interval_coverage(kf_dir):
    """多桶 + k=3：每桶仍产帧，锚点时间覆盖不变（时间完整性硬保证保持）。"""
    kf_dir.mkdir()
    for i in range(46):  # 0..45s，每秒一帧（近似静止）
        write_kf(kf_dir, float(i), 10 + (i % 5))
    reps = select_representatives(str(kf_dir), interval=10.0, k=3)
    times = [r["t"] for r in reps]
    assert len(reps) >= 3 * 4  # 46s / 10s 至少 4 桶 × 每桶 3 帧（末桶可能不满）
    for a, b in zip(times, times[1:]):
        assert b >= a  # 有序
    assert times[0] == 0.0


# ==================== W1: 自适应桶宽 ====================

def test_adaptive_widens_buckets_in_static_section(kf_dir):
    """静止段（帧间差异 < 3%）：adaptive 桶宽倍增 => 锚点数比固定桶宽更少。

    120 张完全相同的帧（0..119s，每秒一张），interval=30：
    固定模式 4 桶 4 锚点；自适应模式首桶静止触发倍增 => 3 桶 3 锚点。
    """
    kf_dir.mkdir()
    for i in range(120):
        write_kf(kf_dir, float(i), 60)

    fixed = select_representatives(str(kf_dir), interval=30.0, adaptive=False)
    adaptive = select_representatives(str(kf_dir), interval=30.0, adaptive=True)

    assert len(fixed) == 4
    assert len(adaptive) < len(fixed)
    assert adaptive[0]["t"] == 0.0
    # 时间完整性保持：锚点间隔仍 >= interval（静止段只会更稀疏）
    ats = [r["t"] for r in adaptive]
    assert all(b - a >= 30.0 - 0.5 for a, b in zip(ats, ats[1:]))


def test_adaptive_halves_buckets_in_active_section(kf_dir):
    """高活跃段（桶内最大差异 > 20%）：adaptive 桶宽减半 => 锚点数比固定更密。

    100 张黑白交替帧（0..99s），interval=10：固定 10 桶；
    自适应首桶判定高活跃 => 后续桶宽 5s => 约 19 个锚点。
    """
    kf_dir.mkdir()
    for i in range(100):
        write_kf(kf_dir, float(i), 0 if i % 2 == 0 else 255)

    fixed = select_representatives(str(kf_dir), interval=10.0, adaptive=False)
    adaptive = select_representatives(str(kf_dir), interval=10.0, adaptive=True)

    assert len(fixed) == 10
    assert len(adaptive) > len(fixed)


def test_adaptive_off_by_default_matches_fixed(kf_dir):
    """不传 adaptive 时与 adaptive=False 完全一致（默认关闭，向后兼容）。"""
    kf_dir.mkdir()
    for i in range(40):
        write_kf(kf_dir, float(i), 30)
    plain = select_representatives(str(kf_dir), interval=10.0)
    off = select_representatives(str(kf_dir), interval=10.0, adaptive=False)
    assert plain == off


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
