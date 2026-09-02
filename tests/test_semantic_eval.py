"""bench/semantic_eval 指标计算测试（W3）。

不跑管线，只验证核心数学：语义覆盖率的时长加权、冗余度、基尼系数、
未落入场景的代表帧统计，以及 annotate_vlm 的 VLM 回复解析。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench" / "semantic_eval"))

from eval_semantic import assign_reps, evaluate, gini, load_reps  # noqa: E402
from annotate_vlm import parse_scenes_from_response  # noqa: E402

GT = {
    "video": "t.mp4",
    "duration_s": 30.0,
    "source": "human",
    "scenes": [
        {"start_s": 0.0, "end_s": 10.0, "desc": "长场景"},
        {"start_s": 10.0, "end_s": 15.0, "desc": "短场景一"},
        {"start_s": 15.0, "end_s": 30.0, "desc": "长场景二"},
    ],
}


def test_coverage_is_duration_weighted():
    # 只覆盖 10s 的"长场景二"（15-30s）-> 10/30，不看场景个数
    r = evaluate([20.0], GT)
    assert r["semantic_coverage"] == 0.5
    assert r["covered_scenes"] == 1
    assert r["redundancy"] == pytest_approx(1 / 3)


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-3)


def test_full_coverage_and_redundancy_one():
    r = evaluate([5.0, 12.0, 20.0], GT)
    assert r["semantic_coverage"] == 1.0
    assert r["redundancy"] == 1.0
    assert r["gini"] == 0.0  # 每场景恰 1 帧 -> 完全均匀


def test_unassigned_reps_reported():
    counts, unassigned = assign_reps([-1.0, 5.0, 35.0], GT["scenes"])
    assert counts == [1, 0, 0]
    assert unassigned == [-1.0, 35.0]
    r = evaluate([-1.0, 5.0], GT)
    assert r["unassigned_reps"] == [-1.0]
    # 未落入场景的帧不影响覆盖分母（GT 时长）
    assert r["semantic_coverage"] == pytest_approx(10 / 30)


def test_gini_distribution():
    assert gini([]) == 0.0
    assert gini([0, 0, 0]) == 0.0
    assert gini([1, 1, 1]) == pytest_approx(0.0)
    # 全部堆一处：有限 n 下基尼最大值为 (n-1)/n（此归一化下的理论上限）
    assert gini([3, 0, 0]) == pytest_approx(2 / 3)
    assert gini([1, 0]) == pytest_approx(0.5)
    assert 0 < gini([2, 1, 0]) < gini([3, 0, 0])


def test_boundary_frame_at_scene_end_belongs_to_that_scene():
    # t=10.0 属于 [0,10)（左闭右开 + 末点容差），不应漏计
    counts, unassigned = assign_reps([10.0], GT["scenes"])
    assert counts[0] == 1 and unassigned == []


def test_load_reps_accepts_bare_list(tmp_path):
    p = tmp_path / "reps.json"
    p.write_text('[{"t": 3.0, "path": "a.jpg"}, {"t": 1.0, "path": "b.jpg"}]',
                 encoding="utf-8")
    assert load_reps(str(p)) == [1.0, 3.0]


def test_parse_vlm_response_tolerates_markdown_fence():
    text = '好的，以下是场景表：\n```json\n{"scenes": [{"start_s": 0, "end_s": 12, "desc": "A"}]}\n```'
    scenes = parse_scenes_from_response(text)
    assert scenes == [{"start_s": 0.0, "end_s": 12.0, "desc": "A"}]
