"""--max-reps 自适应代表帧测试（进程内调 main()，不起子进程）。"""
import json
import sys

import cv2
import numpy as np
import pytest

from vus.select_representatives import main as select_main


def write_kfs(tmp_path, n, step_s=5.0, colors=None):
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)
    colors = colors or [30, 90, 160, 220]
    for i in range(n):
        img = np.full((48, 64, 3), colors[i % len(colors)], dtype=np.uint8)
        cv2.imwrite(str(kf_dir / f"kf_{i:04d}_t{i*step_s:.1f}s.jpg"), img)
    return kf_dir


def run_cli(tmp_path, kf_dir, *extra):
    out = tmp_path / "reps.json"
    argv = ["select_representatives", "--keyframes", str(kf_dir),
            "--out", str(out), *extra]
    argv = [a if not isinstance(a, float) else str(a) for a in argv]
    monkey = pytest.MonkeyPatch()
    with monkey.context():
        monkey.setattr(sys, "argv", argv)
        select_main()
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture
def dense_kfs(tmp_path):
    """80 张、间隔 5s（跨度 395s）的四色轮换关键帧。"""
    return write_kfs(tmp_path, 80)


def test_max_reps_respects_budget(tmp_path, dense_kfs):
    out = run_cli(tmp_path, dense_kfs, "--max-reps", "20")
    assert out["count"] <= 20
    assert out["count"] >= 10  # 不能因 interval 爆炸而塌缩到个位数


def test_max_reps_small_budget(tmp_path, dense_kfs):
    out = run_cli(tmp_path, dense_kfs, "--max-reps", "5")
    assert out["count"] <= 5


def test_interval_still_works_without_max_reps(tmp_path, dense_kfs):
    out = run_cli(tmp_path, dense_kfs, "--interval", "100")
    # 395s / 100s → 约 4-5 桶
    assert 2 <= out["count"] <= 8
    assert out["interval"] == 100.0
