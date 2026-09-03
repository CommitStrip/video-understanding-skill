"""llm_export 测试：缩放导出/联系表/token 估算（离线纯函数）。"""
import cv2
import numpy as np
import pytest

from vus.llm_export import (
    build_grid,
    estimate_tokens,
    export_llm_pack,
    export_rep_images,
)


def _make_reps(tmp_path, n, w=1920, h=1080):
    reps = []
    for i in range(n):
        img = np.random.RandomState(i).randint(0, 255, (h, w, 3), dtype=np.uint8)
        p = tmp_path / f"kf_{i:03d}.jpg"
        cv2.imwrite(str(p), img)
        reps.append({"t": float(i * 20), "path": str(p)})
    return reps


def test_estimate_tokens_positive():
    assert estimate_tokens(640, 360) == int(640 * 360 / 750) + 1
    assert estimate_tokens(1920, 1080) > estimate_tokens(640, 360)


def test_export_rep_images_downscales(tmp_path):
    reps = _make_reps(tmp_path, 5)
    out = tmp_path / "llm"
    exported = export_rep_images(reps, str(out), max_width=640)
    assert len(exported) == 5
    for f in exported:
        assert f["w"] == 640
        img = cv2.imread(f["file"])
        assert img is not None and img.shape[1] == 640


def test_export_keeps_small_images(tmp_path):
    reps = _make_reps(tmp_path, 2, w=320, h=180)
    exported = export_rep_images(reps, str(tmp_path / "llm"), max_width=640)
    assert all(f["w"] == 320 for f in exported)  # 不放大


def test_build_grid_sheets(tmp_path):
    reps = _make_reps(tmp_path, 10, w=320, h=180)
    exported = export_rep_images(reps, str(tmp_path / "llm"), max_width=320)
    grids = build_grid(exported, str(tmp_path / "llm"), cols=3)
    assert len(grids) == 2  # 10 帧 / 每张 3×3 → 2 张联系表
    for g in grids:
        sheet = cv2.imread(g["file"])
        assert sheet is not None and g["tokens"] > 0


def test_export_llm_pack_summary(tmp_path):
    reps = _make_reps(tmp_path, 9, w=1920, h=1080)
    pack = export_llm_pack(reps, str(tmp_path / "llm"), max_width=640, grid_cols=3)
    assert pack["rep_count"] == 9
    assert pack["tokens_grid"] < pack["tokens_direct"]  # 联系表显著省 token
    assert len(pack["grids"]) == 1
