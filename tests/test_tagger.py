"""W8b T0.5 标签道单元测试：BasicLabeler 标签语义 / 工厂 / CLIP 槽位预留。"""
import numpy as np
import pytest

from vus.live import BasicLabeler, LabelerError, create_labeler


@pytest.fixture(scope="module")
def labeler():
    return BasicLabeler()


def _frame(h=48, w=64, color=(0, 0, 0)):
    return np.full((h, w, 3), color, dtype=np.uint8)


def test_static_scene_labels(labeler):
    out = labeler.label(_frame(), motion_ratio=0.0)
    assert out, "必须产出标签"
    assert out[0]["label"] == BasicLabeler.STATIC
    scores = [d["score"] for d in out]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_motion_labels_threshold(labeler):
    strong = labeler.label(_frame(), motion_ratio=0.02)
    assert strong[0]["label"] == BasicLabeler.STRONG_MOTION
    weak = labeler.label(_frame(), motion_ratio=0.001)
    assert weak[0]["label"] == BasicLabeler.WEAK_MOTION


def test_face_label_presence(labeler):
    """无人脸图上不应给'有人脸'；检测不可用时人脸标签整体缺席（不误报）。"""
    out = labeler.label(_frame(color=(30, 30, 30)), motion_ratio=0.0)
    labels = [d["label"] for d in out]
    if labeler.face_detection_available:
        assert BasicLabeler.FACE not in labels  # 纯色块没有人脸
        assert BasicLabeler.NO_FACE in labels
    else:
        assert BasicLabeler.NO_FACE not in labels


def test_label_runs_fast_enough(labeler):
    """T0.5 的生命线：单帧标签必须毫秒级（100ms 上限是 10 倍冗余的宽松门）。"""
    import time
    frame = _frame(h=240, w=320, color=(60, 60, 60))
    t0 = time.perf_counter()
    for _ in range(5):
        labeler.label(frame, motion_ratio=0.005)
    avg_ms = (time.perf_counter() - t0) / 5 * 1000
    assert avg_ms < 100, f"标签道均耗时 {avg_ms:.1f}ms 超出实时预算"


def test_factory_backends():
    assert isinstance(create_labeler("basic"), BasicLabeler)
    with pytest.raises(LabelerError):
        create_labeler("clip")     # W9 预留
    with pytest.raises(LabelerError):
        create_labeler("nope")
