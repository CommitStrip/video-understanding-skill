"""全局运动补偿 + 运动段锚点保障测试（W9 借鉴外部审查建议）。"""
import numpy as np
import cv2
import pytest

from vus.smart_pipeline import SmartPipeline

FPS = 30.0
W, H = 320, 240


def _tex_base(seed=11):
    """结构化纹理背景（矩形+圆，模拟真实画面内容）：3 通道 BGR，
    含大量角点保证 ORB 特征充足。"""
    base = np.full((H, W, 3), 40, dtype=np.uint8)
    rng = np.random.RandomState(seed)
    for _ in range(40):
        x, y = rng.randint(0, W - 45), rng.randint(0, H - 45)
        w, h = rng.randint(15, 45), rng.randint(15, 45)
        val = int(rng.randint(60, 230))
        cv2.rectangle(base, (x, y), (x + w, y + h), (val, val, val), -1)
    for _ in range(15):
        c = (rng.randint(25, W - 25), rng.randint(25, H - 25))
        cv2.circle(base, c, int(rng.randint(10, 28)), (int(rng.randint(80, 220)),) * 3, -1)
    return base


def pan_frames(n=60, shift=8, block=None):
    """纹理背景逐帧水平平移（模拟镜头横摇）；block=(x0,y0,x1,y1,val) 时叠加固定前景块。"""
    base = _tex_base()
    frames = []
    for i in range(n):
        f = np.roll(base, i * shift, axis=1)
        if block is not None:
            x0, y0, x1, y1, val = block
            f[y0:y1, x0:x1] = val
        frames.append(f)
    return frames


def feed(pipe, frames):
    evs = []
    for i, f in enumerate(frames):
        evs.extend(pipe.process_frame(f, i / FPS))
    return evs


def motion_evs(evs):
    return [e for e in evs if e["type"].startswith("motion")]


# ==================== 全局运动补偿 ====================

def test_camera_pan_suppressed_with_compensation():
    pipe = SmartPipeline({"motion_compensation": True})
    evs = feed(pipe, pan_frames())
    assert motion_evs(evs) == [], "纯镜头平移应被补偿抑制"


def test_camera_pan_fires_without_compensation():
    pipe = SmartPipeline({"motion_compensation": False})
    evs = feed(pipe, pan_frames())
    assert motion_evs(evs), "关闭补偿后平移应触发运动（旧行为）"


def test_foreground_motion_still_fires_with_compensation():
    """平移背景 + 固定前景块：前景是真实变化，补偿后仍应触发。"""
    pipe = SmartPipeline({"motion_compensation": True})
    evs = feed(pipe, pan_frames(n=60, block=(10, 10, 110, 110, 0)))
    assert motion_evs(evs), "前景块变化不应被补偿吞掉"


def test_compensation_off_by_default():
    """默认关闭（特征贫乏场景会误杀真实目标运动），显式开启才生效。"""
    p1 = SmartPipeline()
    p2 = SmartPipeline({"motion_compensation": True})
    assert p1.motion_compensation is False
    assert p2.motion_compensation is True


# ==================== 运动段锚点保障 ====================

def _block_frames(pre=20, move=60, post=30, speed=16):
    """纯灰背景 + 中途出现并移动的高亮方块（快系统运动，慢系统零采纳）。"""
    frames = []
    for _ in range(pre):
        frames.append(np.full((H, W, 3), 40, dtype=np.uint8))
    for i in range(move):
        f = np.full((H, W, 3), 40, dtype=np.uint8)
        x = (i * speed) % (W - 40)
        cv2.rectangle(f, (x, 100), (x + 24, 124), (200, 200, 200), -1)
        frames.append(f)
    for _ in range(post):
        frames.append(np.full((H, W, 3), 40, dtype=np.uint8))
    return frames


def test_segment_anchor_guaranteed():
    """段内慢系统零采纳（小方块分数低于慢阈）→ motion_end 前补发段锚点。"""
    pipe = SmartPipeline()
    evs = feed(pipe, _block_frames())
    kinds = [(e["type"], e.get("reason")) for e in evs]
    assert ("keyframe", "segment_anchor") in kinds, f"缺段锚点: {kinds}"
    # 锚点必须在对应 motion_end 之前
    assert kinds.index(("keyframe", "segment_anchor")) < \
        kinds.index(("motion_end", None))


def test_segment_anchor_only_when_needed():
    """段内已有慢系统采纳（scene_change）→ 不再补发段锚点。"""
    pipe = SmartPipeline()
    rng = np.random.RandomState(5)
    frames = []
    for i in range(60):
        f = np.full((H, W, 3), 40, dtype=np.uint8)
        x = (i * 16) % (W - 40)
        cv2.rectangle(f, (x, 100), (x + 24, 124), (200, 200, 200), -1)
        frames.append(f)
        if i == 30:  # 中途整幅噪声突变（慢系统 scene_change 采纳）
            frames[-1] = rng.randint(0, 255, (H, W, 3), dtype=np.uint8)
    evs = feed(pipe, frames)
    reasons = [e.get("reason") for e in evs if e["type"] == "keyframe"]
    assert "scene_change" in reasons
    assert "segment_anchor" not in reasons
