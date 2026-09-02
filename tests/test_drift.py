"""渐变漂移采纳（gradual drift）单元测试。

复现实战发现的失明条件：内容细粒度演化（模拟课件批注推进）——
像素差分超阈（第一闸过）、pHash 距离 0 / 直方图相关 1.0（第二闸与
pHash 去重全部失明）。旧行为（drift_confirm_checks=0）零发帧，
新行为（默认 3 连检）按 reason=gradual_drift 发帧。
"""
import numpy as np
import cv2
import pytest

from vus.smart_pipeline import SmartPipeline

FPS = 30.0
W, H = 320, 240

# 高对比纹理底（模拟内容丰富的课件）：跨度 130 保证降采样后直方图
# 横跨多个 V-bin，状态间 hist 相关 >0.5、pHash 距离 <12 —— 纯漂移型（双闸全盲）
_TEX = np.random.RandomState(11).randint(25, 156, (H, W, 3)).astype(np.uint8)


def drift_frame(delta=48):
    """高细粒度内容帧：纹理底 + 隔行点阵 +delta（模拟批注推进）。

    实测特征（0.25 降采样口径）：相邻状态（delta 差 48）像素差分 = 12
    （> keyframe_diff=10），pHash 距离 = 0（DCT 低频不变），
    直方图相关 = 0.71-0.87 —— pHash/直方图双盲，仅像素差分可见。
    delta 上限 96：更大值会触顶 255 裁剪，破坏漂移特征。
    """
    f = _TEX.copy()
    f[::2, ::2] = np.clip(f[::2, ::2].astype(int) + delta, 0, 255).astype(np.uint8)
    return f


def feed(pipe, frames):
    events = []
    for i, frame in enumerate(frames):
        events.extend(pipe.process_frame(frame, i / FPS))
    return events


def kfs(pipe):
    return [e for e in pipe.events if e["type"] == "keyframe"]


def test_gradual_drift_emits_keyframes():
    """静止 2s -> 点阵内容出现并保持：漂移应在 3 连检（约 6s）后被采纳。"""
    pipe = SmartPipeline()
    frames = [_TEX.copy()] * 60                             # t=0-2s 静止
    frames += [drift_frame()] * 120                         # t=2-6s 漂移内容
    feed(pipe, frames)

    keys = kfs(pipe)
    assert len(keys) >= 2, f"漂移应产生 keyframe, 实际: {[(k['t'], k.get('reason')) for k in keys]}"
    drifts = [k for k in keys if k.get("reason") == "gradual_drift"]
    assert drifts, f"应含 gradual_drift 发帧, 实际 reasons: {[k.get('reason') for k in keys]}"
    assert drifts[0]["t"] >= 2.0                     # 漂移开始后才发
    assert drifts[0]["t"] <= 2.0 + 3 / 1.5 + 0.1     # 3 连检 ≈ 6s 内采纳


def test_drift_disabled_restores_old_behavior():
    """drift_confirm_checks=0 还原旧行为：双盲内容永不发帧。"""
    pipe = SmartPipeline({"drift_confirm_checks": 0})
    frames = [_TEX.copy()] * 30 + [drift_frame()] * 90
    feed(pipe, frames)
    keys = kfs(pipe)
    assert len(keys) == 1 and keys[0]["reason"] == "first_frame"


def test_drift_no_storm_after_acceptance():
    """采纳后基准更新：内容不变则 streak 复位，不会连环发帧。"""
    pipe = SmartPipeline()
    frames = [_TEX.copy()] * 30
    frames += [drift_frame()] * 300  # 漂移内容持续 10s 不再变化
    feed(pipe, frames)
    drifts = [k for k in kfs(pipe) if k.get("reason") == "gradual_drift"]
    assert len(drifts) == 1, f"稳态内容只应发 1 帧, 实际 {len(drifts)}"


def test_drift_recovery_after_new_content():
    """采纳后内容继续演化：streak 重新累计，能再次发帧。"""
    pipe = SmartPipeline()
    frames = [_TEX.copy()] * 30
    frames += [drift_frame(48)] * 90     # 第一态（t=1-4s）
    frames += [drift_frame(96)] * 270    # 第二态（t=4-13s，留足 3 连检窗口）
    feed(pipe, frames)
    drifts = [k for k in kfs(pipe) if k.get("reason") == "gradual_drift"]
    assert len(drifts) >= 2
    assert drifts[1]["t"] > drifts[0]["t"]


def test_scene_change_path_unchanged():
    """突变场景仍走 is_new_scene 立即采纳（不受漂移逻辑影响）。"""
    pipe = SmartPipeline()
    rng = np.random.RandomState(42)
    a = np.stack([np.full((H, W), 40, dtype=np.uint8)] * 3, axis=-1)
    b = rng.randint(0, 255, (H, W, 3)).astype(np.uint8)
    frames = [a] * 30 + [b] * 10
    events = feed(pipe, frames)
    keys = [e for e in events if e["type"] == "keyframe"]
    assert len(keys) == 2
    assert keys[1]["reason"] == "scene_change"
    assert keys[1]["t"] == pytest.approx(1.0, abs=0.1)  # 切换后立即采纳
