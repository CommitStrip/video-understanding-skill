#!/usr/bin/env python3
"""
evidence.py - 证据包输出单元（W9 借鉴外部审查：可审计/可回放/可按预算删减）
============================================================================
把管线产物聚合为事件级 evidence pack。每个 pack 代表一段"值得关注的区间"：
  - 时间范围 + 标注帧（前/峰/后三张）
  - 语音文本 + OCR 差量 + 运动统计
  - 选取依据（reasons）+ 置信度
下游 LLM 可以按预算删减字段；审计者可以知道为何该帧被选中。
"""

import json
import os


def build_evidence_packs(aligned, keyframes, attention_windows=None,
                         max_frames_per_pack=3):
    """把对齐段 + 关键帧 + 注意力窗口聚合为证据包列表。

    aligned: aligned_output.json 的 aligned_segments（含 ASR + 对齐运动/关键帧计数）
    keyframes: 管线关键帧列表 [{t, reason, ...}]
    attention_windows: ASR 注意力回链窗口 [{t, end, cue}]
    max_frames_per_pack: 每个 pack 最多引用的帧数
    返回 [pack dict] 按时间序
    """
    kf_times = sorted(k.get("t", 0.0) for k in keyframes)
    packs = []
    for seg in aligned:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        if end <= start:
            continue
        # 选帧：前/峰/后 三张（区间内 + 邻域的最近关键帧）
        in_range = [t for t in kf_times if start <= t <= end]
        frames = []
        if in_range:
            frames = in_range[:max_frames_per_pack]
        # 无帧时取区间边界附近最近的关键帧
        if not frames:
            near_start = min(kf_times, key=lambda t: abs(t - start)) if kf_times else None
            near_end = min(kf_times, key=lambda t: abs(t - end)) if kf_times else None
            if near_start is not None and (abs(near_start - start) <= 30.0 or
                                           abs(near_end - end) <= 30.0):
                frames = sorted({near_start, near_end})[:max_frames_per_pack]

        speech = seg.get("text", "")
        motion_count = seg.get("linked_motion_events", 0)
        kf_count = seg.get("linked_keyframes", 0)

        reasons = []
        if speech:
            reasons.append("speech")
        if motion_count > 0:
            reasons.append("motion")
        if kf_count > 0:
            reasons.append("keyframe")
        # 注意力窗口命中
        if attention_windows:
            for w in attention_windows:
                wt = w.get("t", 0.0)
                we = w.get("end", wt)
                if start <= we and wt <= end:
                    reasons.append("attention")
                    break

        if not reasons:
            continue

        # 置信度：简单启发式——多信号命中越多越高
        confidence = min(1.0, len(reasons) / 4.0)

        packs.append({
            "event_id": f"ev_{len(packs):04d}",
            "time_range": [round(start, 2), round(end, 2)],
            "frames": frames,
            "speech": speech[:500],
            "motion_events": motion_count,
            "keyframes_in_range": kf_count,
            "selection_reasons": reasons,
            "confidence": round(confidence, 2),
        })
    return packs


def save_evidence_packs(packs, output_dir):
    """写 evidence_packs.json 到 output_dir（安全路径）。返回写入路径。"""
    from .io_utils import write_json
    path = write_json(output_dir, "evidence_packs.json", packs)
    return path
