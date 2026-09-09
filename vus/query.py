#!/usr/bin/env python3
"""
query.py - 查询回源：两阶段视频上下文编译器的第二阶段
========================================================
第一阶段（integrated_pipeline）生成低成本索引（index.json）。
第二阶段（本模块）：用户提问 → 在 ASR/OCR/注意力窗口中检索候选区间
→ 回源原视频对候选区间做密采样（提高帧密度重跑 Tier2/Tier3）
→ 输出 focused 代表帧 + 字幕切片。

依赖原视频可再访问（离线分析定位；直播不支持，文档声明）。
"""

import json
import os

from .io_utils import write_json
from .index import build_index


def _merge_ranges(ranges, gap_s=10.0):
    """合并重叠/相邻时间区间。"""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s <= merged[-1][1] + gap_s:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def find_candidate_ranges(index, keywords=None, attention_only=False):
    """从 index 中检索候选时间区间。

    keywords: 关键词列表，匹配 ASR/OCR 文本
    attention_only: 仅返回注意力窗口
    返回合并后的 [(start_s, end_s)] 列表
    """
    ranges = []
    if not attention_only and keywords:
        for seg in index.get("asr", []):
            text = seg.get("text", "")
            for kw in keywords:
                if kw.lower() in text.lower():
                    ranges.append((seg["t"], seg.get("end_t", seg["t"] + 2.0)))
                    break
        for o in index.get("ocr", []):
            for kw in keywords:
                if kw.lower() in o.get("text", "").lower():
                    ranges.append((o["t"] - 2.0, o["t"] + 5.0))
                    break
    for w in index.get("attention_windows", []):
        ranges.append((w["t"], w.get("end", w["t"] + 5.0)))
    return _merge_ranges(ranges)


def run_focused_pipeline(video_path, output_dir, ranges, config=None):
    """对候选区间回源原视频，高密度重跑画面链。

    ranges: [(start_s, end_s)]；对每个区间以 kf_hz×2 提取帧。
    返回 focused 结果列表。
    """
    import cv2
    from .smart_pipeline import SmartPipeline

    focused = []
    for i, (start_s, end_s) in enumerate(ranges):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000)
        end_frame = int(end_s * fps)
        # kf_hz 密度加倍
        kf_hz = (config or {}).get("keyframe_interval_hz", 1.5) * 2
        check_interval = max(1, int(fps / kf_hz))
        pipe = SmartPipeline(config)
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or (end_frame > 0 and cap.get(cv2.CAP_PROP_POS_FRAMES) > end_frame):
                break
            t = frame_idx / fps
            if frame_idx % check_interval == 0 or frame_idx == 0:
                pipe.process_frame(frame, t)
            frame_idx += 1
        cap.release()
        focused.append({
            "range": [round(start_s, 2), round(end_s, 2)],
            "keyframes": pipe.keyframes,
            "motion_segments": pipe.motion_segments,
        })
    return focused


def query_video(index_path, video_path=None, keywords=None,
                attention_only=False, output_dir=None):
    """查询感知编译器入口：索引检索 + 候选区间回源。

    返回 {"ranges": [...], "focused": [...], "index": index}
    """
    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    ranges = find_candidate_ranges(index, keywords=keywords,
                                   attention_only=attention_only)
    focused = []
    if video_path and os.path.exists(video_path) and ranges:
        focused = run_focused_pipeline(video_path, output_dir, ranges)
    return {"ranges": ranges, "focused": focused, "index": index}
