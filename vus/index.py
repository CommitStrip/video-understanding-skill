#!/usr/bin/env python3
"""
index.py - 轻量索引（W10：查询感知编译器的第一阶段产物）
==========================================================
聚合管线结构化产物为带时间戳的可检索索引（index.json）。
第二阶段：用户提问时经 query.py 回源原视频候选区间做密采样。
"""

import json
import os

from .io_utils import write_json


def build_index(aligned_output_path, output_dir=None):
    """从 aligned_output.json + pipeline_results.json 聚合 index.json。

    返回 (index dict, index_path)。
    全部条目带时间戳（start/end），可被 query.py 检索。
    """
    base = os.path.dirname(aligned_output_path)
    with open(aligned_output_path, encoding="utf-8") as f:
        aligned = json.load(f)

    asr = []
    for seg in aligned.get("asr_segments", []):
        if seg.get("hallucination"):
            continue
        asr.append({"t": seg.get("t", 0.0),
                     "end_t": seg.get("end_t", seg.get("t", 0.0)),
                     "text": seg.get("text", "")})

    ocr = [{"t": o.get("t", 0.0), "text": o.get("text", "")}
           for o in aligned.get("ocr_events", []) if o.get("text")]

    attention = [{"t": w.get("t", 0.0), "end": w.get("end", w.get("t", 0.0))}
                 for w in aligned.get("attention_windows", [])]

    pr_path = os.path.join(base, "pipeline_results.json")
    keyframes = []
    motion_segments = []
    if os.path.exists(pr_path):
        with open(pr_path, encoding="utf-8") as f:
            pr = json.load(f)
        keyframes = [{"t": k.get("t", 0.0), "reason": k.get("reason", "")}
                     for k in pr.get("keyframes", [])]
        motion_segments = pr.get("motion_segments", [])

    index = {
        "asr": asr,
        "ocr": ocr,
        "attention_windows": attention,
        "keyframes": keyframes,
        "motion_segments": motion_segments,
    }

    if output_dir is None:
        output_dir = base
    out_path = write_json(output_dir, "index.json", index)
    return index, out_path
