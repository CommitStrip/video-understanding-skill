#!/usr/bin/env python3
"""
llm_export.py - 代表帧的 LLM 友好导出（借鉴 claude-real-video 的 token 成本设计）
================================================================================
把 Tier3 代表帧转成"直接可喂 LLM"的形态：
  1. 等比缩放到 max_width（默认 640px）——token 成本约降 9 倍
  2. --grid 联系表拼图（3×3）——60 帧拼 7 张，token 再降约 9 倍
  3. token 估算（≈ 宽×高/750，与 crv README 口径一致）
"""

import cv2
import numpy as np
from pathlib import Path


def estimate_tokens(w, h):
    """粗估单图 base64 喂给视觉模型的 token 数（≈像素/750）。"""
    return int(w * h / 750) + 1


def export_rep_images(reps, out_dir, max_width=640, quality=80):
    """把代表帧等比缩放导出。返回 [{"t", "file", "w", "h", "tokens"}]。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exported = []
    for i, r in enumerate(reps):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if w > max_width:
            nh = max(1, round(h * max_width / w))
            img = cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)
            w, h = max_width, nh
        name = f"rep_{i:02d}_t{int(r['t']):04d}s.jpg"
        p = out / name
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        exported.append({"t": r["t"], "file": str(p), "w": w, "h": h,
                         "tokens": estimate_tokens(w, h)})
    return exported


def build_grid(frames_meta, out_dir, cols=3, prefix="grid"):
    """把已导出的缩放帧拼成联系表（每 cols*rows 一张）。返回拼图元数据列表。"""
    out = Path(out_dir)
    if not frames_meta:
        return []
    rows_per_grid = cols
    per = cols * rows_per_grid
    grids = []
    for g in range(0, len(frames_meta), per):
        batch = frames_meta[g:g + per]
        cell_w = max(f["w"] for f in batch)
        cell_h = max(f["h"] for f in batch)
        rows_n = (len(batch) + cols - 1) // cols
        sheet = 255 * np.ones((rows_n * cell_h, cols * cell_w, 3), dtype=np.uint8)
        for idx, f in enumerate(batch):
            img = cv2.imread(f["file"])
            if img is None:
                continue
            r, c = idx // cols, idx % cols
            sheet[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w][: img.shape[0], : img.shape[1]] = img
        name = f"{prefix}_{len(grids) + 1}.jpg"
        p = out / name
        cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 78])
        grids.append({"file": str(p), "w": cols * cell_w, "h": rows_n * cell_h,
                      "frames": len(batch),
                      "tokens": estimate_tokens(cols * cell_w, rows_n * cell_h)})
    return grids


def export_llm_pack(reps, out_dir, max_width=640, grid_cols=3):
    """一步到位：缩放导出 + 联系表 + token 预算汇总。返回汇总 dict。"""
    exported = export_rep_images(reps, out_dir, max_width=max_width)
    grids = build_grid(exported, out_dir, cols=grid_cols)
    direct_tokens = sum(f["tokens"] for f in exported)
    grid_tokens = sum(g["tokens"] for g in grids)
    summary = {
        "rep_count": len(exported),
        "images": exported,
        "grids": grids,
        "tokens_direct": direct_tokens,
        "tokens_grid": grid_tokens,
    }
    return summary
