# -*- coding: utf-8 -*-
"""CLIP 单帧嵌入基准：CPU vs DirectML（真实关键帧全量）。
用法: python bench/gpu/bench_clip.py <keyframes_dir> [--limit N] [--out-dir DIR]
输出: <out-dir>/gpu_clip_bench.json
"""
import argparse
import glob
import json
import os
import sys
import time

import cv2

from vus.clip_onnx import ClipOnnx
from vus.io_utils import write_json


def run(device, frames):
    enc = ClipOnnx(device=device)
    t0 = time.perf_counter()
    for f in frames[:3]:  # 预热（首帧含会话初始化与显存分配）
        enc.embed(f)
    warm = time.perf_counter() - t0
    t0 = time.perf_counter()
    for f in frames:
        enc.embed(f)
    total = time.perf_counter() - t0
    return {
        "device": enc.device,
        "frames": len(frames),
        "warmup_s": round(warm, 3),
        "total_s": round(total, 2),
        "ms_per_frame": round(total / len(frames) * 1000, 2),
        "fps": round(len(frames) / total, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keyframes")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="bench/gpu")
    ap.add_argument("--dml", action="store_true",
                    help="对比 directml（默认对比 cuda）")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.keyframes, "kf_*.jpg")))
    if args.limit:
        paths = paths[:args.limit]
    frames = [cv2.imread(p) for p in paths]
    frames = [f for f in frames if f is not None]
    print(f"frames: {len(frames)} from {args.keyframes}")

    results = []
    devices = ("cpu", "directml") if "--dml" in sys.argv else ("cpu", "cuda")
    for device in devices:
        try:
            r = run(device, frames)
        except Exception as e:
            r = {"device": device, "error": str(e)[:200]}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    write_json(args.out_dir, "gpu_clip_bench.json", {
        "keyframes_dir": args.keyframes,
        "results": results,
    })
    print(f"saved: {args.out_dir}/gpu_clip_bench.json")


if __name__ == "__main__":
    main()
