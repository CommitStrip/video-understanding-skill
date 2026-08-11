#!/usr/bin/env python3
"""运行本项目基准；外部基线必须显式启用并单独安装。"""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from integrated_pipeline import run_realtime_pipeline
from select_representatives import select_representatives

TESTS = ["aba", "slow", "hue", "static"]


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def run_crv(video, output_dir):
    try:
        from claude_real_video import process
    except ImportError as exc:
        raise RuntimeError(
            "未安装 claude-real-video；不能生成外部基线。请固定版本后再运行 --baseline crv"
        ) from exc

    started = time.perf_counter()
    process(str(video), str(output_dir), do_transcribe=False)
    elapsed = time.perf_counter() - started
    frames_path = output_dir / "frames.json"
    if not frames_path.is_file():
        raise RuntimeError(f"外部基线没有生成预期文件: {frames_path}")
    data = json.loads(frames_path.read_text(encoding="utf-8"))
    frames = data if isinstance(data, list) else data.get("frames", [])
    timestamps = [
        float(item.get("timestamp_sec", item.get("t", item.get("timestamp", 0))))
        for item in frames
        if isinstance(item, dict)
    ]
    return {"elapsed_s": round(elapsed, 4), "frames": len(timestamps), "timestamps": timestamps}


def run_ours(video, output_dir, interval):
    started = time.perf_counter()
    pipe, _, _ = run_realtime_pipeline(
        str(video),
        str(output_dir),
        config={"keyframe_interval_hz": 1.5},
        visual_only=True,
    )
    pipeline_elapsed = time.perf_counter() - started

    selection_started = time.perf_counter()
    representatives = select_representatives(str(output_dir / "keyframes"), interval=interval)
    selection_elapsed = time.perf_counter() - selection_started
    representatives_path = output_dir / "representatives.json"
    representatives_path.write_text(
        json.dumps({"representatives": representatives}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "pipeline_elapsed_s": round(pipeline_elapsed, 4),
        "selection_elapsed_s": round(selection_elapsed, 4),
        "keyframes": len(pipe.keyframes),
        "representatives": len(representatives),
        "representative_timestamps": [item["t"] for item in representatives],
        "processing_fps": round(pipe.frame_count / pipeline_elapsed, 2) if pipeline_elapsed else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="视频代表帧基准")
    parser.add_argument("--bench-dir", default=str(BENCH), help="测试视频和输出目录")
    parser.add_argument("--baseline", choices=["none", "crv"], default="none")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval 必须大于 0")

    bench_dir = Path(args.bench_dir).resolve()
    results = {
        "schema_version": 1,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "commit": _git_commit(),
            "baseline": args.baseline,
        },
        "tests": [],
    }

    for name in TESTS:
        video = bench_dir / f"{name}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"缺少测试视频: {video}；请先运行 gen_bench.py")
        row = {"test": name}
        our_output = bench_dir / f"our_out_{name}"
        our_output.mkdir(parents=True, exist_ok=True)
        row["ours"] = run_ours(video, our_output, args.interval)

        if args.baseline == "crv":
            baseline_output = bench_dir / f"crv_out_{name}"
            baseline_output.mkdir(parents=True, exist_ok=True)
            row["crv"] = run_crv(video, baseline_output)
        results["tests"].append(row)

    output_path = bench_dir / "bench_results.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基准结果: {output_path}")


if __name__ == "__main__":
    main()
