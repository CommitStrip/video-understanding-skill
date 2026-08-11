#!/usr/bin/env python3
"""计算低级像素变化覆盖代理指标；该指标不等同于语义准确率。"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEFAULT_BENCH = Path(__file__).resolve().parent
DIFF_SIZE = 64
CHANGE_THRESHOLD = 3.0


def pixel_diff(img_a, img_b, size=DIFF_SIZE):
    resize = lambda image: cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    first = resize(img_a).astype(np.float32)
    second = resize(img_b).astype(np.float32)
    return float(np.mean(np.abs(first - second)) / 255.0 * 100.0)


def change_profile(video):
    """按 1 秒桶计算首尾帧像素差，作为低级画面变化代理。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开测试视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"测试视频 FPS 无效: {video}")
    duration = frame_count / fps
    profile = []
    for second in range(int(duration)):
        first_index = int(second * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_index)
        ok_first, first = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(first_index + int(fps) - 1, frame_count - 1))
        ok_last, last = cap.read()
        if ok_first and ok_last:
            profile.append({"second": second, "change": pixel_diff(first, last)})
    cap.release()
    return profile


def coverage_proxy(selected_timestamps, profile):
    changed = [item for item in profile if item["change"] > CHANGE_THRESHOLD]
    if not changed:
        return {"changed_buckets": 0, "covered_buckets": 0, "coverage": None}
    covered = sum(
        1
        for item in changed
        if any(item["second"] <= timestamp < item["second"] + 1 for timestamp in selected_timestamps)
    )
    return {
        "changed_buckets": len(changed),
        "covered_buckets": covered,
        "coverage": round(covered / len(changed), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="像素变化覆盖代理指标（非语义准确率）")
    parser.add_argument("--bench-dir", default=str(DEFAULT_BENCH))
    args = parser.parse_args()
    bench_dir = Path(args.bench_dir).resolve()
    results_path = bench_dir / "bench_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"缺少 {results_path}；请先运行 run_bench.py")

    bench_results = json.loads(results_path.read_text(encoding="utf-8"))
    output = {
        "schema_version": 1,
        "metric": "one-second pixel-change coverage proxy",
        "semantic_accuracy": False,
        "change_threshold_percent": CHANGE_THRESHOLD,
        "tests": [],
    }
    for row in bench_results["tests"]:
        name = row["test"]
        profile = change_profile(bench_dir / f"{name}.mp4")
        comparison = {
            "test": name,
            "ours": coverage_proxy(row["ours"]["representative_timestamps"], profile),
        }
        if "crv" in row:
            comparison["crv"] = coverage_proxy(row["crv"]["timestamps"], profile)
        output["tests"].append(comparison)

    output_path = bench_dir / "land_compare_results.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"代理指标结果: {output_path}")
    print("注意：像素变化覆盖率不能证明视频语义理解准确率。")


if __name__ == "__main__":
    main()
