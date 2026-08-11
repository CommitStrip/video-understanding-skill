#!/usr/bin/env python3
"""
validate_realtime.py - 实时预算分配架构验证
在 720p/50fps 与 1080p/30fps 两种规格上，验证画面链处理速率是否达到实时。

判定标准：
  处理速率(proc_fps) >= 视频帧率(fps)  => 实时达标（跟随摄像头实时流）
"""
import cv2
import argparse
import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart_pipeline import SmartPipeline


def gen_test_video(path, w, h, fps, duration, motion=True):
    """生成带运动的测试视频（前景移动物体 + 缓慢场景变化）。"""
    cap_v = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not cap_v.isOpened():
        raise RuntimeError(f"无法创建测试视频（mp4v 编码器不可用）: {path}")
    n = int(fps * duration)
    t = 0.0
    for i in range(n):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # 背景缓慢渐变（模拟光照/场景漂移）
        bg = int(30 + 20 * np.sin(t * 0.5))
        frame[:] = (bg, bg, bg + 10)
        if motion:
            # 移动的大色块（面积足够触发语义门控）
            cx = int(w * (0.2 + 0.6 * (0.5 + 0.5 * np.sin(t * 1.6))))
            cy = int(h * (0.3 + 0.4 * (0.5 + 0.5 * np.cos(t * 1.2))))
            r = int(min(w, h) * 0.15)
            cv2.circle(frame, (cx, cy), r, (0, 200, 120), -1)
            # 说话人头部的类微动（小面积，应被语义门控过滤）
            hx = int(w * 0.7)
            hy = int(h * 0.6) + int(4 * np.sin(t * 3))
            cv2.circle(frame, (hx, hy), int(min(w, h) * 0.02), (200, 200, 200), -1)
        cap_v.write(frame)
        t += 1.0 / fps
    cap_v.release()
    return path


def measure_rt(video_path, fast_scale, kf_hz, sample_frames=3000):
    """对一段视频跑实时管线，返回处理速率与事件统计。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开测试视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"测试视频 FPS 无效: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipe = SmartPipeline({"fast_scale": fast_scale, "keyframe_interval_hz": kf_hz})
    frame_idx = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipe.process_frame(frame, frame_idx / fps)
        frame_idx += 1
        if frame_idx >= sample_frames:
            break
    cap.release()
    elapsed = time.time() - t0
    proc_fps = frame_idx / elapsed if elapsed > 0 else 0
    summary = pipe.get_summary()
    summary["spec"] = f"{w}x{h}@{fps}fps"
    summary["proc_fps"] = round(proc_fps, 1)
    summary["realtime"] = proc_fps >= fps
    summary["elapsed_s"] = round(elapsed, 2)
    summary["frames_processed"] = frame_idx
    process_times_ms = np.asarray(pipe.process_times, dtype=np.float64) * 1000.0
    summary["recent_p95_process_ms"] = round(float(np.percentile(process_times_ms, 95)), 3) if len(process_times_ms) else 0
    return summary


def main():
    parser = argparse.ArgumentParser(description="视觉画面链吞吐量冒烟验证（不代表完整 ASR 管线）")
    parser.add_argument("--output", default=os.path.abspath("rt_validate"), help="测试产物目录")
    parser.add_argument("--duration", type=float, default=10.0, help="每个合成视频时长（秒）")
    parser.add_argument("--sample-frames", type=int, default=3000, help="最多处理帧数")
    parser.add_argument("--fast-scale", type=float, default=0.25)
    parser.add_argument("--kf-hz", type=float, default=1.5)
    args = parser.parse_args()
    if args.duration <= 0 or args.sample_frames <= 0:
        parser.error("duration 和 sample-frames 必须大于 0")

    out_dir = os.path.abspath(args.output)
    os.makedirs(out_dir, exist_ok=True)

    specs = [
        {"name": "720p50", "w": 1280, "h": 720, "fps": 50, "dur": args.duration},
        {"name": "1080p30", "w": 1920, "h": 1080, "fps": 30, "dur": args.duration},
    ]
    # 用同一套参数测两种规格
    fast_scale = args.fast_scale
    kf_hz = args.kf_hz

    results = []
    for sp in specs:
        vpath = os.path.join(out_dir, f"rt_{sp['name']}.mp4")
        print(f"\n[生成] {sp['name']}: {sp['w']}x{sp['h']}@{sp['fps']}fps ...")
        gen_test_video(vpath, sp["w"], sp["h"], sp["fps"], sp["dur"])
        print(f"[生成] 完成: {vpath}")

        print(f"[验证] {sp['name']} 实时预算分配(fast_scale={fast_scale}, kf_hz={kf_hz}) ...")
        r = measure_rt(vpath, fast_scale, kf_hz, sample_frames=args.sample_frames)
        results.append(r)
        print(f"  -> {r['spec']}: {r['proc_fps']}fps, "
              f"{'REALTIME ✓' if r['realtime'] else 'NOT REALTIME ✗'}, "
              f"运动事件{r['motion_events']}, 关键帧{r['keyframes']}")

    # 汇总
    print(f"\n{'='*60}")
    print("实时预算分配架构 - 验证汇总")
    print(f"{'='*60}")
    for r in results:
        status = "✓ 实时达标" if r["realtime"] else "✗ 未达标"
        print(f"  {r['spec']:<16} 处理{r['proc_fps']:>6.1f}fps  {status}")
    all_ok = all(r["realtime"] for r in results)

    with open(os.path.join(out_dir, "rt_results.json"), "w", encoding="utf-8") as f:
        json.dump({"fast_scale": fast_scale, "keyframe_interval_hz": kf_hz,
                   "all_realtime": all_ok, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存: {out_dir}/rt_results.json")
    print(f"吞吐量冒烟判定: {'全部达到输入帧率' if all_ok else '存在未达标项'}")
    print("注意：该结果不包含真实 ASR、模型加载、长稳运行、峰值内存或真实视频准确率。")


if __name__ == '__main__':
    main()
