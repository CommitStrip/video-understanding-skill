#!/usr/bin/env python3
"""
integrated_pipeline.py - 实时预算分配 · 三通道流式编排入口
=============================================================
通道1+2: 画面链（快系统运动流 + 慢系统关键帧）—— 实时前台，逐帧流式输出
通道3:   声音链（流式ASR）—— 后台并行，逐块产出增量文本
对齐层:   时间轴对齐融合 → 宿主可实时消费的结构化语义流

实时预算：
  前台画面链  每帧处理完立即把事件交给宿主（on_event 回调），宿主无需等整段。
  后台声音链  与画面链并行跑，不阻塞画面链的实时性。
"""

import cv2
import json
import time
import os
import sys
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart_pipeline import SmartPipeline
from asr_sherpa import (
    extract_audio, load_wav, load_streaming_recognizer,
    transcribe_streaming
)


class StreamingConsumer:
    """
    流式增量的宿主侧消费者。
    宿主把 process_frame/on_event 产出的每个事件立即喂给它，它实时累积并可由宿主随时读取。
    """

    def __init__(self):
        self.events = []          # 全部事件（时间序）
        self.motion_segments = []  # 已闭合的运动段
        self.keyframes = []        # 关键帧
        self._lock = threading.Lock()

    def consume(self, ev):
        """喂入一个事件，立即记录（线程安全）。"""
        with self._lock:
            self.events.append(ev)
            if ev["type"] == "motion_end":
                self.motion_segments.append({
                    "start": ev.get("segment_start"),
                    "end": ev["t"],
                    "duration": ev.get("duration")
                })
            elif ev["type"] == "keyframe":
                self.keyframes.append(ev)

    def snapshot(self):
        """返回当前已消费事件的快照。"""
        with self._lock:
            return {
                "events": list(self.events),
                "motion_segments": list(self.motion_segments),
                "keyframes": list(self.keyframes),
                "live": True,
                "event_count": len(self.events)
            }


def _asr_job(video_path, wav_path, out_segments, sr=16000):
    """后台声音链：流式ASR，阻塞直到完成，写入 out_segments。"""
    samples, sr2 = load_wav(wav_path, sr)
    if len(samples) == 0:
        return
    recognizer = load_streaming_recognizer()
    segs = transcribe_streaming(recognizer, samples, sr2)
    out_segments.extend(segs)


def run_realtime_pipeline(video_path, output_dir=None, save_keyframes=True,
                          config=None, on_event=None):
    """
    实时流式主流程：画面链前台逐帧 + 声音链后台并行。

    参数:
      on_event: 可选回调 on_event(frame_idx, timestamp, events)。
                每帧的画面事件会立即回调，宿主在这里消费，无需等整段。

    返回: (pipe, aligned, asr_segments)
    """
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or '.'

    print(f"[Pipeline] 视频: {video_path}")
    print(f"[Pipeline] 输出: {output_dir}")

    pipe = SmartPipeline(config)
    consumer = StreamingConsumer()

    # === 通道1+2: 画面链（实时前台）===
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Pipeline] 无法打开视频: {video_path}")
        return None, [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"[Pipeline] 视频: {w}x{h} @ {fps:.1f}fps, {total_frames}帧, {duration:.1f}s")

    keyframes_dir = os.path.join(output_dir, 'keyframes')
    if save_keyframes:
        os.makedirs(keyframes_dir, exist_ok=True)

    # === 通道3: 声音链（后台并行）===
    wav_path = extract_audio(video_path, sr=16000)
    asr_segments = []
    asr_thread = None
    if wav_path and os.path.exists(wav_path):
        asr_thread = threading.Thread(target=_asr_job,
                                      args=(video_path, wav_path, asr_segments),
                                      daemon=True)
        asr_thread.start()

    frame_idx = 0
    t_pipeline_start = time.time()
    kf_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps if fps > 0 else frame_idx * 0.02

        # 画面链：逐帧处理，返回本帧流式事件
        frame_events = pipe.process_frame(frame, timestamp)

        # 立即交给宿主消费（流式增量）
        for ev in frame_events:
            consumer.consume(ev)
        if on_event is not None and frame_events:
            on_event(frame_idx, timestamp, frame_events)

        # 保存关键帧（若本帧产生关键帧）
        if save_keyframes and frame_events:
            for ev in frame_events:
                if ev["type"] == "keyframe":
                    kf_path = os.path.join(keyframes_dir,
                                           f"kf_{kf_count:04d}_t{timestamp:.1f}s.jpg")
                    cv2.imwrite(kf_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    kf_count += 1

        frame_idx += 1
        if frame_idx % 500 == 0:
            elapsed = time.time() - t_pipeline_start
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            gating = "REALTIME" if proc_fps >= fps else "SLOW"
            print(f"  [进度] {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%) "
                  f"{proc_fps:.1f}fps [{gating}]")

    cap.release()
    pipeline_time = time.time() - t_pipeline_start
    proc_fps = frame_idx / pipeline_time if pipeline_time > 0 else 0

    # 等待后台声音链完成
    if asr_thread is not None:
        asr_thread.join()

    print(f"\n[Pipeline] 画面链完成: {frame_idx}帧, 耗时{pipeline_time:.1f}s, {proc_fps:.1f}fps")
    print(f"[Pipeline] 声音链完成: {len(asr_segments)}段")

    # === 时间轴对齐融合 ===
    print("[Pipeline] 时间轴对齐融合...")
    aligned = pipe.align_asr_streaming(asr_segments) if asr_segments else []

    # === 保存结果 ===
    results = pipe.save_results(os.path.join(output_dir, 'pipeline_results.json'))
    with open(os.path.join(output_dir, 'aligned_output.json'), 'w', encoding='utf-8') as f:
        json.dump({
            "live": True,
            "aligned_segments": aligned,
            "asr_segments": asr_segments,
            "pipeline_summary": pipe.get_summary(),
            "stream_event_count": consumer.snapshot()["event_count"]
        }, f, ensure_ascii=False, indent=2)

    summary = pipe.get_summary()
    print(f"\n{'='*60}")
    print(f"实时流式整合管线 - 完成")
    print(f"{'='*60}")
    print(f"  视频规格:       {w}x{h} @ {fps:.1f}fps")
    print(f"  画面链处理速率: {proc_fps:.1f}fps  {'✓ 实时达标' if proc_fps >= fps else '✗ 未达标'}")
    print(f"  运动事件:       {summary['motion_events']}")
    print(f"  运动段:         {summary['motion_segments']}")
    print(f"  关键帧:         {summary['keyframes']}")
    print(f"  ASR段:          {len(asr_segments)}")
    print(f"  对齐段:         {len(aligned)}")
    print(f"  画面链耗时:     {pipeline_time:.1f}s")
    print(f"{'='*60}")

    return pipe, aligned, asr_segments


def run_pipeline(video_path, output_dir=None, save_keyframes=True, config=None):
    """兼容旧调用：等价于实时流式流程。"""
    return run_realtime_pipeline(video_path, output_dir, save_keyframes, config)


def main():
    parser = argparse.ArgumentParser(description='实时流式三通道视频理解管线')
    parser.add_argument('--video', required=True, help='视频文件路径')
    parser.add_argument('--output', default=None, help='输出目录')
    parser.add_argument('--no-keyframes', action='store_true', help='不保存关键帧图片')
    parser.add_argument('--fast-scale', type=float, default=0.25, help='快系统降采样比例')
    parser.add_argument('--kf-hz', type=float, default=1.5, help='慢系统关键帧频率(Hz)')
    args = parser.parse_args()

    config = {"fast_scale": args.fast_scale, "keyframe_interval_hz": args.kf_hz}
    run_realtime_pipeline(
        video_path=args.video,
        output_dir=args.output,
        save_keyframes=not args.no_keyframes,
        config=config
    )


if __name__ == '__main__':
    main()