#!/usr/bin/env python3
"""
integrated_pipeline.py - 实时预算分配 · 三通道流式编排入口
=============================================================
通道1+2: 画面链（快系统运动流 + 慢系统关键帧）—— 实时前台，逐帧流式输出
通道3:   声音链（流式ASR）—— 后台并行，逐块产出增量文本
通道4:   文字链（OCR，W3 可选，默认关，--ocr 开启）—— 只对 Tier2 关键帧
         稀疏执行（每关键帧一次），绝不进逐帧实时路径
对齐层:   时间轴对齐融合 → 宿主可实时消费的结构化语义流

实时预算：
  前台画面链  每帧处理完立即把事件交给宿主（on_event 回调），宿主无需等整段。
  后台声音链  与画面链并行跑，不阻塞画面链的实时性。

用法:
  python -m vus.integrated_pipeline --video 视频.mp4 --output out/
  摄像头:   python -m vus.integrated_pipeline --source cam --camera 0
  RTSP流:   python -m vus.integrated_pipeline --source rtsp --url rtsp://host/stream
  文件实时回放: python -m vus.integrated_pipeline --video x.mp4 --realtime
  （兼容入口: python scripts/integrated_pipeline.py，参数一致）
"""

import cv2
import json
import time
import os
import sys
import argparse
import signal
import threading
from collections import deque

from .io_utils import write_json
from .smart_pipeline import SmartPipeline
from .source import FrameSource, FileSource, CameraSource, RTSPSource
from .asr_sherpa import (
    extract_audio, load_wav, load_streaming_recognizer,
    transcribe_streaming
)


class StreamingConsumer:
    """
    流式增量的宿主侧消费者。
    宿主把 process_frame/on_event 产出的每个事件立即喂给它，它实时累积并可由宿主随时读取。

    max_events: 事件内存上限（W1 修复：原 list 无限增长，长直播内存泄漏）。
                超出上限丢弃最旧事件，丢弃数见 snapshot()["events_dropped"]。
                0 = 无界（旧行为）；默认 100000，与 SmartPipeline 的 max_events 配置一致。
    """

    DEFAULT_MAX_EVENTS = 100000

    def __init__(self, max_events=DEFAULT_MAX_EVENTS):
        self.max_events = int(max_events)
        self.events = deque(maxlen=self.max_events) if self.max_events > 0 else []
        self._events_appended = 0
        self.motion_segments = []  # 已闭合的运动段
        self.keyframes = []        # 关键帧
        self._lock = threading.Lock()

    @property
    def events_dropped(self):
        """因超过 max_events 而被丢弃的最旧事件数（max_events=0 无界时恒为 0）。"""
        return self._events_appended - len(self.events)

    def consume(self, ev):
        """喂入一个事件，立即记录（线程安全）。"""
        with self._lock:
            self.events.append(ev)
            self._events_appended += 1
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
                "event_count": len(self.events),
                "events_dropped": self.events_dropped,
                "max_events": self.max_events
            }


def _asr_job(video_path, wav_path, out_segments, sr=16000):
    """后台声音链：流式ASR，阻塞直到完成，写入 out_segments。"""
    samples, sr2 = load_wav(wav_path, sr)
    if len(samples) == 0:
        return
    recognizer = load_streaming_recognizer()
    segs = transcribe_streaming(recognizer, samples, sr2)
    out_segments.extend(segs)


def run_realtime_pipeline(video_path=None, output_dir=None, save_keyframes=True,
                          config=None, on_event=None, source: FrameSource = None,
                          ocr=False):
    """
    实时流式主流程：画面链前台逐帧 + 声音链后台并行。

    参数:
      on_event: 可选回调 on_event(frame_idx, timestamp, events)。
                每帧的画面事件会立即回调，宿主在这里消费，无需等整段。
      source:   可选 FrameSource（W2 实时源）。传入时忽略 video_path，
                支持摄像头 / RTSP 直播流与实时回放；不传时内部用
                FileSource(video_path) 包一层，行为与旧版完全一致。
      video_path: 视频文件路径；source 为 None 时必填。
      ocr:      W3 可选 OCR 第三通道（默认关）。开启时对每个关键帧稀疏跑
                一次 OCR，事件进 consumer 并落到 aligned_output.json 的
                "ocr_events" 键；rapidocr 未安装时显式报错（不静默降级），
                依赖懒加载——不开 OCR 时零额外依赖。

    时间戳:
      直播源（Camera/RTSP）事件用 source.read() 返回的单调时钟 timestamp；
      文件回放仍用 frame_idx / fps。

    Ctrl-C:
      处理循环内收到 KeyboardInterrupt 不直接抛出——停止取帧后继续走
      保存段，把已产生的部分结果落盘后正常返回。

    返回: (pipe, aligned, asr_segments)；帧源打开失败时 (None, [], [])
    """
    if source is None:
        if not video_path:
            print("[Pipeline] 未提供 video_path 也未提供 source，无法运行")
            return None, [], []
        source = FileSource(video_path)
    else:
        # 文件源才对应磁盘视频文件（供声音链抽取音频）；直播源无音频链
        video_path = getattr(source, 'video_path', video_path)
    live = bool(getattr(source, 'live', False))

    if output_dir is None:
        output_dir = os.path.dirname(video_path) if video_path else '.'

    print(f"[Pipeline] 帧源: {source.stats.get('source', video_path)}"
          f"{'（实时直播）' if live else ''}")
    print(f"[Pipeline] 输出: {output_dir}")

    pipe = SmartPipeline(config)
    # 宿主侧事件缓冲与管线共用同一上限配置（0 = 无界，默认 100000 有界）
    consumer = StreamingConsumer(
        max_events=(config or {}).get('max_events', StreamingConsumer.DEFAULT_MAX_EVENTS))

    # === 打开帧源 ===
    if not source.open():
        err = source.stats.get('last_error') or '未知错误'
        print(f"[Pipeline] 无法打开视频源: {err}")
        return None, [], []

    fps = getattr(source, 'fps', 0.0) or 0.0
    total_frames = getattr(source, 'total_frames', 0) or 0
    if live:
        print("[Pipeline] 实时源已连接，等待首帧（分辨率以首帧为准）...")
    else:
        duration = total_frames / fps if fps > 0 else 0
        print(f"[Pipeline] 视频: {getattr(source, 'width', 0)}x"
              f"{getattr(source, 'height', 0)} @ {fps:.1f}fps, "
              f"{total_frames}帧, {duration:.1f}s")

    os.makedirs(output_dir, exist_ok=True)  # save_keyframes=False 时也保证产物目录存在
    keyframes_dir = os.path.join(output_dir, 'keyframes')
    if save_keyframes:
        os.makedirs(keyframes_dir, exist_ok=True)

    # === 通道4: 文字链（OCR，可选默认关；懒加载，不开零依赖）===
    ocr_channel = None
    if ocr:
        from .ocr_channel import OcrChannel  # 函数内懒加载：不开 --ocr 零依赖
        ocr_channel = OcrChannel()  # rapidocr 未安装时显式抛 RuntimeError
        print("[Pipeline] OCR 第三通道已启用（仅关键帧稀疏执行）")

    # === 通道3: 声音链（后台并行；仅文件源有音频，直播源跳过）===
    wav_path = extract_audio(video_path, sr=16000) if video_path else None
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
    frame_w = getattr(source, 'width', 0) or 0
    frame_h = getattr(source, 'height', 0) or 0
    interrupted = False

    # === 通道1+2: 画面链（实时前台）。Ctrl-C 中断后仍走保存段 ===
    try:
        while True:
            ok, frame, src_ts = source.read()
            if not ok:
                break

            if frame_w == 0 and frame is not None:
                frame_h, frame_w = frame.shape[:2]
                if live:
                    print(f"[Pipeline] 实时源首帧: {frame_w}x{frame_h}")

            # 直播源用单调时钟时间戳；文件回放仍用 frame_idx/fps
            timestamp = src_ts if live else \
                (frame_idx / fps if fps > 0 else frame_idx * 0.02)

            # 画面链：逐帧处理，返回本帧流式事件
            frame_events = pipe.process_frame(frame, timestamp)

            # 立即交给宿主消费（流式增量）
            for ev in frame_events:
                consumer.consume(ev)
            if on_event is not None and frame_events:
                on_event(frame_idx, timestamp, frame_events)

            # 保存关键帧（若本帧产生关键帧）；OCR 通道对关键帧稀疏执行
            if frame_events:
                for ev in frame_events:
                    if ev["type"] == "keyframe":
                        if save_keyframes:
                            kf_path = os.path.join(keyframes_dir,
                                                   f"kf_{kf_count:04d}_t{timestamp:.1f}s.jpg")
                            cv2.imwrite(kf_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            kf_count += 1
                        # 文字链：每关键帧一次 OCR，事件立即进 consumer（流式）
                        if ocr_channel is not None:
                            for ocr_ev in ocr_channel.process(frame, timestamp):
                                consumer.consume(ocr_ev)

            frame_idx += 1
            if frame_idx % 500 == 0:
                elapsed = time.time() - t_pipeline_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                if total_frames > 0:
                    gating = "REALTIME" if (fps > 0 and proc_fps >= fps) else "SLOW"
                    print(f"  [进度] {frame_idx}/{total_frames} "
                          f"({frame_idx/total_frames*100:.1f}%) "
                          f"{proc_fps:.1f}fps [{gating}]")
                else:
                    print(f"  [进度] {frame_idx}帧 {proc_fps:.1f}fps [LIVE]")
    except KeyboardInterrupt:
        interrupted = True
        print("\n[Pipeline] 收到中断信号，停止取帧...")
    finally:
        source.close()

    pipeline_time = time.time() - t_pipeline_start
    proc_fps = frame_idx / pipeline_time if pipeline_time > 0 else 0

    # 等待后台声音链完成（中断时也等，保证部分结果完整）
    if asr_thread is not None:
        try:
            asr_thread.join()
        except KeyboardInterrupt:
            pass

    mode = "画面链（中断）" if interrupted else "画面链"
    print(f"\n[Pipeline] {mode}完成: {frame_idx}帧, 耗时{pipeline_time:.1f}s, "
          f"{proc_fps:.1f}fps")
    print(f"[Pipeline] 声音链完成: {len(asr_segments)}段")

    # === 时间轴对齐融合 ===
    print("[Pipeline] 时间轴对齐融合...")
    aligned = pipe.align_asr_streaming(asr_segments) if asr_segments else []

    # W3: OCR 第三通道事件按时间序并入 aligned_output（新增键，不动现有键）
    ocr_events = sorted(
        (e for e in consumer.snapshot()["events"] if e.get("type") == "ocr"),
        key=lambda e: e["t"])

    # === 保存结果（中断时保存的即部分结果）===
    summary = pipe.get_summary()
    summary["source"] = source.stats  # W2: 帧源统计并入 pipeline_summary
    results = pipe.build_results()
    write_json(output_dir, 'pipeline_results.json', results)
    write_json(output_dir, 'aligned_output.json', {
        "live": True,
        "aligned_segments": aligned,
        "asr_segments": asr_segments,
        "ocr_events": ocr_events,
        "pipeline_summary": summary,
        "stream_event_count": consumer.snapshot()["event_count"]
    })

    print(f"\n{'='*60}")
    print(f"实时流式整合管线 - 完成" + ("（部分结果）" if interrupted else ""))
    print(f"{'='*60}")
    print(f"  帧源:           {summary['source'].get('source')}")
    sst = summary['source']
    if sst.get('frames_dropped') or sst.get('reconnects') or live:
        print(f"  帧源统计:       读{sst.get('frames_read', 0)}帧 / "
              f"丢{sst.get('frames_dropped', 0)}帧 / "
              f"重连{sst.get('reconnects', 0)}次")
    print(f"  视频规格:       {frame_w}x{frame_h} @ {fps:.1f}fps")
    print(f"  画面链处理速率: {proc_fps:.1f}fps  "
          f"{'OK 实时达标' if (fps > 0 and proc_fps >= fps) else '未达标/直播'}")
    print(f"  运动事件:       {summary['motion_events']}")
    print(f"  运动段:         {summary['motion_segments']}")
    print(f"  关键帧:         {summary['keyframes']}")
    print(f"  ASR段:          {len(asr_segments)}")
    if ocr_channel is not None:
        print(f"  OCR事件:        {len(ocr_events)}")
    print(f"  对齐段:         {len(aligned)}")
    print(f"  画面链耗时:     {pipeline_time:.1f}s")
    print(f"{'='*60}")
    if interrupted:
        print("[Pipeline] 收到中断，已保存部分结果")

    return pipe, aligned, asr_segments


def run_pipeline(video_path, output_dir=None, save_keyframes=True, config=None):
    """兼容旧调用：等价于实时流式流程。"""
    return run_realtime_pipeline(video_path, output_dir, save_keyframes, config)


def _install_windows_break_handler():
    """Windows: 把 CTRL_BREAK_EVENT 映射为 KeyboardInterrupt。

    某些控制台/进程组环境下 SIGINT（Ctrl-C）不可达（如 CREATE_NEW_PROCESS_GROUP
    的子进程组默认屏蔽 CTRL_C），而 CTRL_BREAK 无法被屏蔽；CPython 默认对
    SIGBREAK 直接终止进程，这里改为走 KeyboardInterrupt → 优雅落盘路径。
    非 Windows 平台为 no-op。
    """
    if sys.platform == 'win32' and hasattr(signal, 'SIGBREAK'):
        try:
            signal.signal(signal.SIGBREAK, signal.default_int_handler)
        except (ValueError, OSError):
            pass


def main():
    _install_windows_break_handler()
    parser = argparse.ArgumentParser(description='实时流式三通道视频理解管线')
    parser.add_argument('--video', default=None, help='视频文件路径（--source file 时必填）')
    parser.add_argument('--output', default=None, help='输出目录')
    parser.add_argument('--no-keyframes', action='store_true', help='不保存关键帧图片')
    parser.add_argument('--fast-scale', type=float, default=0.25, help='快系统降采样比例')
    parser.add_argument('--kf-hz', type=float, default=1.5, help='慢系统关键帧频率(Hz)')
    parser.add_argument('--source', choices=['file', 'cam', 'rtsp'], default='file',
                        help='帧源类型: file=视频文件(默认,向后兼容) / cam=摄像头 / rtsp=网络流')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引（--source cam 时使用）')
    parser.add_argument('--url', default=None, help='RTSP 地址（--source rtsp 时必填）')
    parser.add_argument('--realtime', action='store_true',
                        help='file 源按视频 fps 节拍实时喂帧（默认尽快读，回放分析）')
    parser.add_argument('--ocr', action='store_true',
                        help='启用 OCR 第三通道（W3, 默认关）：对关键帧稀疏识别画面文字，'
                             '事件并入 aligned_output.json 的 ocr_events。'
                             '需要 pip install -e ".[ocr]"')
    args = parser.parse_args()

    # ---- 按帧源类型构造 FrameSource ----
    video_path = None
    if args.source == 'file':
        if not args.video:
            parser.error('--source file 需要 --video 指向视频文件')
        video_path = args.video
        source = FileSource(video_path, realtime=args.realtime)
    elif args.source == 'cam':
        source = CameraSource(args.camera)
    else:  # rtsp
        if not args.url:
            parser.error('--source rtsp 需要 --url 指定 RTSP 地址')
        source = RTSPSource(args.url)

    config = {"fast_scale": args.fast_scale, "keyframe_interval_hz": args.kf_hz}
    try:
        ret = run_realtime_pipeline(
            video_path=video_path,
            output_dir=args.output,
            save_keyframes=not args.no_keyframes,
            config=config,
            source=source,
            ocr=args.ocr
        )
    except KeyboardInterrupt:
        # 兜底：中断发生在帧循环之外（如对齐/落盘阶段）。
        # 帧循环内的中断由 run_realtime_pipeline 捕获并已落盘。
        print("\n[Pipeline] 收到中断，已保存部分结果")
        return 130
    return 0 if ret[0] is not None else 1


if __name__ == '__main__':
    sys.exit(main())
