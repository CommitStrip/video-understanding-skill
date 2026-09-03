#!/usr/bin/env python3
"""
pipeline.py - 直播编排器：四层实时理解栈的组装与运行
====================================================
把既有资产与 W8 新件接成一路可运行的实时进程：

  FrameSource（现有 File/Camera/RTSP，直播背压/重连/单调时钟）
    → SmartPipeline 快/慢系统（现有，逐帧事件）          T0  0ms
    → BasicLabeler 关键帧即席标签                        T0.5 毫秒级
    → AudioStream(ffmpeg) → StreamingASR → RollingCleaner  语音链（直播源首次补齐）
    → UnderstandingWorker 触发式 VLM                     T2  有界滞后
    → SessionState（滚动落盘 live_state.json/live_context.md）
    → LiveServer（可选 SSE：/state /events /healthz）

与离线管线（integrated_pipeline）的关系：互不影响、组件复用。
离线路径的批式对齐/OCR/Tier3 保持原样；直播路径用滚动对齐与触发式理解。

Ctrl-C：画面链停止取帧后，已产生的理解状态与部分结果照常落盘（沿用
integrated_pipeline 的中断语义）。
"""

import os
import threading
import time

import cv2

from ..asr_sherpa import StreamingASR, load_streaming_recognizer
from ..asr_clean import RollingCleaner
from ..io_utils import write_json
from ..smart_pipeline import SmartPipeline
from ..source import CameraSource, FileSource, FrameSource, RTSPSource
from .audio_source import AudioStream
from .events import EventBus
from .server import LiveServer
from .state import SessionState, StateWriter
from .tagger import create_labeler
from .understanding import UnderstandingWorker
from .vlm_client import MockVLM, OpenAICompatVLM


def build_source(kind="file", video_path=None, camera=0, url=None, realtime=False):
    """按 CLI 语义构造 FrameSource（与 integrated_pipeline.main 对齐）。"""
    if kind == "file":
        if not video_path:
            raise ValueError("--source file 需要 --video 指向视频文件")
        return FileSource(video_path, realtime=realtime)
    if kind == "cam":
        return CameraSource(camera)
    if kind == "rtsp":
        if not url:
            raise ValueError("--source rtsp 需要 --url 指定 RTSP 地址")
        return RTSPSource(url)
    raise ValueError(f"未知帧源类型: {kind}")


def build_vlm(backend="mock"):
    """VLM 后端工厂（off 返回 None = 关闭 T2，纯本地模式只跑 T0/T0.5）。"""
    if backend == "off":
        return None
    if backend == "mock":
        return MockVLM()
    if backend == "openai":
        return OpenAICompatVLM()  # 端点/密钥走 env：VLM_API_BASE/VLM_API_KEY/VLM_MODEL
    from .vlm_client import create_vlm
    return create_vlm(backend)


def _asr_live_job(stream, sasr, cleaner, publish, asr_segments, stop_evt):
    """后台声音链：PCM 块 → 增量解码 → 增量清洗 → asr_final 事件。"""
    while not stop_evt.is_set():
        try:
            chunk = stream.read_chunk(timeout=5.0)
        except OSError:
            break
        if chunk is None:
            break
        for seg in cleaner.feed(sasr.feed(chunk)):
            asr_segments.append(seg)
            publish({"type": "asr_final", **seg})
    for seg in cleaner.feed(sasr.flush()):
        asr_segments.append(seg)
        publish({"type": "asr_final", **seg})


def run_live(source, output_dir=None, config=None, live_cfg=None,
             vlm=None, labeler=None, serve=False, port=8600, audio="auto",
             save_keyframes=True, status_interval=10.0, quiet=False):
    """四层实时理解主流程（阻塞运行直到流结束或 Ctrl-C）。

    参数:
      source:  FrameSource 实例（build_source 构造）。
      config:  SmartPipeline config dict（fast_scale / keyframe_interval_hz …）。
      live_cfg: UnderstandingWorker config（min_call_interval / max_timeline …）。
      vlm:     VLM 后端实例；None = 关闭 T2（纯本地 T0+T0.5 模式）。
      labeler: 标签道实例；True=默认 basic；False=关闭 T0.5。
      audio:   "auto"（文件/RTSP 有音轨就走声音链）/ "off"。
      serve:   起本地 SSE 服务（/state /events /healthz）。
      status_interval: 控制台理解状态行打印间隔（秒）。

    返回: {"summary", "aligned", "asr_segments", "output_dir", "server_url"}；
          帧源打开失败返回 None。
    """
    if isinstance(labeler, bool):
        # True=用默认 basic 标签道；False=关闭 T0.5 标签
        labeler = create_labeler("basic") if labeler is True else False
    tagging_enabled = labeler is not False
    if labeler is None:
        labeler = create_labeler("basic")
    live = bool(getattr(source, "live", False))
    config = dict(config or {})
    live_cfg = dict(live_cfg or {})

    if output_dir is None:
        output_dir = "."
    os.makedirs(output_dir, exist_ok=True)
    keyframes_dir = os.path.join(output_dir, "keyframes")
    if save_keyframes:
        os.makedirs(keyframes_dir, exist_ok=True)

    if not source.open():
        print(f"[Live] 无法打开视频源: {source.stats.get('last_error') or '未知错误'}")
        return None
    fps = getattr(source, "fps", 0.0) or 0.0
    source_desc = str(source.stats.get("source", ""))

    bus = EventBus()
    state = SessionState(source_desc=source_desc,
                         max_timeline=live_cfg.get("max_timeline", 40))
    writer = StateWriter(state, output_dir=output_dir, interval=1.0)
    writer.start()

    # worker 无条件启动：vlm=None 时退化为纯状态采集器（T0/T0.5 仍进状态）
    worker = UnderstandingWorker(bus, state, vlm, config=live_cfg)
    worker.start()

    server = None
    if serve:
        server = LiveServer(bus, state, port=port)
        server.start()

    pipe = SmartPipeline(config)
    tag_source = getattr(labeler, "name", "basic") if tagging_enabled else "off"

    # === 声音链（直播源首次补齐；sherpa 缺失则跳过，绝不喂 mock 假字幕） ===
    asr_segments = []
    audio_thread = None
    stop_evt = threading.Event()
    video_path = getattr(source, "video_path", None)
    if audio != "off" and live_cfg.get("asr_trigger", True):
        recognizer = load_streaming_recognizer()
        if recognizer is None:
            print("[Live] 声音链: sherpa-onnx 不可用，跳过（画面链与理解层不受影响）")
        else:
            stream = None
            if live and video_path is None and getattr(source, "url", None):
                stream = AudioStream(source.url, extra_input_args=["-rtsp_transport", "tcp"])
            elif video_path:
                stream = AudioStream(video_path, realtime=(live or bool(
                    getattr(source, "realtime", False))))
            if stream is not None and stream.open():
                sasr = StreamingASR(recognizer, sr=16000, chunk_sec=2.0)
                cleaner = RollingCleaner()
                audio_thread = threading.Thread(
                    target=_asr_live_job,
                    args=(stream, sasr, cleaner, bus.publish, asr_segments, stop_evt),
                    daemon=True, name="vus-live-asr")
                audio_thread.start()
            else:
                print("[Live] 声音链: 音频流不可用，跳过")

    if server is not None:
        print(f"[Live] SSE 服务: {server.url}/state | /events | /healthz")
    print(f"[Live] 理解层: {'T2=' + (getattr(vlm, 'name', '?') if vlm else 'off')}"
          f" T0.5={tag_source}（Ctrl-C 结束并保存部分结果）")

    frame_idx = 0
    kf_count = 0
    t_start = time.time()
    t_status = t_start
    last_motion_ratio = 0.0

    try:
        while not stop_evt.is_set():
            ok, frame, src_ts = source.read()
            if not ok:
                break

            timestamp = src_ts if live else \
                (frame_idx / fps if fps > 0 else frame_idx * 0.02)

            frame_events = pipe.process_frame(frame, timestamp)
            for ev in frame_events:
                if "motion_ratio" in ev:
                    last_motion_ratio = ev["motion_ratio"]

                if ev["type"] == "keyframe" and save_keyframes:
                    kf_path = os.path.join(
                        keyframes_dir, f"kf_{kf_count:04d}_t{timestamp:.1f}s.jpg")
                    cv2.imwrite(kf_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    kf_count += 1
                    # T0.5 毫秒级标签：关键帧在手即席执行，结果进总线与状态
                    if tagging_enabled:
                        t_tag = time.perf_counter()
                        tags = labeler.label(frame, motion_ratio=last_motion_ratio)
                        tag_ms = (time.perf_counter() - t_tag) * 1000.0
                        bus.publish({"type": "tag", "t": timestamp, "labels": tags,
                                     "source": tag_source, "ms": round(tag_ms, 1)})
                    # 落盘路径并入事件（理解层取图、对齐层归窗都用这一份）
                    bus.publish(dict(ev, path=kf_path))
                else:
                    bus.publish(ev)

            frame_idx += 1
            now = time.time()
            if frame_idx % 500 == 0:
                elapsed = now - t_start
                proc_fps = frame_idx / elapsed if elapsed > 0 else 0
                if not quiet:
                    print(f"  [进度] {frame_idx}帧 {proc_fps:.1f}fps "
                          f"{'[LIVE]' if live else ''}")
            if not quiet and now - t_status >= status_interval:
                t_status = now
                snap = state.snapshot()
                lag = snap["telemetry"].get("lag") or {}
                print(f"  [理解] {snap['t2']['now'] or '（等待首次理解…）'} "
                      f"| T2滞后 {lag.get('t2_s')}s | 标签 {snap['telemetry']['t05_labels']} 次"
                      f" | 调用 {snap['telemetry']['t2_calls']} 次")
    except KeyboardInterrupt:
        print("\n[Live] 收到中断信号，停止取帧...")
    finally:
        stop_evt.set()
        source.close()

    pipeline_time = time.time() - t_start
    proc_fps = frame_idx / pipeline_time if pipeline_time > 0 else 0

    if audio_thread is not None:
        audio_thread.join(timeout=5.0)
    worker.stop()
    aligned = worker.flush_aligner()
    if server is not None:
        server.stop()
    writer.stop()  # 停表并强制落盘最终状态

    summary = pipe.get_summary()
    summary["source"] = source.stats
    write_json(output_dir, "pipeline_results.json", pipe.build_results())
    write_json(output_dir, "aligned_output.json", {
        "live": True,
        "aligned_segments": aligned,
        "asr_segments": asr_segments,
        "ocr_events": [],
        "pipeline_summary": summary,
        "stream_event_count": bus.published,
    })

    if not quiet:
        print(f"\n[Live] 完成: {frame_idx}帧, 耗时{pipeline_time:.1f}s, "
              f"{proc_fps:.1f}fps, ASR {len(asr_segments)}段, 对齐 {len(aligned)}段")
        print(f"[Live] 产物: {os.path.abspath(output_dir)}"
              f"（live_state.json / live_context.md / aligned_output.json）")

    return {
        "summary": summary,
        "aligned": aligned,
        "asr_segments": asr_segments,
        "output_dir": output_dir,
        "server_url": server.url if server else None,
        "proc_fps": round(proc_fps, 1),
    }
