#!/usr/bin/env python3
"""视频画面链 + 严格 ASR + 可追溯输出的编排入口。"""

import argparse
from collections import Counter, deque
import json
import os
import sys
import threading
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smart_pipeline import SmartPipeline
from asr_sherpa import (
    ASRError,
    ASRRuntimeError,
    extract_audio,
    load_streaming_recognizer,
    require_ffmpeg,
    transcribe_wav_streaming,
)


class StreamingConsumer:
    """有界历史的事件消费者；准确总数由 Counter 独立维护。"""

    def __init__(self, history_limit=2000):
        if history_limit < 0:
            raise ValueError("history_limit 不能为负数")
        self.events = deque(maxlen=history_limit)
        self.counts = Counter()
        self.motion_segments = []
        self.keyframes = []
        self._lock = threading.Lock()

    def consume(self, event):
        with self._lock:
            self.events.append(event)
            self.counts[event["type"]] += 1
            if event["type"] == "motion_end":
                self.motion_segments.append({
                    "start": event["segment_start"],
                    "end": event["t"],
                    "duration": event["duration"],
                })
            elif event["type"] == "keyframe":
                self.keyframes.append(event)

    def snapshot(self):
        with self._lock:
            return {
                "events": list(self.events),
                "counts": dict(self.counts),
                "motion_segments": list(self.motion_segments),
                "keyframes": list(self.keyframes),
                "event_count": sum(self.counts.values()),
                "retained_event_history": len(self.events),
            }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2)


def _asr_job(video_path, recognizer, out_segments, state, sr=16000):
    """在后台抽取并流式读取音频；异常通过 state 返回主线程。"""
    wav_path = None
    try:
        wav_path = extract_audio(video_path, sr=sr)
        out_segments.extend(transcribe_wav_streaming(recognizer, wav_path, sr=sr))
        state["status"] = "succeeded"
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def _attach_keyframe_file(pipe, event, relative_path):
    event["file"] = relative_path
    for record in reversed(pipe.keyframes):
        if record.get("frame_idx") == event.get("frame_idx"):
            record["file"] = relative_path
            return
    raise RuntimeError("关键帧事件无法关联到管线元数据")


def run_realtime_pipeline(
    video_path,
    output_dir=None,
    save_keyframes=True,
    config=None,
    on_event=None,
    asr_model_dir=None,
    visual_only=False,
    overwrite_output=False,
):
    """运行视觉流和可选的严格 ASR。

    默认要求真实 ASR 可用。只有显式 visual_only=True 时才跳过 ASR；该模式
    不生成 aligned_output.json，也不会伪造字幕。
    """
    video_path = os.path.abspath(video_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(video_path) or ".", "video-understanding-output")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if not overwrite_output and any(os.scandir(output_dir)):
        raise FileExistsError(
            f"输出目录不是空目录: {output_dir}；请使用新目录或显式启用 overwrite_output"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0:
        cap.release()
        raise ValueError("视频 FPS 无效；固定帧率时间轴无法建立")
    duration = total_frames / fps if total_frames > 0 else None

    pipe = SmartPipeline(config)
    consumer = StreamingConsumer(history_limit=pipe.event_history_limit)
    keyframes_dir = os.path.join(output_dir, "keyframes")
    if save_keyframes:
        os.makedirs(keyframes_dir, exist_ok=True)

    asr_segments = []
    asr_state = {"status": "disabled" if visual_only else "pending", "error": None}
    asr_thread = None
    model_info = None
    manifest_path = os.path.join(output_dir, "run_manifest.json")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "mode": "visual-only" if visual_only else "visual+asr",
        "input": {
            "path": video_path,
            "width": width,
            "height": height,
            "fps": fps,
            "total_frames": total_frames,
            "duration_s": duration,
        },
        "asr": {"status": asr_state["status"], "model": None},
    }
    _write_json(manifest_path, manifest)

    if not visual_only:
        try:
            require_ffmpeg()
            recognizer, model_info = load_streaming_recognizer(asr_model_dir, return_info=True)
        except ASRError as exc:
            cap.release()
            asr_state["status"] = "failed"
            asr_state["error"] = f"{type(exc).__name__}: {exc}"
            manifest["status"] = "failed"
            manifest["asr"] = {**asr_state, "model": None}
            _write_json(manifest_path, manifest)
            raise
        asr_state["status"] = "running"
        asr_thread = threading.Thread(
            target=_asr_job,
            args=(video_path, recognizer, asr_segments, asr_state),
            daemon=False,
            name="video-understanding-asr",
        )
        asr_thread.start()
        manifest["asr"] = {"status": asr_state["status"], "model": model_info}
        _write_json(manifest_path, manifest)

    frame_index = 0
    keyframe_count = 0
    started = time.perf_counter()
    events_path = os.path.join(output_dir, "events.jsonl")

    visual_error = None
    try:
        with open(events_path, "w", encoding="utf-8") as event_log:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp = frame_index / fps
                frame_events = pipe.process_frame(frame, timestamp, fps=fps)

                if save_keyframes:
                    for event in frame_events:
                        if event["type"] != "keyframe":
                            continue
                        filename = f"kf_{keyframe_count:04d}_t{timestamp:.3f}s.jpg"
                        absolute_path = os.path.join(keyframes_dir, filename)
                        if not cv2.imwrite(absolute_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                            raise OSError(f"关键帧写入失败: {absolute_path}")
                        relative_path = os.path.relpath(absolute_path, output_dir).replace("\\", "/")
                        _attach_keyframe_file(pipe, event, relative_path)
                        keyframe_count += 1

                for event in frame_events:
                    consumer.consume(event)
                    event_log.write(json.dumps(event, ensure_ascii=False) + "\n")
                if on_event is not None and frame_events:
                    on_event(frame_index, timestamp, frame_events)

                frame_index += 1
                if total_frames > 0 and frame_index % 500 == 0:
                    elapsed = time.perf_counter() - started
                    processing_fps = frame_index / elapsed if elapsed > 0 else 0
                    print(
                        f"[Pipeline] {frame_index}/{total_frames} "
                        f"({frame_index / total_frames * 100:.1f}%) {processing_fps:.1f} fps"
                    )

            final_timestamp = frame_index / fps
            final_events = pipe.finalize(final_timestamp)
            for event in final_events:
                consumer.consume(event)
                event_log.write(json.dumps(event, ensure_ascii=False) + "\n")
            if on_event is not None and final_events:
                on_event(frame_index, final_timestamp, final_events)
    except Exception as exc:
        visual_error = exc
    finally:
        cap.release()

    if visual_error is not None:
        if asr_thread is not None:
            asr_thread.join()
        manifest["status"] = "failed"
        manifest["error"] = f"{type(visual_error).__name__}: {visual_error}"
        manifest["asr"] = {**asr_state, "model": model_info}
        _write_json(manifest_path, manifest)
        raise visual_error

    elapsed = time.perf_counter() - started
    processing_fps = frame_index / elapsed if elapsed > 0 else 0.0
    pipe.save_results(os.path.join(output_dir, "pipeline_results.json"))

    if asr_thread is not None:
        asr_thread.join()
        if asr_state["status"] != "succeeded":
            manifest["status"] = "failed"
            manifest["asr"] = {**asr_state, "model": model_info}
            manifest["visual_processing"] = {
                "frames": frame_index,
                "elapsed_s": round(elapsed, 3),
                "processing_fps": round(processing_fps, 2),
            }
            _write_json(manifest_path, manifest)
            raise ASRRuntimeError(asr_state["error"] or "ASR 未成功完成")

    video_end = frame_index / fps
    aligned = []
    if not visual_only:
        aligned = pipe.align_asr_streaming(asr_segments, video_end=video_end)
        _write_json(
            os.path.join(output_dir, "aligned_output.json"),
            {
                "schema_version": 1,
                "asr_status": "succeeded",
                "asr_model": model_info,
                "aligned_segments": aligned,
                "asr_segments": asr_segments,
                "pipeline_summary": pipe.get_summary(),
                "event_log": "events.jsonl",
            },
        )

    manifest["status"] = "succeeded"
    manifest["asr"] = {"status": asr_state["status"], "model": model_info}
    manifest["visual_processing"] = {
        "frames": frame_index,
        "elapsed_s": round(elapsed, 3),
        "processing_fps": round(processing_fps, 2),
        "summary": pipe.get_summary(),
        "event_log": "events.jsonl",
    }
    _write_json(manifest_path, manifest)

    return pipe, aligned, asr_segments


def run_pipeline(video_path, output_dir=None, save_keyframes=True, config=None, **kwargs):
    return run_realtime_pipeline(video_path, output_dir, save_keyframes, config, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="视频视觉事件与严格 ASR 管线")
    parser.add_argument("--video", required=True, help="视频文件路径")
    parser.add_argument("--output", default=None, help="输出目录")
    parser.add_argument("--visual-only", action="store_true", help="显式跳过 ASR；不会生成模拟字幕")
    parser.add_argument("--asr-model-dir", default=None, help="sherpa-onnx 模型目录")
    parser.add_argument("--no-keyframes", action="store_true", help="不保存关键帧图片")
    parser.add_argument("--fast-scale", type=float, default=0.25, help="快系统降采样比例 (0,1]")
    parser.add_argument("--kf-hz", type=float, default=1.5, help="慢系统最大检查频率 Hz")
    parser.add_argument("--event-history-limit", type=int, default=2000, help="内存中保留的最近事件数")
    parser.add_argument("--overwrite-output", action="store_true", help="允许复用非空输出目录")
    args = parser.parse_args()

    config = {
        "fast_scale": args.fast_scale,
        "keyframe_interval_hz": args.kf_hz,
        "event_history_limit": args.event_history_limit,
    }
    try:
        run_realtime_pipeline(
            video_path=args.video,
            output_dir=args.output,
            save_keyframes=not args.no_keyframes,
            config=config,
            asr_model_dir=args.asr_model_dir,
            visual_only=args.visual_only,
            overwrite_output=args.overwrite_output,
        )
    except ASRError as exc:
        parser.exit(2, f"ASR 失败: {exc}\n")


if __name__ == "__main__":
    main()
