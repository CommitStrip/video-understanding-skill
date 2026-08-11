#!/usr/bin/env python3
"""严格模式的 sherpa-onnx 流式 ASR。

生产路径不提供模拟字幕。缺少 ffmpeg、sherpa-onnx、模型或识别失败时，
调用方会收到明确异常，避免伪文本污染后续视频理解结果。
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np


class ASRError(RuntimeError):
    """ASR 基础异常。"""


class ASRConfigurationError(ASRError):
    """缺少运行依赖或模型配置。"""


class AudioExtractionError(ASRError):
    """ffmpeg 音频抽取失败。"""


class ASRRuntimeError(ASRError):
    """识别执行失败。"""


def require_ffmpeg():
    """返回 ffmpeg 路径；不可用时立即失败。"""
    executable = shutil.which("ffmpeg")
    if not executable:
        raise ASRConfigurationError("未找到 ffmpeg；严格 ASR 模式不能继续")
    return executable


def extract_audio(video_path, output_wav=None, sr=16000):
    """抽取 16 kHz、单声道、16-bit PCM WAV，不覆盖已有用户文件。"""
    if not os.path.isfile(video_path):
        raise AudioExtractionError(f"视频文件不存在: {video_path}")
    if sr <= 0:
        raise ValueError("sr 必须大于 0")

    ffmpeg = require_ffmpeg()
    if output_wav is None:
        fd, output_wav = tempfile.mkstemp(prefix="video-understanding-", suffix=".wav")
        os.close(fd)
        os.unlink(output_wav)
    else:
        output_wav = os.path.abspath(output_wav)
        if os.path.exists(output_wav):
            raise AudioExtractionError(f"拒绝覆盖已有音频文件: {output_wav}")
        parent = os.path.dirname(output_wav)
        if parent:
            os.makedirs(parent, exist_ok=True)

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-acodec",
        "pcm_s16le",
        output_wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not os.path.isfile(output_wav) or os.path.getsize(output_wav) <= 44:
        if os.path.exists(output_wav):
            os.unlink(output_wav)
        detail = (result.stderr or result.stdout or "未知 ffmpeg 错误").strip()
        raise AudioExtractionError(f"音频抽取失败: {detail}")
    return output_wav


def iter_wav_chunks(wav_path, expected_sr=16000, chunk_sec=2.0):
    """按块读取规范 WAV，避免把长音频整体载入内存。"""
    if chunk_sec <= 0:
        raise ValueError("chunk_sec 必须大于 0")
    try:
        wav = wave.open(wav_path, "rb")
    except (OSError, wave.Error) as exc:
        raise ASRRuntimeError(f"无法打开 WAV: {exc}") from exc

    with wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        if channels != 1 or sample_width != 2 or sample_rate != expected_sr:
            raise ASRRuntimeError(
                "WAV 必须是单声道、16-bit PCM、"
                f"{expected_sr} Hz；实际为 channels={channels}, "
                f"sample_width={sample_width}, sample_rate={sample_rate}"
            )

        block_frames = max(1, int(sample_rate * chunk_sec))
        start_frame = 0
        while True:
            raw = wav.readframes(block_frames)
            if not raw:
                break
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            end_frame = start_frame + len(samples)
            yield start_frame / sample_rate, end_frame / sample_rate, samples
            start_frame = end_frame


def load_wav(wav_path, sr=16000):
    """兼容旧调用；主流程使用 iter_wav_chunks 以保持有界内存。"""
    chunks = [block for _, _, block in iter_wav_chunks(wav_path, sr)]
    if not chunks:
        return np.array([], dtype=np.float32), sr
    return np.concatenate(chunks), sr


def _resolve_model_files(model_dir):
    model_dir = model_dir or os.environ.get("SHERPA_ONNX_MODEL_DIR")
    if not model_dir:
        raise ASRConfigurationError(
            "必须通过 --asr-model-dir 或 SHERPA_ONNX_MODEL_DIR 指定 sherpa-onnx 模型目录"
        )
    model_dir = os.path.abspath(os.path.expanduser(model_dir))
    if not os.path.isdir(model_dir):
        raise ASRConfigurationError(f"ASR 模型目录不存在: {model_dir}")

    matches = {"encoder": [], "decoder": [], "joiner": [], "tokens": []}
    for root, _, files in os.walk(model_dir):
        for name in files:
            lower = name.lower()
            path = os.path.join(root, name)
            if lower.endswith(".onnx"):
                for component in ("encoder", "decoder", "joiner"):
                    if component in lower:
                        matches[component].append(path)
            elif lower == "tokens.txt":
                matches["tokens"].append(path)

    resolved = {}
    for component, candidates in matches.items():
        if len(candidates) != 1:
            raise ASRConfigurationError(
                f"模型目录中 {component} 文件数量必须为 1，实际为 {len(candidates)}"
            )
        resolved[component] = candidates[0]
    resolved["model_dir"] = model_dir
    return resolved


def load_streaming_recognizer(model_dir=None, return_info=False):
    """加载真实 sherpa-onnx 流式识别器；任何失败都会抛出异常。"""
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise ASRConfigurationError("未安装 sherpa-onnx；严格 ASR 模式不能继续") from exc

    files = _resolve_model_files(model_dir)
    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=files["encoder"],
            decoder=files["decoder"],
            joiner=files["joiner"],
            tokens=files["tokens"],
            num_threads=2,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
            decoding_method="greedy_search",
            provider="cpu",
        )
    except Exception as exc:
        raise ASRConfigurationError(f"加载 sherpa-onnx 模型失败: {exc}") from exc

    info = {
        "engine": "sherpa-onnx",
        "model_dir": files["model_dir"],
        "encoder": os.path.basename(files["encoder"]),
        "decoder": os.path.basename(files["decoder"]),
        "joiner": os.path.basename(files["joiner"]),
        "tokens": os.path.basename(files["tokens"]),
    }
    return (recognizer, info) if return_info else recognizer


def _decode_blocks(recognizer, blocks, sample_rate):
    if recognizer is None:
        raise ASRConfigurationError("recognizer 不能为空；禁止模拟 ASR 降级")

    segments = []
    stream = recognizer.create_stream()
    previous = ""
    last_start = 0.0
    last_end = 0.0

    def capture(start, end):
        nonlocal previous
        current = recognizer.get_result(stream) or ""
        if not current or current == previous:
            return
        incremental = current[len(previous):] if current.startswith(previous) else current
        incremental = incremental.strip()
        if incremental:
            segments.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "text": incremental,
            })
        previous = current

    try:
        for start, end, block in blocks:
            last_start, last_end = start, end
            stream.accept_waveform(sample_rate, np.asarray(block, dtype=np.float32))
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            capture(start, end)
            if recognizer.is_endpoint(stream):
                recognizer.reset(stream)
                previous = ""

        input_finished = getattr(stream, "input_finished", None)
        if callable(input_finished):
            input_finished()
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            capture(last_start, last_end)
    except Exception as exc:
        raise ASRRuntimeError(f"流式识别失败: {exc}") from exc

    return segments


def transcribe_streaming(recognizer, samples, sr=16000, chunk_sec=2.0):
    """兼容数组输入的严格流式识别接口。"""
    if chunk_sec <= 0 or sr <= 0:
        raise ValueError("sr 和 chunk_sec 必须大于 0")
    samples = np.asarray(samples, dtype=np.float32)
    chunk = max(1, int(sr * chunk_sec))

    def blocks():
        for start in range(0, len(samples), chunk):
            end = min(start + chunk, len(samples))
            yield start / sr, end / sr, samples[start:end]

    return _decode_blocks(recognizer, blocks(), sr)


def transcribe_wav_streaming(recognizer, wav_path, sr=16000, chunk_sec=2.0):
    """以有界内存流式读取 WAV 并识别。"""
    return _decode_blocks(recognizer, iter_wav_chunks(wav_path, sr, chunk_sec), sr)


def transcribe_offline(*_args, **_kwargs):
    raise NotImplementedError("离线 ASR 尚未实现；禁止返回模拟字幕")


def main():
    parser = argparse.ArgumentParser(description="严格模式 sherpa-onnx 流式 ASR")
    parser.add_argument("--wav", required=True, help="16 kHz 单声道 16-bit PCM WAV")
    parser.add_argument("--model-dir", required=True, help="sherpa-onnx 模型目录")
    args = parser.parse_args()

    recognizer, model_info = load_streaming_recognizer(args.model_dir, return_info=True)
    segments = transcribe_wav_streaming(recognizer, args.wav)
    print(json.dumps({"model": model_info, "segments": segments}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
