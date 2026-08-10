#!/usr/bin/env python3
"""
asr_sherpa.py - 声音链：流式ASR词级落盘
使用 sherpa-onnx 流式模型（中英双语），逐块解码产出增量文本 + 绝对时间戳
若 sherpa-onnx 不可用，提供 mock fallback 供框架验证
"""

import numpy as np
import json
import os
import subprocess
import sys


def extract_audio(video_path, output_wav=None, sr=16000):
    """从视频抽取音频：16kHz 单声道 WAV"""
    if output_wav is None:
        output_wav = video_path.rsplit('.', 1)[0] + '_audio.wav'
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', str(sr),
        '-f', 'wav', output_wav
    ]
    subprocess.run(cmd, capture_output=True, check=False)
    if os.path.exists(output_wav):
        return output_wav
    return None


def load_wav(wav_path, sr=16000):
    """读取WAV音频为numpy数组"""
    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0

        if n_channels > 1:
            samples = samples[::n_channels]

        # 重采样
        if framerate != sr:
            ratio = sr / framerate
            n_out = int(len(samples) * ratio)
            indices = np.linspace(0, len(samples) - 1, n_out)
            samples = np.interp(indices, np.arange(len(samples)), samples)

        return samples, sr
    except Exception as e:
        print(f"[ASR] 读取音频失败: {e}")
        return np.array([]), sr


def load_streaming_recognizer(model_dir=None):
    """
    加载 sherpa-onnx 流式识别器
    模型: sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    """
    try:
        import sherpa_onnx
    except ImportError:
        print("[ASR] sherpa-onnx 未安装，使用 fallback 模式")
        return None

    # 查找模型目录
    if model_dir is None:
        candidates = [
            os.path.expanduser("~/sherpa-onnx-models"),
            "/data/user/work/models",
            "/data/user/models",
            "./models",
        ]
        for c in candidates:
            if os.path.exists(c):
                model_dir = c
                break

    if model_dir is None or not os.path.exists(model_dir):
        print("[ASR] 模型目录未找到，使用 fallback 模式")
        return None

    # 查找流式模型
    encoder = None
    decoder = None
    joiner = None
    tokens = None

    for root, dirs, files in os.walk(model_dir):
        for f in files:
            fp = os.path.join(root, f)
            if 'encoder' in f and f.endswith('.onnx'):
                encoder = fp
            elif 'decoder' in f and f.endswith('.onnx'):
                decoder = fp
            elif 'joiner' in f and f.endswith('.onnx'):
                joiner = fp
            elif f == 'tokens.txt':
                tokens = fp

    if not all([encoder, decoder, joiner, tokens]):
        print("[ASR] 流式模型文件不完整，使用 fallback 模式")
        return None

    try:
        recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=2,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,
            decoding_method="greedy_search",
            provider="cpu",
        )
        print(f"[ASR] 流式识别器加载成功")
        return recognizer
    except Exception as e:
        print(f"[ASR] 加载识别器失败: {e}，使用 fallback 模式")
        return None


def transcribe_streaming(recognizer, samples, sr=16000, chunk_sec=2.0):
    """
    流式ASR词级落盘
    输入: recognizer (sherpa_onnx.OnlineRecognizer 或 None), samples, sr
    输出: [{"t": float, "text": str}, ...]
    """
    if recognizer is None:
        return _mock_transcribe(samples, sr, chunk_sec)

    segments = []
    n = len(samples)
    chunk = int(sr * chunk_sec)
    if chunk == 0:
        chunk = n

    stream = recognizer.create_stream()
    prev = ""
    start = 0

    while start < n:
        end = min(start + chunk, n)
        block = samples[start:end].astype(np.float32)

        stream.accept_waveform(sr, block)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        cur = recognizer.get_result(stream)

        if cur and cur != prev:
            if cur.startswith(prev):
                inc = cur[len(prev):]
            else:
                inc = cur
            inc = inc.strip()
            if inc:
                segments.append({
                    "t": round(start / sr, 3),
                    "text": inc
                })
            prev = cur

        # 端点检测
        if recognizer.is_endpoint(stream):
            recognizer.reset(stream)
            prev = ""

        start += chunk

    # 处理最后一块
    if prev.strip():
        segments.append({
            "t": round((start - chunk) / sr, 3),
            "text": prev.strip()
        })

    return segments


def _mock_transcribe(samples, sr=16000, chunk_sec=2.0):
    """
    Fallback: 当 sherpa-onnx 不可用时，生成模拟ASR段
    按固定间隔产出空文本段，保持时间轴结构完整
    """
    segments = []
    n = len(samples)
    duration = n / sr if sr > 0 else 0
    chunk = int(sr * chunk_sec)

    mock_texts = [
        "[音频段] 此段为模拟ASR输出",
        "[音频段] sherpa-onnx模型未加载",
        "[音频段] 框架验证模式",
    ]

    start = 0
    idx = 0
    while start < n:
        t = start / sr
        segments.append({
            "t": round(t, 3),
            "text": mock_texts[idx % len(mock_texts)]
        })
        start += chunk
        idx += 1

    return segments


def transcribe_offline(wav_path, sr=16000):
    """离线ASR（如果可用）"""
    try:
        import sherpa_onnx
    except ImportError:
        samples, sr = load_wav(wav_path, sr)
        return _mock_transcribe(samples, sr)

    # 尝试加载离线模型
    # ... (类似流式，但使用 OfflineRecognizer)
    samples, sr = load_wav(wav_path, sr)
    return _mock_transcribe(samples, sr)


if __name__ == '__main__':
    # 快速自测
    print("=== ASR 模块自测 ===")
    rec = load_streaming_recognizer()
    if rec is None:
        print("使用 fallback 模式")
        # 模拟10秒音频
        samples = np.random.randn(16000 * 10).astype(np.float32) * 0.1
        segs = transcribe_streaming(None, samples)
        print(f"产出 {len(segs)} 个段")
        for s in segs[:3]:
            print(f"  t={s['t']:.1f}s: {s['text']}")
    else:
        print("识别器加载成功，等待音频输入")
