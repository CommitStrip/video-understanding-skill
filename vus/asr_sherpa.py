#!/usr/bin/env python3
"""
asr_sherpa.py - 声音链：流式ASR词级落盘
使用 sherpa-onnx 流式模型（中英双语），逐块解码产出增量文本 + 绝对时间戳
若 sherpa-onnx 不可用，提供 mock fallback 供框架验证

模型目录查找顺序（可通过环境变量 VUS_SHERPA_MODELS 覆盖）：
  1. 环境变量 VUS_SHERPA_MODELS
  2. ~/sherpa-onnx-models
  3. ./models
"""

import numpy as np
import json
import os
import subprocess


def extract_audio(video_path, output_wav=None, sr=16000):
    """从视频抽取音频：16kHz 单声道 WAV。

    ffmpeg 不可用或抽音失败时返回 None（调用方据此跳过声音链，
    画面链与代表帧核心功能不受影响）。
    """
    if output_wav is None:
        output_wav = video_path.rsplit('.', 1)[0] + '_audio.wav'
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-ac', '1', '-ar', str(sr),
        '-f', 'wav', output_wav
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=False)
    except OSError:
        # ffmpeg 未安装等情况：静默降级，返回 None 让调用方跳过声音链
        return None
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

    # 查找模型目录（共享 model_setup 逻辑：目录必须含完整模型文件）
    if model_dir is None:
        try:
            from .model_setup import find_asr_model
            model_dir = find_asr_model()
        except ImportError:
            candidates = [
                os.environ.get("VUS_SHERPA_MODELS") or "",
                os.path.expanduser("~/sherpa-onnx-models"),
                "./models/sherpa",
                "./models",
            ]
            for c in candidates:
                if c and os.path.exists(c):
                    model_dir = c
                    break

    if model_dir is None or not os.path.exists(model_dir):
        # 默认开箱即用：缺模型时自动从官方源下载（VUS_ASR_AUTO_DOWNLOAD=0 关闭）
        try:
            from .model_setup import ensure_asr_model
            model_dir = ensure_asr_model(model_dir)
        except ImportError:
            model_dir = None
        if model_dir is None:
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


def _token_timestamps(recognizer, stream):
    """读取当前流的 token 级时间戳（按 sherpa-onnx 1.10+ API：recognizer.get_tokens(stream)）。

    返回 [(start_sample, text), ...]；API 不存在 / 结构不符 / 抛异常时返回 []，
    调用方据此回退到 chunk 起点粗粒度时间戳。
    注：本仓库不硬依赖 sherpa-onnx（未安装时走 mock 路径），此实现按上游
    1.10+ 文档编写并用 try/except 兜底，token 属性缺失时安全降级。
    """
    try:
        getter = getattr(recognizer, "get_tokens", None)
        if getter is None:
            return []
        out = []
        for tk in getter(stream):
            start = getattr(tk, "start", None)
            text = getattr(tk, "token", None)
            if start is None or text is None:
                return []
            out.append((start, text))
        return out
    except Exception:
        return []


def _first_new_token_time(tokens, n_consumed, chunk_t, chunk_sec, sr):
    """取 chunk 内首个新 token 的真实时间（t = token.start / sample_rate）。

    tokens: 当前流全部 token 的 [(start_sample, text), ...]（_token_timestamps 产出）
    n_consumed: 此前 chunk 已归属旧文本的 token 数，其后即本 chunk 的新增 token
    无法获取时间戳（API 缺失 / 无新 token）或时间戳不在本 chunk 窗口内
    （兼容 token.start 语义差异与端点重置）时，回退 chunk 起点 chunk_t。
    """
    if not tokens or len(tokens) <= n_consumed:
        return chunk_t
    t_tok = tokens[n_consumed][0] / float(sr)
    if chunk_t - 1e-6 <= t_tok <= chunk_t + chunk_sec + 1e-6:
        return t_tok
    return chunk_t


def transcribe_streaming(recognizer, samples, sr=16000, chunk_sec=2.0):
    """
    流式ASR词级落盘
    输入: recognizer (sherpa_onnx.OnlineRecognizer 或 None), samples, sr
    输出: [{"t": float, "text": str, "end_t": float}, ...]

    时间戳策略（W1 升级）：
      - 优先按 sherpa-onnx 1.10+ 的 get_tokens(stream) 取 token 级时间戳
        （t = token.start / sample_rate），段的 "t" 为该 chunk 内首个新 token 的真实时间；
      - API 不可用 / 时间戳越界时回退到 2s chunk 起点粗粒度（旧行为）；
      - 每段附带 "end_t"（该 chunk 末尾时间），对齐层可选消费；
      - recognizer 为 None 时走 _mock_transcribe（mock 段保持 {"t","text"} 旧结构）。
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
    n_consumed_tokens = 0  # 当前流内已被旧文本归属的 token 数
    start = 0

    while start < n:
        end = min(start + chunk, n)
        block = samples[start:end].astype(np.float32)

        stream.accept_waveform(sr, block)
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)

        chunk_t = start / sr
        tokens = _token_timestamps(recognizer, stream)
        cur = recognizer.get_result(stream)

        if cur and cur != prev:
            if cur.startswith(prev):
                inc = cur[len(prev):]
            else:
                inc = cur
            inc = inc.strip()
            if inc:
                seg_t = _first_new_token_time(tokens, n_consumed_tokens,
                                              chunk_t, chunk_sec, sr)
                segments.append({
                    "t": round(seg_t, 3),
                    "text": inc,
                    "end_t": round(end / sr, 3)
                })
            prev = cur

        # 记录本 chunk 解码出的 token 总数，供下一 chunk 定位"新增 token"
        n_consumed_tokens = len(tokens)

        # 端点检测
        if recognizer.is_endpoint(stream):
            recognizer.reset(stream)
            prev = ""
            n_consumed_tokens = 0

        start += chunk

    # 收尾兜底：正常增量路径已在循环内逐段产出（prev 已完整发射，不再重复落盘）；
    # 仅当全程未能产出任何段但确有文本（如增量均被 strip 清空）时补一段，
    # 时间戳无 chunk 上下文可用，回退最后一个 chunk 起点粗粒度。
    if not segments and prev.strip():
        segments.append({
            "t": round((start - chunk) / sr, 3),
            "text": prev.strip(),
            "end_t": round(start / sr, 3)
        })

    return segments


def _mock_transcribe(samples, sr=16000, chunk_sec=2.0):
    """
    Fallback: 当 sherpa-onnx 不可用时，生成模拟ASR段
    按固定间隔产出空文本段，保持时间轴结构完整
    （保持旧结构 {"t","text"}，不带 end_t，作为契约兜底）
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


# W1（2026-09-02）：删除 transcribe_offline() 空桩——原实现不经任何路径
# 都只会返回 _mock_transcribe 结果（假离线识别），对调用方有误导性。
# 离线识别需求统一走 transcribe_streaming（W2 实时源波次再评估真离线模型）。


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
