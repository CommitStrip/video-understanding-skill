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


# mock 占位文本（_mock_transcribe 与 StreamingASR 的 mock 语义共用）
_MOCK_TEXTS = [
    "[音频段] 此段为模拟ASR输出",
    "[音频段] sherpa-onnx模型未加载",
    "[音频段] 框架验证模式",
]


class StreamingASR:
    """增量流式 ASR 解码状态机（W8）：feed() 样本块 → 产出终稿段。

    从 transcribe_streaming 的循环体抽出，使"文件整段切块喂入"与
    "直播音频实时块喂入"共用同一解码状态机，两条路径段产出语义一致：
      - feed() 内部缓冲，凑满一个 chunk 才解码（块边界与文件路径完全一致）；
      - flush() 解码残余不足一块的样本并补发收尾兜底段；
      - 端点检测触发 reset 后增量基线同步归零（与原实现一致）；
      - recognizer=None 时按 mock 语义逐 chunk 产出占位段（契约同旧结构）。
    线程模型：单线程喂入（decode 内部状态非线程安全），跨线程取结果由调用方自理。
    """

    def __init__(self, recognizer, sr=16000, chunk_sec=2.0):
        self.recognizer = recognizer
        self.sr = sr
        self.chunk_sec = chunk_sec
        self.chunk_samples = max(1, int(sr * chunk_sec))
        self._stream = recognizer.create_stream() if recognizer is not None else None
        self._buf = np.array([], dtype=np.float32)  # 不足一块的残余样本
        self._prev = ""                              # 当前流内已发射文本前缀
        self._n_consumed_tokens = 0                  # 当前流内已被旧文本归属的 token 数
        self._next_start = 0                         # 下一 chunk 起始样本（绝对时间基）
        self._mock_idx = 0
        self._emitted = 0                            # 累计产出段数（收尾兜底判据）
        self._finished = False

    def feed(self, block):
        """喂入任意长度样本块，返回本批新产出的终稿段列表（可能为空）。"""
        if self._finished:
            return []
        block = np.asarray(block, dtype=np.float32)
        if len(block) == 0:
            return []
        if len(self._buf):
            block = np.concatenate([self._buf, block])
        segments = []
        n_full = (len(block) // self.chunk_samples) * self.chunk_samples
        for s in range(0, n_full, self.chunk_samples):
            segments.extend(self._feed_chunk(block[s:s + self.chunk_samples]))
        self._buf = block[n_full:]
        return segments

    def flush(self):
        """收尾：解码残余样本；仅当全程未产出任何段而确有文本时补发兜底段。"""
        if self._finished:
            return []
        self._finished = True
        segments = []
        if len(self._buf) > 0:
            segments.extend(self._feed_chunk(self._buf))
            self._buf = np.array([], dtype=np.float32)
        if (self.recognizer is not None and self._emitted == 0
                and self._prev.strip()):
            segments.append({
                "t": round((self._next_start - self.chunk_samples) / self.sr, 3),
                "text": self._prev.strip(),
                "end_t": round(self._next_start / self.sr, 3),
            })
        return segments

    def _feed_chunk(self, sub):
        """解码一个 chunk（≤ chunk_samples）。

        next_start 无条件按整 chunk 推进、end_t 用实际样本数——与文件路径
        `start += chunk` / `end = min(start+chunk, n)` 的语义逐点一致。
        """
        if self.recognizer is None:
            seg = {
                "t": round(self._next_start / self.sr, 3),
                "text": _MOCK_TEXTS[self._mock_idx % len(_MOCK_TEXTS)],
            }
            self._mock_idx += 1
            self._emitted += 1
            self._next_start += self.chunk_samples
            return [seg]

        rec = self.recognizer
        stream = self._stream
        stream.accept_waveform(self.sr, sub.astype(np.float32))
        while rec.is_ready(stream):
            rec.decode_stream(stream)

        chunk_t = self._next_start / self.sr
        end_t = (self._next_start + len(sub)) / self.sr
        tokens = _token_timestamps(rec, stream)
        cur = rec.get_result(stream)

        segments = []
        if cur and cur != self._prev:
            if cur.startswith(self._prev):
                inc = cur[len(self._prev):]
            else:
                inc = cur
            inc = inc.strip()
            if inc:
                seg_t = _first_new_token_time(tokens, self._n_consumed_tokens,
                                              chunk_t, self.chunk_sec, self.sr)
                segments.append({
                    "t": round(seg_t, 3),
                    "text": inc,
                    "end_t": round(end_t, 3)
                })
                self._emitted += 1
            self._prev = cur

        # 记录本 chunk 解码出的 token 总数，供下一 chunk 定位"新增 token"
        self._n_consumed_tokens = len(tokens)

        # 端点检测
        if rec.is_endpoint(stream):
            rec.reset(stream)
            self._prev = ""
            self._n_consumed_tokens = 0

        self._next_start += self.chunk_samples
        return segments


def transcribe_streaming(recognizer, samples, sr=16000, chunk_sec=2.0):
    """
    流式ASR词级落盘（W8 起内部委托 StreamingASR，输入输出契约不变）
    输入: recognizer (sherpa_onnx.OnlineRecognizer 或 None), samples, sr
    输出: [{"t": float, "text": str, "end_t": float}, ...]

    时间戳策略（W1 升级）：
      - 优先按 sherpa-onnx 1.10+ 的 get_tokens(stream) 取 token 级时间戳
        （t = token.start / sample_rate），段的 "t" 为该 chunk 内首个新 token 的真实时间；
      - API 不可用 / 时间戳越界时回退到 2s chunk 起点粗粒度（旧行为）；
      - 每段附带 "end_t"（该 chunk 末尾时间），对齐层可选消费；
      - recognizer 为 None 时走 mock 语义（mock 段保持 {"t","text"} 旧结构）。
    """
    asr = StreamingASR(recognizer, sr=sr, chunk_sec=chunk_sec)
    segments = asr.feed(samples)
    segments.extend(asr.flush())
    return segments


def _mock_transcribe(samples, sr=16000, chunk_sec=2.0):
    """
    Fallback: 当 sherpa-onnx 不可用时，生成模拟ASR段
    按固定间隔产出空文本段，保持时间轴结构完整
    （保持旧结构 {"t","text"}，不带 end_t，作为契约兜底）
    """
    asr = StreamingASR(None, sr=sr, chunk_sec=chunk_sec)
    segments = asr.feed(samples)
    segments.extend(asr.flush())
    return segments


# W1（2026-09-02）：删除 transcribe_offline() 空桩——原实现不经任何路径
# 都只会返回 _mock_transcribe 结果（假离线识别），对调用方有误导性。
# 离线识别需求统一走 transcribe_streaming（W2 实时源波次再评估真离线模型）。


def load_offline_recognizer(model_dir=None):
    """加载 SenseVoice 离线识别器（非流式，全上下文解码）。

    中文专名与可读性显著优于流式 zipformer（流式上下文窗口短），
    用于文件转写默认路径；对 BGM 鲁棒。模型经 model_setup 自动下载。
    返回识别器对象；模型缺失/加载失败时抛 RuntimeError（默认路径不静默降级）。
    """
    try:
        import sherpa_onnx
    except ImportError as e:
        raise RuntimeError("sherpa-onnx 未安装，离线识别不可用") from e
    from .model_setup import ensure_offline_asr_model, find_offline_asr_model

    model_dir = model_dir or ensure_offline_asr_model() or find_offline_asr_model()
    if not model_dir:
        raise RuntimeError("离线模型未找到且自动下载被关闭（VUS_ASR_AUTO_DOWNLOAD=0）")
    model_file = None
    for cand in ("model.int8.onnx", "model.onnx"):
        p = os.path.join(model_dir, cand)
        if os.path.exists(p):
            model_file = p
            break
    if not model_file:
        for root, _dirs, files in os.walk(model_dir):
            for f in files:
                if f.startswith("model") and f.endswith(".onnx"):
                    model_file = os.path.join(root, f)
                    break
            if model_file:
                break
    if not model_file:
        raise RuntimeError(f"离线模型目录缺少 model onnx: {model_dir}")
    tokens = None
    for root, _dirs, files in os.walk(model_dir):
        if "tokens.txt" in files:
            tokens = os.path.join(root, "tokens.txt")
            break
    if not tokens:
        raise RuntimeError(f"离线模型目录缺少 tokens.txt: {model_dir}")
    recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=model_file,
        tokens=tokens,
        num_threads=2,
        use_itn=True,
    )
    print("[ASR] 离线识别器（SenseVoice int8）加载成功")
    return recognizer


def transcribe_offline(recognizer, samples, sr=16000, window_sec=15.0):
    """离线全局转写：按 window_sec 切窗（离线有全窗上下文，无流式短窗问题）。

    返回段格式与 transcribe_streaming 一致：[{"t": 窗起点, "text": ...}]。
    """
    segments = []
    total = len(samples)
    win = int(sr * window_sec)
    if win <= 0:
        win = int(sr)
    for start in range(0, total, win):
        block = samples[start:start + win]
        if len(block) < sr * 0.2:
            break
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, block)
        recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            segments.append({"t": round(start / sr, 3), "text": text})
    return segments


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
