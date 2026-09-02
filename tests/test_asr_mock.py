"""ASR mock fallback + 真词级时间戳路径 单元测试。

不依赖 sherpa-onnx：
  - transcribe_streaming 传 None recognizer 必须走 mock 路径（旧结构契约）；
  - 用 FakeRecognizer 模拟 sherpa-onnx 1.10+ 的 get_tokens API，
    验证 W1 真词级时间戳路径（t = token.start / sample_rate）与 end_t。
另覆盖 load_wav 的合成 WAV 读取。
"""
import wave

import numpy as np
import pytest

from vus.asr_sherpa import (
    transcribe_streaming, load_wav, _mock_transcribe,
    _first_new_token_time, _token_timestamps,
)


def _make_samples(seconds, sr=16000):
    rng = np.random.RandomState(7)
    return (rng.randn(int(sr * seconds)) * 0.05).astype(np.float32)


# ==================== mock 流式转写 ====================

def test_mock_transcribe_structure_and_monotonic_time():
    sr = 16000
    samples = _make_samples(10.0, sr)
    segs = transcribe_streaming(None, samples, sr, chunk_sec=2.0)

    assert segs, "mock 也应产出时间轴结构"
    ts = [s["t"] for s in segs]
    assert ts == sorted(ts), "时间戳必须单调不减"
    for s in segs:
        assert set(s) == {"t", "text"}
        assert isinstance(s["text"], str) and s["text"]
        assert 0.0 <= s["t"] < 10.0


def test_mock_transcribe_chunk_count():
    sr = 16000
    duration = 10.0
    samples = _make_samples(duration, sr)
    segs = _mock_transcribe(samples, sr, chunk_sec=2.0)
    assert len(segs) == 5  # 10s / 2s = 5 段
    assert segs[0]["t"] == pytest.approx(0.0)
    assert segs[1]["t"] == pytest.approx(2.0)


def test_mock_transcribe_empty_audio():
    segs = _mock_transcribe(np.array([], dtype=np.float32), 16000, chunk_sec=2.0)
    assert segs == []


# ==================== W1: 真词级时间戳（FakeRecognizer 走真实代码路径） ====================

class _FakeToken:
    """模拟 sherpa-onnx Token：token 文本 + start（音频内样本偏移）。"""

    def __init__(self, token, start):
        self.token = token
        self.start = start


class _FakeStream:
    def __init__(self, rec):
        self.rec = rec

    def accept_waveform(self, sr, block):
        self.rec._accept(sr, block)


class _FakeRecognizer:
    """模拟 sherpa-onnx OnlineRecognizer（1.10+ 形态）：
    每接受满 1.0s 音频产出一个词，词起点固定在 0.75s, 1.75s, 2.75s, ...
    get_tokens 返回带 start 样本偏移的 token 列表（增量前缀文本语义）。"""

    def __init__(self, sr=16000):
        self.sr = sr
        self.tokens = []      # [(start_sample, text), ...]
        self._accepted = 0    # 累计接受样本数

    def create_stream(self):
        return _FakeStream(self)

    def accept_waveform(self, sr, block):  # 兼容两种调用面
        self._accept(sr, block)

    def _accept(self, sr, block):
        self._accepted += len(block)
        while True:
            pos_s = 0.75 + len(self.tokens) * 1.0
            pos = int(pos_s * sr)
            if pos < self._accepted:
                self.tokens.append((pos, f"w{len(self.tokens)}"))
            else:
                break

    def is_ready(self, stream):
        return False  # 增量结果随 accept 即时可得

    def decode_stream(self, stream):
        pass

    def is_endpoint(self, stream):
        return False

    def reset(self, stream):
        pass

    def get_result(self, stream):
        return "".join(text for _, text in self.tokens)

    def get_tokens(self, stream):
        return [_FakeToken(text, start) for start, text in self.tokens]


def test_transcribe_streaming_word_level_timestamps():
    """真实路径（非 mock）：段 t 应为 chunk 内首个新 token 的真实时间，非 chunk 起点。

    10s 音频，chunk 2s：词落在 0.75/1.75/2.75/...s，因此 5 段的 t
    应为 [0.75, 2.75, 4.75, 6.75, 8.75]（旧粗粒度实现会是 0/2/4/6/8），
    且每段带 end_t = chunk 末尾。
    """
    sr = 16000
    rec = _FakeRecognizer(sr)
    samples = _make_samples(10.0, sr)
    segs = transcribe_streaming(rec, samples, sr, chunk_sec=2.0)

    assert len(segs) == 5
    assert [s["t"] for s in segs] == [0.75, 2.75, 4.75, 6.75, 8.75]
    assert [s["end_t"] for s in segs] == [2.0, 4.0, 6.0, 8.0, 10.0]
    assert segs[0]["text"] == "w0w1"
    assert segs[-1]["text"].endswith("w9")
    ts = [s["t"] for s in segs]
    assert ts == sorted(ts)


def test_transcribe_streaming_falls_back_without_get_tokens():
    """recognizer 无 get_tokens API（旧版 sherpa-onnx）：回退 chunk 起点粗粒度。"""
    sr = 16000

    class _Legacy(_FakeRecognizer):
        def get_tokens(self, stream):
            raise AttributeError("get_tokens 不存在（旧版 API）")

    rec = _Legacy(sr)
    samples = _make_samples(6.0, sr)
    segs = transcribe_streaming(rec, samples, sr, chunk_sec=2.0)

    assert segs, "回退路径仍应产出段"
    assert [s["t"] for s in segs] == [0.0, 2.0, 4.0]  # chunk 起点
    assert [s["end_t"] for s in segs] == [2.0, 4.0, 6.0]


def test_token_timestamps_safe_degradation():
    """_token_timestamps 对缺属性 / 异常的 token 对象安全降级为 []。"""
    rec = _FakeRecognizer()
    stream = rec.create_stream()

    class _Broken:
        pass  # 无 token/start 属性

    assert _token_timestamps(rec, stream) == []

    class _BadRec:
        def get_tokens(self, stream):
            raise RuntimeError("boom")

    assert _token_timestamps(_BadRec(), stream) == []

    class _NoApi:
        pass

    assert _token_timestamps(_NoApi(), stream) == []


def test_first_new_token_time_fallbacks():
    """_first_new_token_time：取首个新 token 时间；无 token/越界时回退 chunk 起点。"""
    tokens = [(32000, "a"), (48000, "b")]  # 2s / 3s @16k
    assert _first_new_token_time(tokens, 1, 2.0, 2.0, 16000) == pytest.approx(3.0)
    assert _first_new_token_time([], 0, 2.0, 2.0, 16000) == pytest.approx(2.0)
    assert _first_new_token_time(tokens, 2, 2.0, 2.0, 16000) == pytest.approx(2.0)
    assert _first_new_token_time([(0, "x")], 0, 2.0, 2.0, 16000) == pytest.approx(2.0)


# ==================== load_wav ====================

def test_load_wav_reads_synthetic_wav(tmp_path):
    sr = 16000
    wav_path = tmp_path / "audio.wav"
    data = (_make_samples(1.0, sr) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

    samples, out_sr = load_wav(str(wav_path), sr)
    assert out_sr == sr
    assert len(samples) == sr  # 1 秒
    assert samples.dtype == np.float32


def test_load_wav_missing_file_returns_empty():
    samples, sr = load_wav("Z:/definitely/not/here.wav", 16000)
    assert len(samples) == 0  # 优雅降级，不抛异常
