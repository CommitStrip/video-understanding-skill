"""ASR mock fallback 单元测试。

不依赖 sherpa-onnx：transcribe_streaming 传 None recognizer 必须走 mock 路径，
覆盖输出结构、时间递增、chunk 粒度，以及 load_wav 的合成 WAV 读取。
"""
import wave

import numpy as np
import pytest

from vus.asr_sherpa import transcribe_streaming, load_wav, _mock_transcribe


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
