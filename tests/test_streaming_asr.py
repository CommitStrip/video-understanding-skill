"""W8a 直播音频链 单元测试。

覆盖四块，全部不依赖 sherpa-onnx（FakeRecognizer 模拟 1.10+ API）：
  1. StreamingASR 增量喂入（任意块尺寸）与批式 transcribe_streaming 逐段一致
  2. RollingCleaner 增量清洗与批式 clean_asr_segments 等价
  3. RollingAligner 直播序闭窗与批式 align_asr_streaming 等价
  4. AudioStream 切块 / EOF / 超时 / 节拍回放 / 真实 ffmpeg 集成
"""
import shutil
import subprocess
import sys
import time
import wave

import numpy as np
import pytest

from vus.asr_clean import RollingCleaner, clean_asr_segments
from vus.asr_sherpa import StreamingASR, _mock_transcribe, transcribe_streaming
from vus.live import AudioStream, RollingAligner
from vus.smart_pipeline import SmartPipeline


def _make_samples(seconds, sr=16000):
    rng = np.random.RandomState(7)
    return (rng.randn(int(sr * seconds)) * 0.05).astype(np.float32)


# ==================== FakeRecognizer（与 test_asr_mock 同款语义） ====================

class _FakeToken:
    def __init__(self, token, start):
        self.token = token
        self.start = start


class _FakeStream:
    def __init__(self, rec):
        self.rec = rec

    def accept_waveform(self, sr, block):
        self.rec._accept(sr, block)


class _FakeRecognizer:
    """每接受满 1.0s 音频产出一个词，词起点固定在 0.75s, 1.75s, ...。"""

    def __init__(self, sr=16000):
        self.sr = sr
        self.tokens = []
        self._accepted = 0
        self._endpoint_calls = 0

    def create_stream(self):
        return _FakeStream(self)

    def accept_waveform(self, sr, block):
        self._accept(sr, block)

    def _accept(self, sr, block):
        self._accepted += len(block)
        while True:
            pos = int((0.75 + len(self.tokens) * 1.0) * sr)
            if pos < self._accepted:
                self.tokens.append((pos, f"w{len(self.tokens)}"))
            else:
                break

    def is_ready(self, stream):
        return False

    def decode_stream(self, stream):
        pass

    def is_endpoint(self, stream):
        self._endpoint_calls += 1
        return self._endpoint_calls % 3 == 0  # 每 3 个 chunk 触发一次端点

    def reset(self, stream):
        pass

    def get_result(self, stream):
        return "".join(text for _, text in self.tokens)

    def get_tokens(self, stream):
        return [_FakeToken(text, start) for start, text in self.tokens]


# ==================== StreamingASR 增量/批式一致性 ====================

def test_streaming_asr_mock_parity_with_batch():
    """recognizer=None：增量 feed（任意块尺寸）结果必须与批式 mock 逐段一致。"""
    sr = 16000
    samples = _make_samples(10.0, sr)
    batch = transcribe_streaming(None, samples, sr, chunk_sec=2.0)

    asr = StreamingASR(None, sr=sr, chunk_sec=2.0)
    out = []
    off = 0
    for size in (7000, 1, 40000, 50000, 62999):  # 故意歪七扭八的直播块
        out.extend(asr.feed(samples[off:off + size]))
        off += size
    out.extend(asr.flush())

    assert out == batch
    assert out == _mock_transcribe(samples, sr, chunk_sec=2.0)
    assert len(out) == 5


def test_streaming_asr_incremental_parity_with_batch():
    """真解码路径：任意直播块尺寸的增量输出与批式 transcribe_streaming 逐段一致。"""
    sr = 16000
    samples = _make_samples(10.0, sr)
    batch = transcribe_streaming(_FakeRecognizer(sr), samples, sr, chunk_sec=2.0)

    asr = StreamingASR(_FakeRecognizer(sr), sr=sr, chunk_sec=2.0)
    out = []
    off = 0
    for size in (7000, 1, 40000, 50000, 62999):
        out.extend(asr.feed(samples[off:off + size]))
        off += size
    out.extend(asr.flush())

    assert out == batch
    # 本 FakeRecognizer 在第 3 块触发端点重置：第 4 段 t 回退 chunk 起点 6.0，
    # 且重发全量文本（增量基线已清零）——与批式路径语义一致
    assert [s["t"] for s in out] == [0.75, 2.75, 4.75, 6.0, 8.75]
    assert out[3]["text"] == "w0w1w2w3w4w5w6w7"


def test_streaming_asr_flush_after_finish_is_noop():
    """flush() 后继续 feed/flush 必须为空（直播关闭后的防御）。"""
    asr = StreamingASR(None, sr=16000, chunk_sec=2.0)
    assert asr.flush() == []  # 空流收尾
    assert asr.feed(np.ones(16000, dtype=np.float32)) == []  # 已 finished
    assert asr.flush() == []


def test_flush_trailing_fallback_segment():
    """全程无段产出但流内有文本时，flush 补发兜底段（时间回退最后 chunk 起点）。"""
    asr = StreamingASR(_FakeRecognizer(16000), sr=16000, chunk_sec=2.0)
    asr._prev = "  hello  "   # 白盒：模拟增量均被 strip 清空后的残留文本
    asr._next_start = 3 * asr.chunk_samples
    assert asr.flush() == [{"t": 4.0, "text": "hello", "end_t": 6.0}]


# ==================== RollingCleaner 增量清洗 ====================

def test_rolling_cleaner_parity_with_batch():
    """逐段喂入与批式 clean_asr_segments 逐条等价（含跨段去重/折叠/幻觉标记）。"""
    segs = [
        {"t": 0.0, "text": "今天的天气"},
        {"t": 2.0, "text": " 今天的天气 "},   # 相邻重复 → 剔除
        {"t": 4.0, "text": "runrunrun"},      # 连叠折叠 → run
        {"t": 6.0, "text": "SIL"},            # 幻觉标记
        {"t": 8.0, "text": "   "},            # 空段剔除
        {"t": 10.0, "text": "今天的天气"},    # 与上一条（SIL）不同 → 保留
    ]
    rc = RollingCleaner()
    out = []
    for seg in segs:
        out.extend(rc.feed([seg]))
    assert out == clean_asr_segments(segs)
    assert out[-1]["text"] == "今天的天气"
    assert out[1]["cleaned"] is True and out[1]["raw_text"] == "runrunrun"


def test_rolling_cleaner_dedup_across_feeds():
    """相邻去重的状态必须跨 feed 存活（chunk 边界重复在直播下同样被折叠）。"""
    rc = RollingCleaner()
    assert rc.feed([{"t": 0.0, "text": "好"}])[-1]["text"] == "好"
    assert rc.feed([{"t": 2.0, "text": "好"}]) == []  # 跨 feed 相邻重复


# ==================== RollingAligner 直播序闭窗 ====================

_EVENTS = [
    {"type": "motion_start", "t": 0.5},
    {"type": "motion", "t": 1.0},
    {"type": "motion_end", "t": 2.5},
    {"type": "keyframe", "t": 3.0},
    {"type": "motion_start", "t": 5.2},
]
_SEGS = [
    {"t": 0.0, "text": "a", "end_t": 2.0},
    {"t": 2.0, "text": "b", "end_t": 4.0},
    {"t": 4.0, "text": "c", "end_t": 6.0},
]


def _norm(aligned):
    return [{**a, "motion_types": sorted(a["motion_types"])} for a in aligned]


def test_rolling_aligner_parity_with_batch():
    """直播序（事件按时间推进、ASR 段陆续到达）闭窗结果与批式对齐逐条等价。"""
    pipe = SmartPipeline()
    for ev in _EVENTS:
        pipe.events.append(ev)
    batch = _norm(pipe.align_asr_streaming(_SEGS))

    al = RollingAligner()
    out = []
    for ev in _EVENTS:
        al.add_frame_event(ev)
    for seg in _SEGS:
        out.extend(al.add_asr_segment(seg))
    out.extend(al.flush())

    assert _norm(out) == batch
    assert [a["text"] for a in out] == ["a", "b", "c"]
    assert out[1]["linked_keyframes"] == 1  # keyframe@3.0 ∈ [2.0, 4.0)


def test_rolling_aligner_last_segment_end_t_fallback():
    """最后一段缺 end_t 时回退 t+2.0（与批式一致）。"""
    al = RollingAligner()
    al.add_asr_segment({"t": 1.0, "text": "x"})
    assert al.flush() == [{
        "start": 1.0, "end": 3.0, "text": "x",
        "linked_motion_events": 0, "linked_keyframes": 0, "motion_types": [],
    }]
    assert al.flush() == []  # 幂等


# ==================== AudioStream ====================

_FAKE_FFMPEG = """
import sys, time, argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--chunks', type=int, default=6)
p.add_argument('--samples', type=int, default=3200)
p.add_argument('--delay', type=float, default=0.0)
args, _ = p.parse_known_args()
rng = np.random.RandomState(3)
for _ in range(args.chunks):
    sys.stdout.buffer.write((rng.randn(args.samples) * 3000).astype('<i2').tobytes())
    sys.stdout.buffer.flush()
    if args.delay:
        time.sleep(args.delay)
"""


def _fake_ffmpeg_cmd(tmp_path):
    script = tmp_path / "fake_ffmpeg.py"
    script.write_text(_FAKE_FFMPEG, encoding="utf-8")
    return [sys.executable, str(script)]


def test_audio_stream_chunks_and_eof(tmp_path):
    """定长切块、块数正确、EOF 后返回 None。"""
    cmd = _fake_ffmpeg_cmd(tmp_path)
    stream = AudioStream("unused", sr=16000, chunk_sec=0.2,
                         ffmpeg_bin=cmd)  # 3200 样本/块 = fake 每次写入量
    assert stream.open() is True
    sizes = []
    while True:
        chunk = stream.read_chunk(timeout=10.0)
        if chunk is None:
            break
        sizes.append(len(chunk))
    stream.close()
    assert sizes == [3200] * 6
    assert stream.stats["chunks_read"] == 6


def test_audio_stream_partial_final_chunk(tmp_path):
    """EOF 残余不足一块时作为最后一块发出（总量 = 全部样本）。"""
    cmd = _fake_ffmpeg_cmd(tmp_path)
    stream = AudioStream("unused", sr=16000, chunk_sec=0.2, ffmpeg_bin=cmd)
    stream.extra_input_args = ["--chunks", "3", "--samples", "1000"]
    stream.open()
    sizes = []
    while True:
        chunk = stream.read_chunk(timeout=10.0)
        if chunk is None:
            break
        sizes.append(len(chunk))
    stream.close()
    assert sum(sizes) == 3000
    assert sizes[-1] < 3200  # 残余块


def test_audio_stream_read_timeout(tmp_path):
    """源供数慢时 read_chunk(timeout) 超时返回 None，且之后仍能取到数据。"""
    cmd = _fake_ffmpeg_cmd(tmp_path)
    stream = AudioStream("unused", sr=16000, chunk_sec=0.2, ffmpeg_bin=cmd)
    stream.extra_input_args = ["--chunks", "2", "--samples", "3200", "--delay", "0.3"]
    stream.open()
    assert stream.read_chunk(timeout=0.1) is None  # 第一块还没到
    first = stream.read_chunk(timeout=10.0)
    assert first is not None and len(first) == 3200
    stream.close()


def test_audio_stream_realtime_pacing(tmp_path):
    """realtime=True：消费 5 块 × 0.2s 的耗时应 ≥ 1s（节拍回放而非抢跑）。"""
    cmd = _fake_ffmpeg_cmd(tmp_path)
    stream = AudioStream("unused", sr=16000, chunk_sec=0.2,
                         realtime=True, ffmpeg_bin=cmd)
    stream.extra_input_args = ["--chunks", "5", "--samples", "3200"]
    stream.open()
    t0 = time.monotonic()
    n = 0
    while stream.read_chunk(timeout=10.0) is not None:
        n += 1
    elapsed = time.monotonic() - t0
    stream.close()
    assert n == 5
    assert elapsed >= 0.8   # 5×0.2s 减抖动容差
    assert elapsed < 5.0


def test_audio_stream_open_failure_returns_false():
    """ffmpeg 二进制不存在时 open() 返回 False（调用方跳过声音链）。"""
    stream = AudioStream("unused", ffmpeg_bin=["no_such_ffmpeg_binary_xyz"])
    try:
        assert stream.open() is False
    finally:
        stream.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="需要 ffmpeg")
def test_audio_stream_with_real_ffmpeg(tmp_path):
    """真实 ffmpeg 集成：合成 wav 的 PCM 直通块总量 ≈ 原始样本数。"""
    sr = 16000
    wav_path = tmp_path / "audio.wav"
    data = (_make_samples(2.0, sr) * 32767).astype(np.int16)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())

    stream = AudioStream(str(wav_path), sr=sr, chunk_sec=0.5)
    assert stream.open() is True
    total = 0
    while True:
        chunk = stream.read_chunk(timeout=10.0)
        if chunk is None:
            break
        assert chunk.dtype == np.float32
        assert np.abs(chunk).max() <= 1.0
        total += len(chunk)
    stream.close()
    assert abs(total - 2 * sr) < 200  # 重采样边界容差


def test_fake_ffmpeg_script_sanity(tmp_path):
    """防呆：确认 fake ffmpeg 脚本本身可独立运行（测试基建自检）。"""
    cmd = _fake_ffmpeg_cmd(tmp_path)
    r = subprocess.run([*cmd, "--chunks", "1", "--samples", "8"],
                       capture_output=True, timeout=30)
    assert r.returncode == 0
    assert len(r.stdout) == 16  # 8 样本 × 2 字节
