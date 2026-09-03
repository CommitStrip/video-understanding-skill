#!/usr/bin/env python3
"""
audio_source.py - 直播音频链：ffmpeg 子进程 → 环形缓冲 → 定长样本块
===================================================================
W8a：直播源此前完全没有音频链（文件管线靠 ffmpeg 对整文件抽 wav 后批式喂）。
这里用 ffmpeg 把任意源（视频文件 / RTSP 流）的音频解码为 16kHz 单声道
s16le PCM 直通管道：读线程后台灌入缓冲，消费方按 chunk_sec 取块喂
StreamingASR——与文件路径共用同一解码状态机。

realtime=True 时取块按挂钟节拍回放（文件仿真实时的音频侧对应物，与
FileSource(realtime=True) 的画面侧节拍对齐），保证 ASR 时间轴与画面链
同步推进，而不是让 ASR 以解码速度抢跑。

ffmpeg 不可用 / 源无音轨时 open() 返回 False 或 read_chunk() 返回 None，
调用方据此关闭声音链——画面链与理解层主链不受影响。
"""

import subprocess
import threading
import time

import numpy as np


class AudioStream:
    """ffmpeg 音频直通流。

    read_chunk() 返回约 chunk_sec 秒的 float32 样本（-1..1）；
    流结束或出错返回 None。ffmpeg_bin 允许传 list 前缀（测试时注入
    [sys.executable, fake_script] 伪造音频源）。
    """

    def __init__(self, src, sr=16000, chunk_sec=2.0, realtime=False,
                 ffmpeg_bin='ffmpeg', extra_input_args=None):
        self.src = src
        self.sr = sr
        self.chunk_sec = chunk_sec
        self.chunk_bytes = max(2, int(sr * chunk_sec)) * 2  # s16le 每样本 2 字节
        self.realtime = bool(realtime)
        self._cmd_prefix = ([ffmpeg_bin] if isinstance(ffmpeg_bin, str)
                            else list(ffmpeg_bin))
        self.extra_input_args = list(extra_input_args or [])
        self.stats = {"bytes_read": 0, "chunks_read": 0}
        self._proc = None
        self._reader = None
        self._buf = bytearray()
        self._cond = threading.Condition()
        self._eof = False
        self._t0 = 0.0

    def open(self):
        """启动 ffmpeg 子进程与后台读线程；启动失败返回 False。"""
        cmd = [*self._cmd_prefix, *self.extra_input_args,
               '-i', self.src, '-vn', '-ac', '1', '-ar', str(self.sr),
               '-f', 's16le', 'pipe:1']
        try:
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.DEVNULL)
        except OSError:
            return False
        self._t0 = time.monotonic()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        return True

    def _read_loop(self):
        proc = self._proc
        try:
            while True:
                data = proc.stdout.read(8192)
                if not data:
                    break
                with self._cond:
                    self._buf.extend(data)
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._eof = True
                self._cond.notify_all()

    def read_chunk(self, timeout=None):
        """取约 chunk_sec 秒样本；EOF 且缓冲耗尽返回 None。

        timeout 秒内凑不满一块（且流未结束）返回 None（调用方可重试）。
        realtime=True 时按已取块数对齐挂钟节拍：消费快于实时则 sleep，
        消费慢于实时则不等待（天然背压，滞后由遥测暴露）。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        raw = None
        with self._cond:
            while True:
                if len(self._buf) >= self.chunk_bytes:
                    raw = bytes(self._buf[:self.chunk_bytes])
                    del self._buf[:self.chunk_bytes]
                    break
                if self._eof:
                    if self._buf:  # EOF 残余不足一块：作为最后一块发出
                        raw = bytes(self._buf)
                        self._buf.clear()
                    break
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._cond.wait(remaining)
                else:
                    self._cond.wait(0.5)
        if raw is None:
            return None
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        self.stats["bytes_read"] += len(raw)
        self.stats["chunks_read"] += 1
        if self.realtime:
            self._pace()
        return samples

    def _pace(self):
        # 目标：第 N 块的取用时刻 ≈ open 时刻 + N*chunk_sec；允许提前 50ms
        # 喂入以吸收调度抖动。消费落后时 lag<0，不 sleep（背压丢给下游遥测）。
        target = self._t0 + self.stats["chunks_read"] * self.chunk_sec
        lag = target - time.monotonic()
        if lag > 0.05:
            time.sleep(lag - 0.05)

    def close(self):
        """终止子进程并回收读线程（幂等）。"""
        if self._proc is not None:
            try:
                self._proc.terminate()
            except OSError:
                pass
            self._proc = None
        if self._reader is not None:
            self._reader.join(timeout=2.0)
            self._reader = None
