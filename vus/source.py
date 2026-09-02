#!/usr/bin/env python3
"""
source.py - 统一帧源抽象（文件 / 摄像头 / RTSP）
================================================
W2 实时源波次：把取帧逻辑从「文件顺序读」抽象为统一 FrameSource 接口，
让 integrated_pipeline 支持真正的实时源：

  FileSource   本地视频文件。realtime=False 尽快读（回放分析，旧行为）；
               realtime=True 按视频 fps 节拍喂帧（time.sleep 节流，供测试与演示）。
  CameraSource 本地摄像头。时间戳 = time.monotonic()。
  RTSPSource   RTSP 网络流（OpenCV 自带 FFmpeg/RTSP 支持），两大关键增强：
               - 背压取最新帧：后台读线程持续 grab() 只保留最新一帧，
                 消费慢时旧帧被丢弃（frames_dropped 统计），处理链永远拿到最新画面；
               - 断流重连：grab 失败按 reconnect_delay 重试。
                 max_reconnects=0 表示无限重连，N 表示最多 N 次。

时间戳契约：read() 返回 (ok, frame, timestamp)，timestamp 为单调秒——
文件源 = frame_idx / fps（回放时间轴）；直播源 = time.monotonic()（直播语义）。

纯 OpenCV 实现，无新依赖。构造参数全带默认值；打开失败 open() 返回 False
（不抛异常），错误信息写入 stats["last_error"]。
"""

from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np


class FrameSource(ABC):
    """帧源统一接口。read() 返回 (ok, frame, timestamp)；timestamp 单调秒。"""

    #: 直播源（Camera/RTSP）为 True：事件时间戳直接用 read() 返回的单调时钟；
    #: 文件源为 False：回放分析仍用 frame_idx / fps 时间轴。
    live = False

    @abstractmethod
    def open(self) -> bool:
        """打开帧源。失败返回 False（不抛异常），错误信息写入 stats。"""

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        """读一帧，返回 (ok, frame, timestamp)；ok=False 时 frame 为 None。"""

    @abstractmethod
    def close(self) -> None:
        """释放资源。可安全重复调用。"""

    @property
    @abstractmethod
    def stats(self) -> dict:
        """统计快照：frames_read / frames_dropped / reconnects / source 描述。"""


class FileSource(FrameSource):
    """本地视频文件源。

    realtime=False: 尽快顺序读（回放分析，等价旧行为）。
    realtime=True : 按视频 fps * speed 的节拍喂帧（time.sleep 节流），
                    模拟实时源，供演示与测试。fps 未知时按 25fps 节拍。
    时间戳 = frame_idx / fps（fps 未知时回退 frame_idx * 0.02）。
    """

    # fps 未知（CAP_PROP_FPS=0）时 realtime 节拍的兜底帧率
    FALLBACK_FPS = 25.0

    def __init__(self, video_path, realtime: bool = False, speed: float = 1.0):
        self.video_path = video_path
        self.realtime = bool(realtime)
        self.speed = max(float(speed), 0.01)
        self._cap = None
        self._fps = 0.0
        self._total = 0
        self._width = 0
        self._height = 0
        self._frame_idx = 0
        self._frames_read = 0
        self._last_error = None
        self._t0 = None

    def open(self) -> bool:
        self._last_error = None
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self._last_error = f"无法打开视频文件: {self.video_path}"
            return False
        self._cap = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_idx = 0
        self._frames_read = 0
        self._t0 = time.monotonic()
        return True

    # 文件源独有的元信息（integrated_pipeline 打印规格 / 计算进度用）
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def _timestamp(self) -> float:
        return self._frame_idx / self._fps if self._fps > 0 else self._frame_idx * 0.02

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        if self._cap is None:
            return False, None, 0.0
        if self.realtime:
            # 第 i 帧应在 t0 + i/(fps*speed) 送达；落后于节拍则不 sleep
            fps = self._fps if self._fps > 0 else self.FALLBACK_FPS
            period = 1.0 / (fps * self.speed)
            delay = (self._t0 + self._frame_idx * period) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        ok, frame = self._cap.read()
        if not ok:
            return False, None, self._timestamp()
        ts = self._timestamp()
        self._frame_idx += 1
        self._frames_read += 1
        return True, frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def stats(self) -> dict:
        return {
            "source": f"file:{self.video_path}",
            "live": False,
            "frames_read": self._frames_read,
            "frames_dropped": 0,
            "reconnects": 0,
            "last_error": self._last_error,
        }


class CameraSource(FrameSource):
    """本地摄像头源。时间戳 = time.monotonic()（单调时钟，直播语义）。"""

    def __init__(self, camera_index: int = 0,
                 width: Optional[int] = None, height: Optional[int] = None):
        self.camera_index = int(camera_index)
        self.width = width
        self.height = height
        self._cap = None
        self._frames_read = 0
        self._last_error = None

    def open(self) -> bool:
        self._last_error = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
        except Exception as e:  # 个别后端可能抛异常，统一降级为 False
            self._last_error = f"打开摄像头异常: {e}"
            return False
        if not cap.isOpened():
            self._last_error = f"无法打开摄像头: {self.camera_index}"
            return False
        if self.width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        self._cap = cap
        self._frames_read = 0
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        ts = time.monotonic()
        if self._cap is None:
            return False, None, ts
        ok, frame = self._cap.read()
        ts = time.monotonic()
        if not ok:
            return False, None, ts
        self._frames_read += 1
        return True, frame, ts

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def stats(self) -> dict:
        return {
            "source": f"camera:{self.camera_index}",
            "live": True,
            "frames_read": self._frames_read,
            "frames_dropped": 0,
            "reconnects": 0,
            "last_error": self._last_error,
        }


class RTSPSource(FrameSource):
    """RTSP 网络流源：背压取最新帧 + 断流重连。

    背压：后台读线程持续 grab()，只保留最新一帧到帧槽；消费慢时旧帧被
    丢弃并计入 frames_dropped。read()（主线程调用）取走槽内最新帧，
    全程用 threading.Lock 保护（线程安全）。

    重连：grab/retrieve 失败视为断流，按 reconnect_delay 等待后重连；
    max_reconnects=0 无限重连，N 表示最多 N 次，耗尽后 read() 返回
    (False, None, ts)（stats["last_error"] 记录原因）。

    时间戳 = time.monotonic()（单调时钟，直播语义）。
    """

    # read() 无新帧时的条件变量等待分片（秒）：保证 Ctrl-C 能及时打断
    WAIT_SLICE = 0.1
    # close() 等待读线程退出的超时（秒）
    JOIN_TIMEOUT = 2.0

    def __init__(self, url: str, reconnect_delay: float = 2.0,
                 max_reconnects: int = 0,
                 open_timeout_ms: int = 10000, read_timeout_ms: int = 10000):
        self.url = url
        self.reconnect_delay = max(float(reconnect_delay), 0.0)
        self.max_reconnects = max(int(max_reconnects), 0)
        self.open_timeout_ms = int(open_timeout_ms)
        self.read_timeout_ms = int(read_timeout_ms)

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap = None

        self._latest: Optional[np.ndarray] = None
        self._has_frame = False
        self._down = False          # True: 重连耗尽，不再有新帧
        self._opened = False

        self._frames_read = 0
        self._frames_dropped = 0
        self._reconnects = 0
        self._last_error = None

    # ---------- 底层捕获 ----------

    def _open_capture(self):
        """创建底层 VideoCapture（测试可覆写本方法伪造帧源）。

        成功返回已打开的 capture，失败返回 None。
        """
        params = []
        if self.open_timeout_ms and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms]
        if self.read_timeout_ms and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms]
        try:
            if params:
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
            else:
                cap = cv2.VideoCapture(self.url)
        except Exception as e:
            self._last_error = f"打开 RTSP 异常: {e}"
            return None
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            self._last_error = f"无法连接 RTSP 流: {self.url}"
            return None
        return cap

    def open(self) -> bool:
        with self._lock:
            self._last_error = None
            self._down = False
            self._has_frame = False
            self._latest = None
        cap = self._open_capture()
        if cap is None:
            with self._cond:
                self._last_error = self._last_error or f"无法连接 RTSP 流: {self.url}"
                self._down = True
            return False
        self._cap = cap
        self._stop.clear()
        self._opened = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="rtsp-reader", daemon=True)
        self._thread.start()
        return True

    # ---------- 后台读线程：背压取最新帧 + 断流重连 ----------

    def _reader_loop(self):
        cap = self._cap
        while not self._stop.is_set():
            ok = False
            if cap is not None:
                try:
                    ok = bool(cap.grab())
                except Exception:
                    ok = False
            if ok:
                try:
                    got, frame = cap.retrieve()
                except Exception:
                    got, frame = False, None
                if got and frame is not None:
                    with self._cond:
                        if self._has_frame:
                            self._frames_dropped += 1  # 槽内旧帧未被消费即被覆盖
                        self._latest = frame
                        self._has_frame = True
                        self._cond.notify_all()
                    continue

            # ---- grab/retrieve 失败：断流，走重连 ----
            if self._stop.is_set():
                break
            with self._lock:
                exhausted = 0 < self.max_reconnects <= self._reconnects
            if exhausted:
                with self._cond:
                    self._last_error = (f"重连次数耗尽({self.max_reconnects})，"
                                        f"放弃: {self.url}")
                    self._down = True
                    self._cond.notify_all()
                break

            with self._lock:
                self._reconnects += 1
            if cap is not None:
                with self._lock:
                    self._cap = None
                try:
                    cap.release()  # 在读线程内释放，避免与 grab 并发
                except Exception:
                    pass
                cap = None
            self._wait_reconnect_delay()
            if self._stop.is_set():
                break
            new_cap = self._open_capture()
            if new_cap is None:
                continue  # 下一轮按断流处理（受 max_reconnects 总量约束）
            cap = new_cap
            with self._lock:
                self._cap = cap  # 回写：close() 在线程退出后释放当前 capture
            with self._cond:
                self._latest = None
                self._has_frame = False

    def _wait_reconnect_delay(self):
        """等待 reconnect_delay（分片等待，随时响应 stop）。"""
        if self.reconnect_delay <= 0:
            return
        deadline = time.monotonic() + self.reconnect_delay
        while not self._stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            if self._stop.wait(min(remaining, self.WAIT_SLICE)):
                return

    # ---------- 主线程接口 ----------

    def read(self) -> tuple[bool, Optional[np.ndarray], float]:
        if not self._opened:
            return False, None, time.monotonic()
        with self._cond:
            # 等新帧 / 断流终态；分片等待保证 Ctrl-C 可打断
            while not self._has_frame and not self._down and not self._stop.is_set():
                self._cond.wait(self.WAIT_SLICE)
            if self._has_frame:
                frame = self._latest
                self._latest = None
                self._has_frame = False
                self._frames_read += 1
                return True, frame, time.monotonic()
            return False, None, time.monotonic()

    def close(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=self.JOIN_TIMEOUT)
        self._thread = None
        with self._cond:
            if self._has_frame:
                self._frames_dropped += 1  # 未消费的最新帧计为丢弃（账目守恒）
            self._latest = None
            self._has_frame = False
            self._down = True
            self._cond.notify_all()
        # 仅在读线程已退出时才 release，避免与阻塞中的 grab() 并发释放
        cap, self._cap = self._cap, None
        self._opened = False
        if cap is not None and (t is None or not t.is_alive()):
            try:
                cap.release()
            except Exception:
                pass

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "source": f"rtsp:{self.url}",
                "live": True,
                "frames_read": self._frames_read,
                "frames_dropped": self._frames_dropped,
                "reconnects": self._reconnects,
                "max_reconnects": self.max_reconnects,
                "last_error": self._last_error,
            }
