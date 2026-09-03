#!/usr/bin/env python3
"""
events.py - 线程安全有界 EventBus（W8 实时理解层的事件主干）
============================================================
三类生产者（画面链帧循环 / ASR 线程 / 理解层自身）把事件 publish 进来，
若干订阅者（理解 worker、SSE 服务）各自持有一条独立的有界队列。

背压语义（实时链路的生命线）：
  - publish 永不阻塞（帧循环 1.6ms/帧的预算绝不能被慢消费者拖住）；
  - 单个订阅者队列满时丢最旧事件并计数（丢弃可观测，见 dropped）；
  - 事件一律是 dict（JSON 可序列化，SSE 可直接下发）。
"""

import threading
from collections import deque


class Subscription:
    """单个订阅者的有界队列（drop-oldest）。"""

    def __init__(self, maxsize=4096):
        self.maxsize = int(maxsize)
        self.dropped = 0
        self._q = deque(maxlen=maxsize) if maxsize > 0 else []
        self._cond = threading.Condition()

    def _put(self, ev):
        with self._cond:
            if self.maxsize > 0 and len(self._q) >= self.maxsize:
                self.dropped += 1
            self._q.append(ev)
            self._cond.notify_all()

    def get(self, timeout=None):
        """阻塞取一条事件；timeout 秒无事件返回 None。"""
        with self._cond:
            if not self._q and timeout != 0:
                self._cond.wait(timeout)
            if self._q:
                return self._q.popleft() if self.maxsize > 0 else self._q.pop(0)
            return None

    def drain(self, timeout=None):
        """取走当前积压的全部事件；timeout>0 且队列为空时先等到至少一条（或超时）。"""
        with self._cond:
            if timeout and not self._q:
                self._cond.wait(timeout)
            out = list(self._q)
            if self.maxsize > 0:
                self._q.clear()
            else:
                del self._q[:]
            return out

    def __len__(self):
        with self._cond:
            return len(self._q)


class EventBus:
    """扇出总线：publish 非阻塞，订阅者互不影响。"""

    def __init__(self, sub_maxsize=4096):
        self.sub_maxsize = int(sub_maxsize)
        self.published = 0
        self._subs = []
        self._lock = threading.Lock()

    def subscribe(self, maxsize=None):
        """注册一个订阅者（通常在起生产者之前完成）。"""
        sub = Subscription(self.sub_maxsize if maxsize is None else maxsize)
        with self._lock:
            self._subs.append(sub)
        return sub

    def publish(self, event):
        """向所有订阅者扇出一条事件（dict）。永不阻塞、永不抛出。"""
        with self._lock:
            subs = list(self._subs)
        self.published += 1
        for sub in subs:
            sub._put(event)

    @property
    def subscriber_count(self):
        with self._lock:
            return len(self._subs)
