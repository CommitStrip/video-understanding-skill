#!/usr/bin/env python3
"""
scheduler.py - 统一预算调度器（W9 借鉴外部审查：多信号 priority + 最低保障）
============================================================================
把视觉/ASR/OCR 的信号统一提交给一个调度器，从"各自独立生成然后对齐"
升级为"跨模态信号共同决定什么值得保留"。

多信号 priority 评分 + 最低保障规则 + 事件级 provenance。
"""

import time
from collections import deque


class BudgetScheduler:
    """统一预算调度器：多信号优先级 + 最低保障规则。

    接口:
      submit(signal_type, weight, timestamp) → None  # 多通道提交信号
      should_retain(now) → bool                      # 当前是否值得保留/调用
      guaranteed_anchor_due(now) → bool              # 最低保障锚点是否到期
    """

    def __init__(self, config=None):
        cfg = config or {}
        # 多信号权重（默认值可由 config 覆盖）
        self.weight_visual = cfg.get("weight_visual", 1.0)
        self.weight_speech = cfg.get("weight_speech", 0.8)
        self.weight_ocr = cfg.get("weight_ocr", 0.6)
        self.weight_attention = cfg.get("weight_attention", 1.2)
        # 最低保障
        self.anchor_interval_s = cfg.get("anchor_interval_s", 30.0)
        # 阈值
        self.retain_threshold = cfg.get("retain_threshold", 1.5)

        # 信号窗口（最近 30s 内的信号记录）
        self._signals = deque(maxlen=200)

    def submit(self, signal_type, weight, timestamp=None):
        """多通道提交信号。signal_type ∈ visual/speech/ocr/attention。"""
        t = timestamp if timestamp is not None else time.time()
        self._signals.append({"type": signal_type, "weight": weight, "t": t})

    def _window_score(self, now, window_s=30.0):
        """最近 window_s 秒内的加权信号分数。"""
        cutoff = now - window_s
        total = 0.0
        for s in self._signals:
            if s["t"] >= cutoff:
                w = getattr(self, f"weight_{s['type']}", 1.0)
                total += s["weight"] * w
        return total

    def should_retain(self, now=None, window_s=30.0):
        """当前窗口内信号分是否超过保留阈值。"""
        now = now if now is not None else time.time()
        return self._window_score(now, window_s) >= self.retain_threshold

    def guaranteed_anchor_due(self, now, last_anchor_t=None):
        """最低保障锚点是否到期（每 anchor_interval_s 至少 1 帧，防全零信号漏帧）。"""
        now = now if now is not None else time.time()
        if last_anchor_t is None:
            return True
        return (now - last_anchor_t) >= self.anchor_interval_s

    def provenance_id(self, timestamp):
        """事件级 provenance ID：时间戳 + 信号类型指纹。"""
        import hashlib
        types = sorted(set(s["type"] for s in self._signals))
        raw = f"{timestamp:.3f}|{'|'.join(types)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:12]
