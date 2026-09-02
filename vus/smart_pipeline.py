#!/usr/bin/env python3
"""
smart_pipeline.py - 实时预算分配架构 · 画面链
================================================
设计目标：能实时运行在机器人 / 实时产品上。

核心思想（生物快慢双通路）：
  快系统：每帧跑，最轻。只在降采样小图上做帧差法，语义门控 + 状态机流式发射运动事件。
  慢系统：低频(1-2Hz) + 触发式。仅在快系统报有内容 / 强制兜底时才抽关键帧，保住静止语义。
  输出：流式增量。process_frame 每帧返回可立即消费的事件列表，宿主无需等整段。

实时预算分配：
  快系统  占 90% 预算，每帧执行，保证跟随视频帧率。
  慢系统  占 10% 预算，低频插空执行，不拖累快系统。
"""

import cv2
import os
import numpy as np
import json
import time
from collections import deque

from .io_utils import write_json


class SmartPipeline:
    """
    实时预算分配架构的画面链。

    用法（流式）:
        pipe = SmartPipeline()
        for frame, ts in stream:
            for ev in pipe.process_frame(frame, ts):
                host.consume(ev)   # 立即消费，无需等整段
    """

    def __init__(self, config=None):
        cfg = config or {}
        # === 快系统参数（每帧，最轻）===
        self.fast_scale = cfg.get('fast_scale', 0.25)           # 降采样比例，0.25 ≈ 1/16 像素
        self.motion_thresh = cfg.get('motion_thresh', 25)       # 帧差阈值
        self.min_area_ratio = cfg.get('min_area_ratio', 0.001)  # 最小运动面积占小图比例
        self.semantic_gate_ratio = cfg.get('semantic_gate_ratio', 0.003)  # 语义门控（占小图比例）
        self.motion_confirm_frames = cfg.get('motion_confirm_frames', 3)  # 运动起始需连续帧
        self.motion_window = cfg.get('motion_window', 4)  # 时间累积窗口：对比 N 帧前的参考帧捕捉慢速运动

        # === 慢系统参数（低频 + 触发式）===
        self.keyframe_interval_hz = cfg.get('keyframe_interval_hz', 1.5)   # 抽取频率 1.5Hz
        self.keyframe_diff = cfg.get('keyframe_diff', 10)                  # 内容变化打分阈值
        self.phash_threshold = cfg.get('phash_threshold', 12)              # pHash 距离阈值
        self.hist_threshold = cfg.get('hist_threshold', 0.5)               # 直方图相关阈值
        self.dedup_threshold = cfg.get('dedup_threshold', 5)               # 去重 hamming 距离

        # === 快系统状态 ===
        self.prev_small = None
        self._gray_hist = deque(maxlen=max(self.motion_window, 2))  # 时间窗口：慢速运动参考
        self.motion_active = False
        self.motion_confirm = 0
        self.motion_start_t = None
        self.current_segment = None
        self.fast_small_size = None

        # === 慢系统状态 ===
        self.prev_kf_gray = None
        self.prev_phash = None
        self.prev_hist = None
        self.kf_hashes = []
        self.last_keyframe_t = -1.0

        # === 输出（流式累积）===
        self.events = []
        self.motion_segments = []
        self.keyframes = []

        self.frame_count = 0
        self.process_times = deque(maxlen=200)

    # ==================== 主入口：流式 ====================

    def process_frame(self, frame, timestamp, fps=None):
        """
        处理一帧，返回本帧产生的流式事件列表（宿主可立即消费）。

        实时预算：快系统每帧跑（最轻），慢系统按频率+触发插空跑。
        """
        t_start = time.time()

        # 共用降采样小图（快/慢系统复用，省算力）
        if self.fast_scale != 1.0:
            h0 = max(int(frame.shape[0] * self.fast_scale), 8)
            w0 = max(int(frame.shape[1] * self.fast_scale), 8)
            small_color = cv2.resize(frame, (w0, h0), interpolation=cv2.INTER_AREA)
        else:
            small_color = frame
        small_gray = cv2.cvtColor(small_color, cv2.COLOR_BGR2GRAY)
        self.fast_small_size = small_gray.shape

        out_events = []

        # === 快系统：每帧帧差法（最轻，占主要预算）===
        if self.prev_small is not None:
            out_events += self._fast_motion(small_gray, timestamp)
        self.prev_small = small_gray
        self._gray_hist.append(small_gray)

        # === 慢系统：低频 + 触发式关键帧（插空跑）===
        if self._should_check_keyframe(timestamp):
            kf = self._slow_keyframe(small_gray, small_color, timestamp)
            if kf:
                out_events.append(kf)

        # 流式返回 + 累积到历史（供摘要 / 对齐层使用）
        if out_events:
            self.events.extend(out_events)

        self.frame_count += 1
        self.process_times.append(time.time() - t_start)

        return out_events

    # ==================== 快系统 ====================

    def _fast_motion(self, gray, timestamp):
        """快系统：帧差法 + 时间累积窗口 + 语义门控 + 状态机流式事件发射（全部在降采样小图上）"""
        events = []
        h, w = gray.shape

        # 相邻帧差（捕捉快速运动）
        diff = cv2.absdiff(self.prev_small, gray)
        _, mask = cv2.threshold(diff, self.motion_thresh, 255, cv2.THRESH_BINARY)

        # 时间累积窗口：对比窗口内最早的参考帧（捕捉慢速运动）
        if len(self._gray_hist) >= self.motion_window:
            ref_old = self._gray_hist[0]
            diff_old = cv2.absdiff(ref_old, gray)
            _, mask_old = cv2.threshold(diff_old, self.motion_thresh, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(mask, mask_old)

        # 小图上形态学（成本低）
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, None, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        total_area = 0
        min_area_px = self.min_area_ratio * h * w
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area_px:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            boxes.append({
                "bbox": [int(x), int(y), int(cw), int(ch)],
                "centroid": [int(x + cw // 2), int(y + ch // 2)],
                "area": int(area)
            })
            total_area += area

        motion_ratio = total_area / (h * w) if h * w > 0 else 0

        if boxes:
            # 语义门控：运动区域占比过小（如说话头部微动）视为低语义，不发事件
            max_box_area = max(b["area"] for b in boxes)
            if max_box_area < self.semantic_gate_ratio * h * w:
                self.motion_confirm = 0
                return events

            self.motion_confirm += 1
            if self.motion_confirm >= self.motion_confirm_frames and not self.motion_active:
                # 运动开始
                self.motion_active = True
                self.motion_start_t = timestamp
                self.current_segment = {
                    "start": round(timestamp, 3),
                    "end": round(timestamp, 3),
                    "max_ratio": round(motion_ratio, 4)
                }
                events.append({
                    "type": "motion_start",
                    "t": round(timestamp, 3),
                    "boxes": boxes,
                    "motion_ratio": round(motion_ratio, 4)
                })
            elif self.motion_active:
                # 运动持续
                if self.current_segment:
                    self.current_segment["end"] = round(timestamp, 3)
                    self.current_segment["max_ratio"] = max(
                        self.current_segment["max_ratio"], round(motion_ratio, 4))
                events.append({
                    "type": "motion",
                    "t": round(timestamp, 3),
                    "boxes": boxes,
                    "motion_ratio": round(motion_ratio, 4)
                })
        else:
            # 无有效运动
            self.motion_confirm = 0
            if self.motion_active:
                self.motion_active = False
                if self.current_segment:
                    duration = self.current_segment["end"] - self.current_segment["start"]
                    if duration >= 0.5:
                        self.motion_segments.append(self.current_segment)
                    events.append({
                        "type": "motion_end",
                        "t": round(timestamp, 3),
                        "duration": round(duration, 3)
                    })
                self.current_segment = None

        return events

    def _should_check_keyframe(self, timestamp):
        """慢系统触发决策：低频 + 快系统报有内容"""
        if self.last_keyframe_t < 0:
            return True  # 第一帧
        # 至少间隔 1/interval_hz 秒
        if timestamp - self.last_keyframe_t < 1.0 / self.keyframe_interval_hz:
            return False
        # 要么距上次足够久（强制兜底），要么当前有运动
        forced = (timestamp - self.last_keyframe_t) > (3.0 / self.keyframe_interval_hz)
        has_motion = self.motion_active
        return forced or has_motion

    # ==================== 慢系统 ====================

    def _slow_keyframe(self, gray, color, timestamp):
        """慢系统：内容变化打分 + pHash + 直方图 + 去重（低频触发）"""
        if self.prev_kf_gray is None:
            self.prev_kf_gray = gray.copy()
            self.prev_phash = self._compute_phash(gray)
            self.prev_hist = self._compute_hist(color)
            self.kf_hashes.append(self.prev_phash)
            self.last_keyframe_t = timestamp
            self.keyframes.append({
                "t": round(timestamp, 3),
                "frame_idx": self.frame_count,
                "reason": "first_frame"
            })
            return {"type": "keyframe", "t": round(timestamp, 3), "reason": "first_frame"}

        diff = cv2.absdiff(self.prev_kf_gray, gray)
        score = float(np.mean(diff))
        if score <= self.keyframe_diff:
            return None

        phash = self._compute_phash(gray)
        hist = self._compute_hist(color)
        phash_dist = self._hamming_distance(phash, self.prev_phash)
        hist_corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))

        is_new_scene = phash_dist > self.phash_threshold or hist_corr < self.hist_threshold

        # 去重：与最近关键帧比较
        is_duplicate = False
        for existing_hash in self.kf_hashes[-5:]:
            if self._hamming_distance(phash, existing_hash) < self.dedup_threshold:
                is_duplicate = True
                break

        if is_new_scene and not is_duplicate:
            self.prev_kf_gray = gray.copy()
            self.prev_phash = phash
            self.prev_hist = hist
            self.kf_hashes.append(phash)
            self.last_keyframe_t = timestamp
            self.keyframes.append({
                "t": round(timestamp, 3),
                "frame_idx": self.frame_count,
                "score": round(score, 2),
                "phash_dist": int(phash_dist),
                "hist_corr": round(hist_corr, 3)
            })
            return {
                "type": "keyframe",
                "t": round(timestamp, 3),
                "score": round(score, 2),
                "phash_dist": int(phash_dist),
                "hist_corr": round(hist_corr, 3)
            }
        return None

    # ==================== 特征计算 ====================

    def _compute_phash(self, gray):
        """感知哈希（pHash）：DCT低频系数二值化"""
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(small))
        dct_low = dct[:8, :8]
        mean = np.mean(dct_low[1:])  # 排除 DC 分量
        return (dct_low > mean).flatten()

    def _compute_hist(self, color):
        """色彩直方图（HSV，8x8x8 bins）"""
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _hamming_distance(h1, h2):
        return int(np.sum(h1 != h2))

    # ==================== 流式对齐层 ====================

    def align_asr_streaming(self, segments):
        """
        时间轴对齐融合：流式 ASR 增量段 + 画面事件（流式消费友好）
        输入: [{"t": float, "text": str}, ...]
        输出: [{"start","end","text","linked_motion_events","linked_keyframes","motion_types"}, ...]
        """
        motion_events = [e for e in self.events if e["type"].startswith("motion")]
        keyframe_events = [e for e in self.events if e["type"] == "keyframe"]

        aligned = []
        for i, seg in enumerate(segments):
            t0 = seg["t"]
            t1 = segments[i + 1]["t"] if i + 1 < len(segments) else t0 + 2.0
            # 左闭右开 [t0, t1)：事件恰好落在段边界时只归后一段，避免重复计数
            linked_motion = [e for e in motion_events if t0 <= e["t"] < t1]
            linked_kf = [k for k in keyframe_events if t0 <= k["t"] < t1]
            aligned.append({
                "start": round(t0, 2),
                "end": round(t1, 2),
                "text": seg["text"],
                "linked_motion_events": len(linked_motion),
                "linked_keyframes": len(linked_kf),
                "motion_types": list(set(e["type"] for e in linked_motion))
            })
        return aligned

    # ==================== 摘要与输出 ====================

    def get_summary(self):
        avg_process = np.mean(self.process_times) if self.process_times else 0
        avg_fps = 1.0 / avg_process if avg_process > 0 else 0
        return {
            "total_frames": self.frame_count,
            "motion_events": len([e for e in self.events if e["type"].startswith("motion")]),
            "motion_segments": len(self.motion_segments),
            "keyframes": len(self.keyframes),
            "total_events": len(self.events),
            "avg_process_time_ms": round(avg_process * 1000, 2),
            "avg_fps": round(avg_fps, 1),
            "fast_scale": self.fast_scale,
            "keyframe_interval_hz": self.keyframe_interval_hz
        }

    def build_results(self):
        """构造结果 dict（不落盘），供宿主自行写入或再加工。"""
        return {
            "summary": self.get_summary(),
            "motion_segments": self.motion_segments,
            "keyframes": self.keyframes,
            "events_summary": {
                "motion_start": len([e for e in self.events if e["type"] == "motion_start"]),
                "motion": len([e for e in self.events if e["type"] == "motion"]),
                "motion_end": len([e for e in self.events if e["type"] == "motion_end"]),
                "keyframe": len([e for e in self.events if e["type"] == "keyframe"]),
            }
        }

    def save_results(self, output_path):
        results = self.build_results()
        write_json(os.path.dirname(output_path) or '.', os.path.basename(output_path), results)
        return results