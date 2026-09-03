#!/usr/bin/env python3
"""
state.py - SessionState：四层理解状态的单一事实源 + 滚动落盘
============================================================
理解 worker / 标签道 / 画面链事件 / SSE 服务都读写这一个对象（线程安全）。
快照结构（live_state.json，机器可读）：

  session   会话元信息（起始墙钟、当前时间轴、帧计数）
  t0        帧级反射层（最近运动事件/段计数）
  t05       语义标签环（最近 N 条标签，毫秒级）
  asr       最近语音段环
  t2        滚动理解（now 摘要 / timeline / chapters / entities，秒级有界滞后）
  telemetry 分层遥测（lag / 调用次数 / 延迟 / 丢弃）

live_context.md（人/agent 可读）由 markdown() 生成，与 SKILL.md 的
"读压缩产物理解视频"工作流对齐——外部 agent 任何时刻读文件即得当前理解。

落盘用 write_*_atomic（tmp+rename）：高频重写下读者永远看不到半截文件。
"""

import threading
import time

from ..io_utils import write_json_atomic, write_text_atomic

#: timeline 超过 max_timeline 时把最旧的并入一条章节摘要（无 VLM，纯文本合并）
DEFAULT_MAX_TIMELINE = 40
LABEL_RING = 50
ASR_RING = 30


class SessionState:
    """实时理解会话状态（线程安全）。时间一律为流时间轴秒（单调）。"""

    def __init__(self, source_desc="", max_timeline=DEFAULT_MAX_TIMELINE):
        self._lock = threading.Lock()
        self._source_desc = source_desc
        self._max_timeline = int(max_timeline)
        self._t_start_wall = time.time()

        self._t_now = 0.0
        self._frame_count = 0
        self._last_motion = None
        self._motion_segments_n = 0
        self._labels = []           # 环：T0.5 标签
        self._asr = []              # 环：最近 ASR 段
        self._labels_max = LABEL_RING
        self._asr_max = ASR_RING
        self._t2 = {
            "now": "",
            "timeline": [],          # [{"start","end","summary"}]
            "chapters": [],          # 压缩后的粗粒度章节（无界直播的防膨胀层）
            "entities": {},          # {"名称": "说明"}
            "understood_t": None,    # 最近一次调用覆盖到的素材末时间（算滞后用）
            "model": None,
            "updated_at": 0.0,
        }
        self._telemetry = {
            "t2_calls": 0, "t2_errors": 0, "t2_last_latency_s": None,
            "t2_last_error": None, "t2_dropped_windows": 0,
            "t05_labels": 0, "t05_avg_ms": None,
            "events_consumed": 0,
        }

    # ---------- 各生产线的写入入口（被对应线程调用） ----------

    def apply_frame_event(self, ev):
        """T0 层：画面链事件（motion_*/keyframe）推进时钟与反射状态。"""
        with self._lock:
            self._t_now = max(self._t_now, float(ev.get("t", 0.0)))
            self._frame_count += 1
            self._telemetry["events_consumed"] += 1
            ev_type = ev.get("type", "")
            if ev_type == "motion_end":
                self._motion_segments_n += 1
                self._last_motion = {
                    "start": ev.get("segment_start"), "end": ev.get("t"),
                    "duration": ev.get("duration"),
                }
            elif ev_type in ("motion_start", "motion"):
                self._last_motion = {"t": ev.get("t"), "active": True}

    def apply_asr(self, seg):
        """语音链：清洗后的 ASR 终稿段进入展示环。"""
        with self._lock:
            self._t_now = max(self._t_now, float(seg.get("t", 0.0)))
            self._append_ring(self._asr, self._asr_max, {
                "t": seg.get("t"), "end_t": seg.get("end_t"),
                "text": seg.get("text", ""),
                **({"hallucination": True} if seg.get("hallucination") else {}),
            })

    def apply_label(self, entry):
        """T0.5 层：{"t", "labels": [{label,score}], "source", "ms"}。"""
        with self._lock:
            self._t_now = max(self._t_now, float(entry.get("t", 0.0)))
            self._append_ring(self._labels, self._labels_max, entry)
            self._telemetry["t05_labels"] += 1
            ms = entry.get("ms")
            if ms is not None:
                prev = self._telemetry["t05_avg_ms"]
                n = self._telemetry["t05_labels"]
                self._telemetry["t05_avg_ms"] = round(
                    ((prev or 0) * (n - 1) + ms) / n, 1)

    def apply_understanding(self, res, window_t0, window_t1, model):
        """T2 层：一次 VLM 调用的结构化结果合入滚动状态。

        res: {"now": str, "segment": {"summary": ...}, "entities": {}}，
        缺键安全降级（llm 输出不保证完全守约）。
        """
        with self._lock:
            now = str(res.get("now", "")).strip()
            if now:
                self._t2["now"] = now
            seg = res.get("segment")
            summary = str(seg.get("summary", "")).strip() if isinstance(seg, dict) else ""
            self._t2["timeline"].append({
                "start": round(float(window_t0), 2),
                "end": round(float(window_t1), 2),
                "summary": summary or now or "（无有效摘要）",
            })
            entities = res.get("entities")
            if isinstance(entities, dict):
                for k, v in entities.items():
                    if isinstance(k, str) and k.strip():
                        self._t2["entities"][k.strip()] = str(v)
            self._t2["understood_t"] = float(window_t1)
            self._t2["model"] = model
            self._t2["updated_at"] = time.time()
            self._telemetry["t2_calls"] += 1

    def note_call_latency(self, latency_s):
        with self._lock:
            self._telemetry["t2_last_latency_s"] = round(float(latency_s), 2)

    def note_call_error(self, err):
        with self._lock:
            self._telemetry["t2_errors"] += 1
            self._telemetry["t2_last_error"] = str(err)[:200]

    def note_dropped_window(self):
        with self._lock:
            self._telemetry["t2_dropped_windows"] += 1

    def compact_timeline(self):
        """timeline 超限时把最旧溢出部分合并成一条章节（纯文本，零 VLM 成本）。"""
        with self._lock:
            tl = self._t2["timeline"]
            if len(tl) <= self._max_timeline:
                return 0
            overflow = tl[:len(tl) - self._max_timeline]
            keep = tl[-self._max_timeline:]
            merged = {
                "start": overflow[0]["start"],
                "end": overflow[-1]["end"],
                "summary": "；".join(e["summary"] for e in overflow if e.get("summary")),
            }
            self._t2["chapters"].append(merged)
            self._t2["timeline"] = list(keep)
            return len(overflow)

    # ---------- 快照 / 渲染 / 落盘 ----------

    def snapshot(self):
        """完整状态快照（深拷贝语义，供 JSON 落盘与 SSE 下发）。"""
        with self._lock:
            return {
                "session": {
                    "source": self._source_desc,
                    "wall_started": self._t_start_wall,
                    "t_now": round(self._t_now, 3),
                    "frame_count": self._frame_count,
                },
                "t0": {
                    "last_motion": self._last_motion,
                    "motion_segments": self._motion_segments_n,
                },
                "t05": {"labels": list(self._labels)},
                "asr": list(self._asr),
                "t2": {
                    **self._t2,
                    "timeline": list(self._t2["timeline"]),
                    "chapters": list(self._t2["chapters"]),
                    "entities": dict(self._t2["entities"]),
                },
                "telemetry": dict(self._telemetry, **self._lags()),
            }

    def _lags(self):
        """分层滞后：t_now - 各层最近覆盖到的素材时间（秒）。"""
        t2_lag = (None if self._t2["understood_t"] is None
                  else round(max(0.0, self._t_now - self._t2["understood_t"]), 2))
        t05_lag = (None if not self._labels
                   else round(max(0.0, self._t_now - self._labels[-1].get("t", 0.0)), 2))
        return {"lag": {"t_now": round(self._t_now, 3), "t2_s": t2_lag, "t05_s": t05_lag}}

    def markdown(self):
        """live_context.md 内容：与离线 SKILL 工作流衔接的可读上下文。"""
        with self._lock:
            lines = ["# 实时视频理解上下文（滚动更新）", ""]
            lines.append(f"- 数据源: {self._source_desc or '未知'}")
            lines.append(f"- 时间轴: {self._t_now:.1f}s（{self._frame_count} 帧）")
            lines.append(f"- 更新: {time.strftime('%H:%M:%S')}")
            lines.append("")
            lines.append("## 当前正在发生（T2 滚动摘要）")
            lines.append(self._t2["now"] or "（等待首次理解…）")
            lines.append("")
            if self._t2["chapters"]:
                lines.append("## 前情章节（已压缩）")
                for ch in self._t2["chapters"]:
                    lines.append(f"- [{ch['start']:.0f}-{ch['end']:.0f}s] {ch['summary']}")
                lines.append("")
            lines.append("## 最近时间线")
            for e in self._t2["timeline"][-15:]:
                lines.append(f"- [{e['start']:.0f}-{e['end']:.0f}s] {e['summary']}")
            lines.append("")
            if self._labels:
                last = self._labels[-1]
                tag = "、".join(f"{d['label']}({d['score']})" for d in last["labels"])
                lines.append(f"## 最新画面标签（T0.5，{last.get('t', 0):.1f}s）")
                lines.append(tag)
                lines.append("")
            if self._asr:
                lines.append("## 最近语音")
                for seg in self._asr[-8:]:
                    lines.append(f"- [{seg.get('t', 0):.1f}s] {seg.get('text', '')}")
                lines.append("")
            lag = self._lags()["lag"]
            lines.append("## 遥测")
            lines.append(f"- T2 滞后: {lag['t2_s']}s / 调用 {self._telemetry['t2_calls']} 次"
                         f"（延迟 {self._telemetry['t2_last_latency_s']}s，错误 "
                         f"{self._telemetry['t2_errors']}）")
            lines.append(f"- T0.5 标签: {self._telemetry['t05_labels']} 次"
                         f"（均 {self._telemetry['t05_avg_ms']}ms）")
            return "\n".join(lines)

    def write_outputs(self, output_dir):
        """原子落盘 live_state.json + live_context.md（StateWriter 周期调用）。"""
        snap = self.snapshot()
        write_json_atomic(output_dir, "live_state.json", snap)
        write_text_atomic(output_dir, "live_context.md", self.markdown())
        return snap

    # ---------- 内部 ----------

    @staticmethod
    def _append_ring(ring, maxsize, entry):
        """有界环：超过上限丢最旧（长直播内存不膨胀）。"""
        ring.append(entry)
        if len(ring) > maxsize:
            del ring[:len(ring) - maxsize]


class StateWriter:
    """周期原子落盘线程（默认 1s；输出目录为 None 时不启动）。"""

    def __init__(self, state, output_dir=None, interval=1.0):
        self.state = state
        self.output_dir = output_dir
        self.interval = float(interval)
        self._stop = None
        self._thread = None

    def start(self):
        if self.output_dir is None:
            return
        import threading
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="vus-state-writer")
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.state.write_outputs(self.output_dir)
            except OSError:
                pass  # 磁盘抖动不拖垮理解链，下个周期重试

    def stop(self):
        """停线程并把最终状态强制落盘一次。"""
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self.output_dir is not None:
            try:
                self.state.write_outputs(self.output_dir)
            except OSError:
                pass
