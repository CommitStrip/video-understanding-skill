#!/usr/bin/env python3
"""
understanding.py - T2 摘要道：触发式 VLM 理解 worker（实时理解层的核心）
========================================================================
双线程模型，理解永不阻塞摄取、摄取永不压垮理解：

  采集线程  drain EventBus → 推进 SessionState（T0/T0.5/ASR）+ 滚动对齐
            → 累积"理解窗口"素材，按触发策略置位
  调用线程  单飞（in-flight=1）消费窗口 → 编码关键帧 + 拼 prompt → VLM
            → 结构化结果合入滚动状态 → 时间线超限压缩

成本控制（本期重点）：
  - 触发式调用：scene_change/渐变漂移关键帧、长运动段闭合、新语音段才触发；
    安静场景零调用
  - 地板间隔 min_call_interval：触发再密也限频（最坏费用上限 = 时长/间隔 × 单次成本）
  - 单飞合并：VLM 在跑时素材只累积不排队，完成后下一窗取"合并后的最新"，
    滞后有界且不为积压重复计费
  - 时间线压缩：timeline 超限把最旧合并成章节（纯文本，零 VLM 成本），
    小时级直播状态不膨胀
容错：VLM 失败指数退避（可被 stop 打断），失败窗口带素材重新排队，
重试超限丢弃并计数——主链（摄取/标签/落盘）永不依赖 VLM 成功。
"""

import queue
import threading
import time

from .rolling_align import RollingAligner
from .vlm_client import encode_frame_b64, parse_understanding_json

DEFAULT_CONFIG = {
    "min_call_interval": 8.0,     # 地板间隔（秒），费用上限的直接旋钮
    "max_frames_per_call": 2,     # 单次调用附带关键帧数（控制 token 成本）
    "frame_max_side": 448,        # 关键帧缩略边长
    "max_timeline": 40,           # timeline 上限（超出压缩成章节）
    "motion_end_trigger_s": 2.0,  # 运动段闭合触发门槛
    "asr_trigger": True,          # 语音段是否触发（静音直播可关）
    "backoff_base": 2.0,          # 失败退避基数（秒）
    "backoff_max": 60.0,
    "call_timeout": 60.0,
    "max_win_kf": 10,             # 窗口内关键帧内存上限（编码时再取最新 N 张）
    "max_win_asr_chars": 4000,    # 窗口语音文本上限
}


class _Window:
    """一次 VLM 调用的素材窗（采集线程写、调用线程消费）。"""

    __slots__ = ("t0", "t1", "kf", "asr", "motion_n", "retries")

    def __init__(self):
        self.t0 = None           # 窗口起始素材时间（None=空窗）
        self.t1 = 0.0            # 窗口末素材时间（滞后计算的锚点）
        self.kf = []             # [(t, path)] 按时间序
        self.asr = ""            # 语音文本（累积，按字符截断保尾）
        self.motion_n = 0
        self.retries = 0

    def empty(self):
        return self.t0 is None

    def add_asr(self, text, max_chars):
        if not text:
            return
        joined = (self.asr + " " + text).strip()
        self.asr = joined[-max_chars:]

    def merge(self, other, max_kf=10, max_asr_chars=4000):
        """把 other 并入 self（时间外扩、素材拼接、上限截断）。"""
        self.t0 = other.t0 if self.t0 is None else min(self.t0, other.t0)
        self.t1 = max(self.t1, other.t1)
        self.kf = sorted(set(self.kf + other.kf))[-max_kf:]
        self.add_asr(other.asr, max_asr_chars)
        self.motion_n += other.motion_n
        self.retries = max(self.retries, other.retries)
        return self


class UnderstandingWorker:
    """触发式 VLM 理解 worker。start() 后台运行，stop() 优雅收线。"""

    def __init__(self, bus, state, vlm, config=None, clock_fn=time.monotonic):
        self.bus = bus
        self.state = state
        self.vlm = vlm
        self.cfg = dict(DEFAULT_CONFIG)
        self.cfg.update(config or {})
        self._clock = clock_fn

        self._sub = bus.subscribe()
        self._aligner = RollingAligner()
        self._aligned = []            # 已闭窗的对齐段（结束时随产物落盘）

        self._win = _Window()         # 采集线程正在累积的窗口
        self._win_lock = threading.Lock()
        self._triggered = False
        self._win_q = queue.Queue(maxsize=1)
        self._in_flight = threading.Event()
        self._last_call_clock = float("-inf")
        self._err_streak = 0

        self._stop_evt = threading.Event()
        self._threads = []

    # ---------- 生命周期 ----------

    def start(self):
        for name, target in (("vus-collector", self._collect_loop),
                             ("vus-caller", self._caller_loop)):
            t = threading.Thread(target=target, daemon=True, name=name)
            t.start()
            self._threads.append(t)

    def stop(self, timeout=5.0):
        self._stop_evt.set()
        for t in self._threads:
            t.join(timeout=timeout)
        self._threads = []

    def wait_idle(self, timeout=10.0):
        """等触发清零、队列清空且无在飞调用（测试/E2E 用）。"""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (not self._triggered and not self._in_flight.is_set()
                    and self._win_q.empty()):
                return True
            time.sleep(0.01)
        return False

    def aligned_segments(self):
        """已闭窗的 ASR×画面对齐段（供结束落盘 aligned_output.json）。"""
        return list(self._aligned)

    def flush_aligner(self):
        """流结束时闭最后一段对齐窗（end_t/t+2.0 兜底，与批式 flush 语义一致）。"""
        self._aligned.extend(self._aligner.flush())
        return self.aligned_segments()

    # ---------- 采集线程 ----------

    def _collect_loop(self):
        while not self._stop_evt.is_set():
            try:
                evs = self._sub.drain(timeout=0.5)
            except Exception:
                continue
            for ev in evs:
                try:
                    self._on_event(ev)
                except Exception:
                    continue  # 单条坏事件不拖垮采集
            if self._triggered:
                self._maybe_fire()

    def _on_event(self, ev):
        ev_type = ev.get("type", "")

        if ev_type == "tag":                       # T0.5 标签（管线侧打好的）
            self.state.apply_label(ev)
            return

        if ev_type in ("motion_start", "motion", "motion_end", "keyframe"):
            self.state.apply_frame_event(ev)
            self._aligner.add_frame_event(ev)
            if ev_type == "keyframe":
                path = ev.get("path")
                with self._win_lock:
                    if path:
                        self._win.kf.append((float(ev.get("t", 0.0)), path))
                    self._win.t0 = (self._win.t0 if self._win.t0 is not None
                                    else float(ev.get("t", 0.0)))
                    self._win.t1 = max(self._win.t1, float(ev.get("t", 0.0)))
                    # 首帧不算语义事件，其余（scene_change/gradual_drift）强触发
                    if ev.get("reason", "scene_change") != "first_frame":
                        self._triggered = True
            elif ev_type == "motion_end":
                duration = float(ev.get("duration") or 0.0)
                with self._win_lock:
                    self._win.motion_n += 1
                if duration >= self.cfg["motion_end_trigger_s"]:
                    self._triggered = True
            return

        if ev_type == "asr_final":
            self.state.apply_asr(ev)
            for closed in self._aligner.add_asr_segment(ev):
                self._aligned.append(closed)
            with self._win_lock:
                self._win.add_asr(ev.get("text", ""), self.cfg["max_win_asr_chars"])
                self._win.t0 = (self._win.t0 if self._win.t0 is not None
                                else float(ev.get("t", 0.0)))
                self._win.t1 = max(self._win.t1, float(ev.get("end_t", ev.get("t", 0.0))))
            if self.cfg["asr_trigger"]:
                self._triggered = True

    def _maybe_fire(self):
        """触发 → 过地板间隔与单飞检查 → 窗口移交调用线程（满则合并）。

        vlm=None（纯本地 T0+T0.5 模式）：worker 退化为纯状态采集器，
        素材只推进状态，永不发起调用。
        """
        if self.vlm is None:
            self._triggered = False
            with self._win_lock:
                self._win = _Window()
            return
        now = self._clock()
        if self._in_flight.is_set():
            return
        if now - self._last_call_clock < self.cfg["min_call_interval"]:
            return
        with self._win_lock:
            if self._win.empty():
                self._triggered = False
                return
            win, self._win = self._win, _Window()
        self._triggered = False
        self._last_call_clock = now
        try:
            self._win_q.put_nowait(win)
        except queue.Full:
            # 单飞合并：上一窗还没被取走 → 合并成一窗（素材不丢、不重复计费）
            try:
                old = self._win_q.get_nowait()
                old.merge(win, max_kf=self.cfg["max_win_kf"],
                          max_asr_chars=self.cfg["max_win_asr_chars"])
                self._win_q.put_nowait(old)
            except (queue.Empty, queue.Full):
                self.state.note_dropped_window()

    # ---------- 调用线程 ----------

    def _caller_loop(self):
        while not self._stop_evt.is_set():
            try:
                win = self._win_q.get(timeout=0.5)
            except queue.Empty:
                continue
            self._in_flight.set()
            try:
                self._process_window(win)
            except Exception as e:  # 防御：任何异常不得杀死调用线程
                self.state.note_call_error(e)
            finally:
                self._in_flight.clear()

    def _process_window(self, win):
        prompt, frames_b64 = self._build_prompt(win)
        t0 = self._clock()
        try:
            text = self.vlm.understand(prompt, frames_b64,
                                       timeout=self.cfg["call_timeout"])
        except Exception as e:
            self.state.note_call_error(e)
            self._err_streak += 1
            win.retries += 1
            if win.retries <= 3:
                self._requeue(win)
            else:
                self.state.note_dropped_window()
            self._backoff_sleep()
            return
        self._err_streak = 0
        self.state.note_call_latency(self._clock() - t0)

        res = parse_understanding_json(text)
        if res is None:
            res = {"now": (text or "").strip()[:120]}
        self.state.apply_understanding(res, win.t0, win.t1,
                                       model=getattr(self.vlm, "model", self.vlm.name))
        self.state.compact_timeline()
        # 理解结果进总线（SSE 订阅者立即可见）
        self.bus.publish({"type": "understanding", "t": win.t1,
                          "now": res.get("now", "")})

    def _requeue(self, win):
        """失败窗口带素材重新排队（与可能的新窗合并，重试超限由调用方丢弃）。"""
        try:
            self._win_q.put_nowait(win)
        except queue.Full:
            try:
                old = self._win_q.get_nowait()
                old.merge(win, max_kf=self.cfg["max_win_kf"],
                          max_asr_chars=self.cfg["max_win_asr_chars"])
                self._win_q.put_nowait(old)
            except (queue.Empty, queue.Full):
                self.state.note_dropped_window()

    def _backoff_sleep(self):
        delay = min(self.cfg["backoff_max"],
                    self.cfg["backoff_base"] * (2 ** (self._err_streak - 1)))
        self._stop_evt.wait(delay)

    # ---------- prompt 组装 ----------

    def _build_prompt(self, win):
        frames_b64 = []
        for _t, path in win.kf[-self.cfg["max_frames_per_call"]:]:
            b64 = encode_frame_b64(path, max_side=self.cfg["frame_max_side"])
            if b64:
                frames_b64.append(b64)

        snap = self.state.snapshot()
        rolling = snap["t2"]["now"] or "（首轮，无历史摘要）"
        label_hint = ""
        if snap["t05"]["labels"]:
            last = snap["t05"]["labels"][-1]
            label_hint = "；".join(f"{d['label']}({d['score']})"
                                   for d in last["labels"])
        asr_text = win.asr[-1500:] or "（本窗口无语音）"
        entities = "、".join(snap["t2"]["entities"].keys()) or "（无）"

        prompt = (
            "你是实时视频理解引擎，正在持续观看一路监控流。"
            "根据当前滚动摘要与新增素材更新理解，只输出 JSON（不要多余文字）：\n"
            '{"now": "≤60字，当前画面+语音正在发生什么",'
            ' "segment": {"start": %s, "end": %s, "summary": "≤50字，本时间窗小结"},'
            ' "entities": {"出现的关键对象": "一句话说明"}}\n' % (win.t0, win.t1) +
            f"【当前滚动摘要】{rolling}\n"
            f"【已知实体】{entities}\n"
            f"【新增语音】{asr_text}\n"
            f"【画面活动】本窗运动事件 {win.motion_n} 个"
            + (f"；最新标签（本地毫秒级）：{label_hint}" if label_hint else "")
            + (f"；附关键帧 {len(frames_b64)} 张" if frames_b64 else "（无图可附）"))
        return prompt, frames_b64
