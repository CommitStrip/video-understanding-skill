#!/usr/bin/env python3
"""
rolling_align.py - 流式对齐器：批式 align_asr_streaming 的增量版
================================================================
批式对齐（SmartPipeline.align_asr_streaming）在流结束后一次性完成；
直播没有"流结束"，故边喂边闭窗：ASR 段 i 的窗口 [t0, t1) 在段 i+1 到达
时才闭合（t1 = 段 i+1 的 t），最后一段由 flush() 用 end_t / t+2.0 兜底
——窗口语义与批式逐条等价（左闭右开、边界事件归后段）。

事件由调用方随事件流推进 push（直播时钟下 ASR 有解码滞后，窗口内的
画面事件必然先于闭窗到达）。文件管线（可快于实时）继续用批式路径，
两轨互不干扰。
"""


class RollingAligner:
    """增量 ASR×画面事件对齐器，产出结构与批式 aligned_segments 完全一致。"""

    def __init__(self):
        self._motion_events = []
        self._keyframe_events = []
        self._pending = None  # 等待下一边界的最后一个 ASR 段

    def add_frame_event(self, ev):
        """随事件流推进画面事件（只保留对齐需要的 motion/keyframe 两类）。"""
        ev_type = ev.get("type", "")
        if ev_type.startswith("motion"):
            self._motion_events.append(ev)
        elif ev_type == "keyframe":
            self._keyframe_events.append(ev)

    def add_asr_segment(self, seg):
        """喂入一个清洗后的 ASR 终稿段，返回本次闭窗产出的对齐段（0 或 1 条）。"""
        if self._pending is None:
            self._pending = seg
            return []
        closed = self._close(self._pending, seg["t"])
        self._pending = seg
        return [closed]

    def flush(self):
        """收尾：闭最后一段（end_t 缺失回退 t+2.0，与批式一致）。"""
        if self._pending is None:
            return []
        seg, self._pending = self._pending, None
        return [self._close(seg, seg.get("end_t", seg["t"] + 2.0))]

    def _close(self, seg, t1):
        t0 = seg["t"]
        linked_motion = [e for e in self._motion_events if t0 <= e["t"] < t1]
        linked_kf = [k for k in self._keyframe_events if t0 <= k["t"] < t1]
        return {
            "start": round(t0, 2),
            "end": round(t1, 2),
            "text": seg["text"],
            "linked_motion_events": len(linked_motion),
            "linked_keyframes": len(linked_kf),
            "motion_types": list(set(e["type"] for e in linked_motion)),
        }
