#!/usr/bin/env python3
"""
asr_clean.py - ASR 后处理清洗层（去重复 + 幻觉标记）
====================================================
纯后处理，不动解码引擎（速度绝对优先）。清洗三件事：
  1. 连叠折叠：解码循环产生的重复子串（runrunrun / 的的的的）折叠
  2. 相邻去重：流式 chunk 边界的相邻重复段合并
     （只看相邻，不做全局去重——全局会误杀长视频里循环的口头禅）
  3. 幻觉标记：纯 ASCII 短文本（sherpa 英文幻觉几乎全大写：SIL/ER/VER 类）
     标记 hallucination=True，默认保留不删，宿主/LLM 自行取舍
清洗过的段保留 raw_text 原文供审计。
"""

import re

# 连续重复子串折叠：1-4 字符单元连续出现 3 次以上 → 折叠为 1 次
# （runrunrun → run；的的的的 → 的）
_REPEAT_RE = re.compile(r'(.{1,4})\1{2,}')

# 纯 ASCII 短文本（字母/空格/常见标点），不含任何 CJK
_HALLUCINATION_RE = re.compile(r"^[A-Za-z\s'?,.!]{2,60}$")


def _is_likely_hallucination(text):
    """判断是否为疑似英文幻觉段。

    纯 ASCII 基础上再加大写启发：sherpa 的英文幻觉几乎全大写
    （SIL / ER / VER / MONEY INTO THE PACKETS OF SUPER RICH），
    正常英文含小写；token 数 ≤2 的极短段同样视为可疑。
    """
    if not _HALLUCINATION_RE.match(text):
        return False
    letters = [c for c in text if c.isalpha()]
    upper_ratio = (sum(1 for c in letters if c.isupper()) / max(len(letters), 1))
    tokens = text.split()
    return upper_ratio > 0.8 or len(tokens) <= 2


def clean_text(text):
    """单段文本清洗：折叠连叠 + 去首尾空白。返回 (cleaned, 是否变化)。"""
    stripped = text.strip()
    cleaned = _REPEAT_RE.sub(r'\1', stripped)
    return cleaned, cleaned != stripped


def clean_asr_segments(segments):
    """清洗 ASR 段列表（不修改入参，返回新列表）。

    - 文本连叠折叠（变化时保留 raw_text 原文 + cleaned 标记）
    - 空段剔除
    - 仅相邻重复合并（全局去重会误杀长视频里循环出现的口头禅）
    - 疑似英文幻觉段标记 hallucination=True（保留不删）
    """
    cleaned = []
    for seg in segments:
        text, changed = clean_text(seg.get("text", ""))
        if not text:
            continue
        if cleaned and cleaned[-1].get("text") == text:
            continue
        entry = dict(seg)
        entry["text"] = text
        if changed:
            entry["raw_text"] = seg.get("text", "")
            entry["cleaned"] = True
        if _is_likely_hallucination(text):
            entry["hallucination"] = True
        cleaned.append(entry)
    return cleaned


def filter_hallucinations(segments):
    """返回剔除幻觉段后的列表（是否剔除由宿主决定，默认保留在产物里）。"""
    return [s for s in segments if not s.get("hallucination")]


# ==================== 注意力回链（W9 借鉴外部审查：跨模态反向保留） ====================

# 语言注意力线索：说话人引导观众看画面特定位置/记住要点时，对应画面区间
# 应提升保留优先级（跨模态反向触发：语言 → 视觉）
ATTENTION_CUES = ("注意看", "重点", "看这里", "记住", "大家看", "注意",
                  "look at", "pay attention", "important", "watch")


def detect_attention_windows(segments, merge_gap_s=5.0):
    """扫描 ASR 段中的注意力线索，合并相邻命中为注意力窗口。

    返回 [{"t", "end", "cue"}]（按时间序）；幻觉段跳过。
    """
    hits = []
    for s in segments:
        if s.get("hallucination"):
            continue
        text = s.get("text", "")
        for cue in ATTENTION_CUES:
            if cue.lower() in text.lower():
                t = s.get("t", 0.0)
                hits.append({"t": t, "end": s.get("end_t", t), "cue": cue})
                break
    windows = []
    for h in sorted(hits, key=lambda x: x["t"]):
        if windows and h["t"] - windows[-1]["end"] <= merge_gap_s:
            windows[-1]["end"] = max(windows[-1]["end"], h["end"])
        else:
            windows.append(dict(h))
    return windows


class RollingCleaner:
    """clean_asr_segments 的增量版（W8 直播链路）：跨 feed 保持相邻去重状态。

    直播 ASR 段是陆续到达的，批式 clean_asr_segments 无法直接使用——
    相邻去重需要记住上一条已发射文本。逐段清洗语义与批式完全一致：
    空段剔除 / 连叠折叠（保留 raw_text）/ 相邻重复合并 / 幻觉标记。
    """

    def __init__(self):
        self._last_text = None  # 上一条已发射段的清洗后文本（相邻去重基准）

    def feed(self, segments):
        """清洗一批新到达的 ASR 段，返回可发射的段列表（可能为空）。"""
        out = []
        for seg in segments:
            text, changed = clean_text(seg.get("text", ""))
            if not text:
                continue
            if self._last_text is not None and self._last_text == text:
                continue
            entry = dict(seg)
            entry["text"] = text
            if changed:
                entry["raw_text"] = seg.get("text", "")
                entry["cleaned"] = True
            if _is_likely_hallucination(text):
                entry["hallucination"] = True
            self._last_text = text
            out.append(entry)
        return out
