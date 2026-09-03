#!/usr/bin/env python3
"""
reconcile.py - ASR/OCR 跨模态轻量对齐标注
==========================================
对每条 ASR 段，在时间窗 ±window_s 内收集 OCR 文本，计算字符 2-gram
Jaccard 相似度；达到阈值时在段上标注 ocr_hint 字段（只标注不替换，
供下游 LLM 参考——专名同音错字可被画面花字/字幕纠正的线索）。
"""

def _bigrams(text):
    """字符 2-gram 集合（单字符文本退化为自身）。"""
    text = "".join(text.split())
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def _similarity(a, b):
    """字符 2-gram Jaccard 相似度（0-1）。"""
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / len(sa | sb)


def reconcile(asr_segments, ocr_events, window_s=15.0, threshold=0.6):
    """给 ASR 段标注时间窗内最相似的 OCR 文本（原地修改并返回）。

    ocr_events: [{"t": float, "text": str, ...}, ...]
    标注字段: seg["ocr_hint"] = {"text": str, "sim": float}
    """
    for seg in asr_segments:
        t = seg.get("t", 0.0)
        best_text, best_sim = None, 0.0
        for ev in ocr_events:
            if abs(ev.get("t", 0.0) - t) > window_s:
                continue
            sim = _similarity(seg.get("text", ""), ev.get("text", ""))
            if sim > best_sim:
                best_text, best_sim = ev.get("text", ""), sim
        if best_text and best_sim >= threshold:
            seg["ocr_hint"] = {"text": best_text, "sim": round(best_sim, 3)}
    return asr_segments
