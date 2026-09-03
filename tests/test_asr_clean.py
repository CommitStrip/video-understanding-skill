"""asr_clean 清洗层测试——向量来自真实实测批评清单的脏数据形态。

实测背景（2026-09-03 三条真实视频）：
- 连续重复：《豺狼的日子》695 处重复 run / 1117 字冗余；大萧条 478 处 / 601 字
- 英文幻觉：SIL、ER、INE、CUS、MONEY INTO THE PACKETS OF SUPER RICH（音乐段）
- 正常英文必须保留："More than that." 这类大小写混合句
- 全局去重会误杀长视频循环口头禅——只允许相邻去重
"""
import pytest

from vus.asr_clean import (
    clean_asr_segments,
    clean_text,
    filter_hallucinations,
    _is_likely_hallucination,
)


def seg(t, text):
    return {"t": t, "text": text}


# ==================== 连叠折叠 ====================

def test_collapse_decoder_loop_runs():
    assert clean_text("runrunrunrun")[0] == "run"
    # 贪婪捕获组下折叠粒度随重复次数变化，但保证大幅压缩且无 3 连叠残留
    out, changed = clean_text("的的的的的的")
    assert changed and len(out) <= 3 and "的的的" not in out
    out, changed = clean_text("是是是是是是")
    assert changed and len(out) <= 3 and "是是是" not in out


def test_collapse_mixed_sentence():
    text, changed = clean_text("啊在一战的时候啊美国看看看看欧洲")
    assert "看看看看" not in text
    assert "美国看欧洲" in text
    assert changed


def test_normal_text_untouched():
    text, changed = clean_text("咱们今天晚上的主要内容呢是")
    assert text == "咱们今天晚上的主要内容呢是"
    assert not changed


def test_legal_reduplication_boundary():
    """两连叠（合法叠词）不在折叠范围（正则要求 3 次以上）。"""
    text, _ = clean_text("对对对")
    assert text == "对"  # 3 连叠折叠到 1
    text, _ = clean_text("对对")
    assert text == "对对"  # 两连叠保留


# ==================== 幻觉判定 ====================

def test_hallucination_caps_tokens_marked():
    assert _is_likely_hallucination("SIL")
    assert _is_likely_hallucination("ER")
    assert _is_likely_hallucination("MONEY INTO THE PACKETS OF SUPER RICH")


def test_normal_english_not_marked():
    assert not _is_likely_hallucination("More than that.")
    assert not _is_likely_hallucination("The exploitative relationship")


def test_cjk_text_never_marked():
    assert not _is_likely_hallucination("在一战的时候啊 SIL")


# ==================== 段级清洗 ====================

def test_adjacent_duplicates_merged():
    segs = [seg(0, "就是剥削关系"),
            seg(2, "就是剥削关系"),
            seg(4, "就是剥削关系"),
            seg(6, "开始削减工人的工资")]
    out = clean_asr_segments(segs)
    assert [s["text"] for s in out] == ["就是剥削关系", "开始削减工人的工资"]


def test_non_adjacent_duplicates_kept():
    """全局去重会误杀循环口头禅——非相邻的重复段必须保留。"""
    segs = [seg(0, "咱们今天晚上"),
            seg(60, "开始削减工人的工资"),
            seg(120, "咱们今天晚上")]
    out = clean_asr_segments(segs)
    assert len(out) == 3


def test_raw_text_preserved_for_audit():
    out = clean_asr_segments([seg(0, "的的的的的")])
    assert out[0]["text"] == "的"
    assert out[0]["raw_text"] == "的的的的的"
    assert out[0]["cleaned"] is True


def test_hallucination_flag_and_filter():
    segs = [seg(0, "SIL"),
            seg(2, "More than that."),
            seg(4, "MONEY INTO THE PACKETS OF SUPER RICH"),
            seg(6, "开始削减工人的工资")]
    out = clean_asr_segments(segs)
    flagged = [s for s in out if s.get("hallucination")]
    assert len(flagged) == 2
    kept = filter_hallucinations(out)
    assert [s["text"] for s in kept] == ["More than that.", "开始削减工人的工资"]


def test_empty_segments_dropped():
    out = clean_asr_segments([seg(0, "   "), seg(1, "内容")])
    assert [s["text"] for s in out] == ["内容"]


def test_input_not_mutated():
    original = seg(0, "的的的的")
    clean_asr_segments([original])
    assert original["text"] == "的的的的"
    assert "cleaned" not in original
