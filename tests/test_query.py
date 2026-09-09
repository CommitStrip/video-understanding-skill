"""W10 查询回源测试（离线，不依赖视频文件）。"""
import json

import pytest

from vus.query import _merge_ranges, find_candidate_ranges
from vus.index import build_index


def _make_index(asr=None, ocr=None, attention=None):
    return {
        "asr": asr or [],
        "ocr": ocr or [],
        "attention_windows": attention or [],
        "keyframes": [],
        "motion_segments": [],
    }


# ==================== 候选区间检索 ====================

def test_find_ranges_by_asr_keyword():
    index = _make_index(asr=[
        {"t": 100.0, "end_t": 110.0, "text": "大家注意看右下角"},
        {"t": 200.0, "end_t": 210.0, "text": "无关内容"},
    ])
    ranges = find_candidate_ranges(index, keywords=["注意看"])
    assert len(ranges) == 1
    assert ranges[0][0] <= 100.0 <= ranges[0][1]


def test_find_ranges_by_ocr_keyword():
    index = _make_index(ocr=[{"t": 50.0, "text": "FREE SOUP"}])
    ranges = find_candidate_ranges(index, keywords=["soup"])
    assert len(ranges) == 1
    assert ranges[0][0] < 50.0 < ranges[0][1]


def test_find_ranges_attention_only():
    index = _make_index(attention=[{"t": 60.0, "end": 70.0}])
    ranges = find_candidate_ranges(index, attention_only=True)
    assert len(ranges) == 1
    assert ranges[0] == (60.0, 70.0)


def test_find_ranges_case_insensitive():
    index = _make_index(asr=[{"t": 100.0, "end_t": 105.0, "text": "Pay Attention NOW"}])
    ranges = find_candidate_ranges(index, keywords=["pay attention"])
    assert len(ranges) == 1


def test_find_ranges_merge_overlapping():
    index = _make_index(asr=[
        {"t": 100.0, "end_t": 110.0, "text": "注意看这里"},
        {"t": 108.0, "end_t": 115.0, "text": "重点在右边"},
    ])
    ranges = find_candidate_ranges(index, keywords=["注意", "重点"])
    assert len(ranges) == 1
    assert ranges[0][0] <= 100.0 and ranges[0][1] >= 115.0


# ==================== 区间合并 ====================

def test_merge_ranges_basic():
    assert _merge_ranges([(10, 20), (18, 30)]) == [(10, 30)]
    # gap_s=10 默认合并间隔 → 22-20=2 < 10 → 合并
    assert _merge_ranges([(10, 20), (22, 30)]) == [(10, 30)]
    # 远距不合并
    assert _merge_ranges([(10, 20), (32, 40)]) == [(10, 20), (32, 40)]
    assert _merge_ranges([]) == []


# ==================== 索引构建 ====================

def test_build_index_from_aligned(tmp_path):
    aligned = {
        "asr_segments": [{"t": 10.0, "text": "hello", "end_t": 12.0}],
        "ocr_events": [{"t": 20.0, "text": "WORLD"}],
        "attention_windows": [{"t": 30.0, "end": 35.0}],
    }
    p = tmp_path / "aligned_output.json"
    p.write_text(json.dumps(aligned), encoding="utf-8")
    index, out_path = build_index(str(p), str(tmp_path))
    assert len(index["asr"]) == 1
    assert index["asr"][0]["text"] == "hello"
    assert len(index["ocr"]) == 1
    assert (tmp_path / "index.json").exists()


def test_build_index_skips_hallucination(tmp_path):
    aligned = {
        "asr_segments": [{"t": 10.0, "text": "real"}, {"t": 20.0, "text": "SIL", "hallucination": True}],
    }
    p = tmp_path / "aligned_output.json"
    p.write_text(json.dumps(aligned), encoding="utf-8")
    index, _ = build_index(str(p), str(tmp_path))
    assert len(index["asr"]) == 1
    assert index["asr"][0]["text"] == "real"
