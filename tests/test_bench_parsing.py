"""bench.run_bench 的 crv frames.json 解析测试（离线，不装 crv）。

背景（实战 bug，对比基准复跑时发现）：crv 0.10+ 输出
{"frames": [{"timestamp_sec": ...}, ...]}，而 run_bench.py 旧代码
假定裸列表 + frames_json_path 属性，导致竞品帧数恒为 0、对比失真。
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "bench_run_bench", REPO_ROOT / "bench" / "run_bench.py")
rb = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("bench_run_bench", rb)
_spec.loader.exec_module(rb)


def test_parse_crv_010_dict_format():
    data = {"frames": [
        {"timestamp_sec": 0.0},
        {"timestamp_sec": 5.5},
        {"timestamp_sec": 11.2},
    ]}
    assert rb._parse_frames_json(data) == [0.0, 5.5, 11.2]


def test_parse_legacy_bare_list():
    data = [{"t": 0.0}, {"t": 3.0}, 7.0]
    assert rb._parse_frames_json(data) == [0.0, 3.0, 7.0]


def test_parse_empty_and_malformed():
    assert rb._parse_frames_json({}) == []
    assert rb._parse_frames_json({"frames": "not-a-list"}) == []
    assert rb._parse_frames_json([]) == []


def test_find_frames_json_recursive(tmp_path):
    """frames.json 在 crv 产物子目录里也应被发现。"""
    sub = tmp_path / "crv_out" / "frames"
    sub.mkdir(parents=True)
    fj = sub / "frames.json"
    fj.write_text(json.dumps({"frames": [{"timestamp_sec": 1.0}]}), encoding="utf-8")
    hits = list(rb._find_frames_json(str(tmp_path), result=None))
    assert hits and os.path.realpath(hits[0]) == os.path.realpath(str(fj))


import os  # noqa: E402  （供上面的 realpath 断言使用）
