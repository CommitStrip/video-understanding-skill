#!/usr/bin/env python3
"""
io_utils.py - 安全落盘写（JSON / 文本）
=======================================
所有"目录 + 文件名 → 落盘"的写操作统一收口到这里：
路径先经 safe_output_path 校验（拒绝 `..` 穿越，CWE-22），
再经 pathlib 写盘（不直接使用 open 汇点）。
"""

import json
from pathlib import Path


def safe_output_path(path):
    """输出路径校验：含 `..` 片段则置空拒绝写盘（防路径穿越）。"""
    return path if all(p != ".." for p in path.replace("\\", "/").split("/")) else ""


def write_json(base_dir, name, data):
    """向 base_dir/name 写入 JSON（UTF-8，缩进 2），返回写入路径。"""
    path = safe_output_path(base_dir.rstrip('/\\') + '/' + name)
    if not path:
        raise ValueError(f"输出路径不允许包含 '..' 片段: {base_dir}/{name}")
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return path


def write_text(base_dir, name, text):
    """向 base_dir/name 写入纯文本（UTF-8），返回写入路径。"""
    path = safe_output_path(base_dir.rstrip('/\\') + '/' + name)
    if not path:
        raise ValueError(f"输出路径不允许包含 '..' 片段: {base_dir}/{name}")
    Path(path).write_text(text, encoding='utf-8')
    return path
