#!/usr/bin/env python3
"""
pathsafe.py - 输出路径安全校验（防路径穿越，CWE-22）
====================================================
vus 的库 API 与 CLI 普遍接受调用方指定的输出路径（--out/--output 等）。
当库被嵌入 Web 服务等不可信输入场景时，含 `..` 的路径可能写到预期之外的
文件。本项目约定：所有落盘写统一经由 safe_output_path 校验。

策略：拒绝路径中任何 `..` 片段（含 Windows 反斜杠写法）；绝对路径与
普通相对路径不受影响（操作者显式指定即为其意图）。
"""

import os
from pathlib import PurePosixPath

__all__ = ["safe_output_path"]


def safe_output_path(path):
    """校验输出路径不含 `..` 穿越片段，返回原样字符串路径。

    抛出 ValueError：路径任一分量为 `..`（正斜杠/反斜杠写法均拦截）。
    """
    raw = os.fspath(path)
    parts = PurePosixPath(raw.replace("\\", "/")).parts
    if ".." in parts:
        raise ValueError(f"输出路径不允许包含 '..' 片段: {raw}")
    return raw
