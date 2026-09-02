#!/usr/bin/env python3
"""
select_representatives.py - 兼容入口（薄壳）
============================================
实现已迁移至 vus.select_representatives（W0 工程化基线）。
保留本入口以保证既有用法不变:

    python scripts/select_representatives.py --keyframes out/keyframes --interval 60

等价的模块用法:

    python -m vus.select_representatives --keyframes out/keyframes --interval 60
"""

import os
import sys

# 兼容兜底：未 pip install -e . 时，把仓库根加入 sys.path 使 vus 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vus.select_representatives import *  # noqa: F401,F403
from vus.select_representatives import (  # noqa: F401
    load_keyframes, select_representatives, build_context, summarize,
    load_clip_model, main
)

if __name__ == '__main__':
    main()
