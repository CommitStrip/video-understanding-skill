#!/usr/bin/env python3
"""
validate_realtime.py - 兼容入口（薄壳）
=======================================
实现已迁移至 vus.validate_realtime（W0 工程化基线）。
保留本入口以保证既有用法不变（输出目录现在用 --out 指定，默认 ./output/rt_validate）:

    python scripts/validate_realtime.py --out ./output/rt_validate

等价的模块用法:

    python -m vus.validate_realtime --out ./output/rt_validate
"""

import os
import sys

# 兼容兜底：未 pip install -e . 时，把仓库根加入 sys.path 使 vus 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vus.validate_realtime import main

if __name__ == '__main__':
    main()
