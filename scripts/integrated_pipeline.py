#!/usr/bin/env python3
"""
integrated_pipeline.py - 兼容入口（薄壳）
=========================================
实现已迁移至 vus.integrated_pipeline（W0 工程化基线）。
保留本入口以保证既有用法不变:

    python scripts/integrated_pipeline.py --video x.mp4 --output out/

等价的模块用法:

    python -m vus.integrated_pipeline --video x.mp4 --output out/
"""

import os
import sys

# 兼容兜底：未 pip install -e . 时，把仓库根加入 sys.path 使 vus 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vus.integrated_pipeline import main

if __name__ == '__main__':
    main()
