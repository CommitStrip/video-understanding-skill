#!/usr/bin/env python3
"""
smart_pipeline.py - 兼容入口（薄壳）
====================================
实现已迁移至 vus.smart_pipeline（W0 工程化基线）。
本壳把公共 API 原样再导出，保证旧导入方式仍然可用:

    from smart_pipeline import SmartPipeline        # 旧写法（scripts 目录在 sys.path）
    from vus.smart_pipeline import SmartPipeline    # 新写法（推荐）
"""

import os
import sys

# 兼容兜底：未 pip install -e . 时，把仓库根加入 sys.path 使 vus 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vus.smart_pipeline import *  # noqa: F401,F403
from vus.smart_pipeline import SmartPipeline  # noqa: F401

if __name__ == '__main__':
    print("smart_pipeline 已迁移至 vus 包，请使用:")
    print("  python -c \"from vus.smart_pipeline import SmartPipeline\"")
    print("或运行管线: python -m vus.integrated_pipeline --video x.mp4")
