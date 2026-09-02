#!/usr/bin/env python3
"""
asr_sherpa.py - 兼容入口（薄壳）
================================
实现已迁移至 vus.asr_sherpa（W0 工程化基线）。
本壳把公共 API 原样再导出，保证旧导入方式仍然可用:

    from asr_sherpa import extract_audio, load_wav, ...   # 旧写法
    from vus.asr_sherpa import extract_audio, load_wav, ...  # 新写法（推荐）
"""

import os
import sys

# 兼容兜底：未 pip install -e . 时，把仓库根加入 sys.path 使 vus 可导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vus.asr_sherpa import *  # noqa: F401,F403
from vus.asr_sherpa import (  # noqa: F401
    extract_audio, load_wav, load_streaming_recognizer,
    transcribe_streaming
)
# W1（2026-09-02）：transcribe_offline 空桩已删除（原实现恒返回 mock，有误导性），
# 故不再从此处导出。

if __name__ == '__main__':
    # 快速自测（与迁移前行为一致）
    import numpy as np
    print("=== ASR 模块自测 (实现位于 vus.asr_sherpa) ===")
    rec = load_streaming_recognizer()
    if rec is None:
        print("使用 fallback 模式")
        # 模拟10秒音频
        samples = np.random.randn(16000 * 10).astype(np.float32) * 0.1
        segs = transcribe_streaming(None, samples)
        print(f"产出 {len(segs)} 个段")
        for s in segs[:3]:
            print(f"  t={s['t']:.1f}s: {s['text']}")
    else:
        print("识别器加载成功，等待音频输入")
