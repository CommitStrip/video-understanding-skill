#!/usr/bin/env python3
"""
model_setup.py - 模型资产自动准备（默认开箱即用）
==================================================
首次使用时自动下载 ASR 流式模型（小米 sherpa-onnx 官方发布），默认启用：
    VUS_ASR_AUTO_DOWNLOAD=0 可关闭自动下载；
    VUS_SHERPA_MODELS 可指定模型目录。
下载源固定为 github.com 官方 releases（仅 https，下载前校验 host），
经系统 curl 落盘（Windows 10+/macOS/Linux 均内置）。
"""

import os
import subprocess
import tarfile
from pathlib import Path
from urllib.parse import urlparse

# 官方发布直链（小米/k2-fsa sherpa-onnx，中英双语流式 zipformer）
ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
)
# 离线识别模型（SenseVoice int8：非流式全上下文，中文专名/可读性显著优于流式，
# 对 BGM 鲁棒；166MB，用于文件转写默认路径）
OFFLINE_ASR_MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2"
)
OFFLINE_ASR_SUBDIR = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"
# 下载允许的官方主机白名单（含 release 资产重定向目标）
_ALLOWED_HOSTS = {"github.com", "objects.githubusercontent.com", "codeload.github.com"}

# 多源 fallback 下载链：官方源 → 自建源（vus-models 仓库）
# 两个源提供相同的模型文件，任一可用即可。
OFFLINE_ASR_SOURCES = [
    # 官方源（sherpa-onnx releases）
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
    # 自建源（CommitStrip/vus-models 仓库，int8 精简包）
    "https://github.com/CommitStrip/vus-models/releases/download/v1.0/sense-voice-int8.tar.bz2",
]
STREAMING_ASR_SOURCES = [
    # 官方源（sherpa-onnx releases，完整包含 fp32+int8）
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
    # 自建源（int8 精简包，体积 490→170MB）
    "https://github.com/CommitStrip/vus-models/releases/download/v1.0/streaming-zipformer-int8.tar.bz2",
]


def _validate_url(url):
    """下载前校验：仅允许 https 且 host 在官方白名单（防 SSRF，CWE-918）。"""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in _ALLOWED_HOSTS:
        raise ValueError(f"仅允许从官方 https 源下载: {url}")
    return url


def asr_models_dir():
    """ASR 模型目录：VUS_SHERPA_MODELS > ./models/sherpa。"""
    return Path(os.environ.get("VUS_SHERPA_MODELS") or "./models/sherpa")


def find_asr_model(model_dir=None):
    """查找模型**完整**就位的目录（tokens.txt + encoder onnx 同在）。

    只判断目录存在会被残缺骨架（如下载中断残留）骗过。找到返回路径，否则 None。
    """
    candidates = [
        model_dir,
        os.environ.get("VUS_SHERPA_MODELS"),
        os.path.expanduser("~/sherpa-onnx-models"),
        "./models/sherpa",
        "./models/sherpa/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20",
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            for root, _dirs, files in os.walk(c):
                has_tokens = "tokens.txt" in files
                has_encoder = any(
                    f.startswith("encoder") and f.endswith(".onnx") for f in files)
                if has_tokens and has_encoder:
                    return c
    return None


def _download_from_sources(sources, dest_dir, label):
    """多源 fallback 下载：按顺序尝试每个 URL，任一成功即返回。

    sources: [URL 字符串]；dest_dir: 解压目标目录；label: 打印标签。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for url in sources:
        _validate_url(url)
        archive = str(dest) + "/model-download.tar.bz2"
        size_hint = "166MB" if "sense-voice" in url.lower() else "490MB"
        print(f"[ASR] {label}: 尝试下载（约 {size_hint}）…")
        try:
            subprocess.run(
                ["curl", "-L", "--fail", "--retry", "2",
                 "-o", archive, url],
                check=True,
            )
            print(f"[ASR] 下载完成，解压中…")
            with tarfile.open(archive, "r:bz2") as tf:
                tf.extractall(dest, filter="data")
            Path(archive).unlink()
            # 解压后扫描 dest 目录找到包含 tokens.txt + model/encoder onnx 的子目录
            for root, _dirs, files in os.walk(str(dest)):
                has_tokens = "tokens.txt" in files
                has_model = any(f.endswith(".onnx") for f in files)
                if has_tokens and has_model:
                    print(f"[ASR] 模型就位: {root}")
                    return root
            return str(dest)  # 找到但结构不同，仍返回解压目录
        except Exception as e:
            print(f"[ASR] 下载源失败({url.split('/')[-1]}): {e}")
            if Path(archive).exists():
                Path(archive).unlink()
    return None


def download_asr_model(dest_dir=None):
    """多源下载流式 ASR 模型到 dest_dir（默认 ./models/sherpa）。

    下载链: 官方 sherpa-onnx release → CommitStrip/vus-models mirror。
    返回模型目录路径字符串；全部失败返回 None。
    """
    return _download_from_sources(STREAMING_ASR_SOURCES,
                                  dest_dir or asr_models_dir(), "流式 ASR")


def ensure_asr_model(model_dir=None, auto=None):
    """确保 ASR 模型就位：已存在则返回目录；缺失时按默认策略自动下载。

    auto: 缺省读环境变量 VUS_ASR_AUTO_DOWNLOAD（默认开）。
    返回模型目录字符串或 None（未就位）。
    """
    found = find_asr_model(model_dir)
    if found:
        return found
    if auto is None:
        auto = os.environ.get("VUS_ASR_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")
    if not auto:
        return None
    return download_asr_model()


# ==================== 离线识别模型（SenseVoice int8） ====================

def offline_asr_dir():
    """离线模型目录：VUS_OFFLINE_ASR_MODELS > ./models/offline_asr。"""
    return Path(os.environ.get("VUS_OFFLINE_ASR_MODELS") or "./models/offline_asr")


def _dir_has_complete_model(d):
    """目录内（含子目录）应能找到 tokens.txt + 任一 model*.onnx。"""
    for root, _dirs, files in os.walk(d):
        names = set(files)
        if "tokens.txt" in names and any(f.startswith("model") and f.endswith(".onnx")
                                         for f in files):
            return True
    return False


def find_offline_asr_model(model_dir=None):
    """查找离线识别模型**完整**就位的目录，找到返回路径字符串，否则 None。"""
    candidates = [
        model_dir,
        os.environ.get("VUS_OFFLINE_ASR_MODELS"),
        str(offline_asr_dir()),
        str(offline_asr_dir() / OFFLINE_ASR_SUBDIR),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and _dir_has_complete_model(c):
            return c
    return None


def download_offline_asr_model(dest_dir=None):
    """多源下载 SenseVoice 离线模型（166MB int8）。

    下载链: 官方 sherpa-onnx release → CommitStrip/vus-models mirror。
    返回模型目录字符串；全部失败返回 None。
    """
    return _download_from_sources(OFFLINE_ASR_SOURCES,
                                  dest_dir or offline_asr_dir(), "离线 SenseVoice")


def ensure_offline_asr_model(model_dir=None, auto=None):
    """确保离线识别模型就位：已存在返回目录；缺失按默认策略自动下载。"""
    found = find_offline_asr_model(model_dir)
    if found:
        return found
    if auto is None:
        auto = os.environ.get("VUS_ASR_AUTO_DOWNLOAD", "1").lower() not in ("0", "false", "no")
    if not auto:
        return None
    return download_offline_asr_model()
