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
# 下载允许的官方主机白名单（含 release 资产重定向目标）
_ALLOWED_HOSTS = {"github.com", "objects.githubusercontent.com", "codeload.github.com"}


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


def download_asr_model(dest_dir=None):
    """经系统 curl 下载官方 ASR 模型并解压到 dest_dir（默认 ./models/sherpa）。

    返回模型子目录路径字符串；失败打印原因返回 None。
    """
    _validate_url(ASR_MODEL_URL)
    dest = Path(dest_dir) if dest_dir else asr_models_dir()
    dest.mkdir(parents=True, exist_ok=True)
    archive = str(dest) + "/asr-model.tar.bz2"
    print(f"[ASR] 模型未找到，开始自动下载（约 490MB，一次性）")
    try:
        subprocess.run(
            ["curl", "-L", "--fail", "--retry", "3",
             "-o", archive, ASR_MODEL_URL],
            check=True,
        )
        print("[ASR] 下载完成，解压中…")
        with tarfile.open(archive, "r:bz2") as tf:
            tf.extractall(dest, filter="data")  # filter 防 tar 路径穿越
        Path(archive).unlink()
        sub = dest / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
        return str(sub if sub.is_dir() else dest)
    except Exception as e:  # 网络/磁盘/解压失败都走显式降级
        print(f"[ASR] 自动下载失败: {e}")
        print("[ASR] 可手动下载后解压到 models/sherpa/，或设 VUS_ASR_AUTO_DOWNLOAD=0 跳过")
        if Path(archive).exists():
            Path(archive).unlink()
        return None


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
