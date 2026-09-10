#!/usr/bin/env python3
"""
model_setup.py - 模型资产自动准备（默认开箱即用）
==================================================
首次使用时自动下载 ASR 流式模型（小米 sherpa-onnx 官方发布），默认启用：
    VUS_ASR_AUTO_DOWNLOAD=0 可关闭自动下载；
    VUS_SHERPA_MODELS 可指定模型目录。
下载源为多源 fallback 下载链（仅 https，逐源校验 host 白名单）：
    官方 github release → hf-mirror.com 国内镜像（文件直下）→ CommitStrip/vus-models 镜像；
    全部通过后对主模型文件做 sha256 完整性校验，curl 带断点续传。
    VUS_ASR_AUTO_DOWNLOAD=0 可关闭自动下载；
    VUS_SHERPA_MODELS / VUS_OFFLINE_ASR_MODELS 可指定模型目录。
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
# 下载允许的官方/镜像主机白名单（含 release 资产重定向目标与 HF 域族）
_ALLOWED_HOSTS = {
    "github.com", "objects.githubusercontent.com", "codeload.github.com",
    "hf-mirror.com", "huggingface.co", "modelscope.cn",
}
_ALLOWED_HOST_SUFFIXES = (".huggingface.co", ".hf.co", ".xethub.hf.co",
                          ".modelscope.cn")

# 国内可达源（hf-mirror.com = HuggingFace 国内镜像；文件为上游转换者 csukuangfj
# 上传的同一批模型，实测与官方源字节一致，下载后同样过 sha256 校验）。
# 链序：官方 github → hf-mirror（国内直连，绕开 github 不可达）→ 自建镜像（github 兜底）。
_OFFLINE_ASR_HF_FILES = [
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/model.int8.onnx",
     "model.int8.onnx"),
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/resolve/main/tokens.txt",
     "tokens.txt"),
]
_STREAMING_ASR_HF_FILES = [
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/encoder-epoch-99-avg-1.int8.onnx",
     "encoder-epoch-99-avg-1.int8.onnx"),
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/decoder-epoch-99-avg-1.int8.onnx",
     "decoder-epoch-99-avg-1.int8.onnx"),
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/joiner-epoch-99-avg-1.int8.onnx",
     "joiner-epoch-99-avg-1.int8.onnx"),
    ("https://hf-mirror.com/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/resolve/main/tokens.txt",
     "tokens.txt"),
]

# 多源 fallback 下载链：官方源 → 国内镜像（hf-mirror 文件直下）→ 自建源（vus-models 仓库）。
# 字符串项 = tar.bz2 压缩包源；列表项 = [(url, 相对文件名)] 文件直下源（无需解压）。
OFFLINE_ASR_SOURCES = [
    # 官方源（sherpa-onnx releases）
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09.tar.bz2",
    # 国内镜像（hf-mirror.com 直连）
    _OFFLINE_ASR_HF_FILES,
    # 自建源（CommitStrip/vus-models 仓库，int8 精简包）
    "https://github.com/CommitStrip/vus-models/releases/download/v1.0/sense-voice-int8.tar.bz2",
]
STREAMING_ASR_SOURCES = [
    # 官方源（sherpa-onnx releases，完整包含 fp32+int8）
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
    "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
    # 国内镜像（hf-mirror.com 直连，仅 int8 四件套，~190MB）
    _STREAMING_ASR_HF_FILES,
    # 自建源（int8 精简包，体积 490→170MB）
    "https://github.com/CommitStrip/vus-models/releases/download/v1.0/streaming-zipformer-int8.tar.bz2",
]

# 下载完整性校验（防代理截断/传输损坏）：解压后对**主模型文件**做 sha256 校验。
# 官方源与镜像源的压缩包打包方式不同，但模型文件内容一致——因此校验内容而非压缩包。
# (期望文件名, 期望 sha256)；文件缺失或哈希不符视为该源失败，自动尝试下一源。
_OFFLINE_ASR_VERIFY = (
    "model.int8.onnx",
    "12ca1a2ae7ecf3e0019ef2822307ee0b5cadc9196569e379b4c4026f8205276d",
)
_STREAMING_ASR_VERIFY = (
    "encoder-epoch-99-avg-1.int8.onnx",
    "8fa764187a261844f859d7143ebaa563af5d10adfece4c18a8f414c88cba2a9b",
)


def _validate_url(url):
    """下载前校验：仅允许 https 且 host 在白名单（精确或域族后缀，防 SSRF，CWE-918）。"""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    ok = parsed.scheme == "https" and (
        host in _ALLOWED_HOSTS or host.endswith(_ALLOWED_HOST_SUFFIXES))
    if not ok:
        raise ValueError(f"仅允许从官方/镜像 https 源下载: {url}")
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


def _sha256_of(path, bufsize=1024 * 1024):
    """流式计算文件 sha256（模型文件数百 MB，不整读进内存）。"""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_file(root, filename):
    """在目录树中定位指定文件名，返回绝对路径或 None。"""
    for r, _dirs, files in os.walk(root):
        if filename in files:
            return os.path.join(r, filename)
    return None


def _curl_to(url, out_path):
    """系统 curl 下载（-C - 断点续传，弱网中断后重跑可续传不重来）。"""
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-C", "-",
         "-o", out_path, url],
        check=True,
    )


def _download_from_sources(sources, dest_dir, label, verify=None):
    """多源 fallback 下载：按顺序尝试每个源，任一成功且通过校验即返回。

    sources 项两种形态：
      str                     —— tar.bz2 压缩包源（下载后解压）；
      [(url, 相对文件名), ...] —— 文件直下源（如 hf-mirror 已解压仓库，逐文件下载）。
    verify: (期望文件名, 期望 sha256)——完成后校验主模型文件内容完整性，
            不符则清理本次产物并尝试下一源（防代理截断的静默损坏）。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for src in sources:
        archive = str(dest) + "/model-download.tar.bz2"
        try:
            if isinstance(src, str):
                _validate_url(src)
                size_hint = "166MB" if "sense-voice" in src.lower() else "490MB"
                print(f"[ASR] {label}: 尝试下载（约 {size_hint}）…")
                _curl_to(src, archive)
                print(f"[ASR] 下载完成，解压中…")
                with tarfile.open(archive, "r:bz2") as tf:
                    tf.extractall(dest, filter="data")
                Path(archive).unlink()
            else:  # 文件直下源
                src_name = src[-1][0].rsplit("/", 1)[-1]
                print(f"[ASR] {label}: 尝试国内镜像直连（{src_name} 等 "
                      f"{len(src)} 个文件）…")
                for url, relname in src:
                    _validate_url(url)
                    out = dest / relname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    _curl_to(url, str(out))
            # 扫描 dest 找到包含 tokens.txt + model/encoder onnx 的目录
            model_root = None
            for root, _dirs, files in os.walk(str(dest)):
                has_tokens = "tokens.txt" in files
                has_model = any(f.endswith(".onnx") for f in files)
                if has_tokens and has_model:
                    model_root = root
                    break
            if model_root is None:
                model_root = str(dest)  # 找到但结构不同，仍按解压目录处理
            if verify and not _verify_model_file(model_root, *verify):
                raise ValueError(
                    f"校验失败: {verify[0]} 缺失或 sha256 不符（源可能被截断）")
            print(f"[ASR] 模型就位: {model_root}")
            return model_root
        except Exception as e:
            src_desc = src if isinstance(src, str) else src[-1][0].rsplit("/", 1)[-1]
            print(f"[ASR] 下载源失败({src_desc.rsplit('/', 1)[-1]}): {e}")
            if Path(archive).exists():
                Path(archive).unlink()
            if verify:
                expected = _find_file(str(dest), verify[0])
                if expected and _sha256_of(expected) != verify[1]:
                    # 坏文件不留祸根：连解压产物一起清掉，交给下一源重下
                    import shutil
                    shutil.rmtree(os.path.dirname(expected), ignore_errors=True)
    return None


def _verify_model_file(model_root, filename, expected_sha256):
    """校验 model_root 内 filename 的 sha256；缺失或不符打印原因返回 False。"""
    path = _find_file(model_root, filename)
    if path is None:
        print(f"[ASR] 完整性校验: 未找到 {filename}")
        return False
    actual = _sha256_of(path)
    if actual != expected_sha256:
        print(f"[ASR] 完整性校验: {filename} sha256 不符"
              f"（期望 {expected_sha256[:12]}…，实际 {actual[:12]}…）")
        return False
    print(f"[ASR] 完整性校验通过: {filename}")
    return True


def download_asr_model(dest_dir=None):
    """多源下载流式 ASR 模型到 dest_dir（默认 ./models/sherpa）。

    下载链: 官方 sherpa-onnx release → CommitStrip/vus-models mirror，
    逐源带 sha256 完整性校验（_STREAMING_ASR_VERIFY）。
    返回模型目录路径字符串；全部失败返回 None。
    """
    return _download_from_sources(STREAMING_ASR_SOURCES,
                                  dest_dir or asr_models_dir(), "流式 ASR",
                                  verify=_STREAMING_ASR_VERIFY)


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

    下载链: 官方 sherpa-onnx release → CommitStrip/vus-models mirror，
    逐源带 sha256 完整性校验（_OFFLINE_ASR_VERIFY）。
    返回模型目录字符串；全部失败返回 None。
    """
    return _download_from_sources(OFFLINE_ASR_SOURCES,
                                  dest_dir or offline_asr_dir(),
                                  "离线 SenseVoice", verify=_OFFLINE_ASR_VERIFY)


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
