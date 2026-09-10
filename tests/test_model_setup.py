"""model_setup 单元测试（不联网、不下载）。"""
import os

import pytest

from vus import model_setup
from vus.model_setup import _validate_url, find_asr_model


def test_validate_url_accepts_official_https():
    url = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
           "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2")
    assert _validate_url(url) == url


@pytest.mark.parametrize("bad", [
    "http://github.com/x.tar.bz2",                 # 明文 http
    "https://evil.example.com/x.tar.bz2",          # 非白名单 host
    "ftp://github.com/x",                          # 非 http(s) 协议
    "https://127.0.0.1/x",                         # 环回地址
    "https://192.168.1.10/x",                      # 私网地址
])
def test_validate_url_rejects_non_official(bad):
    with pytest.raises(ValueError):
        _validate_url(bad)


def test_find_asr_model_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("VUS_SHERPA_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)  # tmp 下无 ./models/sherpa
    (tmp_path / "home").mkdir()
    monkeypatch.setattr(model_setup.os.path, "expanduser", lambda p: str(tmp_path / "home" / "none"))
    assert find_asr_model() is None


def test_find_asr_model_rejects_incomplete_skeleton(tmp_path, monkeypatch):
    """只有目录骨架（下载中断残留）不算就位——这是实战踩过的坑。"""
    d = tmp_path / "models" / "sherpa" / "model"
    d.mkdir(parents=True)
    (d / "tokens.txt").write_text("")          # 有 tokens 无 encoder
    monkeypatch.setenv("VUS_SHERPA_MODELS", str(d.parent))
    monkeypatch.chdir(tmp_path)
    assert find_asr_model() is None


def test_find_asr_model_env_dir(tmp_path, monkeypatch):
    d = tmp_path / "mymodels"
    d.mkdir()
    (d / "tokens.txt").write_text("a b\n")
    (d / "encoder-1.onnx").write_bytes(b"x")
    monkeypatch.setenv("VUS_SHERPA_MODELS", str(d))
    assert find_asr_model() == str(d)


def test_find_asr_model_recursive_subdir(tmp_path, monkeypatch):
    """官方 tar 解压后模型在子目录里，应能递归找到。"""
    d = tmp_path / "models" / "sherpa" / "sherpa-onnx-streaming-x"
    d.mkdir(parents=True)
    (d / "tokens.txt").write_text("a b\n")
    (d / "encoder-1.onnx").write_bytes(b"x")
    monkeypatch.delenv("VUS_SHERPA_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_setup.os.path, "expanduser", lambda p: str(tmp_path / "none"))
    found = find_asr_model()
    # 返回值是搜索根（相对/绝对均可），其下必须能递归找到完整模型文件
    assert found is not None
    for root, _dirs, files in os.walk(found):
        if "tokens.txt" in files and any(
                f.startswith("encoder") and f.endswith(".onnx") for f in files):
            break
    else:
        pytest.fail(f"返回目录下找不到完整模型文件: {found}")


def test_ensure_auto_disabled_never_downloads(tmp_path, monkeypatch):
    """VUS_ASR_AUTO_DOWNLOAD=0 时缺失即返回 None，不发起任何下载。"""
    monkeypatch.delenv("VUS_SHERPA_MODELS", raising=False)
    monkeypatch.setenv("VUS_ASR_AUTO_DOWNLOAD", "0")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_setup.os.path, "expanduser", lambda p: str(tmp_path / "none"))

    called = False
    def _fail_download(*a, **k):
        nonlocal called
        called = True
        return None
    monkeypatch.setattr(model_setup, "download_asr_model", _fail_download)

    assert model_setup.ensure_asr_model() is None
    assert called is False


def test_ensure_auto_enabled_calls_download(tmp_path, monkeypatch):
    monkeypatch.delenv("VUS_SHERPA_MODELS", raising=False)
    monkeypatch.setenv("VUS_ASR_AUTO_DOWNLOAD", "1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_setup.os.path, "expanduser", lambda p: str(tmp_path / "none"))

    def _fake_download(*a, **k):
        return str(tmp_path / "downloaded")
    monkeypatch.setattr(model_setup, "download_asr_model", _fake_download)

    assert model_setup.ensure_asr_model() == str(tmp_path / "downloaded")


# ==================== v1.1.x: 多源下载 sha256 校验 + 断点续传 + 国内镜像源 ====================

import hashlib
import io
import subprocess as _sp
import tarfile
from pathlib import Path


def _fake_curl(monkeypatch, payloads):
    """伪造 _curl_to：按 url 取内容写到 -o 路径；内容为异常实例则抛出。返回全部 argv。"""
    calls = []

    def _run(url, out_path):
        calls.append(["curl", "-o", str(out_path), url])
        payload = payloads[url]
        if isinstance(payload, Exception):
            raise payload
        Path(out_path).write_bytes(payload)

    monkeypatch.setattr(model_setup, "_curl_to", _run)
    return calls


def _tar_bytes(files):
    """内存构造 tar.bz2（{name: bytes}）。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:bz2") as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_verify_ok(tmp_path, monkeypatch):
    content = b"onnx-bytes"
    good_hash = hashlib.sha256(content).hexdigest()
    url = "https://github.com/x/m.tar.bz2"
    _fake_curl(monkeypatch, {
        url: _tar_bytes({"tokens.txt": b"t", "model.int8.onnx": content})})
    got = model_setup._download_from_sources(
        [url], tmp_path / "d", "测试", verify=("model.int8.onnx", good_hash))
    assert got is not None


def test_curl_args_include_resume_and_retry(tmp_path, monkeypatch):
    seen = []

    def _run(cmd, check=True):
        seen.append(list(cmd))
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"x")

    monkeypatch.setattr(model_setup.subprocess, "run", _run)
    model_setup._curl_to("https://github.com/x/a", str(tmp_path / "a.bin"))
    c = seen[0]
    assert c[0] == "curl" and "--fail" in c
    assert c[c.index("-C") + 1] == "-"          # 断点续传
    assert c[c.index("--retry") + 1] == "3"


def test_download_hash_mismatch_falls_to_next_source(tmp_path, monkeypatch):
    bad, good = b"corrupted", b"onnx-bytes"
    good_hash = hashlib.sha256(good).hexdigest()
    url_bad = "https://github.com/x/bad.tar.bz2"
    url_file = "https://hf-mirror.com/csukuangfj/x/resolve/main/model.int8.onnx"
    url_tok = "https://hf-mirror.com/csukuangfj/x/resolve/main/tokens.txt"
    _fake_curl(monkeypatch, {
        url_bad: _tar_bytes({"tokens.txt": b"t", "model.int8.onnx": bad}),
        url_file: good,
        url_tok: b"t",
    })
    got = model_setup._download_from_sources(
        [url_bad, [(url_file, "model.int8.onnx"), (url_tok, "tokens.txt")]],
        tmp_path / "d", "测试", verify=("model.int8.onnx", good_hash))
    assert got is not None
    assert (tmp_path / "d" / "model.int8.onnx").read_bytes() == good


def test_download_all_fail_returns_none(tmp_path, monkeypatch):
    url = "https://github.com/x/m.tar.bz2"
    _fake_curl(monkeypatch, {url: _sp.CalledProcessError(28, "curl")})
    got = model_setup._download_from_sources(
        [url], tmp_path / "d", "测试", verify=("model.int8.onnx", "ab" * 32))
    assert got is None


def test_expected_file_missing_is_failure(tmp_path, monkeypatch):
    url = "https://github.com/x/m.tar.bz2"
    _fake_curl(monkeypatch, {url: _tar_bytes({"tokens.txt": b"t"})})
    got = model_setup._download_from_sources(
        [url], tmp_path / "d", "测试", verify=("model.int8.onnx", "ab" * 32))
    assert got is None


def test_validate_url_allows_cn_mirrors_rejects_other():
    ok = model_setup._validate_url
    assert ok("https://hf-mirror.com/a/b") == "https://hf-mirror.com/a/b"
    assert ok("https://modelscope.cn/models/x") == "https://modelscope.cn/models/x"
    assert ok("https://cdn-lfs-us-1.hf.co/x") == "https://cdn-lfs-us-1.hf.co/x"
    for bad in ("http://hf-mirror.com/a", "https://example.com/a",
                "https://hf-mirror.com.evil.io/a"):
        try:
            ok(bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, bad


def test_chains_include_cn_file_source():
    assert any(not isinstance(s, str) for s in model_setup.OFFLINE_ASR_SOURCES)
    assert any(not isinstance(s, str) for s in model_setup.STREAMING_ASR_SOURCES)
    flat = [u for s in model_setup.STREAMING_ASR_SOURCES if not isinstance(s, str)
            for u, _ in s]
    assert all(u.startswith("https://hf-mirror.com/") for u in flat)
