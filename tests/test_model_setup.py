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
    # 相对/绝对均可，解析后须指向同一目录
    import os as _os
    assert _os.path.realpath(found) == _os.path.realpath(str(d))


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
