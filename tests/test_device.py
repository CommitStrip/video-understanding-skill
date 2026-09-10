#!/usr/bin/env python3
"""test_device.py - v1.1 设备抽象层（vus/device.py）与 --device 贯通测试。

约束：全部进程内 monkeypatch/sys.modules 伪造引擎能力，不依赖真实 GPU，
不使用 subprocess。CPU 零回归：auto 在无 GPU 引擎时必须解析为 cpu。
"""

import sys
import types

import pytest

from vus import device
from vus.device import (
    ort_provider_list, requested, resolve_device, sherpa_provider,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VUS_DEVICE", raising=False)


@pytest.fixture
def no_gpu(monkeypatch):
    """引擎探测全部置空：等价于没装任何 GPU 轮子的机器。"""
    monkeypatch.setattr(device, "_ort_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(device, "_sherpa_version", lambda: "1.13.7")


# ==================== requested()：CLI > env > auto ====================

def test_requested_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("VUS_DEVICE", "cuda")
    assert requested("cpu") == "cpu"


def test_requested_env_fallback(monkeypatch):
    monkeypatch.setenv("VUS_DEVICE", "DIRECTML")
    assert requested(None) == "directml"


def test_requested_default_auto_and_invalid(monkeypatch):
    assert requested(None) == "auto"
    assert requested("tpu") == "auto"


# ==================== resolve_device(): 解析矩阵与回退 ====================

def test_cpu_request_stays_cpu(no_gpu):
    assert resolve_device("cpu", module="clip", engine="ort") == "cpu"


def test_auto_without_gpu_engines_is_cpu(no_gpu, capsys):
    """CPU 零回归锚点：auto + 无 GPU 引擎 = cpu（v1.0 行为）。"""
    assert resolve_device(None, module="asr", engine="sherpa") == "cpu"
    assert resolve_device(None, module="clip", engine="ort") == "cpu"
    assert resolve_device(None, module="ocr", engine="ort") == "cpu"


def test_explicit_cuda_without_sherpa_flavor_falls_back(no_gpu, capsys):
    assert resolve_device("cuda", module="asr", engine="sherpa") == "cpu"
    assert "回退 cpu" in capsys.readouterr().out


def test_explicit_cuda_with_sherpa_cuda_flavor(monkeypatch):
    monkeypatch.setattr(device, "_sherpa_version",
                        lambda: "1.13.7+cuda12.cudnn9")
    assert resolve_device("cuda", module="asr", engine="sherpa") == "cuda"


def test_asr_rejects_directml_by_engine_limit(monkeypatch, capsys):
    """sherpa 无 DirectML 包：即使 ORT 装了 DML，ASR 也回退 cpu。"""
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(device, "_sherpa_version", lambda: "1.13.7")
    assert resolve_device("directml", module="asr", engine="sherpa") == "cpu"
    assert "不支持 directml" in capsys.readouterr().out


def test_auto_picks_dml_for_clip(monkeypatch):
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(device, "_sherpa_version", lambda: "1.13.7")
    assert resolve_device(None, module="clip", engine="ort") == "directml"


def test_auto_prefers_cuda_over_dml(monkeypatch):
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["CUDAExecutionProvider", "DmlExecutionProvider",
                                 "CPUExecutionProvider"])
    monkeypatch.setattr(device, "_sherpa_version",
                        lambda: "1.13.7+cuda12.cudnn9")
    assert resolve_device(None, module="clip", engine="ort") == "cuda"
    assert resolve_device(None, module="asr", engine="sherpa") == "cuda"


# ==================== EP/provider 映射 ====================

def test_ort_provider_list_cpu_and_gpu():
    assert ort_provider_list("cpu") == ["CPUExecutionProvider"]
    assert ort_provider_list("cuda") == ["CUDAExecutionProvider",
                                         "CPUExecutionProvider"]
    assert ort_provider_list("directml") == ["DmlExecutionProvider",
                                             "CPUExecutionProvider"]


def test_sherpa_provider_mapping():
    assert sherpa_provider("cuda") == "cuda"
    assert sherpa_provider("coreml") == "coreml"
    assert sherpa_provider("directml") == "cpu"
    assert sherpa_provider("cpu") == "cpu"


# ==================== ClipOnnx 设备贯通（fake ORT 捕获 providers） ====================

class _FakeSession:
    def __init__(self, model_path=None, providers=None, **kw):
        self.providers = providers
        inp = types.SimpleNamespace(name="pixel_values", shape=[1, 3, 224, 224],
                                    type="tensor(float32)")
        out = types.SimpleNamespace(name="image_embeds", shape=[1, 512],
                                    type="tensor(float32)")
        self._inputs, self._outputs = [inp], [out]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, names, feed):
        import numpy as np
        return [np.zeros(512, dtype=np.float32)]


@pytest.fixture
def fake_ort_factory(monkeypatch):
    made = []

    def make(model_path, providers=None):
        s = _FakeSession(model_path=model_path, providers=providers)
        made.append(s)
        return s

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = make
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return made


@pytest.fixture
def fake_clip_model(tmp_path, monkeypatch):
    model = tmp_path / "clip-visual-vitb32.onnx"
    model.write_bytes(b"fake")
    monkeypatch.setenv("VUS_CLIP_MODELS", str(tmp_path))


def test_clip_onnx_cpu_explicit(fake_ort_factory, fake_clip_model, no_gpu):
    from vus.clip_onnx import ClipOnnx
    enc = ClipOnnx(device="cpu")
    assert enc.device == "cpu"
    assert fake_ort_factory[0].providers == ["CPUExecutionProvider"]


def test_clip_onnx_device_auto_picks_gpu(monkeypatch, fake_ort_factory,
                                         fake_clip_model):
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    from vus.clip_onnx import ClipOnnx
    enc = ClipOnnx()  # auto
    assert enc.device == "directml"
    assert fake_ort_factory[0].providers[0] == "DmlExecutionProvider"


def test_clip_onnx_unavailable_device_falls_back(fake_ort_factory,
                                                 fake_clip_model, no_gpu,
                                                 capsys):
    from vus.clip_onnx import ClipOnnx
    enc = ClipOnnx(device="cuda")  # 未装 onnxruntime-gpu
    assert enc.device == "cpu"
    assert "回退 cpu" in capsys.readouterr().out


# ==================== OcrChannel 设备贯通（kwargs 转发） ====================

@pytest.fixture
def fake_rapidocr_capture(monkeypatch):
    calls = []

    class _Engine:
        def __call__(self, frame):
            return [], []

    def factory(*a, **kw):
        calls.append(kw)
        return _Engine()

    fake = types.ModuleType("rapidocr_onnxruntime")
    fake.RapidOCR = factory
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake)
    return calls


def test_ocr_cpu_passes_no_kwargs(fake_rapidocr_capture, no_gpu):
    from vus.ocr_channel import OcrChannel
    ch = OcrChannel()
    assert ch.device == "cpu"
    assert fake_rapidocr_capture[0] == {}


def test_ocr_cuda_forwards_use_cuda(monkeypatch, fake_rapidocr_capture):
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    from vus.ocr_channel import OcrChannel
    ch = OcrChannel(device="cuda")
    assert ch.device == "cuda"
    assert fake_rapidocr_capture[0] == {"det_use_cuda": True, "cls_use_cuda": True,
                                        "rec_use_cuda": True}


def test_ocr_dml_forwards_use_dml(monkeypatch, fake_rapidocr_capture):
    monkeypatch.setattr(device, "_ort_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    from vus.ocr_channel import OcrChannel
    ch = OcrChannel()  # auto
    assert ch.device == "directml"
    assert fake_rapidocr_capture[0] == {"det_use_dml": True, "cls_use_dml": True,
                                        "rec_use_dml": True}


# ==================== ASR 加载器：GPU 失败自动回退 cpu ====================

def _make_fake_sherpa(monkeypatch, fail_on):
    """伪造 sherpa_onnx：provider 命中 fail_on 时构造抛异常，否则成功。"""
    attempts = []

    class _Rec:
        pass

    class _Online:
        @staticmethod
        def from_transducer(**kw):
            attempts.append(kw.get("provider"))
            if kw.get("provider") == fail_on:
                raise RuntimeError(f"no {fail_on} EP")
            return _Rec()

    class _Offline:
        @staticmethod
        def from_sense_voice(**kw):
            attempts.append(kw.get("provider"))
            if kw.get("provider") == fail_on:
                raise RuntimeError(f"no {fail_on} EP")
            return _Rec()

    fake = types.ModuleType("sherpa_onnx")
    fake.OnlineRecognizer = _Online
    fake.OfflineRecognizer = _Offline
    fake.__version__ = "1.13.7"
    monkeypatch.setitem(sys.modules, "sherpa_onnx", fake)
    return attempts


@pytest.fixture
def fake_stream_model(tmp_path, monkeypatch):
    for name in ("encoder-1.onnx", "decoder-1.onnx", "joiner-1.onnx",
                 "tokens.txt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setenv("VUS_SHERPA_MODELS", str(tmp_path))
    monkeypatch.setenv("VUS_ASR_AUTO_DOWNLOAD", "0")
    return tmp_path


def test_streaming_loader_cuda_failure_falls_back_cpu(monkeypatch,
                                                      fake_stream_model):
    from vus.asr_sherpa import load_streaming_recognizer
    attempts = _make_fake_sherpa(monkeypatch, fail_on="cuda")
    rec = load_streaming_recognizer(provider="cuda")
    assert rec is not None            # 回退成功，不是 None（None=整体不可用）
    assert attempts == ["cuda", "cpu"]


def test_streaming_loader_cpu_only_single_attempt(monkeypatch, fake_stream_model):
    from vus.asr_sherpa import load_streaming_recognizer
    attempts = _make_fake_sherpa(monkeypatch, fail_on="__never__")
    rec = load_streaming_recognizer(provider="cpu")
    assert rec is not None
    assert attempts == ["cpu"]


def test_offline_loader_cuda_failure_falls_back_cpu(monkeypatch, fake_stream_model):
    from vus.asr_sherpa import load_offline_recognizer
    (fake_stream_model / "model.int8.onnx").write_bytes(b"x")
    attempts = _make_fake_sherpa(monkeypatch, fail_on="cuda")
    rec = load_offline_recognizer(provider="cuda")
    assert rec is not None
    assert attempts == ["cuda", "cpu"]


def test_offline_loader_all_fail_raises(monkeypatch, fake_stream_model):
    from vus.asr_sherpa import load_offline_recognizer
    (fake_stream_model / "model.int8.onnx").write_bytes(b"x")
    _make_fake_sherpa(monkeypatch, fail_on="__never__")  # 不失败此路
    monkeypatch.setattr("vus.asr_sherpa.os.path.exists",
                        lambda p: False)  # 强制模型查找失败
    with pytest.raises(RuntimeError):
        load_offline_recognizer(provider="cuda", model_dir=str(
            fake_stream_model / "missing"))


# ==================== 自检命令（python -m vus.device） ====================

def test_device_main_report_cpu_machine(no_gpu, monkeypatch, capsys):
    monkeypatch.setattr(device, "_nvidia_gpus", lambda: [])
    assert device.main() == 0
    out = capsys.readouterr().out
    assert "vus 设备自检" in out
    assert "未检测到" in out  # NVIDIA 显卡行
    assert "onnxruntime-directml" in out  # 安装建议仍在


def test_device_main_report_nvidia_without_cuda_engine(no_gpu, monkeypatch,
                                                       capsys):
    monkeypatch.setattr(device, "_nvidia_gpus",
                        lambda: ["NVIDIA GeForce RTX 3060"])
    device.main()
    out = capsys.readouterr().out
    assert "RTX 3060" in out
    assert "+cuda12.cudnn9" in out      # sherpa CUDA 轮建议
    assert "onnxruntime-gpu" in out     # ORT GPU 轮建议
    assert "cuda-cn.html" in out        # 中国镜像提示
