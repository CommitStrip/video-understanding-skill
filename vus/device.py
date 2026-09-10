#!/usr/bin/env python3
"""
device.py - 厂商无关的推理设备解析（v1.1）
==========================================
把 `--device` / 环境变量 VUS_DEVICE 的请求解析为各推理模块实际可用的
执行提供者（Execution Provider），统一三类机器的加速入口：

  请求:  auto(默认) / cpu / cuda / directml / coreml / rocm
  auto:  探测已安装引擎实际支持的 EP，按 cuda > directml > coreml > rocm 择优；
         什么都没装就是 cpu——任何机器都不被排除，行为与 v1.0 一致
  回退:  请求的设备不可用、或不被该模块支持时，打印原因回退 cpu，绝不崩溃

各模块的 GPU 支持面由上游引擎决定（如实收窄，不做假承诺）：
  asr   sherpa-onnx  → cuda / coreml（官方轮子按 flavor 提供；无 DirectML 包）
  clip  onnxruntime  → cuda / directml / coreml / rocm（装对应 ORT 轮即生效）
  ocr   rapidocr     → cuda / directml（det/cls/rec 均有 use_cuda / use_dml 配置）

安装指南（python -m vus.device 可在本机直接打印）：
  NVIDIA 显卡（Win/Linux）  pip install onnxruntime-gpu
                            pip install sherpa-onnx==<版本>+cuda12.cudnn9
                            -f https://k2-fsa.github.io/sherpa/onnx/cuda.html
                            （驱动向下兼容 CUDA 12.x 运行时，无需装 CUDA Toolkit）
  AMD / Intel 显卡（Win10+）pip install onnxruntime-directml（CLIP 生效，零版本匹配）
  Mac（Apple Silicon）      标准轮自带 CoreML EP（CLIP 生效）
  无 GPU / 其他             不装，默认 CPU（画面链本就 5.9× 实时，无需 GPU）
"""

import os
import sys

# 合法请求与 auto 择优顺序
DEVICE_REQUESTS = ("auto", "cpu", "cuda", "directml", "coreml", "rocm")
_AUTO_ORDER = ("cuda", "directml", "coreml", "rocm")

# 各模块支持的设备（cpu 之外；由上游引擎能力决定）
SUPPORTED = {
    "asr": ("cuda", "coreml"),
    "clip": ("cuda", "directml", "coreml", "rocm"),
    "ocr": ("cuda", "directml"),
}

# ORT EP 名与设备 token 的映射
_ORT_EP = {
    "cuda": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def requested(cli_value=None):
    """归一化设备请求：CLI 参数 > 环境变量 VUS_DEVICE > auto。非法值回退 auto。"""
    v = str(cli_value or "").strip().lower() \
        or os.environ.get("VUS_DEVICE", "").strip().lower() or "auto"
    if v not in DEVICE_REQUESTS:
        print(f"[Device] 未知设备请求 '{v}'（可选: {', '.join(DEVICE_REQUESTS)}），回退 auto")
        return "auto"
    return v


def _ort_providers():
    """pip onnxruntime 实际可用的 EP 列表；未安装返回 []。"""
    try:
        import onnxruntime
        return list(onnxruntime.get_available_providers())
    except Exception:
        return []


def _sherpa_version():
    """sherpa-onnx 版本串（含轮子 flavor，如 '1.13.7+cuda12.cudnn9'）；未安装返回 ''。"""
    try:
        import sherpa_onnx
        return getattr(sherpa_onnx, "__version__", "") or ""
    except Exception:
        return ""


def engine_available(engine, token):
    """指定引擎家族（'sherpa' | 'ort'）下设备 token 是否真实可用。

    sherpa-onnx 自带独立 onnxruntime（轮子 flavor 决定 GPU 能力），
    与 pip onnxruntime 包互不相干，因此按引擎家族分别探测。
    """
    if token == "cpu":
        return True
    if engine == "sherpa":
        v = _sherpa_version()
        if token == "cuda":
            return "+cuda" in v
        if token == "coreml":
            return "+coreml" in v
        return False
    return _ORT_EP.get(token) in _ort_providers()


def resolve_device(request=None, module="asr", engine="ort"):
    """解析请求为该模块实际可用的设备 token（cpu/cuda/directml/coreml/rocm）。

    module: asr / clip / ocr（决定支持面）；engine: sherpa / ort（决定探测源）。
    auto 无 GPU 可用、或请求被回退时均返回 "cpu"——与 v1.0 行为一致。
    """
    req = requested(request)
    supported = SUPPORTED.get(module, ("cpu",))
    if req == "cpu":
        return "cpu"
    if req == "auto":
        for tok in _AUTO_ORDER:
            if tok in supported and engine_available(engine, tok):
                print(f"[Device] {module}: auto 选择 {tok}")
                return tok
        return "cpu"
    if req not in supported:
        print(f"[Device] {module} 不支持 {req}（上游引擎限制），回退 cpu")
        return "cpu"
    if not engine_available(engine, req):
        print(f"[Device] {req} 引擎未安装，{module} 回退 cpu（python -m vus.device 查看装法）")
        return "cpu"
    print(f"[Device] {module}: 使用 {req}")
    return req


def ort_provider_list(token):
    """设备 token → ORT InferenceSession 的 providers 列表（含 CPU 兜底）。"""
    if token in _ORT_EP:
        return [_ORT_EP[token], "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def sherpa_provider(token):
    """设备 token → sherpa-onnx 的 provider 字符串（DirectML/ROCm 无对应 → cpu）。"""
    return token if token in ("cuda", "coreml") else "cpu"


def _nvidia_gpus():
    """探测 NVIDIA 显卡（nvidia-smi 可用时返回名称列表，否则 []）。"""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                           text=True, timeout=10, check=False)
        if r.returncode != 0:
            return []
        return [ln.split("(")[0].strip(" :") for ln in r.stdout.splitlines()
                if ln.strip().startswith("GPU")]
    except Exception:
        return []


def main():
    """python -m vus.device：打印本机硬件 / 已装引擎 / 每模块生效设备与安装建议。"""
    gpus = _nvidia_gpus()
    sherpa_v = _sherpa_version()
    try:
        import onnxruntime
        ort_desc = f"onnxruntime {onnxruntime.__version__}，EP: {', '.join(_ort_providers())}"
    except Exception:
        ort_desc = "onnxruntime 未安装"

    print("=== vus 设备自检 ===")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"NVIDIA 显卡:  {', '.join(gpus) if gpus else '未检测到'}")
    print(f"推理引擎:     {ort_desc}")
    print(f"sherpa-onnx:  {sherpa_v or '未安装'}"
          + ("（CPU 版轮子）" if sherpa_v and "+cuda" not in sherpa_v and "+coreml" not in sherpa_v else ""))
    print()

    for module, engine in (("asr", "sherpa"), ("clip", "ort"), ("ocr", "ort")):
        token = resolve_device(None, module=module, engine=engine)
        state = "GPU 加速" if token != "cpu" else "CPU"
        print(f"  {module:5s}: {token}（{state}）")
    print()

    print("安装建议（装完重跑本命令验证）:")
    if gpus:
        base = sherpa_v.split("+")[0] or "1.13.7"
        print("  NVIDIA 路径（ASR+CLIP+OCR 全生效）:")
        print(f"    pip install onnxruntime-gpu")
        print(f"    pip install sherpa-onnx=={base}+cuda12.cudnn9 "
              f"-f https://k2-fsa.github.io/sherpa/onnx/cuda.html")
        print("    （中国网络可将 -f 换为 https://k2-fsa.github.io/sherpa/onnx/cuda-cn.html）")
    print("  AMD / Intel 显卡（Windows，CLIP 生效）:")
    print("    pip install onnxruntime-directml")
    print("  Mac（Apple Silicon，CLIP 走 CoreML）: 标准轮自带，无需额外安装")
    print("  说明: onnxruntime 的 cpu/gpu/directml 三种轮子同包名互斥，只能装其一；")
    print("        画面链（OpenCV）无需 GPU。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
