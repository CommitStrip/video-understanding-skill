# GPU 加速实测报告（v1.1）

日期：2026-09-10 ｜ 环境：Windows 11 笔记本，RTX 3060 Laptop（6GB）+ Intel Iris Xe 核显，
驱动 616.56，Python 3.12.10。复现脚本：`bench/gpu/bench_clip.py`、`bench/gpu/bench_asr.py`。

## 结论先行

1. **厂商无关的设备层工作正常**：`--device auto` 在 CPU 轮 / DirectML 轮 / CUDA 轮三种
   组合下均正确择优；请求不可用时自动回退 CPU 并打印原因；回退后转写结果与 CPU 逐字一致。
2. **batch=1 的 CLIP 单帧推理，CPU 反而最快**（实测三种 EP 对比，见下表）。GPU 收益需要
   批量推理——Tier3 现行逐帧打分路径下，强 CPU 机器不必开 GPU；弱 CPU（2 核级别）+ 核显
   机器可能受益，未实测。
3. **Intel 核显经 DirectML 成功运行 CLIP**（25.7ms/帧）——AMD/Intel 显卡用户路径可用性实证。
4. **ASR CUDA（sherpa-onnx Windows 轮）当前不可用**：上游 `1.13.7+cuda12.cudnn9` 轮的
   provider DLL 初始化失败（WinError 1114），独立于本项目代码（依赖 DLL 均可单独加载）。
   自动回退 CPU 已实证有效。Linux 待实测。

## CLIP ViT-B/32 ONNX 单帧嵌入（823 张真实关键帧，11 分钟抖音视频产物）

| 执行路径 | ms/帧 | 帧率 | 备注 |
|---|---|---|---|
| CPU（onnxruntime 1.29.0） | **15.95 / 16.68**（两次运行） | 63 / 60 fps | 基线 |
| DirectML @ RTX 3060 | 26.29 / 26.66（两次运行） | 38 fps | onnxruntime-directml 1.24.4 |
| DirectML @ Intel Iris Xe | 25.7（冒烟，20 帧） | ~39 fps | 与独显 DML 同档 |
| CUDA @ RTX 3060 | 34.53 | 29 fps | onnxruntime-gpu 1.22 + cu12 pip 运行时 |

反面结论说明：合一 CLIPModel 图含文本分支（ORT 报告 348 个 Memcpy 节点），单帧 CPU↔GPU
往返开销远超 ViT-B/32 计算收益。若未来 Tier3 改为批量打分（多帧一次推理），GPU 路径可重新
评估；v1.1 不做此改动。

## 离线 ASR（SenseVoice int8，180 秒真实音频）

| 执行路径 | RTF | 转写文本 | 备注 |
|---|---|---|---|
| CPU | **0.0266–0.0268** | 12 段 / 676 字 | 基线 |
| CUDA（sherpa +cuda12.cudnn9 轮） | （未生效） | 回退 CPU 后**逐字一致** | 上游轮子 Windows 初始化失败 |

sherpa CUDA 轮问题定位记录：`sherpa_onnx/lib/onnxruntime_providers_cuda.dll` 加载时
WinError 1114（初始化例程失败）；其全部 CUDA 依赖（cudart64_12 / cublas64_12 /
cudnn64_9 / cufft64_11 / nvrtc / nvJitLink，来自 pip nvidia-cu12 系列）均可单独加载成功，
排除缺 DLL；无 pip onnxruntime 介入时同样失败，排除版本冲突。判定为上游轮子问题。

## 环境注意事项（复现前必读）

- onnxruntime 的 cpu / gpu / directml 三种轮子**同包名互斥**，只能装其一。
- sherpa-onnx 生态已拆分包：`sherpa-onnx`（Python 前端，依赖）+ `sherpa-onnx-core`
  （原生库）。混装旧版单体轮会导致 ABI 不匹配（"requested API version 27 not available"）
  甚至段错误——重装时用 `pip install sherpa-onnx`（不锁旧版单体轮）。
- NVIDIA 显卡跑 CUDA 路径无需装系统级 CUDA Toolkit：`pip install nvidia-cudnn-cu12
  nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cufft-cu12`（这些 cu12 轮带 Windows
  DLL），vus 会自动预载（`vus/device.py: preload_cuda_dlls`）。
- 本报告所有对比均在同一台笔记本、同一段素材上完成；笔记本热节流会带来 ±10% 波动，
  两轮 CPU 基线（15.95 与 16.68ms）即为波动范围参考。
