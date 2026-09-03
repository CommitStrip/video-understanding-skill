# vus — Video Understanding Skill（视频理解技能）

[![CI](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml)

[English](README.md) | **简体中文**

把一段视频变成**结构化、可被大模型高效消费的内容理解产物**：语义代表帧 + 时间轴对齐的 ASR 字幕 + 运动段。把 30fps 的十几万帧原始视频压缩成几十张多模态模型真正读得过来的代表帧——且不漏关键内容。

实战基准：120 分钟 1080p25 直播课程——**处理速率 147.7fps（5.9× 实时）**、41 个关键帧、3505 段真实中文字幕（约 3.3 万字）、内存全程稳定。

## 核心特性

- **实时预算分配**——快系统（逐帧运动门控，约占 3% 预算）触发慢系统（低频关键帧 + 触发式重活）。能直接跑在机器人/边缘设备上，不只是文件回放。
- **实时源**——视频文件 / 摄像头 / RTSP 流统一 `FrameSource` 接口；RTSP 带最新帧背压、断流自动重连、单调时钟时间戳。
- **三层压缩**——原始帧 → 镜头级关键帧 → 语义代表帧，解决"镜头切换 ≠ 内容变化"的冗余问题。
- **渐变演化不丢帧**——课件批注推进、镜头缓摇这类缓慢内容演化，由漂移确认机制捕获，不再只认硬切换。
- **真 ASR**——基于 sherpa-onnx 的中英双语流式语音识别，词级时间戳；模型缺失时显式降级、绝不静默造假。
- **可选语义增强**——CLIP（ONNX，不依赖 PyTorch）语义选帧；OCR 通道提取课件文字。
- **可安装、有测试**——`pip install -e .`，96 个 pytest 用例，GitHub Actions CI。

## 安装

```bash
pip install -e .                 # 核心（opencv-python + numpy）

# 可选 extras
pip install -e ".[asr]"          # sherpa-onnx 流式 ASR（真字幕）
pip install -e ".[clip]"         # CLIP 语义选帧（ONNX）
pip install -e ".[ocr]"          # OCR 通道
```

模型权重不入库。**真 ASR 默认开箱即用**：首次运行时若缺模型会自动从官方
sherpa-onnx release 下载（约 490MB，一次性）。设 `VUS_ASR_AUTO_DOWNLOAD=0`
可关闭自动下载，或手动获取：

```bash
# ASR 模型（小米 sherpa-onnx 流式 zipformer，中英双语，约 490MB）——默认自动下载
mkdir -p models/sherpa
curl -L -o - https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 \
  | tar -xj -C models/sherpa --strip-components=1

# CLIP 视觉编码器（ONNX，约 600MB）——可选项，经下载脚本获取
bash scripts/download_clip_onnx.sh
```

模型目录可用环境变量 `VUS_SHERPA_MODELS` / `VUS_CLIP_MODELS` 覆盖。

> ⚠️ 未安装 sherpa-onnx 时字幕通道退化为 **mock 占位输出——是假文本，不是真实转写**。严禁把 mock 字幕当作真实内容交付。

## 快速开始

```bash
# 1. 提取结构化产物（关键帧 + 运动段 + 字幕）
python -m vus.integrated_pipeline --video lecture.mp4 --output out/ --kf-hz 1.5

# 直播 / RTSP 实时流变体
python -m vus.integrated_pipeline --source rtsp --url rtsp://host/stream --output out/

# 2. 压缩为语义代表帧（Tier 3）
python -m vus.select_representatives --keyframes out/keyframes \
  --interval 60 --out representatives.json --report context.md

# 多人近景轮换（圆桌/访谈）？保留桶内多样性：
python -m vus.select_representatives --keyframes out/keyframes \
  --interval 60 --k 3 --out representatives.json

# 3. 把代表帧 + context.md + aligned_output.json 交给多模态大模型，
#    生成课程讲义 / 剧情摘要 / 场景分析报告
```

多模态模型只需读 `representatives.json` 里的代表帧和对齐字幕，即可产出结构化理解报告。

## 工作原理

| 层级 | 内容 | 数量级 | 用途 |
|------|------|--------|------|
| 0 | 原始帧 | 30fps（10⁵ 帧） | 播放 |
| 1 | 快系统运动事件 | 逐帧 | "有没有事发生" |
| 2 | 镜头级关键帧 | 1-2s 一张 | 时间轴锚定 |
| 3 | **语义代表帧** | 30-60s 一张 | **大模型理解** |

快系统在降采样灰度小图上逐帧运行（帧差 + 语义门控，约占 3% 预算）；慢系统低频采样、仅在快系统报有内容时触发，用像素差分、pHash、直方图给候选打分——并配**渐变漂移确认**，覆盖感知哈希看不见的缓慢内容演化。

## 性能

### 实战（120 分钟 1080p25 直播课程，18 万帧）

| 指标 | 结果 |
|------|------|
| 处理速率 | 147.7fps（**5.9× 实时**） |
| 关键帧 | 41 个（35 渐变漂移 + 5 场景切换），覆盖 0→7150s 全片 |
| ASR | 3505 段、约 3.3 万字，RTF 0.08（与画面链并行） |
| 内存 | 稳定，ASR 模型释放后约 225MB |

### 合成视频实时率（低配 2 核 Windows）

| 规格 | 速率 | 实时倍数 |
|------|------|---------|
| 720p50 | 247fps | 4.9× |
| 1080p30 | 78fps | 2.6× |

### 与 claude-real-video（crv）对比

4 组 12 秒可控合成片段上的受控对比（细节与复现见 `bench/`）：`static` 片段 crv 完全漏掉片尾突变（覆盖率 0%），本技能 2 帧完整捕获；slow/hue 渐变下同覆盖率帧数减半；端到端快约 3.5 倍。

> 说明：该对比表的覆盖率指标以像素差分定义——与本技能选帧信号同源，结构上偏向本技能（`static` 的结论独立成立）。语义级评估协议（标注指南 + 覆盖率/冗余度指标）见 `bench/semantic_eval/`。

## 仓库结构

```
vus/                       可安装核心（pip install -e .）
  smart_pipeline.py        快/慢双系统画面链
  integrated_pipeline.py   三通道编排（画面 + ASR + 对齐）
  asr_sherpa.py            流式 ASR（sherpa-onnx / 显式降级）
  select_representatives.py Tier3 语义选帧（--k/--adaptive/--clip）
  source.py                FileSource / CameraSource / RTSPSource
  clip_onnx.py             onnxruntime 版 CLIP ViT-B/32（无 torch）
  ocr_channel.py           可选 OCR 通道
  io_utils.py, pathsafe.py 安全落盘写（防路径穿越）
scripts/                   旧命令入口（薄壳，继续可用）
bench/                     crv 对比 + 语义级评估协议
tests/                     96 个 pytest 用例 + 端到端冒烟
```

## 作为 AI 技能使用

本仓库就是一个开箱即用的 agent 技能：把整个目录拷进你的 agent 技能目录
（如 `~/.agents/skills/video-understanding-skill/`），内置的 `SKILL.md`
会教会 agent 何时、如何运行管线——包括模型准备与 mock 字幕陷阱。无需安装：
`scripts/` 旧入口自带路径兜底。

## 硬件资源（实测）

2 核 / 4GB 环境：实时画面链约占 1.2 核 + 166MB 内存；Tier3 离线选帧约
317MB（内存有界）；流式 ASR 解码期间额外 300-500MB。只跑画面链 512MB
内存即可；配 ASR 建议 2GB。

## 致谢与版权

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)——流式 ASR 引擎，
  由小米（k2-fsa）开源维护。本仓库仅在 `vus/asr_sherpa.py` 中做轻量封装；
  默认引用的模型 `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`
  遵循上游 Apache-2.0 协议及模型自身发布条款。对引擎或模型的二次分发、
  商用请自行遵守上游条款。
- [openai/CLIP](https://github.com/openai/CLIP) ViT-B/32——语义编码器（ONNX 导出）。
- [claude-real-video](https://github.com/davecap/claude-real-video)——`bench/` 对比基线。

## License

MIT © 2026
