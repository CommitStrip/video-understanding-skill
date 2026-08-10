# Video Understanding Skill

把一段视频变成 **结构化、可被 LLM 高效消费的内容理解产物**。采用"三层压缩"设计：将 30fps 的十几万帧压缩为几十张语义代表帧 + 时间轴对齐字幕 + 运动段，让 AI 既能理解内容又不漏关键信息。

源自真实项目：75 分钟视频，管线产出 1609 张关键帧，LLM 经语义压缩后只需约 60-80 张代表帧即可完整理解。

## 核心价值

- **实时预算分配**：快系统（逐帧运动检测）+ 慢系统（低频关键帧），可实时跑在机器人/产品上（实测 720p50 达 441fps，1080p30 达 202fps，均超实时数倍）。
- **三层压缩**：原始帧 → 镜头级关键帧 → 语义代表帧，解决"镜头切换 ≠ 内容变化"的冗余问题。
- **流式输出**：每帧产出增量事件，宿主无需等整段即可实时消费。
- **多模态融合**：画面（运动+关键帧）+ 声音（流式 ASR）时间轴对齐，输出结构化语义流。

## 快速开始

```bash
# 1. 安装依赖
pip install opencv-python numpy
# 可选：流式 ASR（缺省走 mock fallback）
pip install sherpa-onnx

# 2. 跑管线提取结构化产物（关键帧 + 运动段 + ASR 字幕）
python scripts/integrated_pipeline.py --video 视频.mp4 --output out/

# 3. 压缩为语义代表帧（供 LLM 内容理解）
python scripts/select_representatives.py \
  --keyframes out/keyframes \
  --interval 60 \
  --out representatives.json \
  --report context.md

# 4. 读取代表帧结合字幕做内容理解，输出报告
```

## 目录结构

```
video-understanding-skill/
├── SKILL.md                        # Skill 定义（含完整工作流指引）
├── README.md
├── LICENSE
├── requirements.txt
└── scripts/
    ├── smart_pipeline.py           # 实时预算分配 · 画面链（快/慢双系统）
    ├── integrated_pipeline.py      # 三通道流式编排（画面+声音+对齐）
    ├── asr_sherpa.py               # 流式 ASR 声音链（sherpa-onnx / fallback）
    ├── select_representatives.py   # Tier3 语义代表帧选择（内容理解层）
    └── validate_realtime.py        # 实时率验证（720p50 / 1080p30）
└── bench/                          # 对比基准（与 claude-real-video 落地实测）
    ├── gen_bench.py                # 生成 4 组可控测试视频
    ├── run_bench.py                # 跑双方管线
    └── land_compare.py             # 输出速度/准确性对比表
```

## 三层压缩理念

| 层级 | 内容 | 数量级 | 用途 |
|------|------|--------|------|
| Tier 0 | 原始帧 | 30fps（13万帧） | 播放 |
| Tier 1 | 快系统运动事件 | 逐帧 | 实时感知"有没有事发生" |
| Tier 2 | 镜头级关键帧 | 1-2s/张（~1600张） | 精确时间定位、镜头切换 |
| Tier 3 | **语义代表帧** | 30-60s/张（~60-150张） | **给 LLM 做内容理解** |

## 与 claude-real-video 对比（落地实测）

我们用 4 组可控测试视频（640x360@30fps，12s），对 [claude-real-video (crv)](https://github.com/davecap/claude-real-video) 与本 Skill 做了同一套产物下的速度与准确性对比。准确性指标：将时间轴切成 1s 桶、取桶首尾像素差分作为"该秒内容变化量"，超过 3% 视为有变化；覆盖率 = 被选中帧覆盖的有变化秒数占比。

| 测试 | 场景 | crv 帧   | crv 覆盖率 | crv 耗时 | 本 Skill 帧 | 本 Skill 覆盖率 | 本 Skill 耗时 |
|------|------|---------|-----------|---------|------------|----------------|--------------|
| aba   | A→B→A 场景切换 | 2   | —（无连续变化） | 1.63s | 2 | —（无连续变化） | 0.45s |
| slow  | 全屏缓慢渐变 | 12  | 100%       | 1.68s | 6 | 100%       | 0.48s |
| hue   | 纯色色相渐变 | 12  | 100%       | 1.62s | 6 | 100%       | 0.45s |
| static| 静态画面+末尾突变 | 1   | **0%**      | 1.66s | 2 | **100%**   | 0.42s |

结论：

- **准确性**：在 `static` 场景中，crv 只保留首帧、完全漏掉末尾（11s 处 Δ22.7%）的场景突变，覆盖率 0%；本 Skill 以 2 帧完整捕获（100%）。crv 的"密度下限"策略在信号稀疏时容易漏掉尾部关键变化。
- **精简度**：`slow`/`hue` 上 crv 每 1s 选 1 帧（12 帧），本 Skill 用时间分桶 + 像素差分去重只需 6 帧即达相同覆盖率，冗余减半，LLM 消费更省 token。
- **速度**：本 Skill 端到端约 **0.42-0.48s**，crv 约 **1.62-1.68s**，快约 **3.5 倍**。

复现（在 `bench/` 目录下）：`python gen_bench.py` 生成 4 组测试视频，`python run_bench.py` 跑双方管线，`python land_compare.py` 输出以上对比表。

## 作为 AI Skill 使用

本目录同时是 [TRAE](https://trae.ai) 兼容的 Skill。将本仓库放入 skills 目录后，AI 会在用户要求"理解这个视频 / 分析视频内容 / 提取视频信息"时自动调用 `SKILL.md` 中的工作流。

## 致谢与版权（重要）

本项目的声音链（ASR）基于 **sherpa-onnx** 流式语音识别引擎，该引擎由 **Xiaomi（小米）** 开源维护。使用本项目前请务必周知：

- **上游项目**：本仓库的 `scripts/asr_sherpa.py` 依赖 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（k2-fsa / Xiaomi 维护的流式中英双语 ASR 引擎）。
- **模型权重**：默认引用的流式模型为 `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`，其权重与使用条款遵循 sherpa-onnx 上游的 [Apache-2.0 开源协议](https://github.com/k2-fsa/sherpa-onnx) 及模型自身发布说明。
- **本仓库仅封装**：本项目仅在 `asr_sherpa.py` 中对 sherpa-onnx 做了轻量封装（抽音频 → 流式解码 → 增量落盘），未修改其核心逻辑。任何对 sherpa-onnx 的二次分发、商用，请自行遵守其上游许可证与模型条款。
- **fallback 说明**：未安装 sherpa-onnx 时，ASR 自动退化为 mock fallback（仅占位，不产出真实字幕），不影响画面链与代表帧核心功能。

## 使用注意事项

为获得正确的内容理解结果，请务必遵守以下约束：

1. **必须使用具备视觉能力的多模态模型**。本 Skill 的内容理解依赖"读取代表帧图片"（Tier 3 语义代表帧 + 关键帧），因此下游 LLM **必须支持图像输入**（如带视觉的多模态模型）。若使用纯文本模型（如部分仅文本的 API），将无法读取画面，内容理解会完全失效。这是本项目最关键的软性前提。
2. **ASR 为可选项**。字幕（ASR）用于补充声音信息，但画面理解不依赖它。若未安装 sherpa-onnx 或不具备音频条件，画面链与代表帧理解仍可正常工作。
3. **硬件与实时性**。实时率（实测 720p50 达 441fps、1080p30 达 202fps，75 分钟视频 221fps）基于当前 CPU 环境测得；在资源受限的嵌入式设备上，请通过 `--fast-scale`、`--kf-hz` 调整预算。
4. **运行方式**。管线应作为**常驻进程/线程**运行（如机器人实时流场景），而非一次性后台任务——后台进程可能被宿主回收导致中断（详见下方"已知经验"）。

## 已知经验（踩坑记录）

在真实长视频（75 分钟普法节目，135,744 帧）实测中确认：

- **处理速率**：画面链稳定 221.6fps，超实时 7.4 倍；端到端 612 秒处理完 75 分钟视频。
- **三层压缩**：1549 张镜头级关键帧 → 70 张语义代表帧（压缩到 4.5%），人工抽帧验证内容连贯、覆盖充分、无漏帧。
- **后台进程陷阱**：用 `nohup ... &` 后台启动时，进程会被运行环境回收导致"假死"（日志停在某帧、进程消失、无报错）。应使用前台常驻进程或 `blocking=false` 方式运行。

## License

MIT © 2026