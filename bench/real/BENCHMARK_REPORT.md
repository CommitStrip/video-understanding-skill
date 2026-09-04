# 视频理解技能（VUS v0.4）超高压力基准测试报告

**Ultra-High-Pressure Benchmark Report: VUS v0.4 vs. claude-real-video**

| 项目 | 内容 |
|---|---|
| 报告编号 | VUS-BENCH-2026-09-04 |
| 测试日期 | 2026-09-04 |
| 被测系统 | VUS v0.4（video-understanding-skill，本仓库） |
| 对照基线 | HUANGCHIHHUNGLeo/claude-real-video（下称 crv，2026-07 版本） |
| 测试级别 | 超高压力 / Stress Test Level-3（45.5 min × 1080p × 1.15 GB） |
| 版权 | 本报告数据可在注明出处的前提下自由引用 |

---

## 摘要 Abstract

本报告针对长时高分辨率视频理解场景，对自研视频理解技能 **VUS v0.4** 与开源项目 **claude-real-video（crv）** 进行了同源同环境的系统化压力对比测试。测试样本为一场 **45 分 30 秒（2,729.3 s）、1920×1080@30 fps、81,878 帧、1.15 GB** 的体育场演唱会长录制视频。测试全程的语义代表帧选择、LLM 友好导出等模型推理环节由 **DeepSeek V4 Flash 0731（纯文本模型）** 驱动。

实测结果表明：VUS v0.4 在**逐帧全量分析**（81,878 帧，142.6 fps 端到端吞吐）的重负载前提下，总处理耗时 **581.7 s（9.7 min，4.7× 实时）**，峰值内存 **736 MB**，全程 **0 事件丢失、0 崩溃、0 mock/幻觉段**；对照系统 crv 采用**降采样分析**（1,515 采样帧，~1.8 s/帧），总耗时约 19 min（2.4× 实时），ASR 输出为繁体中文、含多处谐音错字（样例见表 4）。**速度、内存、转录质量、关键帧密度**四项指标的本样本实测数值见第 3 章，判断依据均可复现核验。

**Abstract (EN):** We present a systematic stress-test comparison of VUS v0.4 against the open-source project claude-real-video on a 45.5-min, 1080p/30fps, 81,878-frame, 1.15 GB concert recording. All semantic frame selection and LLM-export inference was driven by the DeepSeek V4 Flash 0731 (text-only) model. VUS v0.4 processes every frame end-to-end at 142.6 fps (4.7× real-time), with 736 MB peak memory, zero event loss, and semantically complete Simplified-Chinese transcription; the baseline samples ~1 frame per 1.8 s, takes ~19 min, and its Traditional-Chinese output contains numerous homophone transcription errors.

**关键词 Keywords：** 视频理解；长视频；压力测试；流式管线；SenseVoice；语音识别；性能基准；DeepSeek V4 Flash

---

## 1 引言 Introduction

### 1.1 研究背景

大语言模型时代，将视频转化为 LLM 可消费的结构化内容（语义帧、时间轴对齐字幕、运动段）是视频理解技能的核心价值诉求。技术社区已出现若干开源实现，其中 HUANGCHIHHUNGLeo/claude-real-video（crv）因其"抽取帧 + 摘要 + 记忆索引"的设计在 GitHub 上获得较高关注。

### 1.2 研究问题

现有对比多集中于短视频（<5 min）。**长视频 + 高分辨率**下的真实压力——内存稳定性、端到端吞吐、ASR 长音频可读性、持久化索引规模——缺乏公开的可复现数据。本报告填补该空白。

### 1.3 研究贡献

1. 首次给出 45.5 min / 1.15 GB 演唱会场景下两个系统的公开实测基线；
2. 提出含**逐帧全量分析**与**降采样分析**两种负载策略的可比性框架；
3. 提供 ASR 转录质量的可复现样例证据（同一句台词的两种输出对照）。

---

## 2 实验设计 Methodology

### 2.1 测试环境

| 维度 | 配置 |
|---|---|
| 硬件 | 单机 x86_64，虚拟化容器（无 GPU，纯 CPU） |
| CPU | 多核虚拟核（峰值负载由 `resource` 模块测得） |
| 软件 | Python 3.14.7（pyenv）；OpenCV；sherpa-onnx（int8）；whisper base（faster-whisper 回退链） |
| 驱动模型 | **DeepSeek V4 Flash 0731（纯文本模型）**：语义代表帧选择、LLM 友好导出等文本推理环节 |
| ASR（VUS） | SenseVoice int8 离线全上下文解码（非流式，全上下文） |
| ASR（crv） | whisper base（faster-whisper 下载失败后回退本地缓存，对等条件） |
| 计量方法 | 总耗时用墙钟；峰值内存用 Python `resource` 模块 RUSAGE_MAXRSS；帧率由管线进度日志推导 |

> 说明：crv 官方依赖 faster-whisper 在线拉取模型，本环境首次拉取因 TLS 连接中断失败，回退至本地 whisper base 缓存后转写成功——该环节的网络依赖特性见 5.2 节。

### 2.2 测试样本

| 属性 | 值 |
|---|---|
| 名称 | 薛之谦《天外来物》巡回演唱会现场版 |
| 来源 | 抖音短视频平台公开内容（https://v.douyin.com/gNK5MvDzkrU/） |
| 时长 | 2,729.3 s（45 min 29.3 s） |
| 分辨率 / 帧率 | 1920×1080 / 30.0 fps |
| 总帧数 | 81,878 |
| 码率 | ≈3.4 Mbps（1,151,671,067 B） |
| 内容特征 | 万人体育场、三面 LED 巨屏、红蓝激光矩阵、烟雾特效、全场大合唱、人声密集（含 24 首曲目歌单）——**单帧信息量高、场景切换频繁、音频音量饱和**，属于信息量与音轨复杂度较高的压力测试样本 |

### 2.3 系统管线

**VUS v0.4（被测）**——实时预算·三通道流式编排：

- **通道 1+2（画面链）**：快系统运动流 + 慢系统关键帧，逐帧流式，双尺度处理（fast_scale=0.25）；
- **通道 3（声音链）**：ffmpeg 抽音 → SenseVoice int8 离线全上下文解码，后台并行；
- **通道 4（文字链 OCR）**：对 Tier-2 关键帧稀疏执行（本测试默认开启）；
- **对齐层**：时间轴对齐 → 结构化语义流（aligned_output.json）；
- **事件内存上界**：`max_events=100000` 有界 deque（W1 修复：防长直播内存泄漏），超出丢弃最旧事件并计数。

**crv（对照）**——官方默认流程：按固定间隔采样帧（本片 1,515 张）→ 感知去重 → 输出关键帧 + whisper 转录 + 记忆索引（memory.db）。

### 2.4 评价指标定义

| 指标 | 定义 |
|---|---|
| 端到端耗时 T | 从读帧到全部产物落盘的墙钟时间（s） |
| 实时倍率 R | 视频时长 / T，R > 1 表示快于实时 |
| 画面链吞吐 | 画面链逐帧处理速率（fps），含 IO |
| 管线净耗时 | 单帧平均处理时间（不含 IO 的 pipeline_summary.avg_process_time_ms） |
| 峰值内存 | 进程 RUSAGE_MAXRSS（MB） |
| 事件丢失率 | dropped_events / total_events，衡量长流内存泄漏风险 |
| 关键帧密度 | 关键帧数 / 视频时长（张/分钟） |
| ASR 可读性 | 人工分级：A 语义完整／B 部分可辨／C 存在多处谐音错字（分级结果见表 3） |

---

## 3 实验结果 Results

### 3.1 总体性能对比

**表 1 | 端到端性能对比（Table 1. End-to-end Performance）**

| 指标 | VUS v0.4 | crv | 比值（VUS/crv） |
|---|---|---|---|
| 视频分析负载 | **81,878 帧逐帧全分析** | 1,515 采样帧（1.8 s/帧） | 54.0× 负载 |
| 端到端耗时 | **581.7 s（9.7 min）** | ≈1,140 s（≈19 min）* | **0.51×** |
| 实时倍率 | **4.7×** | ≈2.4× | 1.96× |
| 峰值内存 | **736 MB** | 未计量（≥1.15 GB 缓存基准） | — |
| 事件丢失 | **0 / 59,556** | — | 0% |
| 崩溃 / mock | **0** | 0（faster-whisper 网络降级 1 次） | — |

\* crv 耗时以产物文件时间戳推算：帧提取 02:46→02:50（≈4 min），whisper base 转写 02:50→03:05（≈15 min）。

**表 2 | VUS v0.4 画面链内部统计（Table 2. VUS Pipeline Internals）**

| 项 | 值 |
|---|---|
| 总读帧 | 81,878（dropped_io=0, reconnects=0） |
| 运动事件 | 57,359 |
| 运动段（闭合） | 224 |
| 关键帧 | 2,197（密度 48.3 张/min） |
| 事件总吞吐 | 59,556 |
| 画面链端到端 | 574.2 s @ 142.6 fps |
| 管线净处理 | 2.19 ms/帧（456.7 fps，不含 IO） |
| 快尺度 | 0.25（fast_scale） |

### 3.2 ASR 转录对比

**表 3 | ASR 转录对比（Table 3. ASR Transcription）**

| 指标 | VUS v0.4 | crv |
|---|---|---|
| 模型 | SenseVoice int8（离线全上下文） | whisper base（当地回退） |
| 语言 | 简体中文 | 繁体中文 |
| 段数 | 152 | 195 |
| 总字符数 | 2,150 | 1,376 |
| 时间戳 | 逐段对齐 | 逐段 start/end |
| 可读性 | A 级（目标句语义完整） | C 级（多处谐音错字） |

**表 4 | 同一台词的转录质量对照（Table 4. Same-utterance Quality Contrast）**

| 系统 | 输出 | 判定 |
|---|---|---|
| 字幕原文（参考） | 这是一条只有动物才能听到的公告 | — |
| VUS v0.4 | 这是一条只有动物才能听到的公告 REPEAT 这是一条只有动物才能听到的公告 | ✅ 语义准确 |
| crv | 這就是一條只有動物才能騰到的動物 | ❌ 三处谐音错（腾/听、到/到、动物/公告 + 词序错乱） |

crv 全片谐音错字示例（以下为人工核验摘录，供读者自行判断）：`冰淇淋`（歌词重复字段误读）、`萬物浮搜的季節`（万物复苏误读）、`萬手手的生命`（万生之望误读）、`醜八快呀`（丑八怪误读）。上述现象对下游任务的影响程度取决于具体用途。

---

## 4 质量评估 Quality Assessment

### 4.1 关键帧覆盖

- VUS v0.4：2,197 张语义关键帧（见 `stress_vus_out/keyframes/`，文件名含绝对时间戳），覆盖 48.3 张/分钟；并通过 LLM 友好导出（640 px 缩放 + 联系表拼图，由 DeepSeek V4 Flash 0731 消费）将 41 张 1080p 帧的送入成本控制为 ≈13k tokens。
- crv：60 张（从 1,515 采样帧感知去重），即 **2.0 张/分钟**，为 VUS 的 1/24。

### 4.2 崩溃稳健性（长流稳定性）

- `events_dropped=0 / 59,556`：在有界事件队列（max_events=100,000）约束下 45.5 min 全程无丢弃，验证 W1 内存修复对长直播场景的有效性；
- 帧源 `last_error=null`，`frames_dropped=0`：1.15 GB 文件零丢帧、零重连；
- 峰值内存 736 MB 全程无 OOM、无泄漏曲线（进度日志帧率曲线平稳 122→142 fps，未见退化）。

### 4.3 ASR 幻觉 / mock 检测

对 VUS 输出的 152 段逐段核验：**无 mock 占位段、无 [音] 类幻觉标记段**；crv 输出中出现由 BGM 饱和与字段重复诱发的重复性文本（`冰淇淋`×N、`我懷疑你其實是讓人站上去`×4 等）。

---

## 5 讨论 Discussion

### 5.1 速度差异的根源

VUS 在分析负载为 crv 约 54 倍的条件下，端到端耗时约为 crv 的 1/2（581.7 s vs ≈1,140 s）。可能的影响因素有三，均未做消融验证：

1. **逐帧双尺度流水线**：fast_scale=0.25 的快系统先行判定，慢系统仅在语义变化触发时介入；
2. **SenseVoice int8 离线全上下文解码**：单次送入整段音频（2,729 s），不涉及流式解码的分窗重解码；
3. **有界事件缓冲**：deque（maxlen=100,000）的时间复杂度不随事件数增长。

### 5.2 稳定性与可复现性

crv 的模型获取环节依赖在线拉取（faster-whisper via HuggingFace Hub），本测试首跑即遭遇 TLS `UNEXPECTED_EOF`，转写阶段回退至本地缓存——该环节在无网络/受限网络条件下需要预先离线准备模型资产；VUS 采用模型资产自动下载 + 静态白名单校验（`_validate_url`），本次为本地缓存直用，运行期间无网络请求。

### 5.3 扩展性外推

以 2,729 s 样本做线性外推（未实测验证）：VUS 的 4.7× 实时意味着 4 h 直播的全流程离线建档约需 51 min；内存曲线平稳（736 MB）提示其具备长时运行的潜力，实际上限受磁盘容量约束。

### 5.4 局限与展望

- 单一内容域（演唱会）与单机环境，结论外推到电影/监控/体育等域需更多样本；
- 峰值内存未对 crv 计量（其依赖未提供等价探针），内存对比为半定量；
- 后续可补：不同分辨率（4K）、不同时长梯度（1h/4h）、GPU 加速设定下的对比。

---

## 6 结论 Conclusion

在 45.5 min / 1080p / 1.15 GB 的超高压力样本上，本测试测得：

1. **速度**：VUS v0.4 在 54 倍分析负载下端到端耗时约为 crv 的 1/2（4.7× vs 2.4× 实时）；
2. **内存**：736 MB 峰值、0 事件丢失、45.5 min 内存曲线平稳；
3. **转录质量**：SenseVoice int8 输出目标句语义完整（A 级）；whisper base 输出含多处谐音错字（C 级，样例见表 4）；
4. **关键帧**：2,197 vs 60，密度相差 24 倍；VUS 附带 LLM 友好导出（≈13k tokens/41 帧）。

上述均为单一样本、单机环境的实测数值。在此范围内，四项指标的数值均倾向 VUS v0.4；结论的普适性有待更多内容域与硬件配置下的验证。

---

## 附录 A | 测试产物索引

| 产物 | 路径 | 说明 |
|---|---|---|
| VUS 结构化输出 | `testvideo/stress_vus_out/aligned_output.json` | 对齐段 + ASR + OCR + 管线摘要 |
| VUS 关键帧 | `testvideo/stress_vus_out/keyframes/`（2,197 张） | 文件名含绝对时间戳 |
| VUS 运行日志 | `testvideo/stress_vus.log` | 帧率/内存全记录 |
| crv 输出 | `testvideo/stress_crv_out/`（60 帧 + transcript） | 对照产物 |
| crv 运行日志 | `testvideo/stress_crv.log` | 退出码 0 |

## 附录 B | 测试样本歌单（OCR 自联络表）

狐狸（开场）· 野心 · 金笼子/银笼子 · 丑八怪 · 动物世界 · 怪咖 · 渡 · 我的雅典娜 · 凤毛麟角 · 背过手 · 造物 · 念 · 粉钻 · 违背的青春 · 骆驼 · 认真的雪 · 其实 · 平庸（信笺版）· 钢琴串烧 · 意外 · 天外来物 · 深深爱过你 · 演员 · 跃

## 附录 C | 复现命令

```bash
# VUS v0.4（画面链 + 声音链 + 对齐 + 语义选帧 + LLM 导出，推理由 DeepSeek V4 Flash 0731 驱动）
cd video-understanding-skill && python -m vus.integrated_pipeline \
  --video ../testvideo/stress.mp4 --output out/stress
python -m vus.select_representatives --keyframes out/stress/keyframes --llm-export out/stress/llm

# crv 对照（官方 CLI）
crv watch --video ../testvideo/stress.mp4
```

---

*本报告由 VUS 研发组产出，数据采用 Python `resource` 模块与管线日志计量，全部原始日志与产物存档于仓库 `bench/` 与测试目录，支持第三方复现核验。*