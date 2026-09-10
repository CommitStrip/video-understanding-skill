---
name: "video-understanding-skill"
description: "把视频（尤其直播课程、讲座、长视频）变成结构化的内容理解产物：语义代表帧 + 时间轴对齐的真 ASR 字幕 + 运动段，供 LLM 输出课程讲义/剧情摘要/场景分析报告。当用户要求'理解这个视频''分析视频内容''提取视频信息''转写字幕''抽关键帧''做课程讲义'，或给出 RTSP 实时流/摄像头画面要做实时理解时调用——即使没明说'视频理解'。"
---

# Video Understanding（视频内容理解）

把一段视频变成**结构化、可被 LLM 高效消费的内容理解产物**：把 30fps 的十几万帧压缩成几十张语义代表帧 + 时间轴对齐字幕 + 运动段。已实测 1080p 直播课程 120 分钟：147.7fps 处理（5.9× 实时）、41 关键帧、3505 段真实中文字幕。

## 前置条件

### 必装（安装技能时一起完成，无选择）

```bash
pip install -e ".[asr]"
```

这会安装 sherpa-onnx + 自动下载 **SenseVoice int8 离线模型**（166MB，
sherpa-onnx 官方源或 CommitStrip/vus-models mirror，首次运行自动下载）。
这是文件转写的**默认 ASR 通道**——没有它字幕退化为 mock 假文本，管线不可用。

安装命令拆解：
```bash
pip install -e . && pip install sherpa-onnx
# 模型自动下载（首次运行时触发，166MB，一次性）
```

🔴 **警告：sherpa-onnx 未安装或自动下载被关闭且模型缺失时，字幕是
"mock 占位假文本"（固定提示语，非真实内容）——禁止把 mock 字幕当作
真实转写交付给用户**，必须在报告里注明字幕缺失。

### 选装（agent 必须询问用户，附优缺点）

以下模型**不随技能自动安装**。agent 在首次遇到对应场景时应向用户说明
优缺点，由用户决定是否安装：

<details>
<summary>🔍 CLIP 语义选帧（--clip）</summary>

```bash
pip install -e ".[clip]" && bash scripts/download_clip_onnx.sh
```

| | |
|---|---|
| ✅ 优点 | 桶内选帧从"像素变化最大"升级为"语义最相关"——多人轮换/复杂场景代表帧质量提升 |
| ❌ 缺点 | 额外下载 ~600MB ONNX 模型；每张候选帧推理 ~370ms（CPU）；体积大 |
| 适用 | 讲座/课程/会议等语义密集型内容 |
| 不适用 | 运动类/快节奏视频（像素差分已足够） |
</details>

<details>
<summary>📝 OCR 花字通道（--ocr）</summary>

```bash
pip install -e ".[ocr]"
# 管线加 --ocr
```

| | |
|---|---|
| ✅ 优点 | 提取画面内嵌文字（课件标题/花字/双语字幕）——对理解"画面上写了什么"至关重要 |
| ❌ 缺点 | 额外安装 rapidocr-onnxruntime；密集文字视频 OCR 耗时增加（仅 Tier3 代表帧，不影响管线速度） |
| 适用 | 课件/字幕/含文字画面的视频 |
| 不适用 | 纯风景/无文字画面 |
</details>

<details>
<summary>📡 流式 zipformer ASR（直播场景）</summary>

```bash
pip install sherpa-onnx
# 模型自动下载（约 490MB）
```

| | |
|---|---|
| ✅ 优点 | 实时逐词转写（词级时间戳），RTSP/摄像头直播场景唯一选择 |
| ❌ 缺点 | 额外下载 ~490MB（比离线 SenseVoice 大 3 倍）；流式上下文短 → 中文专名错字多于离线 |
| 适用 | RTSP 直播流 / 摄像头实时理解 |
| 不适用 | 文件转写（已有更好的离线 SenseVoice） |

模型目录可用环境变量 `VUS_SHERPA_MODELS` 指定。
</details>

<details>
<summary>⚡ GPU 加速（--device auto，v1.1）</summary>

```bash
python -m vus.device   # 自检：打印本机硬件、已装引擎与精确安装命令
```

| | |
|---|---|
| ✅ 优点 | 文件转写 RTF 显著下降（离线 SenseVoice 是最大受益者）；CLIP 语义选帧、OCR 同样提速；NVIDIA/AMD/Intel/Mac 全覆盖 |
| ❌ 缺点 | 需按显卡安装对应 onnxruntime 轮（与 CPU 轮同包名互斥）；sherpa CUDA 轮另需 ~190MB 下载；DirectML/CoreML 不加速 ASR |
| 适用 | NVIDIA 显卡（全套收益）；任何想缩短转写耗时的场景 |
| 不适用 | 无 GPU 机器（CPU 本就 5.9× 实时）；画面链无 GPU 路径（设计如此） |

启用：命令加 `--device auto`（或环境变量 `VUS_DEVICE=auto`）。不可用自动回退 CPU 并打印原因。
未启用 GPU 时在报告注明"GPU 加速未启用（CPU 运行）"。
</details>

> **Agent 行为规范**：安装技能时执行"必装"部分的命令。遇到 CLIP/OCR/直播
> 场景时，向用户展示上面对照表并询问是否安装。用户拒绝时不安装、不影响
> 核心功能，但在产出报告中注明"XX 增强未启用"。

## 工作流

### 第 1 步：跑管线提取结构化产物

```bash
python -m vus.integrated_pipeline --video <视频路径> --output <输出目录> --kf-hz 1.5
# 直播/RTSP 实时流变体：
python -m vus.integrated_pipeline --source rtsp --url rtsp://主机/流 --output <输出目录>
# 带花字/内嵌字幕的视频加 --ocr（只对 Tier3 代表帧执行，不拖慢管线）
```

产出：`<输出目录>/keyframes/`（镜头级关键帧）、`pipeline_results.json`（时间表+运动段）、`aligned_output.json`（对齐字幕）。已有产物时可跳过本步。

⏱ **耗时预估**：处理速率约 130-160fps，预计耗时 ≈ 视频时长 ÷ 130 × 2（长视频务必给足
超时；60 分钟视频约需 8-10 分钟）。输出中文日志在 PowerShell 下可能乱码：设
`PYTHONIOENCODING=utf-8`。

### 第 2 步：压缩为语义代表帧（Tier 3）

```bash
python -m vus.select_representatives --keyframes <输出目录>/keyframes \
  --max-reps 60 --llm-export <输出目录>/llm --out representatives.json --report context.md
```

参数选择：`--max-reps 60` 按 LLM 上下文预算自适应选帧（推荐默认）；`--llm-export`
同步产出 640px 缩放帧 + 3×3 联系表 + token 估算；多人近景轮换（圆桌/访谈）加
`--k 3` 每桶保留 3 张互不冗余的代表帧；内容单调的监控流加 `--adaptive` 自动放宽；
语义增强加 `--clip`。

### 第 3 步：读代表帧做内容理解

🔴 **图片读取预算（必须遵守）**：多数模型提供商限制**单请求 ≤30 张图**，且已读图片
永久占据会话上下文——超限后整个会话无法恢复。因此：

1. **先读联系表**（`--llm-export` 产出的 `grid_*.jpg`，1-3 张拼图即可覆盖全片概貌）
2. **单帧按需精读 ≤8 张**：首帧、尾帧 + 字幕提示的关键转折
3. **整次会话累计读图 ≤15 张**（要给用户截图、中间产物留余量）
4. 理解主力是**字幕文本**（aligned_output.json，零图片成本），图片只做视觉锚定

阅读顺序与要点：

1. **首帧与尾帧必读**——锁定节目类型 + 主题 + 最终结论
2. 场景构成、人物角色（画面 + 字幕交叉验证）
3. 话题时间线（运动段密度 + 代表帧变化 + 字幕关键词）

字幕字段说明：ASR 输出已经过清洗（连叠折叠 + 相邻去重），`hallucination: true`
的段是音乐/静音段的高概率英文幻觉（如 SIL/ER），**不要当作真实台词引用**；
带 `ocr_hint` 的段表示画面 OCR 文本与该段高度相似，可作专名纠错参考。

### 第 4 步：输出结构化报告

按用户偏好交付（默认 HTML，可 Markdown）。最小结构示例：

```markdown
# 《视频标题》内容理解报告
## 基本信息
时长 120 分钟 · 1080p · 直播课程（高中地理）
## 内容概要
本讲围绕雅鲁藏布江与长江的水文特征展开……
## 时间线
| 时间 | 内容 | 依据 |
|------|------|------|
| 00:00-22:00 | 课程引入：雅鲁藏布江 | 代表帧 rep_00 + 字幕 t=60s"咱们今天晚上的主要内容" |
## 关键截图
- rep_03（t=823s）：课件第 2 页板书
```

每条结论都应引用代表帧文件名或字幕时间戳作为依据。

## 实时流理解（v0.4：python -m vus.live）

用户给出 **RTSP 流 / 摄像头 / 直播画面**，或明确要求"边看边理解 / 实时理解"时，
不要走上面的离线批处理，改用实时理解栈（四层：帧级反射 → 毫秒级本地标签 →
触发式 VLM 滚动理解）：

```bash
# 文件仿真实时（开发/验收默认路径；mock 后端零成本）
python -m vus.live --video <视频路径> --realtime --vlm mock --serve

# RTSP 直播 + 真实 VLM（OpenAI 兼容端点走环境变量）
VLM_API_BASE=https://open.bigmodel.cn/api/paas/v4 VLM_API_KEY=<密钥> VLM_MODEL=glm-4v-plus \
  python -m vus.live --source rtsp --url rtsp://主机/流 --vlm openai --serve

# 纯本地免费模式（零 API 成本，只有帧级反射 + 毫秒标签）
python -m vus.live --video <视频路径> --realtime --vlm off --serve
```

理解结果三路消费：

1. **滚动文件（agent 首选）**——`live_context.md`（人/agent 可读摘要）与
   `live_state.json`（全量状态：滚动摘要/时间线/实体/分层滞后遥测）在输出目录
   持续原子更新。回答"现在画面里在干什么"直接 Read 这两个文件，无需等进程结束。
2. **SSE 服务**——`--serve` 后 `GET /state`（快照）、`GET /events`（增量流）、
   `GET /healthz`（探活），供程序订阅。
3. **控制台**——周期打印当前摘要与滞后遥测。

成本与延迟要点：T2 是**触发式**调用（场景切换/长运动段闭合/新语音段才调用），
`--min-call-interval`（默认 8s）是费用上限旋钮；理解滞后 = 触发间隔 + VLM 延迟，
有界不增长。毫秒级语义（有没有人/强运动）来自本地标签道，不经 API。

🔴 **警告：`--vlm mock` 的理解输出是占位假文本（同 mock 字幕红线）——禁止把
mock 理解当作真实内容交付**，必须在报告里注明理解层未接真实模型。

## 注意事项

- **必须用多模态模型**读代表帧，纯文本模型无法完成画面理解。
- 内容缓慢演化的视频（课件批注推进等）由渐变漂移检测自动覆盖，无需调参。
- 低配环境遇 OpenBLAS 内存报错：设 `OPENBLAS_NUM_THREADS=1`。
- 验证安装可用 `python -m vus.validate_realtime --out ./output/rt_check`（可选）。
