# Video Understanding Skill

一个本地视频预处理管线：把视频转换为可追溯的运动事件、镜头级关键帧、严格 ASR 字幕和语义代表帧，供下游多模态模型或业务程序继续分析。

本项目不内置 LLM，也不承诺仅凭固定数量的代表帧就能完整理解任意视频。它解决的是“如何用有界资源生成结构化视频上下文”，不是完整的视频问答产品。

## 设计原则

- **ASR 不降级**：缺少 ffmpeg、sherpa-onnx、模型文件或识别失败时，整合流程明确失败，不生成模拟字幕。
- **显式视觉模式**：仅需画面处理时使用 `--visual-only`；该模式不会伪造字幕，也不会生成 `aligned_output.json`。
- **有界内存**：音频按块读取，内存中的事件历史有上限；完整事件流增量写入 `events.jsonl`。
- **结果可追溯**：关键帧元数据包含实际 JPG 路径，运行清单记录输入规格、模式、ASR 状态和处理结果。
- **验证不过度外推**：合成视频吞吐量和像素变化覆盖率只作为代理指标，不等同于真实语义准确率。

## 处理层级

| 层级 | 产物 | 作用 |
| --- | --- | --- |
| Tier 0 | 原始视频帧 | 原始输入 |
| Tier 1 | 运动开始、持续、结束事件 | 快速发现明显画面变化 |
| Tier 2 | 镜头级关键帧 | 时间定位和场景候选 |
| Tier 3 | 首尾锚点与时间桶代表帧 | 控制下游视觉输入规模 |

Tier 1 的候选框来自运动轮廓，不是目标检测框。Tier 3 只能从 Tier 2 已保存的关键帧中选择，不能恢复上游已经漏掉的画面。

## 环境要求

- Python 3.9+
- OpenCV 与 NumPy
- 完整 ASR 模式额外需要：
  - 系统可执行的 ffmpeg
  - `sherpa-onnx`
  - 一个包含唯一 `encoder*.onnx`、`decoder*.onnx`、`joiner*.onnx` 和 `tokens.txt` 的模型目录

### 安装视觉依赖

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux 激活环境：

```bash
source .venv/bin/activate
```

### 安装 ASR Python 依赖

```bash
python -m pip install -r requirements-asr.txt
ffmpeg -version
```

模型权重不包含在仓库中。请自行下载并审核相应模型的来源、许可证和哈希。

## 快速开始

### 完整视觉 + ASR 模式

ASR 是默认必需能力：

```bash
python scripts/integrated_pipeline.py \
  --video input.mp4 \
  --output out \
  --asr-model-dir /path/to/sherpa-model
```

也可以设置环境变量：

```bash
export SHERPA_ONNX_MODEL_DIR=/path/to/sherpa-model
python scripts/integrated_pipeline.py --video input.mp4 --output out
```

Windows PowerShell：

```powershell
$env:SHERPA_ONNX_MODEL_DIR = "D:\models\sherpa-model"
python scripts/integrated_pipeline.py --video input.mp4 --output out
```

### 显式视觉模式

```bash
python scripts/integrated_pipeline.py \
  --video input.mp4 \
  --output out \
  --visual-only
```

该模式只表示“用户明确不需要音频理解”，不是 ASR 失败后的自动降级。

## 输出

```text
out/
├── run_manifest.json       # 输入规格、运行模式、ASR 状态和处理结果
├── events.jsonl            # 完整增量视觉事件日志
├── pipeline_results.json   # 事件统计、闭合运动段和关键帧清单
├── aligned_output.json     # 仅完整 ASR 成功时生成
└── keyframes/
    └── kf_*.jpg
```

如果真实 ASR 失败，命令返回非零状态，`run_manifest.json` 标记失败，并且不会创建伪造的成功对齐结果。

## 语义代表帧

```bash
python scripts/select_representatives.py \
  --keyframes out/keyframes \
  --interval 60 \
  --out representatives.json \
  --report context.md
```

选择器始终保留首帧和尾帧锚点，并在各时间桶中选取差异较大的候选帧。可以显式开启连续锚点去重：

```bash
python scripts/select_representatives.py \
  --keyframes out/keyframes \
  --interval 60 \
  --dedup-threshold 8 \
  --out representatives.json
```

可选 CLIP 增强只用于离线 Tier 3 选帧：

```bash
python scripts/select_representatives.py \
  --keyframes out/keyframes \
  --interval 60 \
  --clip \
  --w-pix 0.5 \
  --out representatives_clip.json
```

CLIP 依赖和权重未列入核心依赖；启用前需单独审查和安装。缺少依赖或权重时会明确失败。

## 流式视觉 API

```python
from scripts.smart_pipeline import SmartPipeline

pipeline = SmartPipeline({
    "fast_scale": 0.25,
    "keyframe_interval_hz": 1.5,
    "event_history_limit": 2000,
})

for frame, timestamp in video_stream:
    for event in pipeline.process_frame(frame, timestamp):
        host.consume(event)

for event in pipeline.finalize(last_timestamp):
    host.consume(event)
```

宿主应持续消费事件；内存中只保留最近一段历史，准确总数和完整事件保存在输出统计与 JSONL 日志中。

## 验证

### 单元测试

```bash
python -m unittest discover -s tests -v
python -m compileall scripts bench tests
```

### 视觉吞吐量冒烟测试

```bash
python scripts/validate_realtime.py --output rt_validate --duration 10
```

该测试只检查合成视频上的视觉处理吞吐量，不包含真实 ASR、长稳运行、峰值内存或真实视频准确率。

### 可复现基准

```bash
python bench/gen_bench.py --output bench
python bench/run_bench.py --bench-dir bench --baseline none
python bench/land_compare.py --bench-dir bench
```

外部 `claude-real-video` 基线必须显式启用并固定版本：

```bash
python bench/run_bench.py --bench-dir bench --baseline crv
```

`land_compare.py` 输出的是“一秒像素变化覆盖代理指标”，不是语义准确率、人物识别准确率或内容理解召回率。

## 已知限制

- 固定阈值的帧差法对镜头抖动、全局光照变化和复杂压缩噪声较敏感。
- 当前视频时间轴使用固定 FPS 计算，不适合需要严格 PTS 对齐的可变帧率视频。
- ASR 段时间来自分块解码，不是词级强制对齐结果。
- 项目不提供目标检测、OCR、人物跟踪、说话人分离或最终自然语言报告生成。
- “代表帧覆盖像素变化”不能证明“代表帧覆盖全部语义”。生产使用前必须在目标视频集上建立人工标注验收集。

## 项目结构

```text
video-understanding-skill/
├── SKILL.md
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-asr.txt
├── requirements-dev.txt
├── scripts/
│   ├── smart_pipeline.py
│   ├── integrated_pipeline.py
│   ├── asr_sherpa.py
│   ├── select_representatives.py
│   └── validate_realtime.py
├── bench/
│   ├── gen_bench.py
│   ├── run_bench.py
│   └── land_compare.py
└── tests/
```

## 数据与安全

核心流程不调用云端 LLM。完整 ASR 使用本地 ffmpeg、sherpa-onnx 和本地模型；临时 WAV 使用唯一临时文件，并在识别结束后删除。任何外部模型下载、CLIP 权重或下游服务的数据传输都需要部署者单独审核。

## License

[MIT License](LICENSE)。第三方库和模型权重遵循各自上游许可证。
