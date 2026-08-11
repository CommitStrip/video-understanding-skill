---
name: "video-understanding"
description: "把视频预处理为运动事件、关键帧、严格 ASR 字幕和语义代表帧，供具备视觉能力的下游模型继续分析。完整模式要求真实 ASR 成功，禁止模拟字幕降级。"
---

# Video Understanding

本 Skill 提供视频理解前的结构化预处理，不直接完成最终语义推理。它不能保证固定数量的代表帧覆盖任意视频的全部关键内容。

## 使用条件

- 下游模型必须支持图像输入，才能理解代表帧。
- 完整模式必须提供 ffmpeg、sherpa-onnx 和真实模型目录。
- ASR 缺失或失败时必须停止完整流程，不得把占位文本作为字幕。
- 如果用户明确只需要画面处理，可以使用 `--visual-only`；该模式不生成字幕或对齐结果。

## 工作流

### 1. 运行结构化管线

完整视觉 + ASR：

```bash
python scripts/integrated_pipeline.py \
  --video <视频路径> \
  --output <输出目录> \
  --asr-model-dir <sherpa模型目录>
```

显式视觉模式：

```bash
python scripts/integrated_pipeline.py \
  --video <视频路径> \
  --output <输出目录> \
  --visual-only
```

重点检查：

- `run_manifest.json` 的整体状态与 ASR 状态；
- `events.jsonl` 的完整视觉事件流；
- `pipeline_results.json` 的运动段和关键帧文件映射；
- 完整 ASR 成功时生成的 `aligned_output.json`。

如果完整模式没有生成成功状态，停止后续字幕理解，不得自动切换到 mock、空字幕或推测字幕。

### 2. 生成 Tier 3 代表帧

```bash
python scripts/select_representatives.py \
  --keyframes <输出目录>/keyframes \
  --interval 60 \
  --out representatives.json \
  --report context.md
```

选择器保留首尾锚点，并在时间桶中选取代表帧。它只能处理 Tier 2 已经保存的画面，不能恢复上游漏检。

### 3. 下游内容理解

按 `representatives.json` 的时间顺序读取图片，并结合真实 `aligned_output.json` 分析：

- 视频类型与主题；
- 场景和人物候选；
- 关键时间点；
- 字幕与画面变化的对应关系；
- 无法从现有证据确认的内容。

输出中必须区分直接观察、字幕证据和模型推断，不应把像素变化覆盖率当作语义准确率。

## 验证边界

- `validate_realtime.py` 只是合成视频视觉吞吐量冒烟测试。
- `bench/land_compare.py` 只是低级像素变化覆盖代理指标。
- 生产使用前必须用目标视频集验证视觉漏检、误报、ASR 字错率、端到端延迟、峰值内存和长稳运行。

## 禁止事项

- 禁止在 ASR 不可用时生成模拟字幕。
- 禁止声称固定 30、60 或 80 帧能够可靠覆盖所有长视频。
- 禁止把单机合成测试速度当作生产环境性能承诺。
- 禁止把运动轮廓框描述为人物或物体检测框。
