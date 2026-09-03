---
name: "video-understanding-skill"
description: "把视频（尤其直播课程、讲座、长视频）变成结构化的内容理解产物：语义代表帧 + 时间轴对齐的真 ASR 字幕 + 运动段，供 LLM 输出课程讲义/剧情摘要/场景分析报告。当用户要求'理解这个视频''分析视频内容''提取视频信息''转写字幕''抽关键帧''做课程讲义'，或给出 RTSP 实时流/摄像头画面要做实时理解时调用——即使没明说'视频理解'。"
---

# Video Understanding（视频内容理解）

把一段视频变成**结构化、可被 LLM 高效消费的内容理解产物**：把 30fps 的十几万帧压缩成几十张语义代表帧 + 时间轴对齐字幕 + 运动段。已实测 1080p 直播课程 120 分钟：147.7fps 处理（5.9× 实时）、41 关键帧、3505 段真实中文字幕。

## 前置条件

1. 依赖：`pip install -e .`（或直接跑——`scripts/` 下的旧命令入口内置了路径兜底，无需安装）。必需 `opencv-python`、`numpy`；ffmpeg 用于抽音频。
2. **真字幕**（默认开箱即用）：`pip install sherpa-onnx` 后，首次运行缺模型会
   自动从官方源下载（约 490MB，一次性；`VUS_ASR_AUTO_DOWNLOAD=0` 关闭）。
   也可手动下载到 `models/sherpa/`：
   ```bash
   curl -L -o - https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 | tar -xj -C models/sherpa --strip-components=1
   ```
   模型目录可用环境变量 `VUS_SHERPA_MODELS` 指定。
   🔴 **警告：sherpa-onnx 未安装或自动下载被关闭且模型缺失时，字幕是
   "mock 占位假文本"（固定提示语，非真实内容）——禁止把 mock 字幕当作
   真实转写交付给用户**，必须在报告里注明字幕缺失。
3. 可选增强：`pip install -e ".[clip]"` + `bash scripts/download_clip_onnx.sh`（CLIP 语义选帧，权重目录 `VUS_CLIP_MODELS`）；`pip install -e ".[ocr]"` + 管线 `--ocr`（花字/内嵌字幕提取，Tier3 定点执行）。

## 工作流

### 第 1 步：跑管线提取结构化产物

```bash
python -m vus.integrated_pipeline --video <视频路径> --output <输出目录> --kf-hz 1.5
# 直播/RTSP 实时流变体：
python -m vus.integrated_pipeline --source rtsp --url rtsp://主机/流 --output <输出目录>
# 带花字/内嵌字幕的视频加 --ocr（只对 Tier3 代表帧执行，不拖慢管线）
```

产出：`<输出目录>/keyframes/`（镜头级关键帧）、`pipeline_results.json`（时间表+运动段）、`aligned_output.json`（对齐字幕）。已有产物时可跳过本步。

### 第 2 步：压缩为语义代表帧（Tier 3）

```bash
python -m vus.select_representatives --keyframes <输出目录>/keyframes \
  --max-reps 60 --out representatives.json --report context.md
```

参数选择：`--max-reps 60` 按 LLM 上下文预算自适应选帧（推荐默认）；多人近景
轮换（圆桌/访谈）加 `--k 3` 每桶保留 3 张互不冗余的代表帧；内容单调的监控流
加 `--adaptive` 自动放宽；语义增强加 `--clip`。

### 第 3 步：读代表帧做内容理解

用 Read 工具读取 `representatives.json` 里的代表帧图片，结合 `context.md`
与 `aligned_output.json` 理解：

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

## 注意事项

- **必须用多模态模型**读代表帧，纯文本模型无法完成画面理解。
- 内容缓慢演化的视频（课件批注推进等）由渐变漂移检测自动覆盖，无需调参。
- 低配环境遇 OpenBLAS 内存报错：设 `OPENBLAS_NUM_THREADS=1`。
- 验证安装可用 `python -m vus.validate_realtime --out ./output/rt_check`（可选）。
