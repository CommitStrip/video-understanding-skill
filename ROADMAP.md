# ROADMAP — vus 优化计划

## 最终优化目标

把 video-understanding-skill 从「作者机器上可跑的演示管线」升级为
**「可安装、可接实时流、理解质量可评估、工程可信的视频理解核心库 v0.2」**。

成功标准：

1. `pip install -e .` 可装，任何机器零硬编码路径跑通冒烟
2. 支持 RTSP/摄像头实时源 + 背压丢帧
3. Tier3 代表帧桶内多样性可选（不再每桶只留 1 帧）
4. pytest 单测覆盖核心算法，GitHub Actions CI 绿
5. 语义级评估基准替代"像素差分自评"（消除评估同构偏差）
6. 仓库目录内分层 core / apps / devices 清晰
7. 每波 git tag 可回溯

## 波次计划

因账号并发限额，开发 agent 串行调度：每波一个 agent，主会话验收后
commit + push + 打 tag，自动进入下一波；agent 不可用时主会话直接接手。

| 波次 | 范围 | 验收标准 | 状态 |
|------|------|----------|------|
| W0 工程化基线 | 建 `vus` 包 + 薄壳兼容、清沙箱硬编码路径、pyproject、tests/、CI、ROADMAP/README | `pip install -e .` 成功；pytest 全绿；双入口 `--help` 可用；无 `/workspace`、`/data/user` 残留 | [x] 完成于 2026-09-02 |
| W1 核心算法升级 | Tier3 桶内多样性 top-k（最远点采样，默认 k=1 兼容）+ 自适应桶宽；events 有界 deque；ASR 真词级时间戳；删空桩 | 合成关键帧上 k=3 多样性生效；长列表内存测试通过 | [x] 完成于 2026-09-02 |
| W2 实时源 | `vus/source.py`：File/Camera/RTSP 统一接口、断流重连、背压取最新帧、单调时钟时间戳；CLI `--source`；Ctrl-C 优雅落盘 | 限速回放下丢帧统计正确；事件时间轴单调 | [x] 完成于 2026-09-02 |
| W3 语义增强与评估 | CLIP ONNX 化（去 torch）；OCR 第三通道（默认关）；`bench/semantic_eval/` 标注协议 + 语义覆盖率/冗余度指标；修正 crv 对比报告 | 新评估报告落库；CLIP ONNX 路径跑通 | [x] 完成于 2026-09-02 |
| W4 目录内分层 | `vus/`（核心）+ `apps/anti_drone/`（检测栈）+ `devices/`（定位/云台/协同） | pytest 全绿 + CLI 冒烟 + SKILL.md 工作流可用 | [x] 完成于 2026-09-02 |

交付物：每波一个 tag（wave-0 … wave-4）+ 最终 `v0.2.0`。

## 实战发现（2026-09-02，120 分钟直播课程视频首测）

v0.2.0 发布后用真实视频（1080p25 · 120.2min · 180,311 帧）实战验证，性能全部达标
（142.8fps = 5.7× 实时、内存稳定 ~225MB），但暴露一个结构性缺陷与若干环境注意项：

### BUG-1（已修复于 v0.2.1）：慢系统双闸对"渐变演化"内容失明

- **现象**：2 小时课程仅产出 3 个关键帧（t=0/1.5/133.8s），其后 118 分钟画面持续
  渐变演化（课件批注推进）却零发帧
- **根因**：`SmartPipeline._slow_keyframe` 的第二道闸 `is_new_scene`（pHash>12 或
  直方图相关<0.5）对渐变漂移系统性失明——实测漂移点 pHash 距离仅 4-7/64、
  直方图相关 1.00，第一道闸（像素差分 14.3>10）明明已通过却被一票否决。
  双闸为"突变切镜"降噪的设计，把"缓慢累积的内容演化"（直播课程最典型形态）也吞掉了
- **修复**：新增"持续漂移采纳"——第一闸连续 `drift_confirm_checks`（默认 3，
  约 6 秒）超阈值即采纳为关键帧（reason=gradual_drift）；pHash/直方图退回
  突变场景的快速通道而非一票否决。设 0 可还原旧行为
- **CLIP 旁证（实测数据）**：CLIP 语义距离能看见该渐变（漂移点 0.070-0.085 vs
  同内容≈0，达真实切换 0.144 的一半），验证了"CLIP 模式"可行；但 370ms/帧
  不宜进实时检查路径，作为 Tier3 离线语义层（已接入）更合适。若在 Tier2 用
  CLIP 确认，仅在第一闸已过的稀疏候选上懒加载运行

### 环境注意项（低配 Windows 实测）

- **OpenBLAS 多线程内存分配失败**：2 核机器上 opencv+numpy 混跑可能报
  "Memory allocation still failed after 10 retries"，`OPENBLAS_NUM_THREADS=1` 规避
- **cv2.VideoCapture seek 规律**：同一实例"顺序 read 后再 POS_MSEC seek"可能失败，
  每个时间点新开实例 seek 可靠；FileSource 顺序读不受影响
- **ffmpeg 中文路径**：Windows 控制台下 ffmpeg 读中文路径报" No such file"，
  需经 Python subprocess 传参或改 ASCII 路径

### 待办（下轮候选）

- [ ] Tier3 CLIP 语义去重与漂移视频的语义评估实测（PROTOCOL 已就绪）
- [ ] sherpa-onnx 真实字幕链路实测（本机未装，直播课程是理想素材）
- [ ] `apps/anti_drone/models/` 4 个旧权重（~96MB）的 LFS 迁移或下载脚本化（需重写历史，独立决策）
