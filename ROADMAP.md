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
| W1 核心算法升级 | Tier3 桶内多样性 top-k（最远点采样，默认 k=1 兼容）+ 自适应桶宽；events 有界 deque；ASR 真词级时间戳；删空桩 | 合成关键帧上 k=3 多样性生效；长列表内存测试通过 | [ ] |
| W2 实时源 | `vus/source.py`：File/Camera/RTSP 统一接口、断流重连、背压取最新帧、单调时钟时间戳；CLI `--source`；Ctrl-C 优雅落盘 | 限速回放下丢帧统计正确；事件时间轴单调 | [ ] |
| W3 语义增强与评估 | CLIP ONNX 化（去 torch）；OCR 第三通道（默认关）；`bench/semantic_eval/` 标注协议 + 语义覆盖率/冗余度指标；修正 crv 对比报告 | 新评估报告落库；CLIP ONNX 路径跑通 | [ ] |
| W4 目录内分层 | `vus/`（核心）+ `apps/anti_drone/`（检测栈）+ `devices/`（定位/云台/协同） | pytest 全绿 + CLI 冒烟 + SKILL.md 工作流可用 | [ ] |

交付物：每波一个 tag（wave-0 … wave-4）+ 最终 `v0.2.0`。
