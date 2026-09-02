#!/usr/bin/env python3
"""
gen_synthetic_gt.py - 合成端到端示例：旧像素指标 vs 语义指标的对照实验
======================================================================
生成**已知语义结构**的合成视频（4 个语义场景）：

    纯色 A（亮度抖动，语义单一） -> 纯色 B（稳定） -> 纹路 C（棋盘格） -> 纯色 A'（与 A 同语义）

自动产出 GT，再分别用两种尺子评同一套代表帧：

  - 旧像素指标（bench/land_compare.py 同款）：1s 桶、桶首尾像素差分 >3% 记为
    "有变化秒"，覆盖率 = 被选帧覆盖的有变化秒占比。**尺子本身就是像素差分**——
    与选帧算法同构；
  - 语义指标（PROTOCOL.md）：语义覆盖率 / 冗余度 / 场景保真。

预期结论（纯色抖动的误判实证）：
  - 场景 A 与 A' 语义上各只有一个场景（换 1 次内容都不算），但亮度每秒抖一次，
    像素尺子把 A、A' 内**几乎每一秒**都判成"有内容变化"——全部是语义误报；
  - 在这把尺子下，语义上完美的 4 帧选择（覆盖 4 场景、冗余度 1.0）得分很低，
    而 Tier2 密集关键帧（十几倍帧量）反而满分——尺子奖励的是"追着像素噪声跑"；
  - 语义指标下 4 帧即满分：coverage=1.0, redundancy=1.0, gini=0.0。

用法:
  python bench/semantic_eval/gen_synthetic_gt.py [--out-dir out_synthetic]
产物（均在 <out-dir>/ 下，已 gitignore）:
  synthetic_4scene.mp4 / ground_truth.json / representatives.json /
  comparison_results.json（旧指标+语义指标全量数字）
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vus.io_utils import write_json  # noqa: E402
from vus.integrated_pipeline import run_realtime_pipeline  # noqa: E402
from vus.select_representatives import (  # noqa: E402
    select_representatives,
)
from eval_semantic import evaluate as semantic_evaluate  # noqa: E402  同目录

# ---------------- 合成视频定义 ----------------
W, H, FPS, DUR = 320, 180, 10, 48.0
SCENES = [
    # (start, end, desc) —— A' 与 A 语义相同（回到纯色空镜），像素上亮度相位不同
    {"start_s": 0.0, "end_s": 12.0, "desc": "纯色A：蓝色空屏+回弹小标记，亮度每秒抖动（语义单一）"},
    {"start_s": 12.0, "end_s": 24.0, "desc": "纯色B：红色画面+回弹小标记，亮度缓慢漂移（语义单一）"},
    {"start_s": 24.0, "end_s": 36.0, "desc": "纹路C：黑白棋盘格纹理+回弹小标记，亮度缓慢漂移"},
    {"start_s": 36.0, "end_s": 48.0, "desc": "纯色A'：回到蓝色空屏（与场景1同语义），亮度抖动"},
]
REP_INTERVAL = 12.0  # Tier3 分桶间隔：4 场景 x 12s -> 每场景恰好 1 帧


def _marker(frame, t):
    """四个场景共有的回弹小标记（语义上是同一元素，不含场景切换信息）。

    作用：让 Tier2 的 pHash/运动通道有可感知内容（纯色画面 pHash 失明，
    见 select_representatives.py 文档），避免关键帧基线退化为 1-2 张；
    标记在旧尺子 64x64 降采样下面积占比 <0.5%，不会触发 >3% 的"有变秒"。
    """
    cx = int(30 + (t * 55) % (W - 60))
    cy = H // 2 + int(25 * np.sin(t * np.pi / 2))
    cv2.circle(frame, (cx, cy), 8, (120, 120, 120), -1)
    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)


def _drift(t, rate=6.4, span=64):
    """三角波亮度漂移：任意 1s 窗口内变化 rate 级（≈2.5% < 旧尺子 3% 阈值，
    旧尺子对它沉默），但 Tier2 相邻关键帧（>=2s 强制检查）差 ≈2*rate 级
    （>10 级打分阈值），保证稳定场景也有关键帧产出。"""
    phase = (t * rate) % (2 * span)
    return phase if phase < span else 2 * span - phase


def scene_frame(t):
    """第 t 秒的合成帧。

    场景 A/A' 的亮度抖动周期取 0.5s（比旧尺子的 1s 桶更细）：保证每个 1s 桶
    首尾各落在一个相位上，桶内像素差分恒为 |95-50|/255≈17.6% > 3%——
    旧尺子会把抖动段每一秒都判成"有内容变化"，而这在语义上是单一场景。
    场景 B/C 用慢三角漂移（<3%/s）：语义上同样静止，旧尺子却恰好沉默——
    对比出旧尺子阈值（3%）的任意性。
    """
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    if t < 12.0:  # 场景 A：蓝底亮度抖动
        v = 50 if int(t * 2) % 2 == 0 else 95
        frame[:] = (v, v // 2, 0)  # BGR 偏蓝
    elif t < 24.0:  # 场景 B：红底 + 慢漂移
        d = _drift(t)
        frame[:] = (d, 40 + d // 2, 160)
    elif t < 36.0:  # 场景 C：黑白棋盘格（静态纹路）+ 慢漂移
        d = _drift(t)
        c = 16
        yy, xx = np.mgrid[0:H, 0:W]
        frame[:] = (np.where(((yy // c) + (xx // c)) % 2 == 0,
                             200 - d // 2, 40 + d // 2)
                    .astype(np.uint8))[:, :, None]
    else:  # 场景 A'：与 A 同语义，相位偏移的亮度抖动
        v = 50 if (int(t * 2) + 1) % 2 == 0 else 95
        frame[:] = (v, v // 2, 0)
    _marker(frame, t)
    return frame


def gen_video(path):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (W, H))
    assert vw.isOpened(), "VideoWriter 打开失败"
    n = int(FPS * DUR)
    for i in range(n):
        vw.write(scene_frame(i / FPS))
    vw.release()
    return path


# ---------------- 旧像素指标（与 bench/land_compare.py 同款） ----------------
DIFF_SIZE = 64
CHANGE_THRESHOLD = 3.0


def pixel_diff(img_a, img_b, size=DIFF_SIZE):
    small = lambda im: cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    a, b = small(img_a).astype(np.float32), small(img_b).astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0 * 100.0)


def truth_profile(video):
    """1s 桶，桶内首尾两帧像素差分 = 该秒"内容变化量"（旧尺子的真值定义）。"""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prof = []
    for st in range(0, int(DUR)):
        idx = int(st * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx + int(fps) - 1, n - 1))
        ok2, f2 = cap.read()
        if ok1 and ok2:
            prof.append({"t": st, "change": pixel_diff(f1, f2)})
    cap.release()
    return prof


def coverage_old(selected_ts, profile):
    """旧覆盖率：有变化秒中被选帧（±1s 容差）覆盖的占比。"""
    changed = [s for s in profile if s["change"] > CHANGE_THRESHOLD]
    if not changed:
        return 1.0, 0, 0
    covered = sum(1 for s in changed
                  if any(abs(t - s["t"]) <= 1.0 for t in selected_ts))
    return covered / len(changed), covered, len(changed)


def scene_of(t):
    for i, s in enumerate(SCENES):
        if s["start_s"] <= t < s["end_s"]:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description="合成端到端：旧像素指标 vs 语义指标")
    here = Path(__file__).resolve().parent
    ap.add_argument("--out-dir", default=str(here / "out_synthetic"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) 合成视频 + 自动 GT ----
    video = gen_video(out / "synthetic_4scene.mp4")
    gt = {"video": "synthetic_4scene.mp4", "duration_s": DUR,
          "source": "human", "_note": "合成脚本即真值：4 个语义场景，边界精确",
          "scenes": SCENES}
    write_json(str(out), "ground_truth.json", gt)
    print(f"[Gen] 合成视频: {video} ({W}x{H}@{FPS}fps, {DUR:.0f}s, 4 语义场景)")

    # ---- 2) 跑本管线（Tier2 关键帧 -> Tier3 代表帧）----
    # 注意：Windows 上 OpenCV 5 的 cv2.imwrite/imread 对**绝对**非 ASCII 路径
    # 会静默失败（相对路径正常）。管线产物统一用相对路径喂给 vus，
    # JSON 落盘仍走 vus.io_utils（pathlib，不受影响）。
    try:
        pipe_out_rel = str(Path(os.path.relpath(out / "pipeline_out")))
        kf_dir_rel = str(Path(os.path.relpath(out / "pipeline_out" / "keyframes")))
    except ValueError:  # 跨盘符无法 relpath：退回系统临时 ASCII 目录
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="vus_semantic_eval_"))
        pipe_out_rel, kf_dir_rel = str(tmp / "pipeline_out"), str(tmp / "pipeline_out" / "keyframes")
    pipe, _aligned, _asr = run_realtime_pipeline(
        str(video), pipe_out_rel, save_keyframes=True,
        config={"fast_scale": 0.25, "keyframe_interval_hz": 1.5})
    kf_dir = out / "pipeline_out" / "keyframes"
    tier2_ts = [k["t"] for k in pipe.keyframes]

    reps = select_representatives(kf_dir_rel, interval=REP_INTERVAL)
    write_json(str(out), "representatives.json",
               {"count": len(reps), "interval": REP_INTERVAL,
                "representatives": reps})
    tier3_ts = [r["t"] for r in reps]
    print(f"[Gen] Tier2 关键帧 {len(tier2_ts)} 张 -> Tier3 代表帧 {len(tier3_ts)} 张 "
          f"@ t={[round(t,1) for t in tier3_ts]}")

    # ---- 3) 旧像素指标 ----
    prof = truth_profile(video)
    changed = [s for s in prof if s["change"] > CHANGE_THRESHOLD]
    # 语义误报：距任何 GT 边界 >1.5s 的"有变化秒"——不可能是场景切换造成
    bounds = [s["start_s"] for s in SCENES] + [SCENES[-1]["end_s"]]
    false_pos = [s for s in changed
                 if all(abs(s["t"] - b) > 1.5 for b in bounds)]
    cov3_old, c3, n3 = coverage_old(tier3_ts, prof)
    cov2_old, c2, n2 = coverage_old(tier2_ts, prof)
    per_scene_changed = {}
    for s in changed:
        si = scene_of(float(s["t"]))
        per_scene_changed[si] = per_scene_changed.get(si, 0) + 1

    # ---- 4) 语义指标（Tier3 代表帧）----
    sem = semantic_evaluate(tier3_ts, gt)

    # ---- 5) 对比汇总 ----
    rows = {
        "video": "synthetic_4scene.mp4",
        "gt_scenes": len(SCENES),
        "tier3_reps": len(tier3_ts),
        "tier3_rep_ts": [round(t, 2) for t in tier3_ts],
        "tier2_keyframes": len(tier2_ts),
        "old_pixel_metric": {
            "changed_seconds_total": n3,
            "changed_seconds_per_scene": per_scene_changed,
            "intra_scene_false_changes": len(false_pos),
            "intra_scene_false_ratio": round(len(false_pos) / max(1, n3), 3),
            "tier3_coverage": round(cov3_old, 3),
            "tier2_coverage": round(cov2_old, 3),
            "note": "1s桶像素差分>3%记为有变化秒；覆盖率=被选帧(±1s)覆盖的有变化秒占比",
        },
        "semantic_metric": {
            "semantic_coverage": sem["semantic_coverage"],
            "covered_scenes": sem["covered_scenes"],
            "redundancy": sem["redundancy"],
            "gini": sem["gini"],
        },
    }
    write_json(str(out), "comparison_results.json", rows)

    print()
    print("=" * 72)
    print("合成对照实验：旧像素指标 vs 语义指标（同一套 Tier3 代表帧）")
    print("=" * 72)
    print(f"合成视频 4 个语义场景（纯色A抖动 -> 纯色B -> 纹路C -> 纯色A'），"
          f"Tier3 选帧 {len(tier3_ts)} 张（interval={REP_INTERVAL:.0f}s）")
    print()
    print(f"[旧像素尺子] 判定\"有内容变化\"的秒数: {n3}s / 48s"
          f"  （其中距场景边界>1.5s 的纯场景内部变化: {len(false_pos)}s, "
          f"占 {rows['old_pixel_metric']['intra_scene_false_ratio']*100:.0f}% —— 全是纯色抖动误报）")
    scene_labels = {0: "A", 1: "B", 2: "C", 3: "A'", None: "区间外"}
    per_scene_str = ", ".join(f"场景{scene_labels[k]}: {v}s"
                              for k, v in sorted(per_scene_changed.items(),
                                                 key=lambda kv: (kv[0] is None, kv[0])))
    print(f"  逐场景有变秒: {per_scene_str}"
          f"   <- 场景 A/A' 语义单一，却被判为逐秒变化")
    print(f"  Tier3 ({len(tier3_ts)}帧) 旧覆盖率: {cov3_old*100:.1f}%")
    print(f"  Tier2 ({len(tier2_ts)}帧) 旧覆盖率: {cov2_old*100:.1f}%"
          f"  <- 像素尺子用 {len(tier2_ts)/max(1,len(tier3_ts)):.0f} 倍帧量才追平自己造出的\"变化\"")
    print()
    print(f"[语义尺子 ] 语义覆盖率: {sem['semantic_coverage']*100:.0f}% "
          f"({sem['covered_scenes']}/{sem['n_scenes']} 场景)  "
          f"冗余度: {sem['redundancy']:.2f}  场景保真(基尼): {sem['gini']:.2f}")
    print()
    print("结论: 像素尺子把\"语义相同、像素抖动\"的纯色段判成持续内容变化")
    print("      （语义误报率 100%），并据此惩罚语义上完美的稀疏选帧；")
    print("      语义尺子下 4 帧即满分。评估应采用 bench/semantic_eval 协议。")
    print(f"\n[Gen] 对比结果已保存: {out / 'comparison_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
