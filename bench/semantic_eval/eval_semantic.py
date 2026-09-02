#!/usr/bin/env python3
"""
eval_semantic.py - 语义级评估：语义覆盖率 / 冗余度 / 场景保真
==============================================================
按 PROTOCOL.md 的定义，把代表帧（representatives.json）对照语义场景标注
（ground_truth.json）打分，打破"像素差分尺子评像素差分方法"的同构偏差。

用法:
  python bench/semantic_eval/eval_semantic.py \
      --reps  out/representatives.json \
      --gt    bench/semantic_eval/ground_truth.json \
      [--report report.md]

ground_truth.json schema（详见 PROTOCOL.md）:
  {"video": "...", "duration_s": 48.0, "source": "human|vlm-assisted",
   "scenes": [{"start_s": 0.0, "end_s": 12.0, "desc": "..."}, ...]}

指标:
  语义覆盖率 = 有 >=1 张代表帧落入的 GT 场景时长 / 总 GT 场景时长（按时长加权）
  冗余度     = 代表帧总数 / GT 场景数（越接近 1 越精简；k>1 预期 >1）
  场景保真   = 各场景代表帧数分布的基尼系数（可选，0=均匀 1=全部堆一处）
"""
import argparse
import json
import sys
from pathlib import Path

# 相对定位仓库根：本文件位于 <repo>/bench/semantic_eval/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vus.io_utils import write_text  # noqa: E402  落盘统一走 vus.io_utils

_EPS = 1e-6


def load_reps(path):
    """读 representatives.json -> 排序后的代表帧时间戳列表。

    兼容两种形态：管线输出 {"representatives": [{"t","path"},...]}，
    或裸列表 [{"t","path"},...]。
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data["representatives"] if isinstance(data, dict) else data
    return sorted(float(it["t"]) for it in items)


def load_gt(path):
    """读 ground_truth.json -> dict（含 scenes 列表）。"""
    gt = json.loads(Path(path).read_text(encoding="utf-8"))
    scenes = sorted(gt.get("scenes", []), key=lambda s: s["start_s"])
    if not scenes:
        raise ValueError(f"{path} 中没有 scenes 标注")
    return gt


def assign_reps(rep_ts, scenes):
    """把代表帧逐个分配到场景 [start, end)（末场景闭区间兼容帧恰在视频末尾）。

    返回 (per_scene_counts, unassigned_ts)。
    """
    counts = [0] * len(scenes)
    unassigned = []
    for t in rep_ts:
        hit = None
        for i, s in enumerate(scenes):
            if s["start_s"] - _EPS <= t < s["end_s"] + _EPS:
                hit = i
                break
        if hit is None:
            unassigned.append(t)
        else:
            counts[hit] += 1
    return counts, unassigned


def gini(counts):
    """基尼系数：各场景代表帧数分布的不均匀度（0=完全均匀, 1=全部堆一处）。"""
    xs = sorted(float(c) for c in counts)
    n = len(xs)
    total = sum(xs)
    if n == 0 or total == 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2.0 * cum - (n + 1) * total) / (n * total)


def evaluate(rep_ts, gt):
    """计算语义指标，返回结果 dict（含逐场景明细）。"""
    scenes = sorted(gt.get("scenes", []), key=lambda s: s["start_s"])
    if not scenes:
        raise ValueError("ground_truth.json 中没有 scenes 标注")
    counts, unassigned = assign_reps(rep_ts, scenes)
    durations = [max(0.0, s["end_s"] - s["start_s"]) for s in scenes]
    total_dur = sum(durations)
    covered_dur = sum(d for d, c in zip(durations, counts) if c >= 1)

    coverage = covered_dur / total_dur if total_dur > 0 else 0.0
    redundancy = len(rep_ts) / len(scenes) if scenes else 0.0
    return {
        "video": gt.get("video", ""),
        "gt_source": gt.get("source", ""),
        "n_scenes": len(scenes),
        "n_reps": len(rep_ts),
        "total_duration_s": round(total_dur, 2),
        "semantic_coverage": round(coverage, 4),
        "covered_scenes": sum(1 for c in counts if c >= 1),
        "redundancy": round(redundancy, 4),
        "gini": round(gini(counts), 4),
        "unassigned_reps": unassigned,
        "scenes": [
            {
                "idx": i,
                "start_s": scenes[i]["start_s"],
                "end_s": scenes[i]["end_s"],
                "dur_s": round(durations[i], 2),
                "desc": scenes[i].get("desc", ""),
                "reps": counts[i],
                "covered": counts[i] >= 1,
            }
            for i in range(len(scenes))
        ],
    }


def format_report(r):
    """渲染人类可读的评估表（纯文本 + Markdown 兼容）。"""
    lines = []
    lines.append("# 语义级评估报告")
    lines.append("")
    if r["video"]:
        lines.append(f"- 视频: {r['video']}  （GT 来源: {r['gt_source'] or '未标注'}）")
    lines.append(f"- GT 场景数: {r['n_scenes']}  代表帧数: {r['n_reps']}"
                 f"  GT 总时长: {r['total_duration_s']}s")
    lines.append("")
    lines.append("## 指标")
    lines.append("")
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("|------|----|------|")
    lines.append(f"| 语义覆盖率 | {r['semantic_coverage']*100:.1f}% | "
                 f"{r['covered_scenes']}/{r['n_scenes']} 场景有代表帧（按时长加权） |")
    lines.append(f"| 冗余度 | {r['redundancy']:.2f} | 代表帧数/GT场景数，"
                 "越接近 1 越精简（k>1 预期 >1） |")
    lines.append(f"| 场景保真(基尼) | {r['gini']:.3f} | 每场景代表帧数分布，"
                 "覆盖率打满时越低越均匀 |")
    lines.append("")
    lines.append("## 逐场景明细")
    lines.append("")
    lines.append("| # | 区间(s) | 时长(s) | 代表帧数 | 覆盖 | 描述 |")
    lines.append("|---|---------|---------|----------|------|------|")
    for s in r["scenes"]:
        lines.append(f"| {s['idx']} | {s['start_s']:.1f}-{s['end_s']:.1f} "
                     f"| {s['dur_s']:.1f} | {s['reps']} | "
                     f"{'Y' if s['covered'] else '**N**'} | {s['desc']} |")
    if r["unassigned_reps"]:
        lines.append("")
        lines.append(f"警告: {len(r['unassigned_reps'])} 张代表帧未落入任何 GT 场景: "
                     f"{[round(t, 2) for t in r['unassigned_reps']]}"
                     "（多为视频首尾超出标注区间的帧，请检查 GT 完整性）")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="语义级评估（覆盖率/冗余度/场景保真）")
    here = Path(__file__).resolve().parent
    ap.add_argument("--reps", required=True, help="representatives.json 路径")
    ap.add_argument("--gt", required=True, help="ground_truth.json 路径")
    ap.add_argument("--report", default=None, help="输出 Markdown 报告路径")
    ap.add_argument("--out-json", default=None,
                    help="输出机器可读指标 JSON 路径（默认与 gt 同目录 eval_metrics.json）")
    args = ap.parse_args()

    rep_ts = load_reps(args.reps)
    gt = load_gt(args.gt)
    r = evaluate(rep_ts, gt)

    text = format_report(r)
    print(text)

    from vus.io_utils import write_json
    out_json = args.out_json or str(
        Path(args.gt).resolve().parent / "eval_metrics.json")
    write_json(str(Path(out_json).parent), Path(out_json).name, r)
    print(f"[Eval] 指标 JSON 已保存: {out_json}")

    if args.report:
        write_text(str(Path(args.report).resolve().parent),
                   Path(args.report).name, text)
        print(f"[Eval] 报告已保存: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
