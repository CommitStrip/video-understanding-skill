#!/usr/bin/env python3
"""
select_representatives.py - 语义代表帧选择（Tier 3 / 内容理解层）
=================================================================
把管线输出的"镜头级关键帧"（Tier 2，通常 1-2s 一张）进一步压缩为
"语义代表帧"（通常 30-60s 一张），供 LLM 做高效、无冗余的内容理解。

历史教训（来自真实 75 分钟视频验证）：
  - 管线原样输出 1609 张关键帧，但 LLM 只抽样 15 张就理解了全部内容。
  - 因为表层运动（切镜头、说话人近景交替）≠ 语义内容变化。
  - 本脚本用两种互补信号做二次筛选：
      1) 像素差分（downscale RGB 均值差 %）—— 去视觉冗余，对慢变化/纯色色相敏感
      2) 时间均匀锚点 —— 保证最长时间间隔内至少保留 1 帧（守住稀疏区）

为什么用像素差分而非 pHash（借鉴 claude-real-video 的实测结论）：
  - pHash 在"等亮度色相变化"和"纯色画面"上失明（灰度哈希对色相迟钝）。
  - 实测：slow 渐变视频相邻帧 pHash 仅 12（被判相似），但像素差分 14.3%（应保留）。
  - 像素差分把相邻帧差异从 [48,80] 提升到稳定 14-20%，贴近人眼感知。

可选 CLIP 语义增强（P0，默认关闭）：
  - 像素差分对"纯低级信号"（色相渐变/切变）敏感，但对"语义内容"不敏感。
  - 例如同一场景的不同机位、说话人交替，像素差分可能很大但语义相同；反之语义切换但画面相似。
  - 启用 --clip 后，桶内选帧打分 = w_pix*像素差分 + (1-w_pix)*CLIP 语义距离，兼顾视觉与语义。
  - 实测结论（2026-08-10）：CLIP ViT-B/32 单帧 CPU 推理 157.7ms / 1029MB，
    绝不可进逐帧实时路径；仅作为**可选的 Tier3 离线增强**（对桶锚点候选计算）。
  - 依赖真实 CLIP 权重（openai/clip ViT-B/32），需外网下载；沙箱/离线环境会显式报错而非静默降级。

用法:
  python select_representatives.py --keyframes <dir> [--out json] [--interval 60]
  python select_representatives.py --keyframes <dir> --report context.md
"""

import os
import sys
import re
import json
import argparse
import numpy as np
import cv2


_DIFF_SIZE = 64  # 像素差分对比的分辨率（downscale 后）

def _pixel_diff(img_a, img_b, size=_DIFF_SIZE):
    """像素差分：降采样到 size x size 后的 RGB 平均差分百分比（0-100）。
    等价于 crv 的 downscaled-RGB 全局差分通道，对慢变化和等亮度色相变化敏感。"""
    small = lambda im: cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    a = small(img_a).astype(np.float32)
    b = small(img_b).astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0 * 100.0)


def load_clip_model(device=None):
    """延迟加载真实 CLIP 模型（P0 离线增强用）。

    返回 (preprocess, model, device)。只应在显式 --clip 时调用。
    依赖 openai/CLIP 的 ViT-B/32 权重；缺包或缺权重时抛 RuntimeError，
    由调用方显式报错，而非静默降级为像素差分（避免用户误以为语义增强已生效）。
    """
    try:
        import torch
        import clip
    except ImportError as e:
        raise RuntimeError(
            "启用 --clip 需要安装推理依赖: pip install torch openai-clip（真实 CLIP 权重需外网下载）"
        ) from e
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except Exception as e:
        raise RuntimeError(
            f"加载 CLIP ViT-B/32 权重失败: {e}。"
            "请确认外网可达且权重已缓存；离线/被墙环境无法启用语义增强。"
        ) from e
    model.eval()
    return preprocess, model, device


def _clip_embed(img_bgr, preprocess, model, device):
    """对单帧提取 CLIP 语义嵌入向量 (D,)。"""
    import torch
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = preprocess(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    if hasattr(feat, "float"):
        feat = feat.float()
    return feat.cpu().numpy().squeeze()


def _clip_dist(a, b):
    """语义距离：1 - 余弦相似度（0-1，越大越不相似）。"""
    va, vb = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - float(np.dot(va, vb) / (na * nb))


def load_keyframes(kf_dir):
    """扫描 keyframes 目录，返回 [(t, path), ...] 按时间排序。"""
    items = []
    for f in sorted(os.listdir(kf_dir)):
        m = re.search(r'_t([0-9.]+)s\.jpg', f)
        if m:
            items.append((float(m.group(1)), os.path.join(kf_dir, f)))
    items.sort(key=lambda x: x[0])
    return items


def select_representatives(kf_dir, interval=60.0, dedup_threshold=0.0, clip=None, w_pix=0.5):
    """
    分层选择语义代表帧。

    策略（时间完整性优先）：
      1) 时间分桶：每 interval 秒一个桶，桶内选与桶首像素差分最大的帧作为该时段锚点。
         桶锚点天然间隔 ≥ interval，保证稀疏区不被漏掉。
      2) 桶锚点无条件保留：不因视觉相似而丢弃 —— 这是"每 interval 秒至少 1 帧"的硬保证。
      3) 可选去重（仅当 dedup_threshold > 0）：像素差分 < 阈值的连续锚点视为冗余合并，
         用于内容高度单调的视频进一步压缩，默认关闭。

    dedup_threshold 单位：像素差分百分比（0-100）。推荐 8（对应 64x64 下 8% 像素变化）。

    clip: 可选 (preprocess, model, device) 元组（来自 load_clip_model）。为 None 时纯像素差分。
        启用时，桶内选帧打分 = w_pix*像素差分 + (1-w_pix)*CLIP 语义距离。
    w_pix: 像素差分权重（0-1），默认 0.5，与 CLIP 语义距离各占一半。

    返回: [{"t": float, "path": str}, ...]
    """
    kfs = load_keyframes(kf_dir)
    if not kfs:
        return []

    def _score(cand, first):
        """桶内候选帧相对首帧的综合差异分。"""
        pix = _pixel_diff(cand['img'], first['img']) / 100.0  # 归一化 0-1
        if clip is None:
            return pix
        sem = _clip_dist(cand['emb'], first['emb'])  # 0-1
        return w_pix * pix + (1.0 - w_pix) * sem

    # 步骤1：时间分桶，桶内选与桶首差异最大帧作为锚点
    bucket_start = 0.0
    bucket = []
    candidates = []
    for t, p in kfs:
        img = cv2.imread(p)
        if img is None:
            continue
        emb = _clip_embed(img, *clip) if clip is not None else None
        node = {'t': t, 'p': p, 'img': img, 'emb': emb}
        if t >= bucket_start + interval:
            if bucket:
                first = bucket[0]
                candidates.append(max(bucket, key=lambda x: _score(x, first)))
            bucket = [node]
            bucket_start = t
        else:
            bucket.append(node)
    if bucket:
        first = bucket[0]
        candidates.append(max(bucket, key=lambda x: _score(x, first)))

    # 步骤2：默认全部保留（时间完整性优先）；仅当显式开启去重时压缩单调内容
    if dedup_threshold <= 0:
        return [{"t": round(c['t'], 1), "path": c['p']} for c in candidates]

    final = []
    final_hashes = []
    for c in candidates:
        is_redundant = False
        if final_hashes:
            is_redundant = _pixel_diff(c['img'], final_hashes[-1]) < dedup_threshold
        if not is_redundant:
            final.append({"t": round(c['t'], 1), "path": c['p']})
            final_hashes.append(c['img'])

    return final


def build_context(representatives, kf_dir, sources=None, out_context=None):
    """
    生成给 LLM 的分析上下文（按时间序的帧清单 + 可选的额外素材）。
    sources: 额外的结构化素材清单，如 [{"t":..,"type":"subtitle","text":..}]
    """
    lines = []
    lines.append("# 视频内容理解上下文")
    lines.append("")
    lines.append("## 代表帧清单（按时间序）")
    lines.append("")
    lines.append("| 时间点 | 帧文件 |")
    lines.append("|--------|--------|")
    for r in representatives:
        name = os.path.basename(r["path"])
        lines.append(f"| {r['t']:.1f}s | {name} |")
    lines.append("")
    if sources:
        lines.append("## 附加素材")
        lines.append("")
        for s in sources:
            lines.append(f"- [{s.get('t'):.1f}s] [{s.get('type')}] {s.get('text')}")
        lines.append("")
    text = "\n".join(lines)
    if out_context:
        with open(out_context, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def summarize(kf_dir):
    """打印关键帧数量与时间分布，便于判断是否需要压缩。"""
    kfs = load_keyframes(kf_dir)
    if not kfs:
        return {"count": 0}
    ts = [t for t, _ in kfs]
    gaps = np.diff(ts)
    return {
        "count": len(kfs),
        "span_s": round(ts[-1] - ts[0], 1),
        "avg_interval_s": round(np.mean(gaps), 2) if len(gaps) else 0,
        "p95_interval_s": round(np.percentile(gaps, 95), 2) if len(gaps) else 0,
        "max_interval_s": round(gaps.max(), 1) if len(gaps) else 0,
    }


def main():
    ap = argparse.ArgumentParser(description="语义代表帧选择（内容理解层）")
    ap.add_argument("--keyframes", required=True, help="关键帧目录")
    ap.add_argument("--interval", type=float, default=60.0, help="时间分桶间隔(秒)")
    ap.add_argument("--dedup-threshold", type=float, default=0.0,
                    help="像素差分去重阈值%%(0-100, 0=关闭,时间完整性优先; 推荐8)")
    ap.add_argument("--clip", action="store_true",
                    help="启用 CLIP 语义增强(可选,P0)。仅对桶锚点候选计算,离线环境若无权重会显式报错")
    ap.add_argument("--w-pix", type=float, default=0.5,
                    help="CLIP 混合权重中像素差分的占比(0-1, 默认0.5, 其余为CLIP语义距离)")
    ap.add_argument("--out", default=None, help="输出代表帧 JSON 路径")
    ap.add_argument("--report", default=None, help="输出 LLM 上下文 Markdown 路径")
    args = ap.parse_args()

    # 仅当显式 --clip 才加载真实 CLIP 模型；缺依赖/权重时显式报错，不静默降级
    clip_engine = None
    if args.clip:
        import time
        t0 = time.time()
        try:
            clip_engine = load_clip_model()
        except RuntimeError as e:
            sys.exit(f"[Select] 错误: {e}")
        print(f"[Select] CLIP 语义增强已启用 (模型加载 {time.time()-t0:.1f}s, "
              f"设备 {clip_engine[2]})")

    info = summarize(args.keyframes)
    print(f"[Select] 管线关键帧: {info['count']} 帧, 跨度 {info['span_s']}s, "
          f"平均间隔 {info['avg_interval_s']}s")

    reps = select_representatives(args.keyframes, args.interval,
                                  args.dedup_threshold, clip_engine, args.w_pix)
    print(f"[Select] 语义代表帧: {len(reps)} 张 "
          f"(压缩到 {len(reps)/info['count']*100:.1f}%)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"count": len(reps), "interval": args.interval,
                       "representatives": reps}, f, ensure_ascii=False, indent=2)
        print(f"[Select] 已保存: {args.out}")

    if args.report:
        ctx = build_context(reps, args.keyframes)
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(ctx)
        print(f"[Select] 已保存上下文: {args.report}")


if __name__ == "__main__":
    main()