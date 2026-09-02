#!/usr/bin/env python3
"""
export_clip_onnx.py - 把 CLIP ViT-B/32 视觉塔导出为 ONNX
=========================================================
在**装有 torch + openai-clip** 的机器上运行（本仓库运行时不需要 torch）：

    pip install torch openai-clip onnx
    python scripts/export_clip_onnx.py --out-dir ./models

产物:
    <out-dir>/clip-visual-vitb32.onnx   纯视觉塔（输入 NCHW float32 1x3x224x224，
                                        输出图像嵌入 512 维）
    <out-dir>/clip-visual-vitb32.readme.txt  导出说明（来源、输入输出签名）

说明:
  - openai-clip 首次运行需外网下载 ViT-B/32 权重（约 350MB，缓存在 ~/.cache/clip）。
  - 导出为固定 batch=1：Tier3 语义增强本身就是逐帧离线计算，固定形状模型更小更稳。
    vus/clip_onnx.ClipOnnx 也兼容动态 batch / 文本视觉合一的第三方导出。
  - 无 torch 环境请改用 scripts/download_clip_onnx.sh 直接下载现成 ONNX。
"""

import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser(description="导出 CLIP ViT-B/32 视觉塔为 ONNX")
    ap.add_argument("--out-dir", default="./models", help="输出目录（默认 ./models）")
    args = ap.parse_args()

    # ---- 依赖探测：本机无 torch 时打印清晰指引退出（不静默失败）----
    missing = []
    try:
        import torch  # noqa: F401
    except ImportError:
        missing.append("torch")
    try:
        import clip  # noqa: F401
    except ImportError:
        missing.append("openai-clip")
    try:
        import onnx  # noqa: F401
    except ImportError:
        missing.append("onnx")
    if missing:
        print("[Export] 缺少依赖: " + ", ".join(missing))
        print("[Export] 请在有 torch 环境的机器上执行:")
        print("           pip install torch openai-clip onnx")
        print("           python scripts/export_clip_onnx.py --out-dir ./models")
        print("[Export] 不想装 torch？改用下载脚本获取现成 ONNX:")
        print("           bash scripts/download_clip_onnx.sh")
        print("           （镜像: https://hf-mirror.com，"
              "示例仓库 onnx-community/clip-vit-base-patch32-ONNX）")
        return 1

    import torch
    import clip

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clip-visual-vitb32.onnx")

    print("[Export] 加载 CLIP ViT-B/32（首次运行会下载权重，需外网）...")
    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()
    visual = model.visual  # VisionTransformer: 输入 1x3x224x224，输出 1x512

    dummy = torch.randn(1, 3, 224, 224)
    print(f"[Export] torch.onnx.export -> {out_path} ...")
    with torch.no_grad():
        torch.onnx.export(
            visual,
            (dummy,),
            out_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=14,
            dynamic_axes={"pixel_values": {0: "batch"},
                          "image_embeds": {0: "batch"}},
            do_constant_folding=True,
        )
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[Export] 完成: {out_path} ({size_mb:.1f} MB)")

    note = (
        "clip-visual-vitb32.onnx 导出说明\n"
        "================================\n"
        "来源: openai/CLIP ViT-B/32 视觉塔（scripts/export_clip_onnx.py 导出）\n"
        "输入: pixel_values, float32, [batch, 3, 224, 224],\n"
        "      预处理 = resize短边224 -> 中心裁剪224 -> BGR2RGB -> /255\n"
        "      -> (x-mean)/std, mean=[0.481,0.457,0.408] std=[0.269,0.261,0.275]\n"
        "输出: image_embeds, float32, [batch, 512]（未归一化，调用方需 L2 归一化）\n"
        "配套推理代码: vus/clip_onnx.ClipOnnx\n"
        "验证: python -m pytest tests/test_clip_onnx.py -v\n"
    )
    from pathlib import Path
    Path(os.path.join(out_dir, "clip-visual-vitb32.readme.txt")).write_text(note,
                                                                             encoding="utf-8")
    print("[Export] 说明已写: clip-visual-vitb32.readme.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
