#!/usr/bin/env bash
# download_clip_onnx.sh - 下载 CLIP ViT-B/32 视觉 ONNX 权重到 ./models/
# =========================================================================
# 用途: vus/clip_onnx.ClipOnnx 需要 <models_dir>/clip-visual-vitb32.onnx。
#       本脚本从国内镜像 hf-mirror.com（回退 huggingface.co 原站）下载现成
#       ONNX 并落为该文件名，装 torch 很重，推荐用本脚本获取权重。
#
# 用法:
#   bash scripts/download_clip_onnx.sh            # 下载到 ./models
#   MODELS_DIR=/path/to/models bash scripts/download_clip_onnx.sh
#
# 说明:
#   - openai/clip-vit-base-patch32 官方仓库**没有** onnx 目录（2026-09 实测
#     onnx/model.onnx 为 404），实际可用的是 onnx-community 的 CLIPModel
#     导出（文本+视觉合一，~605MB fp32）。vus/clip_onnx.ClipOnnx 会自动
#     探测视觉入口（pixel_values -> image_embeds），只消费视觉塔。
#   - curl 带超时（整体 600s 级）与断点续传/重试。
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-./models}"
OUT="$MODELS_DIR/clip-visual-vitb32.onnx"

# 候选源（按优先级）：镜像在前，原站兜底。文件为 fp32 CLIPModel 合一导出。
URLS=(
  "https://hf-mirror.com/onnx-community/clip-vit-base-patch32-ONNX/resolve/main/onnx/model.onnx"
  "https://huggingface.co/onnx-community/clip-vit-base-patch32-ONNX/resolve/main/onnx/model.onnx"
)

if [ -s "$OUT" ]; then
  echo "[Download] 已存在: $OUT ($(du -h "$OUT" | cut -f1))，跳过下载"
  exit 0
fi

mkdir -p "$MODELS_DIR"

echo "[Download] 目标: $OUT"
for url in "${URLS[@]}"; do
  echo "[Download] 尝试: $url"
  # --max-time 590 单次整体超时; --retry 3 断点重试; -C - 断点续传
  if curl -L --connect-timeout 20 --max-time 590 --retry 3 --retry-delay 5 \
        -C - --fail -sS -o "$OUT.part" "$url"; then
    mv "$OUT.part" "$OUT"
    echo "[Download] 完成: $OUT ($(du -h "$OUT" | cut -f1))"
    echo "[Download] 验证: python -m pytest tests/test_clip_onnx.py -v -k real_model"
    exit 0
  fi
  echo "[Download] 该源失败，尝试下一候选..." >&2
  rm -f "$OUT.part"
done

echo "[Download] 所有源均失败。可手动下载后重命名放入 $MODELS_DIR/:" >&2
echo "  curl -L -o $OUT ${URLS[0]}" >&2
echo "或在装有 torch+openai-clip 的机器上: python scripts/export_clip_onnx.py" >&2
exit 1
