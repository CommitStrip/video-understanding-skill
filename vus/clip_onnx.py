#!/usr/bin/env python3
"""
clip_onnx.py - CLIP 视觉塔 ONNX 推理（去 torch 依赖的语义增强引擎）
====================================================================
W3（2026-09-02）：把 Tier3 代表帧选择的 CLIP 语义增强从 torch + openai-clip
迁移到 onnxruntime CPU 推理，安装体积从数 GB（torch CUDA 全家桶）降到
几十 MB（onnxruntime CPU），且权重一次性下载、离线可用。

模型文件约定: <model_dir>/clip-visual-vitb32.onnx
  来源两种：
    1) scripts/export_clip_onnx.py 在装有 torch+openai-clip 的机器上导出
       （纯视觉塔，输入 NCHW float32，输出图像嵌入）；
    2) scripts/download_clip_onnx.sh 从 hf-mirror 下载现成 ONNX
       （可能文本+视觉合一，如 onnx-community 的 CLIPModel 导出；
        本模块通过输入/输出名自动探测视觉入口，只消费视觉塔）。

预处理（与 openai/CLIP 官方 preprocess 对齐）：
  resize 短边 224 → 中心裁剪 224x224 → BGR2RGB → /255
  → (x-mean)/std，mean=[0.481,0.457,0.408] std=[0.269,0.261,0.275]
  → NCHW float32

设计哲学（延续项目约定）：模型缺失 / onnxruntime 缺失时构造抛 RuntimeError
并给出明确指引（下载脚本 + 镜像地址），**绝不静默降级**为纯像素差分——
否则用户会误以为语义增强已生效。
"""

import os

import cv2
import numpy as np

MODEL_FILENAME = "clip-visual-vitb32.onnx"

# openai/CLIP ViT-B/32 官方归一化参数（RGB 顺序）
CLIP_MEAN = (0.481, 0.457, 0.408)
CLIP_STD = (0.269, 0.261, 0.275)
CLIP_INPUT_SIZE = 224

DOWNLOAD_HINT = (
    "未找到 CLIP ONNX 模型 {model_path}。获取方式二选一：\n"
    "  1) 运行 scripts/download_clip_onnx.sh 从国内镜像 hf-mirror.com 下载（推荐）；\n"
    "  2) 在装有 torch+openai-clip 的机器上运行 scripts/export_clip_onnx.py 自行导出。\n"
    "镜像地址示例: https://hf-mirror.com/onnx-community/clip-vit-base-patch32-ONNX/"
    "resolve/main/onnx/model.onnx （下载后重命名为 {model_filename} 放入模型目录）。\n"
    "也可用环境变量 VUS_CLIP_MODELS 或 --clip-models-dir 指定已有模型目录。"
)


def resolve_model_dir(model_dir=None):
    """模型目录查找顺序：参数 > 环境变量 VUS_CLIP_MODELS > ./models。"""
    if model_dir:
        return model_dir
    env_dir = os.environ.get("VUS_CLIP_MODELS", "").strip()
    if env_dir:
        return env_dir
    return os.path.join(".", "models")


class ClipOnnx:
    """CLIP 视觉塔 ONNX 推理器（onnxruntime CPU）。

    用法:
        enc = ClipOnnx("./models")
        vec = enc.embed(bgr_frame)   # L2 归一化的一维向量
        d = cosine_dist(vec_a, vec_b)
    """

    def __init__(self, model_dir=None):
        model_path = os.path.join(resolve_model_dir(model_dir), MODEL_FILENAME)
        if not os.path.isfile(model_path):
            raise RuntimeError(DOWNLOAD_HINT.format(
                model_path=model_path, model_filename=MODEL_FILENAME))
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError(
                "ClipOnnx 需要 onnxruntime: pip install -e \".[clip]\" "
                "（或 pip install onnxruntime）"
            ) from e
        self.model_path = model_path
        # CPU 固定：语义增强仅作离线 Tier3 增强，绝无 GPU 假设
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name, self.output_name, self.text_inputs = self._probe_io()

    # CLIP 词表 EOS token id（文本塔占位输入用；图像嵌入与其数值无关）
    _EOS_TOKEN_ID = 49407

    def _probe_io(self):
        """探测视觉塔的输入/输出名。

        兼容两种导出：
          - 纯视觉塔（export_clip_onnx.py 产物）：输入通常叫 pixel_values/input，
            输出为图像嵌入；
          - 文本+视觉合一（optimum/onnx-community 的 CLIPModel 导出）：
            输入含 input_ids/attention_mask/pixel_values，输出含 image_embeds。
            这类图把文本输入声明为必填，推理时喂最小占位 token（EOS）即可——
            image_embeds 只依赖视觉分支，数值上与文本输入无关。
        返回 (image_input_name, output_name, text_input_names)。
        """
        inp = None
        for i in self.session.get_inputs():
            name = (i.name or "").lower()
            shape = i.shape or []
            if "pixel" in name:
                inp = i.name
                break
            # 形状为 [batch, 3, H, W] 的输入即图像输入
            if len(shape) == 4 and (shape[1] == 3 or shape[1] == "3"):
                inp = inp or i.name
        if inp is None:
            inp = self.session.get_inputs()[0].name

        out = None
        for o in self.session.get_outputs():
            name = (o.name or "").lower()
            if "image_embed" in name or "image_feat" in name:
                out = o.name
                break
        if out is None:
            out = self.session.get_outputs()[-1].name

        text_names = [i.name for i in self.session.get_inputs()
                      if (i.name or "").lower() in ("input_ids", "attention_mask")]
        return inp, out, text_names

    def _dtype_of(self, input_name):
        """把 ORT 类型字符串（tensor(int64) 等）映射为 numpy dtype。"""
        t = next((i.type for i in self.session.get_inputs() if i.name == input_name), "")
        for dt in (np.dtype(np.int64), np.dtype(np.int32)):
            if dt.name in (t or ""):
                return dt.type
        return np.int64

    def _build_feed(self, x):
        """构造推理 feed：图像输入 + 文本塔占位输入（合一模型需要）。"""
        feed = {self.input_name: x}
        for name in self.text_inputs:
            lname = (name or "").lower()
            if lname == "attention_mask":
                feed[name] = np.ones((1, 1), dtype=self._dtype_of(name))
            else:  # input_ids：单个 EOS token 占位
                feed[name] = np.full((1, 1), self._EOS_TOKEN_ID, dtype=self._dtype_of(name))
        return feed

    def preprocess(self, bgr_frame):
        """BGR 帧 → NCHW float32（与 openai/CLIP 官方 preprocess 对齐）。"""
        if bgr_frame is None or getattr(bgr_frame, "size", 0) == 0:
            raise ValueError("preprocess 需要非空 BGR 帧")
        h, w = bgr_frame.shape[:2]
        # 1) resize 短边到 224（保持纵横比，双线性，与官方 Resize 一致）
        scale = CLIP_INPUT_SIZE / min(h, w)
        new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
        img = cv2.resize(bgr_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # 2) 中心裁剪 224x224
        ch, cw = img.shape[:2]
        top, left = max(0, (ch - CLIP_INPUT_SIZE) // 2), max(0, (cw - CLIP_INPUT_SIZE) // 2)
        img = img[top:top + CLIP_INPUT_SIZE, left:left + CLIP_INPUT_SIZE]
        if img.shape[0] != CLIP_INPUT_SIZE or img.shape[1] != CLIP_INPUT_SIZE:
            img = cv2.resize(img, (CLIP_INPUT_SIZE, CLIP_INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        # 3) BGR2RGB → /255 → (x-mean)/std → CHW
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(CLIP_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(CLIP_STD, dtype=np.float32).reshape(3, 1, 1)
        chw = (rgb.transpose(2, 0, 1) - mean) / std
        return chw[np.newaxis, ...].astype(np.float32)  # NCHW

    def embed(self, bgr_frame):
        """单帧 → L2 归一化的语义嵌入向量 (D,) float32。"""
        x = self.preprocess(bgr_frame)
        out = self.session.run([self.output_name], self._build_feed(x))[0]
        vec = np.asarray(out, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        return vec

    def cosine_dist(self, a, b):
        """实例方法别名，见模块级 cosine_dist。"""
        return cosine_dist(a, b)


def cosine_dist(a, b):
    """余弦距离 1 - cos(a,b)（0-1，越大越不相似）。零向量定义为距离 0。"""
    va, vb = np.asarray(a, dtype=np.float32).reshape(-1), np.asarray(b, dtype=np.float32).reshape(-1)
    na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
    if na == 0 or nb == 0:
        return 0.0
    return 1.0 - float(np.dot(va, vb) / (na * nb))
