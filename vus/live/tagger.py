#!/usr/bin/env python3
"""
tagger.py - T0.5 语义标签道（毫秒级、零 API 成本、零模型下载）
==============================================================
在关键帧产生的瞬间打上语义标签，让监控 HUD / 机器人反射层在 VLM 摘要
（秒级）到来之前就有语义可用。

本期默认实现 BasicLabeler：
  - Haar 级别人脸检测（opencv 自带 cascade，无额外下载）→ "有人脸/无人脸"
  - 运动面积比（快系统语义门控同源统计）→ 强运动/弱运动/静态
单帧 5-20ms（320px 灰度图），1.5Hz 关键帧频率下占用 <5% 单核。

CLIP 零样本文本标签（标签集可配）需要完整 CLIP ONNX（含文本塔）+ 分词器，
现有 clip_onnx.py 是纯视觉塔（只能做图像-图像距离）——归入 W9 与本地 VLM
一起接入。本模块的 Labeler 接口（label -> [{label, score}]）保持稳定，
到时以新后端替换即可，理解层与状态层零改动。
"""

import os
import threading


class LabelerError(RuntimeError):
    pass


class BasicLabeler:
    """毫秒级基础标签道：人脸检测 + 运动强度。线程安全（cascade 只读）。"""

    #: 输出标签语义（score ∈ [0,1]，sorted 降序）
    FACE = "有人脸"
    NO_FACE = "无人脸"
    STRONG_MOTION = "强运动"
    WEAK_MOTION = "弱运动"
    STATIC = "静态画面"

    #: motion_ratio ≥ 此值判强运动（与快系统语义门控 0.003 同量级）
    STRONG_MOTION_RATIO = 0.01

    def __init__(self, face_scale_width=320):
        self.face_scale_width = int(face_scale_width)
        self._cascade = None
        self._cascade_err = None
        self._lock = threading.Lock()
        self._load_cascade()

    def _load_cascade(self):
        """加载 OpenCV 自带 Haar 正脸 cascade；失败不致命（只损失人脸标签）。"""
        try:
            import cv2
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if os.path.exists(cascade_path):
                self._cascade = cv2.CascadeClassifier(cascade_path)
                if self._cascade.empty():
                    self._cascade = None
                    self._cascade_err = "cascade 加载为空"
            else:
                self._cascade_err = f"cascade 文件缺失: {cascade_path}"
        except Exception as e:  # opencv 变体不带 cv2.data 等
            self._cascade_err = str(e)

    @property
    def face_detection_available(self):
        return self._cascade is not None

    def label(self, frame_bgr, motion_ratio=0.0):
        """关键帧 → 标签列表 [{"label", "score"}]（降序）。

        motion_ratio: 快系统本帧运动面积比（0-1）。score 为启发式置信度。
        """
        import cv2
        import numpy as np
        labels = []
        if motion_ratio >= self.STRONG_MOTION_RATIO:
            labels.append({self.STRONG_MOTION: 0.9})
        elif motion_ratio > 0:
            labels.append({self.WEAK_MOTION: 0.6})
        else:
            labels.append({self.STATIC: 0.7})

        face_score = self._face_score(cv2, np, frame_bgr)
        if face_score is not None:
            labels.append({self.FACE if face_score > 0 else self.NO_FACE: 0.8})
        return self._normalize(labels)

    def _face_score(self, cv2, np, frame_bgr):
        """检测到人脸返回人脸数（>0）；无人脸返回 0；检测不可用返回 None。"""
        if self._cascade is None or frame_bgr is None:
            return None
        with self._lock:
            h, w = frame_bgr.shape[:2]
            scale = self.face_scale_width / max(w, 1)
            small = cv2.resize(frame_bgr, (self.face_scale_width,
                                           max(1, round(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self._cascade.detectMultiScale(gray, scaleFactor=1.2,
                                                   minNeighbors=5, minSize=(24, 24))
        return len(faces)

    @staticmethod
    def _normalize(pairs):
        out = []
        for d in pairs:
            for label, score in d.items():
                out.append({"label": label, "score": round(float(score), 2)})
        out.sort(key=lambda x: -x["score"])
        return out


def create_labeler(kind="basic", **kwargs):
    """标签道工厂。clip 零样本文本标签后端为 W9 预留。"""
    if kind == "basic":
        return BasicLabeler(**kwargs)
    if kind == "clip":
        raise LabelerError(
            "CLIP 零样本文本标签需要完整 CLIP ONNX（含文本塔）+ 分词器，W9 接入；"
            "当前可用后端: basic")
    raise LabelerError(f"未知标签道后端: {kind}（可用: basic / clip 预留）")
