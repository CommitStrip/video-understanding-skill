#!/usr/bin/env python3
"""
ocr_channel.py - OCR 第三通道（画面文字链，默认关）
====================================================
W3（2026-09-02）：在"快系统运动流 + 慢系统关键帧 + 流式 ASR"三通道之外，
新增可选的 OCR 通道：识别画面中的文字（字幕/路牌/幻灯片标题/聊天框等），
产出与 ASR 同构的文字事件。

设计约束（延续项目哲学）：
  - **默认关闭**：OCR 有额外推理开销，只在显式 --ocr 时启用；
  - **稀疏执行**：只对 Tier2 关键帧逐张跑 OCR（约 1-2s 一张），
    绝不进逐帧实时路径（与 CLIP 增强同一条红线）；
  - **懒加载**：rapidocr_onnxruntime 在构造时才 import——不开 --ocr 的
    环境零依赖（装了本包不装 rapidocr 也能全量跑通测试）；
  - **显式报错**：开启 --ocr 但 rapidocr 未安装时抛 RuntimeError 并给出
    安装指引，绝不静默降级成"无文字事件"（避免用户误以为已生效）。

事件结构（与 subtitle 事件同构，进 consumer / aligned_output）：
    [{"t": 12.3, "type": "ocr", "text": "场地入侵告警", "conf": 0.93}, ...]
"""

# rapidocr_onnxruntime 包名（extras: pip install -e ".[ocr]"）
_OCR_PACKAGE = "rapidocr-onnxruntime"


class OcrChannel:
    """rapidocr-onnxruntime 封装：关键帧 -> 文字事件列表。

    用法:
        ocr = OcrChannel()                      # 缺包时这里抛 RuntimeError
        events = ocr.process(frame_bgr, 12.3)   # [{"t","type":"ocr","text","conf"}]
    """

    def __init__(self):
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as e:
            raise RuntimeError(
                f"启用 OCR 通道需要安装 {_OCR_PACKAGE}: "
                f"pip install -e \".[ocr]\"（或 pip install {_OCR_PACKAGE}）"
            ) from e
        # RapidOCR 自带检测+方向分类+识别三模型，默认配置即可用
        self._engine = RapidOCR()

    def process(self, frame_bgr, timestamp):
        """对单帧跑 OCR，返回按置信度过滤后的文字事件列表。

        frame_bgr: BGR ndarray（来自实时管线的关键帧）
        timestamp: 该帧时间戳（秒），直通进事件的 "t"
        """
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return []
        result, _elapse = self._engine(frame_bgr)
        events = []
        if result:
            for item in result:
                # rapidocr 条目: [box(4点坐标), text, score]
                box, text, score = item[0], item[1], item[2]
                text = str(text).strip()
                if not text:
                    continue
                events.append({
                    "t": round(float(timestamp), 3),
                    "type": "ocr",
                    "text": text,
                    "conf": round(float(score), 4),
                })
        return events
