"""vus.live - 实时理解层（W8 波次）
=================================
在既有"快系统/慢系统/ASR"三层压缩之上，把理解（LLM）拉进实时环路：

  T0   帧级反射    现有 SmartPipeline（快/慢系统），0 滞后，本包不改动
  T0.5 语义标签    tagger.BasicLabeler 毫秒级标签（CLIP 零样本 W9）
  T2   滚动理解    understanding.UnderstandingWorker 触发式 VLM，滞后有界

模块：
  audio_source   直播音频链（ffmpeg 直通 PCM 流 → 定长样本块）
  rolling_align  流式对齐器（批式 align_asr_streaming 的增量版）
  events         线程安全有界 EventBus
  vlm_client     VLM 后端注册表（openai / mock，ollama 槽位预留）
  tagger         T0.5 毫秒级标签道（basic；CLIP 零样本 W9）
  state          SessionState + 滚动落盘（live_state.json / live_context.md）
  understanding  理解 worker（触发策略 / 单飞合并 / 时间线压缩）
  server         SSE 服务（/state /events /healthz）
  pipeline       直播编排器 + CLI（python -m vus.live）
"""

from .audio_source import AudioStream
from .rolling_align import RollingAligner
from .events import EventBus, Subscription
from .state import SessionState, StateWriter
from .tagger import BasicLabeler, LabelerError, create_labeler
from .vlm_client import (
    MockVLM, OpenAICompatVLM, VLMError, create_vlm, parse_understanding_json,
)
from .understanding import UnderstandingWorker
from .server import LiveServer
from .pipeline import build_source, build_vlm, run_live

__all__ = [
    "AudioStream", "RollingAligner", "EventBus", "Subscription",
    "SessionState", "StateWriter", "BasicLabeler", "LabelerError",
    "create_labeler", "MockVLM", "OpenAICompatVLM", "VLMError",
    "create_vlm", "parse_understanding_json", "UnderstandingWorker",
    "LiveServer", "build_source", "build_vlm", "run_live",
]
