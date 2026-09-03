"""vus - Video Understanding Skill 核心库

可安装的视频理解核心：实时预算分配画面链（快/慢双系统）+ 流式 ASR 声音链
+ 时间轴对齐融合 + Tier3 语义代表帧选择。

用法:
    from vus import SmartPipeline, run_realtime_pipeline, select_representatives
    from vus import FileSource, CameraSource, RTSPSource  # W2 实时源

    # 模块方式运行
    python -m vus.integrated_pipeline --video x.mp4 --output out/
    python -m vus.integrated_pipeline --source rtsp --url rtsp://host/stream
"""

__version__ = "0.3.0"

from .smart_pipeline import SmartPipeline
from .integrated_pipeline import run_realtime_pipeline
from .select_representatives import select_representatives
from .source import FrameSource, FileSource, CameraSource, RTSPSource

__all__ = [
    "__version__",
    "SmartPipeline",
    "run_realtime_pipeline",
    "select_representatives",
    "FrameSource",
    "FileSource",
    "CameraSource",
    "RTSPSource",
]
