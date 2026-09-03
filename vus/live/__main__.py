#!/usr/bin/env python3
"""python -m vus.live —— 四层实时理解栈 CLI 入口。

用法:
  文件仿真实时（开发/验收默认路径）:
    python -m vus.live --video x.mp4 --realtime --vlm mock
  RTSP 直播实战:
    python -m vus.live --source rtsp --url rtsp://host/stream --vlm openai --serve
  摄像头:
    python -m vus.live --source cam --camera 0 --serve
  纯本地免费模式（只跑 T0 帧反射 + T0.5 毫秒标签，零 API 成本）:
    python -m vus.live --video x.mp4 --realtime --vlm off --serve
"""

import argparse
import sys

from .pipeline import build_source, build_vlm, run_live


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m vus.live", description="实时视频流 → LLM 分层理解（W8）")
    parser.add_argument("--video", default=None, help="视频文件路径（--source file 时必填）")
    parser.add_argument("--output", default=None, help="输出目录（默认当前目录）")
    parser.add_argument("--source", choices=["file", "cam", "rtsp"], default="file",
                        help="帧源类型: file=视频文件(默认) / cam=摄像头 / rtsp=网络流")
    parser.add_argument("--camera", type=int, default=0, help="摄像头索引")
    parser.add_argument("--url", default=None, help="RTSP 地址（--source rtsp 时必填）")
    parser.add_argument("--realtime", action="store_true",
                        help="file 源按视频 fps 节拍实时喂帧（仿真实时验收）")
    parser.add_argument("--fast-scale", type=float, default=0.25, help="快系统降采样比例")
    parser.add_argument("--kf-hz", type=float, default=1.5, help="慢系统关键帧频率(Hz)")
    parser.add_argument("--no-keyframes", action="store_true", help="不保存关键帧图片")
    parser.add_argument("--vlm", choices=["mock", "openai", "off"], default="mock",
                        help="T2 理解后端: mock=回放测试 / openai=OpenAI兼容API(env 配置) /"
                             " off=纯本地免费模式(只跑 T0+T0.5)")
    parser.add_argument("--min-call-interval", type=float, default=8.0,
                        help="T2 触发式调用的地板间隔秒数（费用上限旋钮，默认 8）")
    parser.add_argument("--audio", choices=["auto", "off"], default="auto",
                        help="声音链开关（auto=文件/RTSP 有音轨即启用）")
    parser.add_argument("--serve", action="store_true", help="起 SSE 状态服务")
    parser.add_argument("--port", type=int, default=8600, help="SSE 服务端口（默认 8600）")
    parser.add_argument("--quiet", action="store_true", help="减少控制台输出")
    args = parser.parse_args(argv)

    source = build_source(args.source, video_path=args.video, camera=args.camera,
                          url=args.url, realtime=args.realtime)
    vlm = build_vlm(args.vlm)
    config = {"fast_scale": args.fast_scale, "keyframe_interval_hz": args.kf_hz}
    live_cfg = {"min_call_interval": args.min_call_interval}

    try:
        ret = run_live(source, output_dir=args.output, config=config,
                       live_cfg=live_cfg, vlm=vlm, serve=args.serve,
                       port=args.port, audio=args.audio,
                       save_keyframes=not args.no_keyframes, quiet=args.quiet)
    except KeyboardInterrupt:
        print("\n[Live] 收到中断，已保存部分结果")
        return 130
    return 0 if ret is not None else 1


if __name__ == "__main__":
    sys.exit(main())
