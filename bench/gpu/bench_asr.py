# -*- coding: utf-8 -*-
"""离线 ASR（SenseVoice int8）转写基准：CPU vs CUDA。
用法: python bench/gpu/bench_asr.py <audio.wav> [--out-dir DIR]
输出: <out-dir>/gpu_asr_bench.json（含两路 RTF 与转写文本一致性对照）
"""
import argparse
import json
import os
import time

from vus.asr_sherpa import load_offline_recognizer, load_wav, transcribe_offline
from vus.io_utils import write_json


def run(provider, samples, sr):
    t_load0 = time.perf_counter()
    rec = load_offline_recognizer(provider=provider)
    load_s = time.perf_counter() - t_load0
    rec.create_stream()  # 触发会话初始化（CUDA 首次含 kernel 编译/显存分配）
    segs = transcribe_offline(rec, samples[: sr * 5], sr)  # 预热 5s
    t0 = time.perf_counter()
    segs = transcribe_offline(rec, samples, sr)
    decode_s = time.perf_counter() - t0
    audio_s = len(samples) / sr
    text = "".join(s.get("text", "") for s in segs)
    return {
        "provider": provider,
        "load_s": round(load_s, 2),
        "decode_s": round(decode_s, 2),
        "audio_s": round(audio_s, 1),
        "rtf": round(decode_s / audio_s, 4),
        "segments": len(segs),
        "chars": len(text),
        "text_head": text[:60],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--out-dir", default="bench/gpu")
    args = ap.parse_args()

    samples, sr = load_wav(args.wav)
    print(f"audio: {args.wav} ({len(samples)/sr:.1f}s)")

    results = []
    for provider in ("cpu", "cuda"):
        try:
            r = run(provider, samples, sr)
        except Exception as e:
            r = {"provider": provider, "error": str(e)[:200]}
        results.append(r)
        print(json.dumps(r, ensure_ascii=False))

    write_json(args.out_dir, "gpu_asr_bench.json", {
        "wav": args.wav,
        "results": results,
    })
    print(f"saved: {args.out_dir}/gpu_asr_bench.json")


if __name__ == "__main__":
    main()
