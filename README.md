# vus — Video Understanding Skill

[![CI](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![GitHub tag](https://img.shields.io/github/v/tag/CommitStrip/video-understanding-skill)](https://github.com/CommitStrip/video-understanding-skill/tags)

**English** | [简体中文](README-CN.md)

Turn a video into **structured, LLM-ready understanding artifacts**: semantic
representative frames + timeline-aligned ASR subtitles + motion segments.
Compress 30 fps raw video (hundreds of thousands of frames) into a few dozen
frames a multimodal model can actually read — without missing what matters.
The same architecture also runs **live**: point it at an RTSP stream or a
camera and the LLM understands while watching, with lag that stays bounded.

**Stress-tested** on a 45.5-min 1080p30 concert recording (1.15 GB,
81,878 frames): **4.7× realtime end-to-end**, 736 MB peak memory, **0 dropped
events out of 59,556**, A-grade Simplified-Chinese transcription, and a
41-frame LLM export at ≈13k tokens. Full data in
[Benchmarks](#benchmarks).

## Key features

- **Realtime budget allocation** — a fast system (per-frame motion gating,
  ~3% of budget) triggers a slow system (low-frequency keyframing + triggered
  heavy work). Runs live on robots/edge devices, not just file replay.
- **Live understanding** — `vus.live`: a four-layer stack with millisecond
  local tagging + trigger-based rolling VLM understanding (capped-cost knobs)
  + an SSE state service, built for robot realtime vision.
- **Live sources** — video file, camera, or RTSP stream through one
  `FrameSource` interface; RTSP gets latest-frame backpressure, automatic
  reconnection and monotonic timestamps.
- **Three-tier compression** — raw frames → shot-level keyframes → semantic
  representative frames. Solves "shot changes ≠ content changes".
- **Drift-resilient keyframing** — gradual content evolution (e.g. slide
  annotations building up in a lecture) is captured via drift confirmation,
  not just hard scene cuts.
- **Bilingual ASR, two lanes** — file transcription defaults to an offline
  SenseVoice int8 model (full-context decoding, A-grade readability, robust
  to background music); live audio uses the streaming zipformer lane with
  word-level timestamps. Degrades loudly, never silently, when a model is
  absent.
- **LLM-friendly export** — representative frames auto-downscaled to 640 px,
  3×3 contact sheets, and token estimation up front. Know the context cost
  before you run.
- **Optional semantic enhancement** — CLIP (ONNX, no PyTorch) for semantic
  frame selection; OCR channel for burned-in text.
- **Installable & tested** — `pip install -e .`, 200+ pytest cases, GitHub
  Actions CI.

## Installation

```bash
pip install -e .                 # core (opencv-python + numpy)

# optional extras
pip install -e ".[asr]"          # sherpa-onnx ASR (real subtitles)
pip install -e ".[clip]"         # CLIP semantic frame selection (ONNX)
pip install -e ".[ocr]"          # OCR channel
```

Models are not bundled; on first use the required model is downloaded
automatically from the official sherpa-onnx release:

| Lane | Model | Size | When |
|------|-------|------|------|
| File transcription (default) | SenseVoice int8 (zh+en, offline) | ~166 MB | first pipeline run |
| Live audio (RTSP/camera) | streaming zipformer (zh+en) | ~490 MB | first live run |
| CLIP semantic selection (optional) | ViT-B/32 ONNX | ~600 MB | `--clip` / download script |

To opt out set `VUS_ASR_AUTO_DOWNLOAD=0` and fetch manually (see
`vus/model_setup.py` for the exact URLs). Model directories can be overridden
with `VUS_SHERPA_MODELS` / `VUS_OFFLINE_ASR_MODELS` / `VUS_CLIP_MODELS`.

> ⚠️ Without a downloaded model the subtitle channel falls back to **mock
> output — placeholder text, not real transcription**. Never present mock
> subtitles as real content.

## Quick start

```bash
# 1. Extract structured artifacts (keyframes + motion segments + subtitles)
#    File transcription runs the offline SenseVoice lane by default.
python -m vus.integrated_pipeline --video lecture.mp4 --output out/ --kf-hz 1.5

# live stream variant
python -m vus.integrated_pipeline --source rtsp --url rtsp://host/stream --output out/

# with OCR (runs on Tier-3 representative frames only — stays out of the
# frame loop; ASR output is auto-cleaned: loop-collapse + dedup + hallucination flags)
python -m vus.integrated_pipeline --video lecture.mp4 --output out/ --ocr

# 2. Compress to semantic representative frames (Tier 3) and export the
#    LLM pack: 640 px frames + 3×3 contact sheets + token estimate
python -m vus.select_representatives --keyframes out/keyframes \
  --max-reps 60 --llm-export out/llm --out representatives.json --report context.md

# multi-speaker roundtables? keep per-bucket diversity:
python -m vus.select_representatives --keyframes out/keyframes \
  --interval 60 --k 3 --out representatives.json

# 3. Feed out/llm/ images + context.md + aligned_output.json to a
#    multimodal LLM for understanding / report generation
```

## Live understanding (v0.4)

The offline pipeline answers "compress a video for an LLM to read"; `vus.live`
answers "a stream is coming in — understand it as it happens". Four layers,
each running at its own physical limit:

| Layer | Output | Latency | Cost |
|-------|--------|---------|------|
| T0 frame reflex | motion events + boxes | 0 ms (~1.6 ms/frame) | zero (fast system) |
| T0.5 semantic tags | face/motion-intensity tags per keyframe | ms-level | zero (local, no model download) |
| T2 rolling understanding | current summary / timeline / entities | bounded lag (VLM latency + trigger interval) | per call; trigger-based with a floor interval |

> Millisecond-scale semantics come from T0+T0.5. Rich understanding is bounded
> by VLM inference latency — the architectural guarantee is that **lag stays
> bounded and never grows**: while a VLM call is in flight, material only
> accumulates (single-flight coalescing), and the next call takes the merged
> latest window.

```bash
# file-as-live (default dev/acceptance path; mock backend costs nothing)
python -m vus.live --video lecture.mp4 --realtime --vlm mock --serve

# RTSP live + real VLM (OpenAI-compatible env: VLM_API_BASE / VLM_API_KEY / VLM_MODEL)
python -m vus.live --source rtsp --url rtsp://host/stream --vlm openai --serve

# pure-local free mode (T0+T0.5 only, zero API cost)
python -m vus.live --video x.mp4 --realtime --vlm off --serve
```

Cost knobs:

- **Trigger-based calls** — only scene changes / long motion segments closing /
  new speech fire a call; quiet scenes cost nothing;
- **Floor interval** `--min-call-interval` (default 8 s) — worst-case spend =
  duration ÷ interval × per-call cost;
- **Slim calls** — the latest 1–2 keyframes at 448 px + incremental speech
  text + compacted motion stats;
- `--vlm off` calls nothing at all.

Three consumption channels (usable together):

- **Rolling files** — `live_state.json` (machine-readable) +
  `live_context.md` (human/agent-readable), atomically written; any agent can
  read the current understanding at any moment (mirrors the offline SKILL flow);
- **SSE service** — with `--serve`: `GET /state` snapshot, `GET /events`
  incremental stream, `GET /healthz` liveness — the subscription entry point
  for robots and monitoring dashboards;
- **Console** — periodic summary, per-layer lag and call telemetry.

Long-session anti-bloat: when the understanding timeline overflows, the oldest
entries merge into "previous chapters" (plain text, zero VLM cost); speech and
tag rings are bounded, so memory does not grow with duration.

## How it works

| Tier | Content | Scale | Purpose |
|------|---------|-------|---------|
| 0 | raw frames | 30 fps (10⁵ frames) | playback |
| 1 | fast-system motion events | per frame | "is anything happening" |
| 2 | shot-level keyframes | 1–2 s apart | timeline anchoring |
| 3 | **semantic representative frames** | 30–60 s apart | **LLM understanding** |

The fast system runs every frame on a downscaled grayscale image (frame
difference + semantic gate, ~3% of budget). The slow system samples at low
frequency and only when the fast system reports content, scoring candidates
with pixel difference, pHash and histogram — plus **gradual-drift
confirmation** (hard window + sustained soft lane) for content that evolves
slowly (slide annotations, camera pan), which perceptual hashes are blind to.

## Benchmarks

All numbers below are measured, single-machine (CPU-only), with repro scripts
in `bench/`.

### Stress test — 45.5-min 1080p30 concert recording (81,878 frames, 1.15 GB)

| Metric | vus v0.4 | claude-real-video (baseline) |
|--------|----------|------------------------------|
| Analysis load | **81,878 frames, every frame analyzed** | ~1,515 sampled frames (1 per 1.8 s) |
| End-to-end time | **581.7 s (9.7 min)** | ≈19 min |
| Realtime factor | **4.7×** | ≈2.4× |
| Peak memory | **736 MB**, flat curve | not instrumented |
| Dropped events | **0 / 59,556** | — |
| Keyframe density | 2,197 (48.3/min) | 60 (2.0/min) |
| ASR readability | **A-grade** Simplified Chinese (SenseVoice int8, offline) | C-grade Traditional Chinese, multiple homophone errors (whisper base) |
| LLM export | 41 frames @ 640 px ≈ **13k tokens** | 60 frames @ 640 px |

vus analyzed **54× the per-frame load** of the baseline and still finished in
about half the wall-clock time.

### Real world (120-min 1080p25 live course, 180k frames)

| Metric | Result |
|--------|--------|
| Processing rate | 147.7 fps (**5.9× realtime**, streaming-ASR era) |
| Keyframes | 41 (35 gradual-drift + 5 scene-change), full coverage 0→7150 s |
| ASR | 3,505 segments, ~33k chars, RTF 0.08 (runs in parallel) |
| Memory | stable, ~225 MB after ASR model release |

### Synthetic realtime rates (low-spec 2-core Windows)

| Spec | Rate | Realtime factor |
|------|------|-----------------|
| 720p50 | 247 fps | 4.9× |
| 1080p30 | 78 fps | 2.6× |

### vs. claude-real-video (crv) — synthetic clips

Controlled comparison on 4 synthetic 12 s clips (details and repro in
`bench/`): on the `static` clip crv missed the end-of-video change entirely
(0% coverage) while vus captured it with 2 frames; on slow/hue ramps vus
reached the same coverage with half the frames; under the semantic-level
protocol in `bench/semantic_eval/`, vus redundancy was **1.0 (4 frames / 4
scenes)** vs crv's 12.0 — a 12× LLM-context saving at equal coverage.

> Honesty notes: the pixel-coverage metric is defined by the same signal vus
> selects frames with (the `static` result stands on its own); the stress test
> above covers one content domain (concert) on one machine. Repro scripts and
> the semantic evaluation protocol live in `bench/`.

## Repository layout

```
vus/                       installable core (pip install -e .)
  smart_pipeline.py        fast/slow dual-system vision chain
  integrated_pipeline.py   four-channel orchestration (vision + ASR + OCR + align)
  asr_sherpa.py            two ASR lanes: offline SenseVoice (files, default) +
                           streaming zipformer (live), shared cleaning
  asr_clean.py             ASR output cleaning (loop-collapse + dedup + hallucination flags)
  select_representatives.py Tier-3 semantic frame selection (--k/--adaptive/--clip/--max-reps)
  llm_export.py            LLM-friendly export (640 px downscale + contact sheets + token estimate)
  source.py                FileSource / CameraSource / RTSPSource
  clip_onnx.py             CLIP ViT-B/32 via onnxruntime (no torch)
  ocr_channel.py           optional OCR channel
  reconcile.py             ASR/OCR cross-modal hint annotation
  model_setup.py           model auto-download (official-source allowlist)
  io_utils.py, pathsafe.py safe output writing (traversal-guarded)
  live/                    live understanding layer (v0.4)
    pipeline.py            four-layer orchestrator (python -m vus.live)
    understanding.py       trigger-based VLM worker (coalescing / compaction / backoff)
    state.py               SessionState + rolling atomic persistence
    server.py              SSE state service (/state /events /healthz)
    tagger.py              T0.5 ms-level tagging lane
    vlm_client.py          VLM backend registry (openai/mock)
    audio_source.py        live audio chain (ffmpeg PCM → fixed blocks)
    events.py              bounded EventBus
    rolling_align.py       streaming aligner (incremental twin of batch align)
scripts/                   legacy entry points (thin shims, still work)
bench/                     crv comparison, real-video evidence reports, semantic
                           evaluation protocol
tests/                     pytest cases + end-to-end smoke (incl. file-as-live)
```

## Use as an agent skill

This repository is a ready-to-drop agent skill: copy it into your agent's
skills directory (e.g. `~/.agents/skills/video-understanding-skill/`) and the
bundled `SKILL.md` teaches the agent when and how to run the pipeline —
including model setup and the mock-subtitle pitfall. No installation
required: the legacy `scripts/` entries resolve paths on their own.

## Hardware

Measured on a 2-core / 4 GB box: realtime vision chain ~1.2 cores + 166 MB
RSS; Tier-3 offline selection ~317 MB (bounded memory); ASR adds 300–500 MB
while decoding. 512 MB RAM is enough for the vision chain alone; 2 GB
recommended with ASR. The 45.5-min stress test peaked at 736 MB including
models.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — ASR engine maintained
  by Xiaomi (k2-fsa). This repo only wraps it in `vus/asr_sherpa.py` /
  `vus/model_setup.py`; models used here
  (`sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09`,
  `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20`) follow the
  upstream Apache-2.0 license and their own model terms. Any redistribution or
  commercial use of the engine or models must comply with upstream terms.
  SenseVoice is a model from the FunAudioLLM / Alibaba speech team.
- [openai/CLIP](https://github.com/openai/CLIP) ViT-B/32 — semantic encoder
  (ONNX export).
- [claude-real-video](https://github.com/HUANGCHIHHUNGLeo/claude-real-video) —
  comparison baseline in `bench/`.

## License

MIT © 2026
