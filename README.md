# vus — Video Understanding Skill

[![CI](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml/badge.svg)](https://github.com/CommitStrip/video-understanding-skill/actions/workflows/ci.yml)

**English** | [简体中文](README-CN.md)

Turn a video into **structured, LLM-ready understanding artifacts**: semantic
representative frames + timeline-aligned ASR subtitles + motion segments.
Compress 30 fps raw video (hundreds of thousands of frames) into a few dozen
frames a multimodal model can actually read — without missing what matters.

Real-world benchmark — a 120-minute 1080p25 live course:
**147.7 fps processing (5.9× realtime)**, 41 keyframes, 3,505 segments of real
Mandarin subtitles (~33k characters), stable memory footprint.

## Key features

- **Realtime budget allocation** — a fast system (per-frame motion gating,
  ~3% of budget) triggers a slow system (low-frequency keyframing + triggered
  heavy work). Runs live on robots/edge devices, not just file replay.
- **Live sources** — video file, camera, or RTSP stream through one
  `FrameSource` interface; RTSP gets latest-frame backpressure, automatic
  reconnection and monotonic timestamps.
- **Three-tier compression** — raw frames → shot-level keyframes → semantic
  representative frames. Solves "shot changes ≠ content changes".
- **Drift-resilient keyframing** — gradual content evolution (e.g. slide
  annotations building up in a lecture) is captured via drift confirmation,
  not just hard scene cuts.
- **Real ASR** — streaming bilingual (zh/en) speech recognition via
  sherpa-onnx, with word-level timestamps. Degrades loudly, never silently,
  when the model is absent.
- **Optional semantic enhancement** — CLIP (ONNX, no PyTorch) for semantic
  frame selection; OCR channel for slide text.
- **Installable & tested** — `pip install -e .`, 96 pytest cases, GitHub
  Actions CI.

## Installation

```bash
pip install -e .                 # core (opencv-python + numpy)

# optional extras
pip install -e ".[asr]"          # sherpa-onnx streaming ASR (real subtitles)
pip install -e ".[clip]"         # CLIP semantic frame selection (ONNX)
pip install -e ".[ocr]"          # OCR channel
```

Models are not bundled. **Real ASR is the default**: on first run the missing
model is downloaded automatically from the official sherpa-onnx release
(~490 MB, once). To opt out set `VUS_ASR_AUTO_DOWNLOAD=0`, or fetch manually:

```bash
# ASR model (Xiaomi sherpa-onnx streaming zipformer, zh+en, ~490 MB) — auto by default
mkdir -p models/sherpa
curl -L -o - https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2 \
  | tar -xj -C models/sherpa --strip-components=1

# CLIP visual encoder (ONNX, ~600 MB) — optional, via download script
bash scripts/download_clip_onnx.sh
```

Model directories can be overridden with `VUS_SHERPA_MODELS` /
`VUS_CLIP_MODELS`.

> ⚠️ Without sherpa-onnx the subtitle channel falls back to **mock output —
> placeholder text, not real transcription**. Never present mock subtitles as
> real content.

## Quick start

```bash
# 1. Extract structured artifacts (keyframes + motion segments + subtitles)
python -m vus.integrated_pipeline --video lecture.mp4 --output out/ --kf-hz 1.5

# live stream variant
python -m vus.integrated_pipeline --source rtsp --url rtsp://host/stream --output out/

# with OCR (runs on Tier-3 representative frames only — stays out of the
# frame loop; ASR output is auto-cleaned: loop-collapse + dedup + hallucination flags)
python -m vus.integrated_pipeline --video lecture.mp4 --output out/ --ocr

# 2. Compress to semantic representative frames (Tier 3)
#    --max-reps: budget-oriented adaptive selection (recommended for LLM use)
python -m vus.select_representatives --keyframes out/keyframes \
  --max-reps 60 --out representatives.json --report context.md

# multi-speaker roundtables? keep per-bucket diversity:
python -m vus.select_representatives --keyframes out/keyframes \
  --interval 60 --k 3 --out representatives.json

# 3. Feed representative frames + context.md + aligned_output.json to a
#    multimodal LLM for understanding / report generation
```

Reading `representatives.json` and the aligned subtitles is all a multimodal
model needs to produce a lecture notes / plot summary / scene report.

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
confirmation** for content that evolves slowly (slide annotations, camera
pan), which perceptual hashes are blind to.

## Performance

### Real world (120-min 1080p25 live course, 180k frames)

| Metric | Result |
|--------|--------|
| Processing rate | 147.7 fps (**5.9× realtime**) |
| Keyframes | 41 (35 gradual-drift + 5 scene-change), full coverage 0→7150 s |
| ASR | 3,505 segments, ~33k chars, RTF 0.08 (runs in parallel) |
| Memory | stable, ~225 MB after ASR model release |

### Synthetic realtime rates (low-spec 2-core Windows)

| Spec | Rate | Realtime factor |
|------|------|-----------------|
| 720p50 | 247 fps | 4.9× |
| 1080p30 | 78 fps | 2.6× |

### vs. claude-real-video (crv)

Controlled comparison on 4 synthetic 12 s clips (details and repro in
`bench/`): on the `static` clip crv missed the end-of-video change entirely
(0% coverage) while vus captured it with 2 frames; on slow/hue ramps vus
reached the same coverage with half the frames; end-to-end ~3.5× faster.

> Note: the coverage metric in that table is pixel-difference-defined — the
> same signal vus selects frames with, so it structurally favors vus (the
> `static` result stands on its own). A semantic-level evaluation protocol
> (annotation guide + coverage/redundancy metrics) lives in
> `bench/semantic_eval/`.

## Repository layout

```
vus/                       installable core (pip install -e .)
  smart_pipeline.py        fast/slow dual-system vision chain
  integrated_pipeline.py   three-channel orchestration (vision + ASR + align)
  asr_sherpa.py            streaming ASR (sherpa-onnx / explicit fallback)
  select_representatives.py Tier-3 semantic frame selection (--k/--adaptive/--clip)
  source.py                FileSource / CameraSource / RTSPSource
  clip_onnx.py             CLIP ViT-B/32 via onnxruntime (no torch)
  ocr_channel.py           optional OCR channel
  io_utils.py, pathsafe.py safe output writing (traversal-guarded)
scripts/                   legacy entry points (thin shims, still work)
bench/                     crv comparison + semantic evaluation protocol
tests/                     96 pytest cases + end-to-end smoke
```

## Use as an agent skill

This repository is a ready-to-drop agent skill: copy it into your agent's
skills directory (e.g. `~/.agents/skills/video-understanding-skill/`) and the
bundled `SKILL.md` teaches the agent when and how to run the pipeline —
including model setup and the mock-subtitle pitfall. No installation
required: the legacy `scripts/` entries resolve paths on their own.

## Hardware

Measured on a 2-core / 4 GB box: realtime vision chain ~1.2 cores + 166 MB
RSS; Tier-3 offline selection ~317 MB (bounded memory); streaming ASR adds
300–500 MB while decoding. 512 MB RAM is enough for the vision chain alone;
2 GB recommended with ASR.

## Acknowledgments

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — streaming ASR engine,
  maintained by Xiaomi (k2-fsa). This repo only wraps it in
  `vus/asr_sherpa.py`; the bundled model
  `sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20` follows the
  upstream Apache-2.0 license and its own model terms. Any redistribution or
  commercial use of the engine or models must comply with upstream terms.
- [openai/CLIP](https://github.com/openai/CLIP) ViT-B/32 — semantic encoder
  (ONNX export).
- [claude-real-video](https://github.com/davecap/claude-real-video) —
  comparison baseline in `bench/`.

## License

MIT © 2026
