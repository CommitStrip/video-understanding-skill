import os, sys, time, json
from pathlib import Path
import numpy as np

# 相对定位仓库根：本文件位于 <repo>/bench/，仓库根为 parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vus.smart_pipeline import SmartPipeline
from vus.select_representatives import load_keyframes, select_representatives
from vus.io_utils import write_json

BENCH = os.path.dirname(os.path.abspath(__file__))
TESTS = ['aba', 'slow', 'hue', 'static']

def _find_frames_json(outdir, result=None):
    """crv 产物 frames.json 的可能位置：结果对象属性优先，其次 outdir 递归搜索。"""
    fj = getattr(result, "frames_json_path", None) if result is not None else None
    if fj and os.path.exists(fj):
        yield fj
    if outdir and os.path.isdir(outdir):
        for root, _dirs, files in os.walk(outdir):
            if "frames.json" in files:
                yield os.path.join(root, "frames.json")


def _parse_frames_json(data):
    """解析 crv frames.json，兼容两种格式：

    - 旧假设：裸列表 [{"t": ...}, ...] 或 [秒数, ...]
    - crv 0.10+ 实际：{"frames": [{"timestamp_sec": ...}, ...]}
    返回时间戳列表。
    """
    frames = data
    if isinstance(data, dict):
        frames = data.get("frames", [])
    tts = []
    if not isinstance(frames, list):
        return tts
    for d in frames:
        if isinstance(d, dict):
            v = d.get("timestamp_sec", d.get("t", d.get("timestamp", d.get("time", 0))))
            tts.append(v)
        elif isinstance(d, (int, float)):
            tts.append(d)
    return [t for t in tts if isinstance(t, (int, float))]


def run_crv(video):
    """用 claude-real-video 处理，返回 (elapsed, n_frames, timestamps)"""
    from claude_real_video import process
    outdir = os.path.join(BENCH, f'crv_out_{os.path.basename(video)[:-4]}')
    t0 = time.time()
    r = process(video, outdir, do_transcribe=False)
    elapsed = time.time() - t0
    # 统计帧：兼容 crv 0.10+ 的 {"frames":[...]}+timestamp_sec 格式
    tts = []
    for fj in _find_frames_json(outdir, r):
        try:
            with open(fj) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        tts = _parse_frames_json(data)
        if tts:
            break
    n = len(tts)
    return elapsed, n, tts

def run_ours(video):
    """用我们的管线处理，返回 (pipeline_elapsed, pipe, reps, rep_ts)"""
    from vus.integrated_pipeline import run_realtime_pipeline
    outdir = os.path.join(BENCH, f'our_out_{os.path.basename(video)[:-4]}')
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    pipe, aligned, asr = run_realtime_pipeline(video, outdir, config={'keyframe_interval_hz': 1.5})
    p_elapsed = time.time() - t0
    # 代表帧
    kfdir = os.path.join(outdir, 'keyframes')
    t1 = time.time()
    reps = select_representatives(kfdir, interval=2.0)
    r_elapsed = time.time() - t1
    rep_ts = [r['t'] for r in reps]
    return p_elapsed, pipe, len(pipe.keyframes), rep_ts, r_elapsed

results = []
for name in TESTS:
    video = f'{BENCH}/{name}.mp4'
    print(f'\n===== {name} =====')
    row = {'test': name}

    # crv
    try:
        e, n, tts = run_crv(video)
        row['crv_s'] = round(e, 2)
        row['crv_frames'] = n
        row['crv_ts'] = [round(t,1) for t in tts]
        print(f'  crv: {e:.2f}s, {n} frames')
    except Exception as ex:
        row['crv_s'] = f'ERR:{ex}'
        print(f'  crv ERROR: {ex}')

    # ours
    try:
        e, pipe, kf_n, rep_ts, re = run_ours(video)
        # 总帧数
        row['our_s'] = round(e, 2)
        row['our_keyframes'] = kf_n
        row['our_reps'] = rep_ts
        row['our_fps'] = pipe.frame_count / e if e > 0 else 0
        row['our_repcalc_s'] = round(re, 2)
        print(f'  ours: {e:.2f}s ({row["our_fps"]:.0f}fps), {kf_n} keyframes -> {len(rep_ts)} reps')
    except Exception as ex:
        row['our_s'] = f'ERR:{ex}'
        print(f'  ours ERROR: {ex}')

    results.append(row)

print('\n\n===== 汇总 =====')
for r in results:
    print(json.dumps(r, ensure_ascii=False))

write_json(BENCH, 'bench_results.json', results)
print(f'\n已保存: {BENCH}/bench_results.json')