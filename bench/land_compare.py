#!/usr/bin/env python3
"""
land_compare.py - 落地对比：视频内容理解 Skill vs claude-real-video
===================================================================
修正上一轮的缺陷（crv 帧解析用了不存在的 r.frames_json_path 导致统计为 0），
直接读取双方已有产物做正确统计，并新增"内容变化覆盖度"这一准确性指标。

准确性度量的核心思路：
  一个选帧方法若准确，应把视频中"画面变化较大"的区间都覆盖到。
  定义：对每个测试视频，把时间轴切成 1s 的桶，桶内取首尾两帧的像素差分作为
  "该秒的内容变化量"。任一方法选出的帧若落在这个桶内，则视为该秒被覆盖。
  覆盖率 = 内容变化量 > 阈值 的秒数中，被方法覆盖的占比。

依赖：必须已有 bench 产物（gen_bench.py + 双方管线已跑过）。
"""
import os, sys, json, glob
import numpy as np
import cv2

BENCH = os.path.dirname(os.path.abspath(__file__))
TESTS = ['aba', 'slow', 'hue', 'static']
DIFF_SIZE = 64
CHANGE_THRESHOLD = 3.0  # 像素差分百分比，超过视为"该秒有内容变化"


def pixel_diff(img_a, img_b, size=DIFF_SIZE):
    small = lambda im: cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    a, b = small(img_a).astype(np.float32), small(img_b).astype(np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0 * 100.0)


def truth_profile(video):
    """把视频切成 1s 桶，返回 [{t, change}], change=桶首尾像素差分。"""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = n / fps if fps > 0 else 0
    prof = []
    # 每整秒采样 1 帧
    for st in range(0, int(dur)):
        idx = int(st * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok1, f1 = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(idx + int(fps) - 1, n - 1))
        ok2, f2 = cap.read()
        if ok1 and ok2:
            prof.append({'t': st, 'change': pixel_diff(f1, f2)})
    cap.release()
    return prof


def crv_frames(name):
    """从 crv 产物读帧时间戳（正确的解析）。"""
    p = f'{BENCH}/crv_out_{name}/frames.json'
    if not os.path.exists(p):
        return []
    with open(p) as f:
        data = json.load(f)
    frames = data.get('frames', [])
    return [float(fr['timestamp_sec']) for fr in frames]


def our_keyframes(name):
    """从我们产物读关键帧时间戳。"""
    p = f'{BENCH}/our_out_{name}/pipeline_results.json'
    if not os.path.exists(p):
        return []
    with open(p) as f:
        d = json.load(f)
    return [float(k['t']) for k in d.get('keyframes', [])]


def coverage(selected_ts, profile):
    """计算覆盖率：有内容变化的秒中，被选中帧覆盖的比例。"""
    changed = [s for s in profile if s['change'] > CHANGE_THRESHOLD]
    if not changed:
        return 1.0, 0, 0  # 无变化秒数，视为完全覆盖
    sel = sorted(selected_ts)
    covered = 0
    for s in changed:
        # 选中帧落在该秒 或 该秒前后相邻桶也被覆盖
        if any(abs(t - s['t']) <= 1.0 for t in sel):
            covered += 1
    return covered / len(changed), covered, len(changed)


def main():
    rows = []
    print('='*84)
    print('落地对比：视频内容理解 Skill vs claude-real-video')
    print('='*84)
    print('测试视频: 640x360@30fps, 12s/360帧')
    print(f'准确性指标: 每秒内容变化(像素差分)>{CHANGE_THRESHOLD}% 视为有变化, 覆盖率=被选中帧覆盖的有变化秒数占比')
    print()

    for name in TESTS:
        video = f'{BENCH}/{name}.mp4'
        prof = truth_profile(video)
        crv_ts = crv_frames(name)
        our_ts = our_keyframes(name)

        c_cov, c_covn, c_chg = coverage(crv_ts, prof)
        o_cov, o_covn, o_chg = coverage(our_ts, prof)

        # 已保存的耗时（来自上一轮 run_bench.json）
        br = json.load(open(f'{BENCH}/bench_results.json'))
        row = [r for r in br if r['test'] == name][0]
        crv_s = row.get('crv_s'); our_s = row.get('our_s')

        # 内容变化最多的秒（用于诊断）
        top = sorted(prof, key=lambda s: -s['change'])[:3]
        top_s = [f"{s['t']}s(Δ{s['change']:.1f}%)" for s in top]

        print(f'--- {name} ---')
        print(f'  内容变化(top3秒): {", ".join(top_s)}')
        print(f'  [crv ] 选帧={len(crv_ts):2d}  覆盖率={c_cov*100:5.1f}% ({c_covn}/{c_chg}有变秒)  耗时~{crv_s}s')
        print(f'  [ours] 选帧={len(our_ts):2d}  覆盖率={o_cov*100:5.1f}% ({o_covn}/{o_chg}有变秒)  耗时~{our_s}s')
        rows.append({
            'test': name, 'changes_sec': c_chg,
            'crv_n': len(crv_ts), 'crv_cov': round(c_cov, 3), 'crv_s': crv_s,
            'our_n': len(our_ts), 'our_cov': round(o_cov, 3), 'our_s': our_s,
        })

    print('\n' + '='*84)
    print('汇总表')
    print('='*84)
    hdr = f"{'测试':<7}{'有变秒':<7}{'crv帧':<7}{'crv覆盖率':<10}{'crv耗时':<9}{'our帧':<7}{'our覆盖率':<10}{'our耗时':<9}"
    print(hdr); print('-'*len(hdr))
    for r in rows:
        def fs(v):
            return f"{v}s" if isinstance(v, (int, float)) else str(v)
        print(f"{r['test']:<7}{r['changes_sec']:<7}{r['crv_n']:<7}{r['crv_cov']*100:<9.1f}%{fs(r['crv_s']):<9}{r['our_n']:<7}{r['our_cov']*100:<9.1f}%{fs(r['our_s']):<9}")

    with open(f'{BENCH}/land_compare_results.json', 'w') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f'\n已保存: {BENCH}/land_compare_results.json')


if __name__ == '__main__':
    main()