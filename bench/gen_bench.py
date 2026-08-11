import argparse
import os

import cv2
import numpy as np


def video_writer(path, w, h, fps):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建测试视频（mp4v 编码器不可用）: {path}")
    return writer

def gen_aba(path, w=640, h=360, fps=30, dur=12):
    vw = video_writer(path, w, h, fps)
    n = int(fps * dur)
    for i in range(n):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if t < 4:
            frame[:] = (0, 0, 60); cv2.circle(frame, (200, 180), 60, (0, 0, 200), -1)
        elif t < 8:
            frame[:] = (60, 60, 0); cv2.circle(frame, (440, 180), 60, (200, 130, 0), -1)
        else:
            frame[:] = (0, 0, 60); cv2.circle(frame, (200, 180), 60, (0, 0, 200), -1)
        vw.write(frame)
    vw.release()

def gen_slow(path, w=640, h=360, fps=30, dur=12):
    vw = video_writer(path, w, h, fps)
    n = int(fps * dur)
    for i in range(n):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        bg = int(30 + 200 * t / dur)
        frame[:] = (bg, bg, bg)
        cx = int(80 + 480 * t / dur)
        cv2.rectangle(frame, (cx, 120), (cx + 60, 240), (0, 200, 120), -1)
        vw.write(frame)
    vw.release()

def gen_hue(path, w=640, h=360, fps=30, dur=12):
    vw = video_writer(path, w, h, fps)
    n = int(fps * dur)
    # OpenCV 8-bit HSV 的 H 范围是 0-179。
    hsv = np.zeros((180, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.arange(180, dtype=np.uint8)
    hsv[:, 0, 1] = 200
    hsv[:, 0, 2] = 100
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(180, 3).astype(np.uint8)
    x_idx = np.arange(w)
    for i in range(n):
        t = i / fps
        phase = t / dur * 180
        hue_idx = ((phase + x_idx * 0.15) % 180).astype(np.int32)
        colors = bgr[hue_idx]  # (w,3)
        frame = np.repeat(colors[np.newaxis, :, :], h, axis=0)
        vw.write(frame)
    vw.release()

def gen_static(path, w=640, h=360, fps=30, dur=12):
    vw = video_writer(path, w, h, fps)
    n = int(fps * dur)
    for i in range(n):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 50)
        cv2.putText(frame, 'STATIC SLIDE', (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        if t > 11:
            frame[:] = (200, 50, 50)
        vw.write(frame)
    vw.release()

def main():
    parser = argparse.ArgumentParser(description="生成可复现的合成视频基准集")
    parser.add_argument("--output", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()
    output = os.path.abspath(args.output)
    os.makedirs(output, exist_ok=True)

    for name, fn in [('aba', gen_aba), ('slow', gen_slow), ('hue', gen_hue), ('static', gen_static)]:
        path = os.path.join(output, f'{name}.mp4')
        fn(path)
        print(f'{name}: {os.path.getsize(path)} bytes')
    print(f'测试视频生成完毕: {output}')


if __name__ == '__main__':
    main()
