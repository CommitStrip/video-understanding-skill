#!/usr/bin/env python3
"""
annotate_vlm.py - 语义场景 GT 半自动标注脚手架（VLM 辅助）
==========================================================
把管线产物（代表帧 + 对齐字幕）整理成给多模态 VLM 的标注材料，
产出 ground_truth.json 草稿。两种用法：

1) 缺省（无 API 环境）：
     python bench/semantic_eval/annotate_vlm.py --reps ... --keyframes ... --aligned ...
   只生成:
     - annotation_manifest.json   待标注清单（帧时间/文件 + 逐帧附近字幕）
     - vlm_prompt.md              给多模态 VLM 的 prompt 模板（含清单表格）
     - ground_truth.json          空草稿（含 schema 注释），人工填或喂给 VLM 后校对

2) 自动调用（OpenAI 兼容 API，环境变量）：
     export VLM_API_BASE=https://.../v1     # 如 https://open.bigmodel.cn/api/paas/v4
     export VLM_API_KEY=sk-...
     export VLM_MODEL=glm-4v-plus           # 可选，默认 glm-4v-plus
     python bench/semantic_eval/annotate_vlm.py ... --api
   把代表帧缩略图（base64）+ 字幕发给模型，解析出场景表写入 ground_truth.json
   （source="vlm-assisted"）。**VLM 输出只是草稿，边界与 desc 必须人工校对**
   （校对完成后把 source 改为 "human"）。

标注协议见同目录 PROTOCOL.md。
"""
import argparse
import base64
import http.client
import json
import os
import re
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vus.io_utils import write_json, write_text  # noqa: E402

DEFAULT_WINDOW_S = 5.0   # 代表帧前后各多少秒内的字幕视为相关
DEFAULT_MAX_IMAGES = 16  # 自动调用时最多随请求附带的代表帧数（控制 payload）


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_rep_items(reps_path):
    """representatives.json -> [{"t": float, "path": str}, ...]（按时间序）。"""
    data = load_json(reps_path)
    items = data["representatives"] if isinstance(data, dict) else data
    return sorted(items, key=lambda x: float(x["t"]))


def load_subtitles(aligned_path):
    """aligned_output.json -> [{"start","end","text"}]（无则退化为 asr 段）。"""
    doc = load_json(aligned_path)
    segs = doc.get("aligned_segments") or []
    if segs:
        return [{"start": float(s["start"]), "end": float(s["end"]),
                 "text": s.get("text", "")} for s in segs if s.get("text")]
    return [{"start": float(s["t"]), "end": float(s.get("end_t", s["t"] + 2.0)),
             "text": s.get("text", "")} for s in doc.get("asr_segments", [])
            if s.get("text")]


def subtitles_near(t, subs, window=DEFAULT_WINDOW_S):
    return [s for s in subs if s["start"] <= t + window and s["end"] >= t - window]


def build_manifest(rep_items, subs):
    """待标注清单：每张代表帧 + 附近字幕。"""
    manifest = []
    for i, it in enumerate(rep_items):
        t = float(it["t"])
        near = subtitles_near(t, subs)
        manifest.append({
            "idx": i,
            "t": round(t, 2),
            "file": Path(it["path"]).name,
            "path": it["path"],
            "nearby_subtitles": [
                {"start": round(s["start"], 2), "end": round(s["end"], 2),
                 "text": s["text"]} for s in near],
        })
    return manifest


PROMPT_TEMPLATE = """# 任务：视频语义场景切分标注

你是视频标注助手。下面是一段视频的按时间序代表帧（Tier3，约每 {interval}s 一张）
及其附近的字幕。请把视频按**语义场景**切段并输出 JSON。

## 切分判据
- 场景切换：说话人 / 机位 / 地点 / 主题 / 画面功能（空镜 vs 讲解 vs 字幕板）变化；
- 不切换：同场景内的镜头运动、光照抖动、纯色底色明暗变化、字幕滚动；
- 每段给一句话中文描述（desc）；
- 边界用代表帧之间的时间中点估计，允许 ±1s 误差（后续人工校对）。

## 输出格式（只输出 JSON，不要多余文字）
{{
  "scenes": [
    {{"start_s": 0.0, "end_s": 12.0, "desc": "..."}},
    ...
  ]
}}

## 待标注材料

{manifest_table}

## 字幕全文（按时间序）

{subtitle_dump}
"""


def build_prompt(manifest, subs, interval):
    rows = ["| idx | 时间(s) | 帧 | 附近字幕 |", "|-----|---------|-----|----------|"]
    for m in manifest:
        sub_text = " / ".join(s["text"] for s in m["nearby_subtitles"]) or "（无）"
        rows.append(f"| {m['idx']} | {m['t']:.1f} | {m['file']} | {sub_text} |")
    sub_dump = "\n".join(
        f"[{s['start']:.1f}-{s['end']:.1f}s] {s['text']}" for s in subs) or "（无字幕）"
    return PROMPT_TEMPLATE.format(interval=interval,
                                  manifest_table="\n".join(rows),
                                  subtitle_dump=sub_dump)


def _b64_jpeg(path, max_side=448):
    """读代表帧并缩略成 JPEG base64（控制请求体大小）。

    用 np.fromfile + cv2.imdecode 读图（兼容中文/unicode 路径）。
    """
    import cv2
    import numpy as np
    buf = np.fromfile(str(path), dtype=np.uint8)  # 只读不写，安全
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))))
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(jpg.tobytes()).decode("ascii") if ok else None


def validate_api_base(api_base):
    """校验 VLM API 端点，防 SSRF（CWE-918）。

    仅允许 https 外网端点，或 http 的 localhost/127.0.0.1/::1（本地推理服务，
    如 Ollama / LMDeploy）。其余（http 明文外网、内网地址、file/ftp 等）拒绝。
    """
    parsed = urllib.parse.urlparse(api_base or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme == "https" and host:
        return api_base
    if parsed.scheme == "http" and host in ("localhost", "127.0.0.1", "::1"):
        return api_base
    raise ValueError(
        f"VLM_API_BASE 仅允许 https 端点或 http://localhost（本地服务）: {api_base!r}")


def _post_json(url, payload_bytes, api_key, timeout=180):
    """POST JSON 到已通过 validate_api_base 校验的端点，返回响应字节。

    用 http.client 直连（host 经 validate_api_base 白名单后才建连）。
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    try:
        conn.request("POST", parsed.path or "/", body=payload_bytes,
                     headers={"Content-Type": "application/json",
                              "Authorization": f"Bearer {api_key}"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status != 200:
            raise RuntimeError(f"VLM API HTTP {resp.status}: {data[:200]!r}")
        return data
    finally:
        conn.close()


def call_vlm(prompt, manifest, api_base, api_key, model):
    """OpenAI 兼容 /chat/completions 调用（stdlib urllib，零新依赖）。

    只带前 DEFAULT_MAX_IMAGES 张代表帧缩略图，避免超长 payload。
    """
    api_base = validate_api_base(api_base)
    content = [{"type": "text", "text": prompt}]
    for m in manifest[:DEFAULT_MAX_IMAGES]:
        b64 = _b64_jpeg(m["path"])
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.2,
    }).encode("utf-8")
    url = urllib.parse.urljoin(api_base, "/chat/completions")
    body = _post_json(url, payload, api_key, timeout=180)
    return json.loads(body.decode("utf-8"))["choices"][0]["message"]["content"]


def parse_scenes_from_response(text):
    """从模型回复中提取第一个 JSON 对象的 scenes 数组（容错 markdown 代码块）。"""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("VLM 回复中未找到 JSON")
    obj = json.loads(m.group(0))
    scenes = obj.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("VLM 回复 JSON 中没有 scenes 数组")
    return [{"start_s": float(s["start_s"]), "end_s": float(s["end_s"]),
             "desc": str(s.get("desc", ""))} for s in scenes]


def main():
    ap = argparse.ArgumentParser(description="语义场景 GT 半自动标注（VLM 辅助）")
    ap.add_argument("--reps", required=True, help="representatives.json")
    ap.add_argument("--keyframes", default=None,
                    help="关键帧目录（校验代表帧存在；不传则跳过校验）")
    ap.add_argument("--aligned", default=None, help="aligned_output.json（字幕来源）")
    ap.add_argument("--out-dir", default=None,
                    help="产物目录（默认 bench/semantic_eval/annotations/）")
    ap.add_argument("--api", action="store_true",
                    help="自动调用 VLM（需环境变量 VLM_API_BASE/VLM_API_KEY）")
    ap.add_argument("--interval", type=float, default=None,
                    help="代表帧分桶间隔（prompt 里说明用；默认从 reps 文件读）")
    args = ap.parse_args()

    rep_items = load_rep_items(args.reps)
    if not rep_items:
        sys.exit(f"[Annotate] {args.reps} 中没有代表帧")
    subs = load_subtitles(args.aligned) if args.aligned else []
    interval = args.interval
    if interval is None:
        doc = load_json(args.reps)
        interval = float(doc.get("interval", 60.0)) if isinstance(doc, dict) else 60.0

    out_dir = Path(args.out_dir or (Path(__file__).resolve().parent / "annotations"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 待标注清单
    manifest = build_manifest(rep_items, subs)
    if args.keyframes:
        kf = Path(args.keyframes)
        for m in manifest:
            if not (kf / m["file"]).exists():
                print(f"[Annotate] 警告: 关键帧缺失 {m['file']}")
    write_json(str(out_dir), "annotation_manifest.json", {
        "count": len(manifest),
        "interval_s": interval,
        "rep_items": manifest,
    })
    print(f"[Annotate] 清单已写: {out_dir / 'annotation_manifest.json'} "
          f"({len(manifest)} 帧, 字幕 {len(subs)} 段)")

    # 2) prompt 模板
    prompt = build_prompt(manifest, subs, interval)
    write_text(str(out_dir), "vlm_prompt.md", prompt)
    print(f"[Annotate] prompt 已写: {out_dir / 'vlm_prompt.md'}")

    # 3) ground_truth.json 草稿
    reps_doc = load_json(args.reps)
    video_name = str(reps_doc.get("video", "")) if isinstance(reps_doc, dict) else ""
    scenes = []
    source = "human"
    if args.api:
        api_base = os.environ.get("VLM_API_BASE", "").strip()
        api_key = os.environ.get("VLM_API_KEY", "").strip()
        model = os.environ.get("VLM_MODEL", "glm-4v-plus").strip()
        if not api_base or not api_key:
            sys.exit("[Annotate] --api 需要环境变量 VLM_API_BASE 与 VLM_API_KEY")
        print(f"[Annotate] 调用 VLM: {api_base} (model={model}) ...")
        content = call_vlm(prompt, manifest, api_base, api_key, model)
        scenes = parse_scenes_from_response(content)
        source = "vlm-assisted"
        print(f"[Annotate] VLM 产出 {len(scenes)} 个场景（草稿，务必人工校对）")
    else:
        print("[Annotate] 未启用 --api：ground_truth.json 为空草稿，"
              "可把 vlm_prompt.md 粘给多模态模型或人工填写")

    duration = max((s["end_s"] for s in scenes), default=0.0)
    draft = {
        "video": video_name,
        "duration_s": duration,
        "source": source,
        "_note": "VLM 输出仅为草稿；人工校对边界与 desc 后把 source 改为 human",
        "scenes": scenes,
    }
    write_json(str(out_dir), "ground_truth.json", draft)
    print(f"[Annotate] GT 草稿已写: {out_dir / 'ground_truth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
