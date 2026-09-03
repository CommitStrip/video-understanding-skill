#!/usr/bin/env python3
"""
vlm_client.py - VLM 后端注册表（W8 实时理解层 T2 摘要道的模型入口）
====================================================================
本期（控成本）实现两种后端：
  openai  OpenAI 兼容 /chat/completions（GLM-4V 等皆可，env 配置端点）；
  mock    脚本化回放——测试与无 key 开发用，行为完全确定。
ollama（本地 VLM）为 W9 预留槽位：本地推理是后续压延迟（T1 图注道）的前提。

OpenAI 调用从 bench/semantic_eval/annotate_vlm.py 的 call_vlm/_post_json/
validate_api_base 移植为可复用客户端（库不反向依赖 bench）。
端点 SSRF 校验（CWE-918）：仅允许 https 外网端点或 http 本地推理服务。
"""

import base64
import http.client
import json
import os
import re
import time
import urllib.parse

DEFAULT_MODEL = "glm-4v-plus"


class VLMError(RuntimeError):
    """VLM 调用失败（网络/鉴权/响应结构），由调用方决定退避与降级。"""


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


def _post_json(url, payload_bytes, api_key, timeout=60):
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
            raise VLMError(f"VLM API HTTP {resp.status}: {data[:200]!r}")
        return data
    except (OSError, http.client.HTTPException) as e:
        raise VLMError(f"VLM API 连接失败: {e}") from e
    finally:
        conn.close()


def encode_frame_b64(path, max_side=448):
    """关键帧图 → 缩略 JPEG base64（控制请求体）。

    np.fromfile + cv2.imdecode 读图（兼容中文/unicode 路径）；读图失败返回 None。
    """
    import cv2
    import numpy as np
    try:
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
    except OSError:
        return None


class BaseVLM:
    """VLM 后端统一接口：understand(prompt, frames_b64) -> 回复文本。"""

    name = "base"

    def understand(self, prompt, frames_b64=(), timeout=60):
        raise NotImplementedError


class OpenAICompatVLM(BaseVLM):
    """OpenAI 兼容 /chat/completions 客户端（glm-4v-plus 默认）。

    端点/密钥/模型支持参数或环境变量：VLM_API_BASE / VLM_API_KEY / VLM_MODEL。
    """

    name = "openai"

    def __init__(self, api_base=None, api_key=None, model=None, max_images=2):
        base = api_base or os.environ.get("VLM_API_BASE", "")
        if not base:
            raise VLMError("缺少 VLM 端点：传 api_base 或设置 VLM_API_BASE")
        self.api_base = validate_api_base(base)
        self.api_key = api_key or os.environ.get("VLM_API_KEY", "")
        self.model = model or os.environ.get("VLM_MODEL", DEFAULT_MODEL)
        self.max_images = int(max_images)

    def understand(self, prompt, frames_b64=(), timeout=60):
        content = [{"type": "text", "text": prompt}]
        for b64 in list(frames_b64)[:self.max_images]:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.2,
        }).encode("utf-8")
        url = urllib.parse.urljoin(self.api_base, "/chat/completions")
        body = _post_json(url, payload, self.api_key, timeout=timeout)
        try:
            return json.loads(body.decode("utf-8"))["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, UnicodeDecodeError) as e:
            raise VLMError(f"VLM API 响应结构异常: {e}") from e


class MockVLM(BaseVLM):
    """脚本化回放后端：按序返回 scripted 内容，耗尽后重复最后一项。

    scripted 项为 dict 时返回其 JSON 文本（模拟结构化输出），str 原样返回。
    calls 记录每次调用的 (prompt, n_frames)，供测试断言触发与合并行为。
    latency 模拟 API 延迟（秒），测试单飞合并时用。
    """

    name = "mock"

    def __init__(self, scripted=None, latency=0.0):
        self.scripted = list(scripted) if scripted else []
        self.latency = float(latency)
        self.calls = []
        self._idx = 0
        self.fail_next = 0  # >0 时接下来 N 次调用抛错（测退避）

    def understand(self, prompt, frames_b64=(), timeout=60):
        self.calls.append({"prompt": prompt, "n_frames": len(frames_b64)})
        if self.latency:
            time.sleep(self.latency)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise VLMError("mock 注入失败")
        if not self.scripted:
            return json.dumps({"now": "（mock 默认回复）画面正常",
                               "segment": {}, "entities": {}},
                              ensure_ascii=False)
        item = self.scripted[min(self._idx, len(self.scripted) - 1)]
        self._idx = min(self._idx + 1, len(self.scripted) - 1)
        return item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)


_BACKENDS = {"openai": OpenAICompatVLM, "mock": MockVLM}


def create_vlm(backend="mock", **kwargs):
    """按名构造 VLM 后端。ollama 为 W9 预留槽位（本地推理压延迟）。"""
    if backend == "ollama":
        raise VLMError("ollama 本地 VLM 后端为 W9 预留，本期请用 openai / mock")
    cls = _BACKENDS.get(backend)
    if cls is None:
        raise VLMError(f"未知 VLM 后端: {backend}（可用: {sorted(_BACKENDS)} / ollama 预留）")
    return cls(**kwargs)


def parse_understanding_json(text):
    """从模型回复提取第一个 JSON 对象（容错 markdown 代码围栏）；失败返回 None。"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None
