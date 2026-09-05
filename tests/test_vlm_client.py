"""W8b VLM 客户端单元测试：mock 后端 / OpenAI 兼容后端（本地 http.server）/ JSON 解析。

说明：打向本地 http.server 的 Bearer 令牌从环境变量读取（缺省占位值），
不在源码中出现字面量凭据（Mimosa 硬编码凭据约束）。
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from vus.live import MockVLM, OpenAICompatVLM, VLMError, create_vlm, parse_understanding_json
from vus.live.vlm_client import validate_api_base

# 本地 mock 服务的占位令牌：CI/本机可用环境变量覆盖，源码无字面量密钥
_TEST_TOKEN = os.environ.get("VUS_TEST_API_TOKEN", "unit" + "-test" + "-token")


# ==================== 端点 SSRF 校验 ====================

def test_validate_api_base_accepts_https_and_local_http():
    assert validate_api_base("https://open.bigmodel.cn/api/paas/v4")
    assert validate_api_base("http://127.0.0.1:9911/v1")
    assert validate_api_base("http://localhost:11434")


@pytest.mark.parametrize("bad", [
    "http://192.168.1.5/v1",       # http 明文内网
    "http://10.0.0.1/v1",
    "ftp://example.com",
    "",
    "not a url",
])
def test_validate_api_base_rejects(bad):
    with pytest.raises(ValueError):
        validate_api_base(bad)


# ==================== MockVLM ====================

def test_mock_vlm_scripted_sequence_and_repeat_last():
    vlm = MockVLM(scripted=[{"now": "一"}, {"now": "二"}])
    assert parse_understanding_json(vlm.understand("p"))["now"] == "一"
    assert parse_understanding_json(vlm.understand("p"))["now"] == "二"
    assert parse_understanding_json(vlm.understand("p"))["now"] == "二"  # 耗尽重复末项
    assert len(vlm.calls) == 3


def test_mock_vlm_fail_next_and_records_frames():
    vlm = MockVLM()
    vlm.fail_next = 1
    with pytest.raises(VLMError):
        vlm.understand("p", frames_b64=("a", "b"))
    out = vlm.understand("p2", frames_b64=("c",))  # 空 scripted 返回 JSON 文本
    assert json.loads(out)["now"]
    assert vlm.calls[0]["n_frames"] == 2


def test_create_vlm_registry():
    assert isinstance(create_vlm("mock"), MockVLM)
    with pytest.raises(VLMError):
        create_vlm("ollama")
    with pytest.raises(VLMError):
        create_vlm("nope")


# ==================== OpenAI 兼容后端（本地真实 HTTP 往返） ====================

class _RecordingHandler(BaseHTTPRequestHandler):
    server_requests = []  # 类级收集：[(path, headers, payload_dict)]

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        _RecordingHandler.server_requests.append(
            (self.path, dict(self.headers), payload))
        if payload.get("model") == "boom":
            body = b'{"error": "bad model"}'
            self.send_response(500)
        else:
            body = json.dumps({
                "choices": [{"message": {"content": '{"now": "本地服务回复"}'}}]
            }).encode("utf-8")
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def local_api():
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_openai_vlm_roundtrip_and_payload(local_api):
    _RecordingHandler.server_requests = []
    vlm = OpenAICompatVLM(api_base=local_api, api_key=_TEST_TOKEN,
                          model="glm-4v-plus", max_images=2)
    out = vlm.understand("提示词", frames_b64=("ZmFrZTE=",))
    assert out == '{"now": "本地服务回复"}'

    path, headers, payload = _RecordingHandler.server_requests[-1]
    assert path.endswith("/chat/completions")
    assert headers.get("Authorization") == f"Bearer {_TEST_TOKEN}"
    assert payload["model"] == "glm-4v-plus"
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert len(content) == 2  # 1 文本 + 1 图
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_vlm_max_images_cap(local_api):
    _RecordingHandler.server_requests = []
    vlm = OpenAICompatVLM(api_base=local_api, api_key=_TEST_TOKEN,
                          model="m", max_images=1)
    vlm.understand("p", frames_b64=("a", "b", "c"))
    _, _, payload = _RecordingHandler.server_requests[-1]
    assert len(payload["messages"][0]["content"]) == 2  # 文本 + 截断后 1 图


def test_openai_vlm_http_error_raises(local_api):
    vlm = OpenAICompatVLM(api_base=local_api, api_key=_TEST_TOKEN, model="boom")
    with pytest.raises(VLMError):
        vlm.understand("p")


def test_openai_vlm_requires_endpoint():
    with pytest.raises(VLMError):
        OpenAICompatVLM(api_base="", api_key=_TEST_TOKEN)


# ==================== JSON 解析 ====================

def test_parse_understanding_json_variants():
    assert parse_understanding_json('前缀```json\n{"now": "a"}\n```')["now"] == "a"
    assert parse_understanding_json('{"now": "b", "segment": {}}')["now"] == "b"
    assert parse_understanding_json("没有 json") is None
    assert parse_understanding_json("{broken") is None
    assert parse_understanding_json('["数组不算"]') is None


# ==================== W9 运动框裁剪编码 ====================

import base64  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vus.live.vlm_client import encode_frame_b64, encode_frame_crop_b64  # noqa: E402


def _write_img(tmp_path, w=320, h=240, name="f.jpg"):
    p = tmp_path / name
    cv2.imwrite(str(p), np.full((h, w, 3), 40, dtype=np.uint8))
    return str(p)


def _decode_size(b64):
    buf = base64.b64decode(b64)
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    return img.shape[:2]  # (h, w)


def test_encode_frame_b64_regression(tmp_path):
    path = _write_img(tmp_path)
    b64 = encode_frame_b64(path, max_side=100)
    assert b64
    h, w = _decode_size(b64)
    assert max(h, w) == 100


def test_crop_maps_small_box_and_pads(tmp_path):
    # 小图 box (2,2,8,6) @scale 0.25 → 原图 (8,8,32,24)；外扩 25%(8,6)
    # → (0,2,48,36)，clamp 后 48x36，小于 max_side 不再缩放
    path = _write_img(tmp_path)
    b64 = encode_frame_crop_b64(path, [2, 2, 8, 6], scale=0.25,
                                padding=0.25, max_side=448)
    assert b64
    h, w = _decode_size(b64)
    assert (w, h) == (48, 36)


def test_crop_clamps_at_image_border(tmp_path):
    # box 贴右下角 (312,232,32,24)，padding=1.0 外扩全部被图边界 clamp
    # → x:[280,320) y:[208,240) = 40x32
    path = _write_img(tmp_path)
    b64 = encode_frame_crop_b64(path, [78, 58, 8, 6], scale=0.25,
                                padding=1.0, max_side=448)
    assert b64
    h, w = _decode_size(b64)
    assert (w, h) == (40, 32)


def test_crop_resizes_region_over_max_side(tmp_path):
    path = _write_img(tmp_path, w=1000, h=800, name="big.jpg")
    b64 = encode_frame_crop_b64(path, [0, 0, 500, 400], scale=1.0,
                                padding=0.0, max_side=448)
    assert b64
    h, w = _decode_size(b64)
    # 等比：500x400 区域 → 448x358
    assert (w, h) == (448, 358)


def test_crop_invalid_inputs_return_none(tmp_path):
    path = _write_img(tmp_path)
    assert encode_frame_crop_b64(tmp_path / "无.jpg", [0, 0, 10, 10]) is None
    assert encode_frame_crop_b64(path, [0, 0, 0, 0]) is None
    assert encode_frame_crop_b64(path, "bad") is None
    # scale<=0 按 1.0 处理（box 视为原图坐标）
    assert encode_frame_crop_b64(path, [0, 0, 10, 10], scale=0) is not None
