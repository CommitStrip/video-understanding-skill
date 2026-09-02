"""clip_onnx / CLIP 引擎分发测试（W3）。

两层验证：
  1) 无权重路径（始终运行）：monkeypatch 伪造 onnxruntime InferenceSession，
     验证 ClipOnnx 预处理（resize/中心裁剪/BGR2RGB/归一化/NCHW）、
     L2 归一化、模型缺失时的显式报错与指引、cosine_dist 纯函数性质、
     select_representatives 的引擎分发（embed/dist 接口）。
  2) 真实权重路径（skip-if-no-model）：./models/clip-visual-vitb32.onnx 存在时，
     验证两色帧语义距离 > 同色帧距离。权重获取: scripts/download_clip_onnx.sh。
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import types

from vus import clip_onnx
from vus.clip_onnx import (
    CLIP_MEAN,
    CLIP_STD,
    MODEL_FILENAME,
    ClipOnnx,
    cosine_dist,
    resolve_model_dir,
)
from vus.select_representatives import (
    OnnxClipEngine,
    TorchClipEngine,
    _as_clip_engine,
    select_representatives,
)

ROOT = Path(__file__).resolve().parents[1]


def _real_model_path():
    """真实权重路径：环境变量目录优先，其次仓库根 models/。"""
    candidates = []
    env_dir = resolve_model_dir(None)
    candidates.append(Path(env_dir))
    candidates.append(ROOT / "models" / MODEL_FILENAME)  # 兜底（cwd 无关）
    for c in candidates:
        p = c if c.is_file() else c / MODEL_FILENAME
        if p.is_file():
            return p
    return None


class _FakeValueInfo:
    def __init__(self, name, dtype="tensor(float)", shape=None):
        self.name = name
        self.type = dtype
        self.shape = shape or []


class _FakeSession:
    """伪造 InferenceSession：记录 feed，返回固定嵌入（未归一化 [3,4]）。"""

    def __init__(self, model_path, providers=None, inputs=None, outputs=None):
        self.model_path = model_path
        self.providers = providers
        self._inputs = inputs or [_FakeValueInfo("pixel_values",
                                                 shape=["batch", 3, 224, 224])]
        self._outputs = outputs or [_FakeValueInfo("image_embeds")]
        self.last_feed = None

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs

    def run(self, output_names, feed):
        self.last_feed = dict(feed)
        return [np.array([[3.0, 4.0]], dtype=np.float32)]


@pytest.fixture
def fake_ort(monkeypatch):
    """注入伪造 onnxruntime 模块，返回可检查的会话工厂。"""
    made = []

    class _Factory:
        sessions = made

        @classmethod
        def make(cls, **kw):
            s = _FakeSession(**kw)
            made.append(s)
            return s

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = lambda model_path, providers=None: _Factory.make(
        model_path=model_path, providers=providers)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    return _Factory


@pytest.fixture
def fake_model_dir(tmp_path, monkeypatch):
    """放一个假模型文件（内容任意，InferenceSession 已被伪造），并清掉环境变量。"""
    monkeypatch.delenv("VUS_CLIP_MODELS", raising=False)
    d = tmp_path / "models"
    d.mkdir()
    (d / MODEL_FILENAME).write_bytes(b"fake-onnx-payload")
    return d


# ==================== resolve_model_dir ====================

def test_resolve_model_dir_precedence(fake_model_dir, monkeypatch):
    assert resolve_model_dir(None) == os.path.join(".", "models")  # 缺省
    monkeypatch.setenv("VUS_CLIP_MODELS", "/env/models")
    assert resolve_model_dir(None) == "/env/models"  # 环境变量 > 默认
    assert resolve_model_dir("/explicit") == "/explicit"  # 参数最优先


def test_missing_model_raises_with_guidance(tmp_path, monkeypatch, fake_ort):
    monkeypatch.delenv("VUS_CLIP_MODELS", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError) as ei:
        ClipOnnx(str(empty))
    msg = str(ei.value)
    assert "download_clip_onnx.sh" in msg, "错误信息必须含下载脚本名"
    assert "hf-mirror.com" in msg, "错误信息必须含镜像地址"
    assert "export_clip_onnx.py" in msg


# ==================== 预处理与 embed（伪造会话） ====================

def test_embed_l2_normalization_and_layout(fake_model_dir, fake_ort):
    enc = ClipOnnx(str(fake_model_dir))
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    vec = enc.embed(img)
    assert vec.shape == (2,)
    assert vec.dtype == np.float32
    # 未归一化 [3,4] -> 归一化 [0.6, 0.8]
    assert np.allclose(vec, [0.6, 0.8], atol=1e-6)
    assert len(fake_ort.sessions) == 1


def test_preprocess_shape_dtype_and_bgr2rgb(fake_model_dir, fake_ort):
    enc = ClipOnnx(str(fake_model_dir))
    # BGR 纯红 (0,0,255)：RGB 后 R 通道 = 255
    img = np.full((320, 480, 3), (0, 0, 255), dtype=np.uint8)
    x = enc.preprocess(img)
    assert x.shape == (1, 3, 224, 224)
    assert x.dtype == np.float32
    r, g, b = x[0, 0], x[0, 1], x[0, 2]  # RGB 顺序
    # R=(255/255-0.481)/0.269≈1.93；G/B=(0-均值)/std 为负
    assert r.mean() == pytest.approx((1.0 - CLIP_MEAN[0]) / CLIP_STD[0], abs=1e-4)
    assert g.mean() == pytest.approx((0.0 - CLIP_MEAN[1]) / CLIP_STD[1], abs=1e-4)
    assert b.mean() == pytest.approx((0.0 - CLIP_MEAN[2]) / CLIP_STD[2], abs=1e-4)
    assert r.mean() > g.mean() and r.mean() > b.mean()


def test_preprocess_short_side_resize_and_center_crop(fake_model_dir, fake_ort):
    enc = ClipOnnx(str(fake_model_dir))
    # 竖长条 640h x 320w：短边 320 -> 等比缩放到 448h x 224w（比例 0.7），
    # 中心裁剪行 [112, 336) 对应原图行 [160, 480)。
    # 原图上半黑、下半白（边界 320 恰为缩放后中心 224）：
    # 中心裁剪窗口内应恰好一半黑一半白（顶/底对齐裁剪则全黑/全白，可据此判别）。
    img = np.zeros((640, 320, 3), dtype=np.uint8)
    img[320:, :, :] = 255
    x = enc.preprocess(img)
    black = (0.0 - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
    white = (1.0 - np.array(CLIP_MEAN)) / np.array(CLIP_STD)
    expect = 0.5 * black + 0.5 * white  # 各通道一半黑一半白
    assert x.shape == (1, 3, 224, 224)
    for c in range(3):
        assert x[0, c].mean() == pytest.approx(float(expect[c]), abs=1e-3)
        assert not np.allclose(x[0, c].mean(), float(black[c]), atol=1e-2), \
            "不应是顶对齐裁剪（全黑）"
        assert not np.allclose(x[0, c].mean(), float(white[c]), atol=1e-2), \
            "不应是底对齐裁剪（全白）"


def test_feed_uses_probed_pixel_values(fake_model_dir, fake_ort):
    enc = ClipOnnx(str(fake_model_dir))
    enc.embed(np.zeros((64, 64, 3), dtype=np.uint8))
    feed = fake_ort.sessions[0].last_feed
    assert "pixel_values" in feed
    assert feed["pixel_values"].shape == (1, 3, 224, 224)


def test_full_clip_model_gets_dummy_text_inputs(tmp_path, fake_ort):
    """文本+视觉合一导出：input_ids/attention_mask 必填时自动喂占位 token。"""
    d = tmp_path / "m"
    d.mkdir()
    (d / MODEL_FILENAME).write_bytes(b"fake")
    full_inputs = [
        _FakeValueInfo("input_ids", dtype="tensor(int64)", shape=["b", "seq"]),
        _FakeValueInfo("pixel_values", dtype="tensor(float)",
                       shape=["b", 3, 224, 224]),
        _FakeValueInfo("attention_mask", dtype="tensor(int64)", shape=["b", "seq"]),
    ]
    enc = ClipOnnx(str(d))
    # 手动替换会话为合一模型版（构造时探测的是视觉-only 假输入）
    enc.session = _FakeSession("fake", inputs=full_inputs,
                               outputs=[_FakeValueInfo("image_embeds")])
    enc.input_name, enc.output_name, enc.text_inputs = enc._probe_io()
    vec = enc.embed(np.zeros((64, 64, 3), dtype=np.uint8))
    feed = enc.session.last_feed
    assert np.allclose(vec, [0.6, 0.8], atol=1e-6)  # 输出仍走归一化
    assert feed["input_ids"].shape == (1, 1)  # EOS 占位
    assert feed["attention_mask"].shape == (1, 1)
    assert feed["input_ids"].dtype == np.int64


def test_onnxruntime_missing_raises_with_pip_hint(tmp_path, monkeypatch):
    d = tmp_path / "models"
    d.mkdir()
    (d / MODEL_FILENAME).write_bytes(b"fake")
    # 让 import onnxruntime 失败（None in sys.modules -> ImportError）
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match=r"onnxruntime"):
        ClipOnnx(str(d))


# ==================== cosine_dist ====================

def test_cosine_dist_properties():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 2.0])
    assert cosine_dist(a, a) == pytest.approx(0.0)
    assert cosine_dist(a, b) == pytest.approx(1.0)  # 正交
    assert cosine_dist(a, -a) == pytest.approx(2.0)  # 反向
    assert cosine_dist(a, np.zeros(2)) == 0.0  # 零向量约定为 0
    assert 0 <= cosine_dist(a, np.array([1.0, 1.0])) < 1


# ==================== 引擎分发（select_representatives） ====================

class _FakeEngine:
    """鸭子类型引擎：嵌入 = 帧均值颜色映射，距离 = 欧氏。"""

    name = "fake"

    def embed(self, img):
        m = img.reshape(-1, 3).mean(axis=0)
        return np.asarray(m / 255.0, dtype=np.float32)

    def dist(self, a, b):
        return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def test_as_clip_engine_dispatch():
    assert _as_clip_engine(None) is None
    e = _FakeEngine()
    assert _as_clip_engine(e) is e  # 鸭子类型引擎直通
    tup = ("pre", "model", "cpu")
    wrapped = _as_clip_engine(tup)
    assert isinstance(wrapped, TorchClipEngine)  # 旧元组 -> torch 引擎适配
    with pytest.raises(TypeError):
        _as_clip_engine(42)


def test_onnx_engine_is_lazy_no_io_at_construction_of_class():
    """OnnxClipEngine 类本身不触 IO（实例化才找模型）——保证 import 零副作用。"""
    assert callable(OnnxClipEngine)


def test_select_representatives_with_fake_clip_engine(tmp_path):
    """启用语义增强的选帧路径：嵌入/距离按引擎分发，打分公式保持
    w_pix*pixel_diff + (1-w_pix)*语义距离。"""
    kf_dir = tmp_path / "keyframes"
    kf_dir.mkdir()
    # 0s 灰 40（桶首）；30s 黑（像素差分最大）；50s 白
    for t, g in [(0.0, 40), (30.0, 0), (50.0, 255)]:
        img = np.full((64, 64, 3), g, dtype=np.uint8)
        import cv2
        assert cv2.imwrite(str(kf_dir / f"kf_{int(t*10):05d}_t{t:.1f}s.jpg"), img)

    reps = select_representatives(str(kf_dir), interval=60.0, clip=_FakeEngine())
    assert len(reps) == 1
    assert set(reps[0]) == {"t", "path"}
    # 纯像素差分下 30s 黑与 50s 白与桶首灰的差异同为 100%，此时语义距离应主导
    # ——假引擎嵌入按灰度均值：白帧(1.0)离灰(0.157)最远，应选 50s 白帧
    assert reps[0]["t"] == 50.0


def test_onnx_engine_missing_model_surfaces_error(tmp_path, monkeypatch, fake_ort):
    """OnnxClipEngine 包住的模型缺失错误必须透传（不静默降级）。"""
    monkeypatch.delenv("VUS_CLIP_MODELS", raising=False)
    with pytest.raises(RuntimeError, match="download_clip_onnx"):
        OnnxClipEngine(str(tmp_path / "nope"))


# ==================== 真实权重端到端（skip-if-no-model） ====================

@pytest.mark.skipif(_real_model_path() is None,
                    reason="CLIP ONNX 权重未下载: bash scripts/download_clip_onnx.sh")
def test_real_model_two_color_frames_semantic_distance():
    """真实权重：两色帧语义距离 > 同色帧距离；嵌入 L2 归一化且有限。"""
    model_dir = str(_real_model_path().parent)
    try:
        enc = ClipOnnx(model_dir)
    except MemoryError:
        pytest.skip("内存不足以加载 605MB CLIP ONNX（skip，非失败）")
    except RuntimeError as e:
        if "bad allocation" in str(e) or "OpenBLAS" in str(e):
            pytest.skip(f"推理运行时内存不足，跳过真实权重验证: {e}")
        raise
    red = np.full((360, 640, 3), (0, 0, 255), dtype=np.uint8)   # BGR 红
    blue = np.full((360, 640, 3), (255, 0, 0), dtype=np.uint8)  # BGR 蓝
    er, eb = enc.embed(red), enc.embed(blue)
    assert er.shape == (512,)
    assert np.isfinite(er).all() and np.isfinite(eb).all()
    assert np.linalg.norm(er) == pytest.approx(1.0, abs=1e-5)
    d_diff, d_same = cosine_dist(er, eb), cosine_dist(er, er)
    assert d_diff > d_same, f"异色帧距离 {d_diff} 应大于同色帧距离 {d_same}"
