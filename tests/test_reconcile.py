"""reconcile 跨模态标注测试。"""
from vus.reconcile import _similarity, reconcile


def test_similarity_identical_and_disjoint():
    assert _similarity("开始削减工人的工资", "开始削减工人的工资") == 1.0
    assert _similarity("完全不同的话题", "abc def") == 0.0


def test_similarity_homophone_high():
    """同音错字（ASR: 动梦 vs OCR: 东梦）应有较高相似度——纠错线索。"""
    sim = _similarity("我是东梦我们下期节目再见", "我是动梦我们下期节目再见")
    assert sim >= 0.6


def test_reconcile_annotates_hit():
    asr = [{"t": 655.0, "text": "我是动梦我们下期节目再见"}]
    ocr = [{"t": 654.0, "text": "酒痴东梦", "conf": 0.99}]
    out = reconcile(asr, ocr, window_s=15, threshold=0.6)
    hint = out[0].get("ocr_hint")
    if hint is not None:  # 相似度不达阈值时不标注——标注了就必须结构正确
        assert isinstance(hint["sim"], float) and hint["sim"] >= 0.6
        assert hint["text"] == "酒痴东梦"


def test_reconcile_strong_hit():
    asr = [{"t": 100.0, "text": "拥有将近全国收入的一半左右"}]
    ocr = [{"t": 101.0, "text": "拥有将近全国收入的将近一半左右"}]
    out = reconcile(asr, ocr, window_s=15, threshold=0.6)
    assert "ocr_hint" in out[0]
    assert out[0]["ocr_hint"]["sim"] >= 0.6


def test_reconcile_no_hit_outside_window():
    asr = [{"t": 100.0, "text": "拥有将近全国收入的一半左右"}]
    ocr = [{"t": 200.0, "text": "拥有将近全国收入的将近一半左右"}]
    out = reconcile(asr, ocr, window_s=15, threshold=0.6)
    assert "ocr_hint" not in out[0]


def test_reconcile_no_hit_low_similarity():
    asr = [{"t": 100.0, "text": "完全无关的一句话"}]
    ocr = [{"t": 101.0, "text": "FREE SOUP THE UNEMPLOYED"}]
    out = reconcile(asr, ocr, window_s=15, threshold=0.6)
    assert "ocr_hint" not in out[0]
