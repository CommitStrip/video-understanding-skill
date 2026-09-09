"""W9 v0.7 测试：证据包 + 统一预算调度器。"""
import pytest

from vus.evidence import build_evidence_packs
from vus.scheduler import BudgetScheduler


# ==================== 证据包 ====================

def _kf(t, reason="scene_change"):
    return {"t": t, "reason": reason}


def test_evidence_pack_structure():
    aligned = [{"start": 100.0, "end": 120.0, "text": "老师讲了重点",
                "linked_motion_events": 5, "linked_keyframes": 2}]
    kfs = [_kf(102.0), _kf(110.0), _kf(118.0)]
    packs = build_evidence_packs(aligned, kfs)
    assert len(packs) == 1
    p = packs[0]
    assert p["event_id"] == "ev_0000"
    assert p["time_range"] == [100.0, 120.0]
    assert set(p["selection_reasons"]) >= {"speech", "motion", "keyframe"}
    assert p["confidence"] > 0


def test_evidence_pack_attention_boost():
    aligned = [{"start": 50.0, "end": 60.0, "text": "注意看这里",
                "linked_motion_events": 0, "linked_keyframes": 1}]
    kfs = [_kf(55.0)]
    aw = [{"t": 52.0, "end": 58.0, "cue": "注意看"}]
    packs = build_evidence_packs(aligned, kfs, attention_windows=aw)
    assert "attention" in packs[0]["selection_reasons"]
    assert packs[0]["confidence"] >= 0.5


def test_evidence_pack_no_reason_skipped():
    aligned = [{"start": 100.0, "end": 110.0, "text": "",
                "linked_motion_events": 0, "linked_keyframes": 0}]
    packs = build_evidence_packs(aligned, [])
    assert packs == []


def test_evidence_pack_near_boundary_frames():
    """区间内无关键帧但边界附近有 → 取最近邻。"""
    aligned = [{"start": 100.0, "end": 110.0, "text": "内容",
                "linked_motion_events": 1, "linked_keyframes": 0}]
    kfs = [_kf(95.0), _kf(115.0)]
    packs = build_evidence_packs(aligned, kfs)
    assert packs and len(packs[0]["frames"]) >= 1


# ==================== 统一预算调度器 ====================

def test_scheduler_submit_and_score():
    sch = BudgetScheduler()
    sch.submit("visual", 1.0, timestamp=100.0)
    sch.submit("speech", 1.0, timestamp=101.0)
    score = sch._window_score(now=102.0, window_s=30.0)
    assert score > 0


def test_scheduler_should_retain_threshold():
    sch = BudgetScheduler({"retain_threshold": 1.5})
    # 弱信号 → 不保留
    sch.submit("ocr", 0.3, timestamp=100.0)
    assert not sch.should_retain(now=101.0)
    # 强信号 → 保留
    sch.submit("visual", 2.0, timestamp=101.0)
    assert sch.should_retain(now=102.0)


def test_scheduler_guaranteed_anchor():
    sch = BudgetScheduler({"anchor_interval_s": 30.0})
    assert sch.guaranteed_anchor_due(now=100.0)  # 无历史 → 到期
    assert not sch.guaranteed_anchor_due(now=100.0, last_anchor_t=90.0)
    assert sch.guaranteed_anchor_due(now=135.0, last_anchor_t=100.0)


def test_scheduler_provenance_id_stable():
    sch = BudgetScheduler()
    sch.submit("visual", 1.0, timestamp=100.0)
    id1 = sch.provenance_id(100.0)
    id2 = sch.provenance_id(100.0)
    assert id1 == id2  # 同状态同 ID
    assert len(id1) == 12
