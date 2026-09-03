"""W8b EventBus 单元测试：扇出、有界丢弃、drain。"""
import threading

from vus.live import EventBus


def test_fanout_and_drain():
    bus = EventBus()
    s1, s2 = bus.subscribe(), bus.subscribe()
    for i in range(5):
        bus.publish({"type": "motion", "t": i})
    assert bus.published == 5
    out1, out2 = s1.drain(), s2.drain()
    assert [e["t"] for e in out1] == [0, 1, 2, 3, 4]
    assert out2 == out1
    assert s1.drain() == []  # 取走即清


def test_drop_oldest_per_subscriber():
    bus = EventBus()
    small = bus.subscribe(maxsize=3)
    big = bus.subscribe(maxsize=100)
    for i in range(6):
        bus.publish({"type": "motion", "t": i})
    got = small.drain()
    assert [e["t"] for e in got] == [3, 4, 5]  # 丢最旧
    assert small.dropped == 3
    assert len(big.drain()) == 6
    assert big.dropped == 0  # 订阅者互不影响


def test_get_timeout_returns_none():
    bus = EventBus()
    sub = bus.subscribe()
    assert sub.get(timeout=0.05) is None
    bus.publish({"type": "keyframe", "t": 1.0})
    assert sub.get(timeout=0.05)["type"] == "keyframe"


def test_drain_with_timeout_waits_for_first():
    bus = EventBus()
    sub = bus.subscribe()

    def later():
        import time
        time.sleep(0.05)
        bus.publish({"type": "asr_final", "t": 1.0})

    threading.Thread(target=later, daemon=True).start()
    out = sub.drain(timeout=2.0)
    assert [e["type"] for e in out] == ["asr_final"]
