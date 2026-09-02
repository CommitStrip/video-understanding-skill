"""pathsafe 单元测试：拒绝 `..` 穿越，放行正常路径。"""
import pytest

from vus.pathsafe import safe_output_path


def test_rejects_dotdot_segments():
    for bad in ["../evil.json", "out/../../../evil", "a/b/../../c",
                "..\\evil.json", "out\\..\\..\\evil"]:
        with pytest.raises(ValueError):
            safe_output_path(bad)


def test_allows_normal_paths():
    assert safe_output_path("out/results.json") == "out/results.json"
    assert safe_output_path("/abs/dir/results.json") == "/abs/dir/results.json"
    assert safe_output_path("results.json") == "results.json"
    # 名字里含两个点但不是 .. 片段，放行
    assert safe_output_path("out/v0..2.json") == "out/v0..2.json"


def test_result_is_string():
    from pathlib import Path
    out = safe_output_path(Path("tmp") / "x.json")
    assert isinstance(out, str)


# ==================== io_utils 落盘写 ====================

def test_write_json_roundtrip(tmp_path):
    from vus.io_utils import write_json
    out = write_json(str(tmp_path), "data.json", {"a": 1, "b": [2, 3]})
    import json
    assert json.loads(open(out, encoding="utf-8").read()) == {"a": 1, "b": [2, 3]}


def test_write_json_rejects_traversal(tmp_path):
    from vus.io_utils import write_json
    with pytest.raises(ValueError):
        write_json(str(tmp_path), "../evil.json", {"x": 1})
    assert not (tmp_path.parent / "evil.json").exists()


def test_write_text(tmp_path):
    from vus.io_utils import write_text
    out = write_text(str(tmp_path), "note.md", "# hello")
    assert open(out, encoding="utf-8").read() == "# hello"
