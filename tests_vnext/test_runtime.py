import pytest

from gemmafischer.runtime import extract_json_array


def test_extract_json_array_accepts_markdown_fence() -> None:
    output = '<|channel>final\n```json\n[{"kind":"guidance"}]\n```'
    assert extract_json_array(output) == [{"kind": "guidance"}]


def test_extract_json_array_rejects_missing_payload() -> None:
    with pytest.raises(ValueError, match="did not contain"):
        extract_json_array("no structured result")
