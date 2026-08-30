import json

from gemmafischer.cli import main


def test_dev_doctor_needs_no_engine(capsys) -> None:
    assert main(["doctor", "--profile", "dev", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert [item["code"] for item in payload["checks"]] == ["PYTHON_VERSION"]


def test_version_contract(capsys) -> None:
    assert main(["version", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"application": "0.2.0", "api": "v1", "evidence_schema": "2.0"}
