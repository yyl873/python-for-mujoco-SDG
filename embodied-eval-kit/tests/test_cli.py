"""The CLI must work from a bare shell, because that is how people actually try a kit."""

from __future__ import annotations

import json

import pytest

from eeval.cli import main


@pytest.fixture()
def rows_file(tmp_path, real_run):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(real_run), encoding="utf-8")
    return path


def test_report_writes_markdown_json_and_receipt(tmp_path, rows_file):
    md = tmp_path / "report.md"
    js = tmp_path / "report.json"
    rc = main([
        "report", str(rows_file),
        "--tasks", "Put the medicine box into the left wooden box.,"
                   "Put the medicine box into the right wooden box.",
        "--boxes", "target_box,target_box_b",
        "--markdown", str(md), "--json", str(js),
        "--quiet",
    ])
    assert rc == 0
    assert "Verdict" in md.read_text(encoding="utf-8")
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["pairs_both_follow"] == 15


def test_report_accepts_a_protocol_file(tmp_path, rows_file):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "instructions": ["left", "right"],
        "expected_box": {"left": "box_a", "right": "box_b"},
    }), encoding="utf-8")
    rows = tmp_path / "simple.json"
    rows.write_text(json.dumps([
        {"episode_id": 1, "seed": 0, "ix": 0, "landed_box": "box_a"},
        {"episode_id": 1, "seed": 0, "ix": 1, "landed_box": "box_b"},
    ]), encoding="utf-8")
    # one scene is not enough to clear the Wilson lower bound, so the verdict fails and
    # the exit code is 1 (warn) -- see the exit-code table in the CLI docstring
    assert main(["report", str(rows), "--protocol", str(protocol), "--quiet"]) == 1


def test_report_without_a_protocol_is_a_usage_error(rows_file):
    with pytest.raises(SystemExit):
        main(["report", str(rows_file), "--tasks", "only-one", "--boxes", "a,b"])


def test_judge_subcommand(tmp_path):
    episodes = tmp_path / "episodes.jsonl"
    lines = [
        {"lift_mm": [0, 60, 70, 70, 70, 70], "tilt_deg": [0] * 6, "penetration_mm": [0] * 6},
        {"lift_mm": [0, 3, 3, 3, 3, 3], "tilt_deg": [0] * 6, "penetration_mm": [0] * 6},
    ]
    episodes.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
    assert main(["judge", str(episodes)]) == 1  # one of two failed
    assert main(["judge", str(episodes), "--lift", "2"]) == 0  # nothing to fail now


def test_selfcheck_passes(capsys):
    assert main(["selfcheck"]) == 0
    assert "overall" in capsys.readouterr().out


def test_missing_rows_key_is_reported(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"nothing": 1}), encoding="utf-8")
    with pytest.raises(SystemExit):
        main(["report", str(path), "--tasks", "a,b", "--boxes", "x,y"])