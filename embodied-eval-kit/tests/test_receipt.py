"""Receipts: three states, worst-state roll-up, and a refuse-to-lie helper."""

from __future__ import annotations

import json

import pytest

from eeval.receipt import STATUS, Receipt, Stage, exit_code


def test_worst_state_wins():
    receipt = Receipt().add("a", "ok", x=1).add("b", "warn", y=2)
    assert receipt.worst() == "warn"
    receipt.add("c", "fail", z=3)
    assert receipt.worst() == "fail"
    assert [stage.name for stage in receipt.failed()] == ["c"]
    assert [stage.name for stage in receipt.warned()] == ["b"]


def test_exit_codes_follow_the_shell_convention():
    assert exit_code(Receipt().add("a", "ok")) == 0
    assert exit_code(Receipt().add("a", "warn")) == 1
    assert exit_code(Receipt().add("a", "fail")) == 2
    assert exit_code(Receipt()) == 0


def test_invalid_status_is_rejected_at_construction():
    with pytest.raises(ValueError, match="status must be one of"):
        Stage(name="a", status="green")
    assert STATUS == ("ok", "warn", "fail")


def test_require_ok_refuses_to_continue_quietly():
    Receipt().add("a", "ok").require_ok()
    with pytest.raises(RuntimeError, match="warn"):
        Receipt().add("a", "ok").add("b", "warn", reason="box moved 6.85mm").require_ok()


def test_json_and_markdown_round_trip():
    receipt = Receipt().add("collect", "ok", episodes=28).add("train", "warn", steps=5000)
    payload = json.loads(receipt.to_json())
    assert payload["overall"] == "warn"
    assert payload["stages"][0]["facts"] == {"episodes": 28}
    restored = Receipt.from_dict(payload)
    assert restored.worst() == "warn"
    assert restored.stages[0].facts["episodes"] == 28
    markdown = receipt.to_markdown()
    assert "**warn**" in markdown and "collect" in markdown and "overall" in markdown


def test_notes_survive_and_render():
    receipt = Receipt().add("eval", "warn", note="2 rollouts never landed")
    assert json.loads(receipt.to_json())["stages"][0]["note"] == "2 rollouts never landed"
    assert "2 rollouts never landed" in receipt.to_markdown()


def test_merge_keeps_the_worst_of_both():
    first = Receipt().add("a", "ok")
    second = Receipt().add("b", "fail")
    assert first.merge(second).worst() == "fail"
    assert len(first.extend([Stage("c", "ok")]).stages) == 3