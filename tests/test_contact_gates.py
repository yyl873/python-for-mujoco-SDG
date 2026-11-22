"""The six contact gates, and the promise that the documentation agrees with them.

`grasp/contact_gates.py` is the executable version of the table in
`docs/PHYSICS_FACTS.md`.  Two things must stay true, so both are tested:

1. the thresholds in the code equal the thresholds in the document (a stale document is
   how a "0.5 mm budget" quietly becomes "whatever the code does");
2. the numbers from the accepted run pass, and the numbers from the rejected designs fail
   -- otherwise the gates are decoration.
"""

from __future__ import annotations

import re

import pytest

from grasp import contact_gates
from grasp.contact_gates import GATES, AuditReport, evaluate, summarise, threshold_of

#: The accepted run, as published in docs/PHYSICS_FACTS.md §2.
ACCEPTED = {
    "descend_contact_steps": 0,
    "press_mm": 0.39,
    "tilt_at_grasp_end_deg": 1.06,
    "rise_mm": 122.22,
    "gap_fixed_mm": -0.85,
    "gap_jaw_mm": -1.07,
    "pen_depth_mm": 0.02,
}

#: The rejected designs, also from PHYSICS_FACTS.md §2 and §6.
REJECTED_SHELL_DESIGN = dict(ACCEPTED, pen_depth_mm=7.5, press_mm=22.6)
REJECTED_OFFSET_GRASP = dict(ACCEPTED, descend_contact_steps=43, rise_mm=4.51)
REJECTED_SLOW_DESCENT = dict(ACCEPTED, tilt_at_grasp_end_deg=90.0, press_mm=26.48)


def test_the_accepted_run_passes_every_gate():
    report = evaluate(ACCEPTED)
    assert report.ok, report.failed
    assert report.failed == ()
    assert len(report.verdicts) == 6


@pytest.mark.parametrize(
    "measurement,value",
    [
        ("descend_contact_steps", 1),
        ("press_mm", 0.51),
        ("tilt_at_grasp_end_deg", 5.01),
        ("rise_mm", 49.99),
        ("gap_fixed_mm", 1.21),
        ("gap_jaw_mm", -1.21),
        ("pen_depth_mm", 1.61),
    ],
)
def test_each_gate_fails_on_its_own(measurement, value):
    report = evaluate(dict(ACCEPTED, **{measurement: value}))
    assert not report.ok
    expected_key = {
        "gap_fixed_mm": "gap_mm",
        "gap_jaw_mm": "gap_mm",
    }.get(measurement, measurement)
    assert report.failed == (expected_key,), report.failed


def test_thresholds_are_inclusive():
    """A measurement sitting exactly on the line passes -- the line is a budget, not a hint."""
    on_the_line = {
        "descend_contact_steps": 0,
        "press_mm": threshold_of("press_mm"),
        "tilt_at_grasp_end_deg": threshold_of("tilt_at_grasp_end_deg"),
        "rise_mm": threshold_of("rise_mm"),
        "gap_fixed_mm": threshold_of("gap_mm"),
        "gap_jaw_mm": -threshold_of("gap_mm"),
        "pen_depth_mm": threshold_of("pen_depth_mm"),
    }
    assert evaluate(on_the_line).ok


def test_the_rejected_designs_fail_the_gate_that_should_catch_them():
    shell = evaluate(REJECTED_SHELL_DESIGN)
    assert not shell.ok
    assert "pen_depth_mm" in shell.failed
    assert "press_mm" in shell.failed

    offset = evaluate(REJECTED_OFFSET_GRASP)
    assert not offset.ok
    assert offset.failed == ("descend_contact_steps", "rise_mm")

    slow = evaluate(REJECTED_SLOW_DESCENT)
    assert not slow.ok
    assert "tilt_at_grasp_end_deg" in slow.failed


def test_g4_is_the_gate_that_stops_not_grasping_at_all():
    """A perfectly clean descent with no lift means the robot never touched the box."""
    never_touched = {
        "descend_contact_steps": 0,
        "press_mm": 0.0,
        "tilt_at_grasp_end_deg": 0.0,
        "rise_mm": 0.0,
        "gap_fixed_mm": -0.9,
        "gap_jaw_mm": -0.9,
        "pen_depth_mm": 0.0,
    }
    report = evaluate(never_touched)
    assert report.failed == ("rise_mm",)


def test_unmeasured_is_a_failure_not_a_skip():
    report = evaluate(dict(ACCEPTED, gap_jaw_mm=None))
    assert not report.ok
    assert report.failed == ("gap_mm",)
    assert "not measured" in report.to_markdown()


def test_a_missing_measurement_is_a_programming_error():
    broken = dict(ACCEPTED)
    broken.pop("pen_depth_mm")
    with pytest.raises(KeyError, match="pen_depth_mm"):
        evaluate(broken)


def test_legacy_flags_keep_their_historical_names():
    flags = evaluate(ACCEPTED).flag_dict()
    assert flags == {
        "gate_descend_clean": True,
        "gate_press": True,
        "gate_tilt": True,
        "gate_lift": True,
        "gate_gap": True,
        "gate_pen": True,
    }
    assert set(flags) == set(contact_gates.LEGACY_FLAGS.values())


def test_report_serialises_and_renders():
    report = evaluate(REJECTED_SHELL_DESIGN)
    payload = report.as_dict()
    assert payload["ok"] is False
    assert set(payload["failed"]) == {"pen_depth_mm", "press_mm"}
    assert len(payload["gates"]) == 6
    markdown = report.to_markdown()
    for gate in GATES:
        assert gate.label in markdown
    assert "overall: **fail**" in markdown


def test_batch_summary_counts_per_gate():
    reports = [evaluate(ACCEPTED), evaluate(REJECTED_SHELL_DESIGN), evaluate(REJECTED_OFFSET_GRASP)]
    summary = summarise(reports)
    assert summary["n"] == 3 and summary["n_ok"] == 1
    assert summary["by_gate"]["pen_depth_mm"] == 1
    assert summary["by_gate"]["descend_contact_steps"] == 1


def test_unknown_gate_key_raises():
    with pytest.raises(KeyError, match="unknown gate"):
        threshold_of("wrist_roll_deg")


def test_every_gate_documents_what_it_protects():
    for gate in GATES:
        assert gate.protects, gate.key
        assert gate.rationale, f"{gate.key} 没有写为什么是这个阈值"
        assert gate.unit is not None


# ---------------------------------------------------------------------------
# documentation must agree with the code
# ---------------------------------------------------------------------------
GATE_ROW = re.compile(r"^\|\s*G(\d)\b")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_documented_thresholds(physics_facts: str) -> dict[str, float]:
    """Read the measured-threshold column of the acceptance table in PHYSICS_FACTS.md.

    G1 is written as ``0 步`` (zero is the requirement itself, with no comparison sign)
    while the rest read ``≤ 0.5 mm``, so the parser takes the first number in the column
    instead of insisting on a comparison operator.
    """
    documented = {}
    for line in physics_facts.splitlines():
        match = GATE_ROW.match(line.strip())
        if not match:
            continue
        criterion = line.strip().strip("|").split("|")[1]
        number = NUMBER.search(criterion)
        if number:
            documented[f"G{match.group(1)}"] = float(number.group())
    return documented


def test_documented_thresholds_match_the_code(repo_root):
    text = (repo_root / "docs" / "PHYSICS_FACTS.md").read_text(encoding="utf-8")
    documented = parse_documented_thresholds(text)
    assert set(documented) == {"G1", "G2", "G3", "G4", "G5", "G6"}, documented
    code = {
        "G1": threshold_of("descend_contact_steps"),
        "G2": threshold_of("press_mm"),
        "G3": threshold_of("tilt_at_grasp_end_deg"),
        "G4": threshold_of("rise_mm"),
        "G5": threshold_of("gap_mm"),
        "G6": threshold_of("pen_depth_mm"),
    }
    for gate, value in code.items():
        assert documented[gate] == pytest.approx(value), (
            f"{gate} 在 docs/PHYSICS_FACTS.md 里是 {documented[gate]}，代码里是 {value}"
        )