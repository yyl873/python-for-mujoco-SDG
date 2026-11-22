"""Clean-success gates: boundaries, the consecutive-run rule, and cheat detection."""

from __future__ import annotations

import pytest

from eeval.judge import CleanCfg, Episode, judge_episode, judge_many


def episode(lift, tilt=None, penetration=None, landed=None, **kwargs) -> Episode:
    return Episode(
        lift_mm=lift,
        tilt_deg=tilt if tilt is not None else [0.0] * len(lift),
        penetration_mm=penetration if penetration is not None else [0.0] * len(lift),
        landed_box=landed,
        **kwargs,
    )


def test_a_clean_lift_passes():
    verdict = judge_episode(episode([0, 10, 60, 90, 95, 96, 95]))
    assert verdict["ok"]
    assert verdict["reasons"] == []
    assert verdict["max_lift_mm"] == 96.0
    assert verdict["consecutive_ok"]


def test_thresholds_are_inclusive():
    """A trace sitting exactly on the threshold is a pass, not a fail."""
    cfg = CleanCfg()
    at_threshold = episode([cfg.lift_min_mm] * 10,
                           tilt=[cfg.tilt_max_deg] * 10,
                           penetration=[cfg.penetration_max_mm] * 10)
    assert judge_episode(at_threshold, cfg)["ok"]
    just_under = episode([cfg.lift_min_mm - 0.001] * 10)
    assert not judge_episode(just_under, cfg)["ok"]


def test_high_lift_with_penetration_is_rejected():
    """The reward hack from the residual-RL round: dig in, then lift."""
    verdict = judge_episode(
        episode([0, 30, 80, 90, 85, 80, 80, 80],
                tilt=[0, 1, 2, 3, 3, 3, 3, 3],
                penetration=[0, 0.2, 1.5, 2.0, 1.8, 1.5, 1.5, 1.4])
    )
    assert not verdict["ok"]
    assert verdict["failed_gates"] == ["max_penetration_mm"]
    assert "max_penetration_mm" in verdict["reasons"][0]
    assert any("digging" in flag for flag in verdict["cheat_flags"])


def test_instant_lift_is_rejected():
    """Up for one sample and back down is not a grasp."""
    verdict = judge_episode(episode([0, 0, 80, 0, 0, 0, 0]), CleanCfg(lift_consecutive=5))
    assert not verdict["ok"]
    assert verdict["failed_gates"] == ["lift_consecutive"]
    assert verdict["max_lift_mm"] == 80.0
    assert any("held" in flag for flag in verdict["cheat_flags"])


def test_tilt_gate_is_independent_of_lift():
    verdict = judge_episode(episode([0, 70, 70, 70, 70, 70], tilt=[0, 1, 6, 6, 6, 6]))
    assert not verdict["ok"]
    assert verdict["failed_gates"] == ["max_tilt_deg"]
    assert any("angle" in flag for flag in verdict["cheat_flags"])


def test_every_failed_gate_is_reported_not_just_the_first():
    verdict = judge_episode(episode([1, 2, 3], tilt=[9, 9, 9], penetration=[3, 3, 3]))
    assert set(verdict["failed_gates"]) == {
        "max_lift_mm", "lift_consecutive", "max_tilt_deg", "max_penetration_mm"
    }
    assert len(verdict["gates"]) == 4


def test_release_gate_is_optional():
    trace = episode([0, 60, 60, 60, 60, 60], landed=None)
    assert judge_episode(trace, CleanCfg())["ok"]
    strict = judge_episode(trace, CleanCfg(require_release=True))
    assert not strict["ok"]
    assert strict["failed_gates"] == ["landed_in_box"]
    landed = judge_episode(episode([0, 60, 60, 60, 60, 60], landed="box_a"),
                           CleanCfg(require_release=True))
    assert landed["ok"]


def test_empty_traces_are_judged_as_failure_not_as_an_exception():
    verdict = judge_episode(Episode())
    assert not verdict["ok"]
    assert verdict["max_lift_mm"] == 0.0


def test_batch_summary_decomposes_failures():
    good = episode([0, 60, 60, 60, 60, 60])
    low = episode([0, 5, 5, 5, 5, 5])
    digging = episode([0, 70, 70, 70, 70, 70], penetration=[0, 2, 2, 2, 2, 2])
    tilted = episode([0, 70, 70, 70, 70, 70], tilt=[0, 8, 8, 8, 8, 8])
    summary = judge_many([good, low, digging, tilted])
    assert summary["n"] == 4 and summary["n_ok"] == 1
    assert summary["success_rate"] == pytest.approx(0.25)
    assert summary["by_gate"]["max_penetration_mm"] == 1
    assert summary["by_gate"]["max_tilt_deg"] == 1
    assert summary["by_gate"]["max_lift_mm"] == 1  # only the low trace
    assert summary["by_gate"]["lift_consecutive"] == 1
    assert len(summary["per_episode"]) == 4


def test_batch_summary_of_nothing():
    summary = judge_many([])
    assert summary["n"] == 0 and summary["n_ok"] == 0 and summary["success_rate"] is None


# ----------------------------------------------------------------------------------
# input validation
# ----------------------------------------------------------------------------------
def test_mismatched_trace_lengths_are_rejected():
    with pytest.raises(ValueError, match="same length"):
        Episode(lift_mm=[1, 2, 3], tilt_deg=[1, 2], penetration_mm=[1, 2, 3])


def test_non_finite_values_are_rejected():
    with pytest.raises(ValueError, match="non-finite"):
        Episode(lift_mm=[0.0, float("nan")], tilt_deg=[0.0, 0.0], penetration_mm=[0.0, 0.0])


def test_bad_configuration_is_rejected():
    with pytest.raises(ValueError):
        CleanCfg(lift_min_mm=-1.0)
    with pytest.raises(ValueError):
        CleanCfg(lift_consecutive=0)