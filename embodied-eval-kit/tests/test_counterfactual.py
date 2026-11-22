"""The counterfactual protocol: estimands, pairing, validation, and the verdict."""

from __future__ import annotations

import json

import pytest

from eeval.counterfactual import Protocol, Rollout, aggregate, aggregate_pairs, analyze, analyze_rows, verdict

LEFT, RIGHT = "left", "right"
BOX_A, BOX_B = "box_a", "box_b"
EXPECTED = {LEFT: BOX_A, RIGHT: BOX_B}


def protocol(**kwargs) -> Protocol:
    return Protocol(instructions=(LEFT, RIGHT), expected_box=EXPECTED, **kwargs)


def rollout(episode: int, seed: int, ix: int, box: str | None, **kwargs) -> Rollout:
    return Rollout(episode_id=episode, seed=seed, instruction=ix, landed_box=box, **kwargs)


# ----------------------------------------------------------------------------------
# protocol validation
# ----------------------------------------------------------------------------------
def test_protocol_needs_at_least_two_instructions():
    with pytest.raises(ValueError):
        Protocol(instructions=(LEFT,), expected_box=EXPECTED)


def test_protocol_needs_an_expected_box_for_every_instruction():
    with pytest.raises(ValueError):
        Protocol(instructions=(LEFT, RIGHT), expected_box={LEFT: BOX_A})


def test_protocol_rejects_duplicate_instructions():
    with pytest.raises(ValueError):
        Protocol(instructions=(LEFT, LEFT), expected_box=EXPECTED)


def test_protocol_round_trips_through_dict():
    original = protocol()
    restored = Protocol.from_dict(original.as_dict())
    assert restored.instructions == original.instructions
    assert restored.expected_box == original.expected_box


# ----------------------------------------------------------------------------------
# the four estimands
# ----------------------------------------------------------------------------------
def four_scene_run() -> list[Rollout]:
    """Four scenes chosen so every cell of the pair matrix is exercised once."""
    return [
        rollout(1, 0, 0, BOX_A), rollout(1, 0, 1, BOX_B),   # both follow
        rollout(2, 0, 0, BOX_A), rollout(2, 0, 1, BOX_A),   # only left follows
        rollout(3, 0, 0, None), rollout(3, 0, 1, BOX_B),    # only right follows
        rollout(4, 0, 0, BOX_B), rollout(4, 0, 1, BOX_A),   # neither follows
    ]


def test_estimands_and_pair_matrix():
    report = analyze(four_scene_run(), protocol())
    assert report.n_rollouts == 8
    assert report.n_landed == 7
    assert report.n_followed == 4
    assert report.follow_acc == pytest.approx(4 / 7)
    assert report.follow_acc_itt == pytest.approx(4 / 8)
    assert report.complete_pairs == 4
    assert report.pairs_both_follow == 1
    assert report.pairs_any_landed == 4  # every scene put the box down somewhere
    assert report.pairs_neither_landed == 0
    assert report.pattern_counts == {"TT": 1, "TF": 1, "FT": 1, "FF": 1}
    assert report.discordant_p == 1.0  # one discordant pair in each direction


def test_per_instruction_breakdown():
    report = analyze(four_scene_run(), protocol())
    left = report.per_instruction[LEFT]
    right = report.per_instruction[RIGHT]
    assert left["n_rollouts"] == 4 and left["n_landed"] == 3 and left["n_followed"] == 2
    assert right["n_landed"] == 4 and right["n_followed"] == 2
    assert left["expected_box"] == BOX_A


def test_outcome_breakdown_separates_not_landed_from_wrong_box():
    report = analyze(four_scene_run(), protocol())
    assert report.outcome_counts["followed"] == 4
    assert report.outcome_counts["wrong_box"] == 3
    assert report.outcome_counts["not_landed"] == 1
    assert report.landed_box_counts == {BOX_A: 4, BOX_B: 3}


def test_incomplete_pairs_are_excluded_and_reported():
    rows = four_scene_run() + [rollout(5, 0, 0, BOX_A)]  # scene 5 only ran the left instruction
    report = analyze(rows, protocol())
    assert report.n_pairs == 5
    assert report.complete_pairs == 4
    assert any("incomplete" in note for note in report.notes)


def test_itt_denominator_is_every_rollout():
    """Per-protocol accuracy can be high while the task mostly fails; both are published."""
    rows = [rollout(1, 0, 0, BOX_A), rollout(2, 0, 0, BOX_A)]
    report = analyze(rows, Protocol(instructions=(LEFT, RIGHT), expected_box=EXPECTED,
                                    strict_boxes=False))
    assert report.follow_acc == 1.0
    assert report.follow_acc_itt == 1.0
    rows_mixed = rows + [rollout(3, 0, 0, None)]
    mixed = analyze(rows_mixed, Protocol(instructions=(LEFT, RIGHT), expected_box=EXPECTED,
                                         strict_boxes=False))
    assert mixed.follow_acc == 1.0
    assert mixed.follow_acc_itt == pytest.approx(2 / 3)


# ----------------------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------------------
def test_duplicate_rollout_raises():
    rows = [rollout(1, 0, 0, BOX_A), rollout(1, 0, 0, BOX_A)]
    with pytest.raises(ValueError, match="duplicate"):
        analyze(rows, protocol())


def test_unknown_instruction_index_raises():
    with pytest.raises(ValueError, match="instruction index"):
        analyze([rollout(1, 0, 5, BOX_A)], protocol())


def test_unknown_box_raises_by_default():
    with pytest.raises(ValueError, match="not one of the protocol boxes"):
        analyze([rollout(1, 0, 0, "box_z")], protocol())


def test_unknown_box_can_be_tolerated():
    report = analyze(
        [rollout(1, 0, 0, "box_z"), rollout(1, 0, 1, BOX_B)],
        Protocol(instructions=(LEFT, RIGHT), expected_box=EXPECTED, strict_boxes=False),
    )
    assert report.outcome_counts["unknown_box"] == 1
    assert report.n_landed == 2  # a box we do not know still counts as "landed somewhere"


# ----------------------------------------------------------------------------------
# verdicts
# ----------------------------------------------------------------------------------
def test_clean_run_passes():
    rows = [rollout(i, 0, ix, EXPECTED[text]) for i in range(10) for ix, text in enumerate((LEFT, RIGHT))]
    report = analyze(rows, protocol())
    assert report.verdict.passed
    assert report.verdict.label == "instruction followed"
    assert report.pairs_both_follow == 10


def test_degenerate_always_right_policy_is_caught_by_the_pair_metric():
    """The whole point of the protocol: 50% follow_acc is a coin flip, pairs are 0."""
    rows = [rollout(i, 0, ix, BOX_B) for i in range(10) for ix in (0, 1)]
    report = analyze(rows, protocol())
    assert report.follow_acc == pytest.approx(0.5)
    assert report.pairs_both_follow == 0
    assert not report.verdict.passed
    assert "ignored" in report.verdict.label
    assert report.chance_baselines["follow_acc_always_single_box"] == pytest.approx(0.5)
    assert report.side_bias_p is not None and report.side_bias_p < 0.01


def test_high_accuracy_without_a_correct_pair_is_reported_separately():
    """Perfect follow_acc, zero pairs: the pairing itself is broken, not the policy."""
    rows = []
    for episode in range(5):
        rows.append(rollout(episode, 0, 0, BOX_A))       # left follows, right never lands
    for episode in range(5, 10):
        rows.append(rollout(episode, 0, 1, BOX_B))       # right follows, left never lands
    report = analyze(rows, protocol())
    assert report.follow_acc == 1.0
    assert report.pairs_both_follow == 0
    assert not report.verdict.passed
    assert "no pair is fully correct" in report.verdict.label


def test_three_instruction_protocol_works():
    tasks = ("left", "middle", "right")
    boxes = {"left": "a", "middle": "b", "right": "c"}
    rows = [Rollout(1, 0, ix, boxes[text]) for ix, text in enumerate(tasks)]
    report = analyze(rows, Protocol(instructions=tasks, expected_box=boxes))
    assert report.complete_pairs == 1
    assert report.pairs_both_follow == 1
    assert report.discordant_p is None  # McNemar is only defined for two arms


# ----------------------------------------------------------------------------------
# serialisation
# ----------------------------------------------------------------------------------
def test_report_serialises_to_json_and_markdown():
    report = analyze(four_scene_run(), protocol())
    payload = json.loads(report.to_json())
    assert payload["n_rollouts"] == 8
    assert payload["pairs_both_follow"] == 1
    assert payload["verdict"]["passed"] is False
    markdown = report.to_markdown()
    assert "follow_acc" in markdown and "pairs both followed" in markdown


def test_rollout_from_dict_accepts_the_rollout_log_schema():
    row = {"episode_id": 2400, "seed": 0, "ix": 1, "landed_box": "target_box_b",
           "steps": 310, "wall_s": 6.5}
    parsed = Rollout.from_dict(row)
    assert parsed.pair_key == (2400, 0)
    assert parsed.instruction == 1
    assert parsed.steps == 310 and parsed.seconds == 6.5


def test_legacy_aggregate_matches_the_report():
    rows = four_scene_run()
    legacy = aggregate(
        [{"episode_id": r.episode_id, "seed": r.seed, "ix": r.instruction,
          "landed_box": r.landed_box} for r in rows],
        (LEFT, RIGHT),
        {},
        expect_map=EXPECTED,
    )
    report = analyze(rows, protocol())
    assert legacy["n_rollouts"] == report.n_rollouts
    assert legacy["follow_correct"] == report.n_followed
    pairs = aggregate_pairs(
        [{"episode_id": r.episode_id, "seed": r.seed, "ix": r.instruction,
          "landed_box": r.landed_box} for r in rows],
        (LEFT, RIGHT),
        EXPECTED,
    )
    assert pairs == {"n_pairs": report.n_pairs,
                     "pairs_both_follow": report.pairs_both_follow,
                     "pairs_any_landed": report.pairs_any_landed}
    assert verdict(legacy) == "instruction ignored"


def test_analyze_rows_is_the_dict_adapter():
    rows = [{"episode_id": 1, "seed": 0, "ix": 0, "landed_box": BOX_A},
            {"episode_id": 1, "seed": 0, "ix": 1, "landed_box": BOX_B}]
    assert analyze_rows(rows, protocol()).pairs_both_follow == 1