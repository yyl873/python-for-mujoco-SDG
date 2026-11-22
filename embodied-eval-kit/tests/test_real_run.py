"""End-to-end: the real 76-rollout log reproduces the published numbers exactly.

This is the regression test that keeps the kit honest -- if the aggregation ever drifts,
the project README's table is wrong, and this test says so.
"""

from __future__ import annotations

import pytest

from eeval.counterfactual import analyze_rows

PUBLISHED = {
    "n_rollouts": 76,
    "n_landed": 40,
    "n_followed": 40,
    "follow_acc": 1.0,
    "wilson_lower": 0.9124,
    "itt": 0.5263,
    "pairs_both_follow": 15,
    "n_pairs": 38,
    "pairs_any_landed": 25,
    "pairs_neither_landed": 13,
    "left_landed": 25,
    "right_landed": 15,
}


@pytest.fixture(scope="module")
def report(real_run, real_protocol):
    return analyze_rows(real_run["rows"], real_protocol)


def test_counts_match_the_published_table(report):
    assert report.n_rollouts == PUBLISHED["n_rollouts"]
    assert report.n_landed == PUBLISHED["n_landed"]
    assert report.n_followed == PUBLISHED["n_followed"]
    assert report.follow_acc == pytest.approx(PUBLISHED["follow_acc"])
    assert report.follow_wilson[0] == pytest.approx(PUBLISHED["wilson_lower"], abs=5e-4)
    assert report.follow_acc_itt == pytest.approx(PUBLISHED["itt"], abs=5e-4)


def test_pair_metrics_match_the_published_table(report):
    assert report.n_pairs == PUBLISHED["n_pairs"]
    assert report.complete_pairs == PUBLISHED["n_pairs"]
    assert report.pairs_both_follow == PUBLISHED["pairs_both_follow"]
    assert report.pairs_any_landed == PUBLISHED["pairs_any_landed"]
    assert report.pairs_neither_landed == PUBLISHED["pairs_neither_landed"]
    assert report.pattern_counts == {"TT": 15, "TF": 10, "FF": 13}


def test_landing_distribution_and_bias(report):
    assert report.landed_box_counts == {"target_box": PUBLISHED["left_landed"],
                                        "target_box_b": PUBLISHED["right_landed"]}
    assert report.side_bias_p == pytest.approx(0.1539, abs=5e-4)
    # 10 pairs where only "left" followed, none where only "right" did
    assert report.discordant_p == pytest.approx(0.00195, abs=5e-5)


def test_verdict_passes_and_reports_chance_baselines(report):
    assert report.verdict.passed
    assert report.verdict.label == "instruction followed"
    assert report.chance_baselines["follow_acc_uniform_guess"] == pytest.approx(0.5)
    assert report.chance_baselines["follow_acc_always_single_box"] == pytest.approx(0.5)


def test_markdown_and_json_are_renderable(report):
    markdown = report.to_markdown()
    assert "**Verdict: instruction followed**" in markdown
    payload = report.as_dict()
    assert payload["n_rollouts"] == 76
    assert payload["per_instruction"]["Put the medicine box into the left wooden box."]["n_landed"] == 25