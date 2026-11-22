"""Fisher / McNemar / Newcombe / Benjamini-Hochberg at the level a report uses them."""

from __future__ import annotations

import math

import pytest

from eeval.stats import (
    benjamini_hochberg,
    fisher_test_2x2,
    mcnemar_exact,
    newcombe_diff_ci,
    paired_odds_ratio_interval,
    two_proportion_diff,
)


def test_two_proportion_diff_reproduces_the_real_robot_ab():
    """1/12 vs 9/12 grasp success: the number quoted in the project README."""
    result = two_proportion_diff(9, 12, 1, 12)
    assert result.p_value == pytest.approx(0.0028, abs=5e-4)
    assert result.diff == pytest.approx(8 / 12)
    lo, hi = result.ci
    assert lo < result.diff < hi
    assert lo > 0.0, "a 12-vs-0 split should exclude a zero difference"
    text = str(result)
    assert "Fisher p" in text and "12" in text


def test_two_proportion_diff_with_no_effect_straddles_zero():
    result = two_proportion_diff(6, 12, 7, 12)
    lo, hi = result.ci
    assert lo < 0.0 < hi
    assert result.p_value > 0.5


def test_fisher_result_reports_odds_ratio_and_handles_empty_cells():
    result = fisher_test_2x2(9, 3, 1, 11)
    assert result.odds_ratio == pytest.approx(33.0)
    assert result.table == ((9, 3), (1, 11))
    infinite = fisher_test_2x2(12, 0, 0, 12)
    assert math.isinf(infinite.odds_ratio)
    assert infinite.as_dict()["odds_ratio"] is None


def test_fisher_degenerate_tables_return_one():
    assert fisher_test_2x2(0, 0, 0, 0).p_value == 1.0
    assert fisher_test_2x2(5, 0, 3, 0).p_value == 1.0


def test_mcnemar_uses_only_discordant_pairs():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(8, 1) == pytest.approx(0.0391, abs=5e-4)
    assert mcnemar_exact(1, 8) == mcnemar_exact(8, 1)


def test_matched_odds_ratio_interval_excludes_one_for_clear_asymmetry():
    lo, hi = paired_odds_ratio_interval(8, 1)
    assert lo > 1.0 and math.isinf(hi) is False
    lo0, hi0 = paired_odds_ratio_interval(0, 0)
    assert lo0 == 0.0 and math.isinf(hi0)


def test_newcombe_matches_its_definition_for_a_known_table():
    """Recomputing the closed form from the two Wilson intervals must agree exactly."""
    from eeval.stats import wilson_interval

    k1, n1, k2, n2 = 9, 12, 1, 12
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    expected_lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    expected_hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    lo, hi = newcombe_diff_ci(k1, n1, k2, n2)
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)


def test_benjamini_hochberg_matches_a_hand_computation():
    result = benjamini_hochberg([0.01, 0.04], q=0.05)
    assert result.adjusted == pytest.approx((0.02, 0.04))
    assert result.rejected == (True, True)


def test_benjamini_hochberg_is_monotone_in_the_ranking():
    p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    result = benjamini_hochberg(p_values, q=0.05)
    for raw, adjusted in zip(p_values, result.adjusted):
        assert adjusted >= raw - 1e-12
    ordered = [a for _, a in sorted(zip(p_values, result.adjusted))]
    assert ordered == sorted(ordered), "adjusted p-values must follow the raw ordering"
    assert result.adjusted[0] == pytest.approx(0.01)
    assert any(result.rejected) and not all(result.rejected)


def test_benjamini_hochberg_keeps_input_order():
    """Adjustment must be reported against the caller's list, not the sorted one."""
    left = benjamini_hochberg([0.5, 0.001], q=0.05)
    assert left.adjusted[1] < left.adjusted[0]
    assert left.rejected == (False, True)


def test_benjamini_hochberg_empty_input():
    result = benjamini_hochberg([], q=0.05)
    assert result.adjusted == () and result.rejected == ()