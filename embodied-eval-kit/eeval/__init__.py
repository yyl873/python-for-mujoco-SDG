"""eeval -- make "is this policy any good?" a number somebody else can recompute.

Four small modules, no dependencies outside the standard library:

``eeval.stats``
    Exact intervals and tests for small samples: Wilson, Clopper-Pearson, exact
    binomial, Fisher, McNemar, Newcombe's difference interval, Benjamini-Hochberg.
``eeval.judge``
    Clean-success gates: a rollout only counts if it lifted *and* did not tilt,
    penetrate, or fake the lift.
``eeval.counterfactual``
    Paired evaluation of a language-conditioned policy: same scenes, both instructions,
    three estimands that a degenerate policy cannot fake at once.
``eeval.receipt``
    Three-state receipts so a ``warn`` can never be reported as ``ok``.

Typical use::

    from eeval import Protocol, analyze_rows

    report = analyze_rows(rows, Protocol(
        instructions=("left", "right"),
        expected_box={"left": "box_a", "right": "box_b"},
    ))
    print(report.to_markdown())
    report.verdict.passed
"""

from .counterfactual import (
    Protocol,
    Report,
    Rollout,
    Verdict,
    aggregate,
    aggregate_pairs,
    analyze,
    analyze_rows,
    verdict,
)
from .judge import CleanCfg, Episode, GateResult, judge_episode, judge_many
from .receipt import STATUS, Receipt, Stage, exit_code
from .stats import (
    BHRResult,
    FisherResult,
    Proportion,
    TwoProportionDiff,
    benjamini_hochberg,
    binom_cdf,
    binom_pmf,
    binom_sf,
    binom_two_sided,
    clopper_pearson,
    fisher_exact_2x2,
    fisher_test_2x2,
    log_binom_coef,
    mcnemar_exact,
    newcombe_diff_ci,
    paired_odds_ratio_interval,
    proportion,
    two_proportion_diff,
    wilson,
    wilson_interval,
)

__version__ = "0.2.0"

__all__ = [
    # stats
    "BHRResult",
    "FisherResult",
    "Proportion",
    "TwoProportionDiff",
    "benjamini_hochberg",
    "binom_cdf",
    "binom_pmf",
    "binom_sf",
    "binom_two_sided",
    "clopper_pearson",
    "fisher_exact_2x2",
    "fisher_test_2x2",
    "log_binom_coef",
    "mcnemar_exact",
    "newcombe_diff_ci",
    "paired_odds_ratio_interval",
    "proportion",
    "two_proportion_diff",
    "wilson",
    "wilson_interval",
    # judge
    "CleanCfg",
    "Episode",
    "GateResult",
    "judge_episode",
    "judge_many",
    # counterfactual
    "Protocol",
    "Report",
    "Rollout",
    "Verdict",
    "aggregate",
    "aggregate_pairs",
    "analyze",
    "analyze_rows",
    "verdict",
    # receipt
    "STATUS",
    "Receipt",
    "Stage",
    "exit_code",
]