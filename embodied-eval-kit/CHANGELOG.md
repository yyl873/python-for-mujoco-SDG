# Changelog

## 0.2.0

Rewritten around a protocol instead of a handful of helpers. The 0.1.0 surface still
works; nothing imported from it was removed.

### Added
- `eeval.stats`: `proportion`, `clopper_pearson` (exact, by bisection on the binomial
  tails), `binom_pmf / binom_cdf / binom_sf`, `fisher_test_2x2` (returns p-value, odds
  ratio and the table), `newcombe_diff_ci`, `two_proportion_diff`, `paired_odds_ratio_interval`,
  `benjamini_hochberg`, `log_binom_coef`. All log-space, `math.fsum`-summed tails.
- `eeval.counterfactual`: `Protocol` (ordered instructions + expected boxes +
  pre-registered thresholds), `Rollout`, `Report`, `Verdict`, `analyze`, `analyze_rows`.
  Reports now carry per-instruction breakdowns, the pair-pattern matrix, the failure
  decomposition, landing distribution with a bias test, McNemar on discordant pairs, and
  chance baselines.
- `eeval.judge`: `GateResult`, `judge_many` (failure decomposition), cheat-pattern flags,
  input validation for trace lengths and non-finite values.
- `eeval.cli` / `python -m eeval`: `report`, `judge`, `selfcheck`, with exit codes that
  follow the receipt states.
- `eeval.receipt`: `to_dict`, `from_dict`, `merge`, `extend`, `require_ok`, `failed`,
  `warned`, per-stage notes.
- `example/data/language_pair_76rows.json`: a real 76-rollout log, used by
  `examples/demo_real_run.py` and by the regression test.
- Test suite: 134 tests across 8 files, comparing the float code against exact rational
  arithmetic, closed forms, defining equations and Monte-Carlo coverage.

### Changed
- Package, docstrings, reasons and verdict labels are English (0.1.0 was Chinese).
  `verdict()` therefore returns `"instruction followed"` / `"instruction ignored"` where
  0.1.0 returned Chinese strings.
- `fisher_exact_2x2` keeps returning a float; use `fisher_test_2x2` for the odds ratio.
- `judge_episode` reasons are structured (`failed_gates`, `gates`, `cheat_flags`) instead
  of a single Chinese prose list.
- Zero runtime dependencies is now an enforced rule: CI installs nothing but `pytest`.

### Fixed
- `binom_two_sided` recomputed every pmf twice per iteration; it now builds the pmf once.
- Clopper-Pearson's lower bound bracketed the bisection in the wrong direction (it only
  returned sane values for `k = 0`).
- `wilson(k, n)` and `clopper_pearson(k, n)` accepted `n < 0` silently.
- `Episode` with mismatched trace lengths raised nothing and produced a meaningless
  maximum.

## 0.1.0

First cut: `wilson`, `binom_two_sided`, `fisher_exact_2x2`, `mcnemar_exact`,
`CleanCfg`/`Episode`/`judge_episode`, `aggregate`/`aggregate_pairs`/`verdict`,
`Stage`/`Receipt`/`exit_code`.