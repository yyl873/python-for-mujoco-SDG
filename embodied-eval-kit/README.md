# embodied-eval-kit

**Evaluation numbers that survive scrutiny.** Exact small-sample statistics, a
counterfactual pairing protocol for language-conditioned policies, clean-success gates,
and three-state receipts — in ~1,600 lines of dependency-free Python.

```
python -m pip install -e ".[test]"
python examples/demo_real_run.py     # real 76-rollout log, no downloads
python -m pytest tests -q            # 134 tests, ~6 s, standard library only
```

The motivating failure, from the project this kit was extracted from: a policy was
reported at **100% instruction-following accuracy**. It was also true that the policy put
the box on the instructed side in only **15 of 38 scenes**. Both statements came from the
same data. The first used "rollouts that landed in *some* box" as its denominator; the
second used paired counterfactual replays. Choosing the denominator is what decides
whether a robot demo is a result or a story.

| what | why it exists |
|---|---|
| `eeval.stats` | small samples lie: 40/40 is **91.2%** at the lower 95% bound, not 100% |
| `eeval.counterfactual` | one number cannot both measure skill and resist a degenerate policy |
| `eeval.judge` | "it lifted" rewards digging into the object and lifting the debris |
| `eeval.receipt` | a stage that prints `warn` must never roll up to `ok` |

## Try it in 30 seconds

```bash
python -m eeval report examples/data/language_pair_76rows.json \
  --tasks "Put the medicine box into the left wooden box.,Put the medicine box into the right wooden box." \
  --boxes "target_box,target_box_b"
```

```text
**Verdict: instruction followed** (pass)

| estimand | value | denominator |
|---|---|---|
| follow_acc | 100.0% | 40 rollouts that landed |
| follow_acc (ITT) | 52.6% | 76 rollouts |
| pairs both followed | 15 / 38 (39.5%) | complete pairs |
| Wilson 95% of follow_acc | 91.2% - 100.0% | - |
| p vs chance (0.50) | < 1e-6 | - |
| side bias p (landing counts {'target_box': 25, 'target_box_b': 15}) | 0.1539 | - |
```

Exit code `0` when the pre-registered verdict passes, `1` when the report is produced
but the verdict fails, `2` on bad input — so a shell or CI job can gate on it.

## The protocol

A language-conditioned policy is evaluated by replaying **the same scenes** under **both
instructions**. Outcomes then come in pairs, and three estimands fall out, each with a
defensible denominator:

| estimand | denominator | question |
|---|---|---|
| `follow_acc` (per-protocol) | rollouts that landed in **some** box | when it did put the box down, was it the instructed box? |
| `follow_acc_itt` | **every** rollout | did the whole task succeed? |
| `pairs_both_follow` | complete `(scene, seed)` pairs | did both instructions of the same scene land where told? |

Only the third one is unfakeable: a policy that always drops right scores 100% on half
the rollouts, **0 pairs**. `Report` also publishes the failure breakdowns
(`landed_correct` / `landed_wrong_box` / `never_landed`), the landing distribution with a
bias test, McNemar on the discordant pairs, and the chance baselines a reader would
otherwise have to reconstruct. See [`docs/PROTOCOL.md`](docs/PROTOCOL.md) for the formal
definitions, the worked example, and the six ways this analysis is commonly fudged.

## API in one screen

```python
from eeval import Protocol, Rollout, analyze_rows

protocol = Protocol(
    instructions=("left", "right"),               # order == the policy's language slot
    expected_box={"left": "box_a", "right": "box_b"},
    min_follow_acc=0.80, min_wilson_lower=0.50,   # pre-registered, fixed before the run
    min_pairs_both_follow=1,
)
report = analyze_rows(rows, protocol)   # rows: [{episode_id, seed, ix, landed_box}, ...]

report.follow_acc          # 1.0
report.follow_wilson       # (0.9124, 1.0)   -- the honest version of "100%"
report.follow_acc_itt      # 0.5263
report.pairs_both_follow   # 15
report.verdict.passed      # True
report.to_markdown()       # the table above
report.to_json()           # everything, for a results file
```

| module | functions |
|---|---|
| `stats` | `wilson`, `wilson_interval`, `clopper_pearson`, `proportion`, `binom_pmf/cdf/sf`, `binom_two_sided`, `fisher_exact_2x2`, `fisher_test_2x2`, `mcnemar_exact`, `paired_odds_ratio_interval`, `newcombe_diff_ci`, `two_proportion_diff`, `benjamini_hochberg`, `log_binom_coef` |
| `counterfactual` | `Protocol`, `Rollout`, `Report`, `Verdict`, `analyze`, `analyze_rows`, `aggregate`, `aggregate_pairs`, `verdict` |
| `judge` | `CleanCfg`, `Episode`, `GateResult`, `judge_episode`, `judge_many` |
| `receipt` | `Stage`, `Receipt`, `exit_code`, `STATUS` |

Every interval function takes `alpha` (default `0.05`); every test returns a plain
`float`; every result object has `as_dict()` and a readable `__str__`.

## Numbers this code has to reproduce

These are asserted in `tests/`, not pasted from memory:

| input | output | where it comes from |
|---|---|---|
| `wilson(40, 40)` | lower bound **0.9124** | final language-conditioned evaluation, 40/40 |
| `fisher_exact_2x2(1, 11, 9, 3)` | **p = 0.0028** | real-robot A/B: 1/12 vs 9/12 grasp success |
| `fisher_exact_2x2(1, 11, 8, 4)` | **p = 0.0094** | same-session control, only the training data differs |
| `mcnemar_exact(8, 1)` | **p = 0.039** | paired release-success comparison |
| `analyze_rows(language_pair_76rows.json)` | 15/38 pairs, ITT 52.6% | the project README's table |

## Why hand-written statistics

`scipy` would have been one import. The kit is used on robot computers, on CI runners
with no scientific stack, and on an air-gapped lab machine, so it depends on nothing.
Doing it by hand is only acceptable if it is checked, so the tests compare each routine
against:

- **exact rational arithmetic** (`fractions.Fraction` + `math.comb`) — an independent
  implementation that shares no arithmetic with the float code;
- **closed forms** that hold exactly at the extremes (`k = 0`, `k = n`) for
  Clopper-Pearson;
- **the defining equation** of each interval, evaluated at its own bounds (the Wilson
  bounds satisfy the score equation to 1e-6);
- **Monte-Carlo coverage** at the nominal level for Wilson, Clopper-Pearson and
  Newcombe's difference interval.

```
$ python -m pytest tests -q
134 passed in 5.6s
```

| test file | what it protects |
|---|---|
| `test_stats_exact.py` | float code vs exact rationals, tail underflow, monotone symmetries |
| `test_stats_intervals.py` | closed forms, score equation, simulated coverage |
| `test_stats_paired.py` | the published p-values, Newcombe's definition, BH ordering |
| `test_stats_errors.py` | bad input fails loudly instead of producing a wrong denominator |
| `test_counterfactual.py` | estimands, pair matrix, incomplete pairs, degenerate-policy detection |
| `test_real_run.py` | the real 76-rollout log reproduces the published table exactly |
| `test_judge.py` | gate boundaries, consecutive-run rule, reward-hack patterns |
| `test_receipt.py`, `test_cli.py` | three-state roll-up, exit codes, end-to-end CLI |

## What it is not

- **Not a metrics framework.** No plotting, no experiment tracking, no dataset loaders.
  It turns a rollout log into a defensible number and gets out of the way.
- **Not for huge samples.** Exact tests are `O(n)` per call; fine for hundreds of
  rollouts, silly for a million. Use a normal approximation at that scale and say so.
- **Not a substitute for replicates.** A single seed, a single scene set, and n = 12
  trials stay single-sample statements no matter how exact the arithmetic is. The kit
  makes the uncertainty visible; it cannot manufacture evidence.
- **Not free of assumptions.** Independence between scenes, a fixed instruction set, and
  a judge that actually measures what it claims. Everything the protocol assumes is
  written down in [`docs/PROTOCOL.md`](docs/PROTOCOL.md).

## Layout

```
eeval/           stats.py  counterfactual.py  judge.py  receipt.py  cli.py  __main__.py
examples/        demo_real_run.py  demo_synthetic.py  data/language_pair_76rows.json
tests/           8 files, 134 tests, no dependencies beyond pytest
docs/            PROTOCOL.md  (formal definitions, worked example, common mistakes)
```

Zero runtime dependencies · MIT licensed · Python ≥ 3.10 · CI on 3.10 / 3.12 / 3.13