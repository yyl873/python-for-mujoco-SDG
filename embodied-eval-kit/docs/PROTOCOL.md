# The counterfactual pairing protocol

A specification for "does this policy actually use the instruction?", written so someone
who disagrees with the conclusion can redo the analysis and check it.

---

## 1. The problem with success rate

Let a language-conditioned policy `π` control a robot in a scene `s` under an instruction
`t` drawn from a fixed set `T`. A rollout reports where the object ended up.

Success rate answers "did the task work". It cannot answer "did the *language* do
anything", because a policy that ignores `t` entirely and always drops the object in one
place still succeeds on every rollout whose instruction happens to name that place. With
`|T| = 2` and balanced instructions, that degenerate policy scores **50%** — a number
that reads like "half working" and means "zero understanding".

The fix is not a better metric, it is a better *design*: replay each scene under every
instruction, so the instruction is the only thing that changes.

## 2. Setup

| symbol | meaning |
|---|---|
| `S` | set of scenes (episode ids), held out from training |
| `K` | seeds per scene (different initial object poses / noise) |
| `T = (t₀ … t_{m-1})` | the instruction set, **ordered**: index `i` is what the policy receives |
| `b(t)` | the box instruction `t` asks for (`Protocol.expected_box`) |
| `x = (s, k, i)` | one rollout: scene `s`, seed `k`, instruction index `i` |
| `y(x)` | `landed_box` reported by the rollout, `None` if the object never landed |

A **pair** is the set of rollouts sharing `(s, k)` — all `m` instructions on the same
physical scene. A pair is *complete* when every instruction in `T` was replayed.

Two labels per rollout:

```
landed(x)   :=  y(x) is not None
followed(x) :=  y(x) == b(t_i)
```

`landed` and `followed` are deliberately separate: "it never put the box down" is a
different failure from "it put the box in the wrong place", and merging them hides which
one dominates.

## 3. The three estimands

```
follow_acc     = Σ followed(x) / Σ landed(x)        # per-protocol, denominator = landed
follow_acc_itt = Σ followed(x) / Σ 1                # intention-to-treat, denominator = all
pairs_both     = #{pairs p : followed(x) for every x ∈ p}   # denominator = complete pairs
```

Published with intervals, never as bare ratios:

| estimand | interval | why |
|---|---|---|
| `follow_acc`, `follow_acc_itt` | Wilson 95% and Clopper-Pearson 95% | small `n`, and `k = n` must not read as `[1, 1]` |
| `pairs_both` | report the count and its denominator; `k/n` at these `n` is noise | 38 pairs cannot resolve 1–2 pairs |
| difference between two policies | Newcombe hybrid score interval + Fisher exact | one cell is usually 0 at n ≈ 12 |
| two arms of the same scenes | McNemar exact on discordant pairs | pairing removes scene difficulty from the variance |

`follow_acc` is the *conditional* accuracy and the one to quote when asking "when it
worked, was it right"; `follow_acc_itt` is the one to quote when asking "did the task
get done". A report that quotes only the first is describing 40 rollouts out of 76 and
calling it the result.

## 4. What the pairs buy you

For `m = 2` the complete pairs form a 2×2 table of the followed vector `(f₀, f₁)`:

| pattern | count | meaning |
|---|---|---|
| `TT` | both instructions followed | genuine instruction use |
| `TF` | only the first followed | asymmetry or noise |
| `FT` | only the second followed | asymmetry or noise |
| `FF` | neither followed | the task failed, instruction aside |

McNemar's exact test runs on the discordant cells `TF`, `FT`: if the instruction had no
effect, `TF` and `FT` are equally likely, so `p = 2 · P(Binomomial(TF+FT, 0.5) ≤ min(TF, FT))`.

Two extra checks that catch the failure modes metrics miss:

- **Side bias.** Count landings per box across all rollouts and test the largest count
  against a fair `1/|B|` split (`side_bias_p`). A policy that is drifting towards one side
  shows up here before it shows up in accuracy.
- **Chance baselines**, reported in every `Report`:
  - `follow_acc_uniform_guess = 1/|T|` (the policy guesses a box at random);
  - `follow_acc_always_single_box = max_boxes(#rollouts asking for that box) / n` (the
    policy always lands in one place);
  - `pairs_both_follow_always_single_box = 0` — the reason the pair metric exists.

## 5. Pre-registration

The thresholds are part of the protocol, fixed before the data is seen:

```
min_follow_acc        = 0.80
min_wilson_lower      = 0.50
min_pairs_both_follow = 1
```

The verdict is then mechanical (`eeval.counterfactual._pre_registered_verdict`):

| condition | label |
|---|---|
| all three checks pass | `instruction followed` |
| accuracy and interval pass, no correct pair | `landing tracks the instruction, but no pair is fully correct` |
| otherwise | `instruction ignored (landing is indistinguishable from guessing)` |

The middle label is the interesting one: it means the *pairing* is broken (scenes differ
between the two arms, or the two arms were not both replayed), not that the policy failed.

## 6. Worked example (the run this kit was extracted from)

19 held-out scenes × 2 seeds × 2 instructions = **76 rollouts**, `|B| = 2`:

| quantity | value |
|---|---|
| landed in some box | 40 / 76 |
| followed the instruction | 40 |
| `follow_acc` | 100%, Wilson **[0.9124, 1.0]**, exact **[0.9119, 1.0]** |
| `follow_acc_itt` | 52.6% |
| pair patterns | `TT` 15, `TF` 10, `FF` 13, `FT` 0 |
| `pairs_both_follow` | **15 / 38** |
| landing distribution | box A 25, box B 15 → `side_bias_p = 0.154` |
| McNemar on `TF` vs `FT` | p = 0.00195 |
| verdict | `instruction followed` |

Read honestly: when the policy put the box down, it always put it in the right place;
but it only put the box down in 40 of 76 rollouts, and it needed both arms of a scene to
succeed only 15 times out of 38. The asymmetry (`TF = 10, FT = 0`) is a real finding —
the "right" arm landed less often — and the 36 failures are a grasping problem, not a
language problem.

## 7. Six ways this analysis gets fudged

1. **Quoting the conditional accuracy alone.** "100%" hides that 47% of rollouts never
   landed. Always publish `follow_acc` and `follow_acc_itt` together.
2. **Quoting a ratio when the denominator is 2 or 3.** `follow_acc = 100%` from 3 landed
   rollouts is not evidence; the Wilson lower bound says so.
3. **Instructions correlated with scenes.** If instruction `i` is used only on scenes from
   region `i`, accuracy measures scene difficulty. The counterfactual replay is the only
   fix; a weight inspection is not.
4. **Unpaired arms.** Two arms that ran different scenes (or different seeds) cannot be
   compared with McNemar, and their pair metric is meaningless.
5. **Silent denominator changes.** Rollouts that crashed, timed out, or landed outside the
   protocol boxes must be counted and labelled, never dropped. `Report.notes` records
   every exclusion; `strict_boxes=True` refuses unknown boxes outright.
6. **Twelve tests, one p-value.** Ablations multiply comparisons. Use
   `benjamini_hochberg` over the family and report adjusted values.

## 8. Assumptions

- Rollouts are independent across scenes; seeds within a scene are exchangeable.
- The instruction set `T` is fixed and each instruction has exactly one correct box.
- Landing box is a faithful proxy for task success. If the task is "place it *carefully*",
  box identity is not enough — use `eeval.judge` gates as well, or a judge with receipted
  annotation and an agreement statistic.
- The judge is correct. No amount of statistics repairs a judge that counts the wrong
  thing; this is why the kit keeps judging (`judge`) separate from aggregating
  (`counterfactual`) and makes both explicit.

## 9. Minimal checklist for a report

```
[ ] scene set is held out from training, and stated
[ ] every scene replayed under every instruction, same seeds
[ ] follow_acc AND follow_acc_itt with Wilson + exact intervals
[ ] pairs_both_follow with its denominator
[ ] landing distribution + bias test
[ ] failure breakdown: never landed / wrong box / correct
[ ] thresholds fixed before the run, verdict mechanical
[ ] exclusions and crashes counted, not deleted
```