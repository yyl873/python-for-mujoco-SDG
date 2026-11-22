"""Counterfactual paired evaluation: does the landing side follow the instruction?

The protocol
------------
A language-conditioned policy is evaluated by replaying *the same scenes* twice, once
per instruction, and asking whether the outcome tracks the instruction.  Success rate
alone cannot answer that: a policy that always drops the box on the right scores 100%
on every rollout whose instruction happens to say "right".

Every report therefore carries three estimands with different denominators, because
picking one of them is how evaluation results get inflated:

======================  ==========================  ==================================
estimand                denominator                 question it answers
======================  ==========================  ==================================
``follow_acc``          rollouts that landed        when it did put the box down, was
                        in *some* box               it the instructed box?
``follow_acc_itt``      **every** rollout           did the whole task succeed?
                        (intention-to-treat)
``pairs_both_follow``   complete (scene, seed)      did *both* instructions of the same
                        pairs                       scene land where they were told?
======================  ==========================  ==================================

The third one is the only one a degenerate policy cannot fake: always dropping right
gives ``pairs_both_follow == 0`` no matter how good ``follow_acc`` looks.

Chance baselines are reported next to the estimates (``chance_baselines``) so a reader
never has to reconstruct them.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .stats import Proportion, binom_two_sided, mcnemar_exact, proportion

__all__ = [
    "Protocol",
    "Report",
    "Rollout",
    "Verdict",
    "aggregate",
    "aggregate_pairs",
    "analyze",
    "analyze_rows",
    "verdict",
]

FOLLOWED = "followed"
NOT_LANDED = "not_landed"
WRONG_BOX = "wrong_box"
UNKNOWN_BOX = "unknown_box"


# --------------------------------------------------------------------------------------
# protocol and data types
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Protocol:
    """What counts as "following the instruction" for one evaluation run.

    ``instructions`` is ordered: the index of an instruction is what the policy receives
    as its language condition (a one-hot slot, a task string, ...).  ``expected_box`` maps
    every instruction to the box name the rollout report should contain on success.
    """

    instructions: Sequence[str]
    expected_box: Mapping[str, str]
    min_follow_acc: float = 0.80
    min_wilson_lower: float = 0.50
    min_pairs_both_follow: int = 1
    strict_boxes: bool = True

    def __post_init__(self) -> None:
        if len(self.instructions) < 2:
            raise ValueError(
                f"a counterfactual protocol needs >= 2 instructions, got {len(self.instructions)}"
            )
        if len(set(self.instructions)) != len(self.instructions):
            raise ValueError("instructions must be unique")
        missing = [t for t in self.instructions if t not in self.expected_box]
        if missing:
            raise ValueError(f"expected_box is missing an entry for {missing!r}")

    # -- helpers -----------------------------------------------------------------
    @property
    def boxes(self) -> tuple[str, ...]:
        return tuple(self.expected_box[t] for t in self.instructions)

    @property
    def chance(self) -> float:
        """Follow-accuracy a policy reaches by ignoring the instruction (uniform guess)."""
        return 1.0 / len(self.instructions)

    def box_for(self, instruction_index: int) -> str:
        return self.expected_box[self.instructions[instruction_index]]

    def instruction_index(self, text: str) -> int:
        try:
            return list(self.instructions).index(text)
        except ValueError as exc:
            raise ValueError(f"unknown instruction {text!r}; known: {list(self.instructions)!r}") from exc

    def as_dict(self) -> dict:
        return {
            "instructions": list(self.instructions),
            "expected_box": dict(self.expected_box),
            "thresholds": {
                "min_follow_acc": self.min_follow_acc,
                "min_wilson_lower": self.min_wilson_lower,
                "min_pairs_both_follow": self.min_pairs_both_follow,
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "Protocol":
        return cls(
            instructions=tuple(data["instructions"]),
            expected_box=dict(data["expected_box"]),
            strict_boxes=bool(data.get("strict_boxes", True)),
        )

    @classmethod
    def from_json_file(cls, path) -> "Protocol":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


@dataclass(frozen=True)
class Rollout:
    """One episode playback under one instruction."""

    episode_id: int
    seed: int
    instruction: int
    landed_box: str | None = None
    steps: int | None = None
    seconds: float | None = None
    reason: str | None = None  # free-form, e.g. "timeout" / "early_terminate"

    @property
    def pair_key(self) -> tuple[int, int]:
        """Rollouts sharing this key are the two halves of one counterfactual pair."""
        return (self.episode_id, self.seed)

    def landed(self, protocol: Protocol) -> bool:
        return self.landed_box is not None

    def followed(self, protocol: Protocol) -> bool:
        return self.landed_box == protocol.box_for(self.instruction)

    @classmethod
    def from_dict(
        cls,
        row: Mapping,
        *,
        instruction_key: str = "ix",
        box_key: str = "landed_box",
        step_key: str = "steps",
        time_key: str = "wall_s",
        reason_key: str = "reason",
    ) -> "Rollout":
        return cls(
            episode_id=int(row["episode_id"]),
            seed=int(row["seed"]),
            instruction=int(row[instruction_key]),
            landed_box=row.get(box_key),
            steps=None if row.get(step_key) is None else int(row[step_key]),
            seconds=None if row.get(time_key) is None else float(row[time_key]),
            reason=row.get(reason_key),
        )


@dataclass(frozen=True)
class Verdict:
    """A pre-registered decision, with the checks that produced it."""

    label: str
    passed: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "passed": self.passed,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
        }


# --------------------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------------------
@dataclass
class Report:
    """Everything a counterfactual run should publish, in one object."""

    protocol: Protocol
    n_rollouts: int
    n_landed: int
    n_followed: int
    n_pairs: int
    complete_pairs: int
    pairs_both_follow: int
    pairs_any_landed: int
    pairs_neither_landed: int
    pattern_counts: Mapping[str, int]
    per_instruction: Mapping[str, dict]
    outcome_counts: Mapping[str, int]
    landed_box_counts: Mapping[str, int]
    side_bias_p: float | None
    discordant_p: float | None
    chance_baselines: Mapping[str, float]
    verdict: Verdict
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- derived estimates --------------------------------------------------------
    @property
    def follow(self) -> Proportion:
        return proportion(self.n_followed, self.n_landed)

    @property
    def itt(self) -> Proportion:
        return proportion(self.n_followed, self.n_rollouts)

    @property
    def follow_acc(self) -> float | None:
        return self.follow.point

    @property
    def follow_acc_itt(self) -> float | None:
        return self.itt.point

    @property
    def follow_wilson(self) -> tuple[float, float]:
        return self.follow.wilson

    # -- serialisation ------------------------------------------------------------
    def as_dict(self, ndigits: int = 4) -> dict:
        return {
            "protocol": self.protocol.as_dict(),
            "n_rollouts": self.n_rollouts,
            "n_landed_in_some_box": self.n_landed,
            "follow_correct": self.n_followed,
            "follow_acc": _round(self.follow_acc, ndigits),
            "follow_acc_itt": _round(self.follow_acc_itt, ndigits),
            "follow_wilson": [round(v, ndigits) for v in self.follow_wilson],
            "follow_acc_clopper_pearson": [round(v, ndigits) for v in self.follow.clopper_pearson],
            "binom_p_vs_chance": _round(self.binom_p, 6),
            "n_pairs": self.n_pairs,
            "complete_pairs": self.complete_pairs,
            "pairs_both_follow": self.pairs_both_follow,
            "pairs_any_landed": self.pairs_any_landed,
            "pairs_neither_landed": self.pairs_neither_landed,
            "pair_pattern_counts": dict(self.pattern_counts),
            "per_instruction": {k: dict(v) for k, v in self.per_instruction.items()},
            "outcome_counts": dict(self.outcome_counts),
            "landed_box_counts": dict(self.landed_box_counts),
            "side_bias_p": _round(self.side_bias_p, 6),
            "discordant_mcnemar_p": _round(self.discordant_p, 6),
            "chance_baselines": dict(self.chance_baselines),
            "verdict": self.verdict.as_dict(),
            "notes": list(self.notes),
        }

    def to_json(self, ndigits: int = 4, indent: int = 1) -> str:
        return json.dumps(self.as_dict(ndigits), ensure_ascii=False, indent=indent, sort_keys=False)

    @property
    def binom_p(self) -> float | None:
        if not self.n_landed:
            return None
        return binom_two_sided(self.n_followed, self.n_landed, self.protocol.chance)

    def to_markdown(self) -> str:
        lines = [
            f"**Verdict: {self.verdict.label}** ({'pass' if self.verdict.passed else 'fail'})",
            "",
            "| estimand | value | denominator |",
            "|---|---|---|",
            f"| follow_acc | {_pct(self.follow_acc)} | {self.n_landed} rollouts that landed |",
            f"| follow_acc (ITT) | {_pct(self.follow_acc_itt)} | {self.n_rollouts} rollouts |",
            f"| pairs both followed | {self.pairs_both_follow} / {self.complete_pairs} "
            f"({_pct(self.pairs_both_follow / self.complete_pairs if self.complete_pairs else None)})"
            f" | complete pairs |",
            f"| Wilson 95% of follow_acc | {_pct(self.follow_wilson[0])} - "
            f"{_pct(self.follow_wilson[1])} | - |",
            f"| p vs chance ({self.protocol.chance:.2f}) | {_fmt_p(self.binom_p)} | - |",
            f"| side bias p (landing counts {dict(self.landed_box_counts)}) | "
            f"{_fmt_p(self.side_bias_p)} | - |",
        ]
        if self.notes:
            lines.extend(["", "Notes:"] + [f"- {note}" for note in self.notes])
        return "\n".join(lines)


def _round(value: float | None, ndigits: int) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return "< 1e-6" if value < 1e-6 else f"{value:.4g}"


# --------------------------------------------------------------------------------------
# core analysis
# --------------------------------------------------------------------------------------
def analyze(rollouts: Iterable[Rollout], protocol: Protocol) -> Report:
    """Run the full counterfactual analysis.

    Raises ``ValueError`` on data that would silently corrupt the denominators:
    duplicate ``(episode_id, seed, instruction)`` rows, unknown instruction indices,
    and landing boxes outside ``protocol.expected_box`` (unless ``strict_boxes=False``).
    """
    rollouts = list(rollouts)
    n_instructions = len(protocol.instructions)
    known_boxes = set(protocol.boxes)

    seen: dict[tuple[int, int, int], Rollout] = {}
    for rollout in rollouts:
        if not 0 <= rollout.instruction < n_instructions:
            raise ValueError(
                f"rollout {rollout.pair_key} has instruction index {rollout.instruction}, "
                f"outside 0..{n_instructions - 1}"
            )
        if rollout.landed_box is not None and protocol.strict_boxes and rollout.landed_box not in known_boxes:
            raise ValueError(
                f"rollout {rollout.pair_key} landed in {rollout.landed_box!r}, "
                f"which is not one of the protocol boxes {sorted(known_boxes)!r}"
            )
        key = (*rollout.pair_key, rollout.instruction)
        if key in seen:
            raise ValueError(
                f"duplicate rollout for (episode={rollout.episode_id}, seed={rollout.seed}, "
                f"instruction={rollout.instruction}); each pair must be replayed exactly once"
            )
        seen[key] = rollout

    outcome_counts: Counter[str] = Counter()
    landed_box_counts: Counter[str] = Counter()
    per_instruction: dict[str, dict] = {}

    for index, text in enumerate(protocol.instructions):
        rows = [r for r in rollouts if r.instruction == index]
        landed = [r for r in rows if r.landed(protocol)]
        followed = sum(1 for r in landed if r.followed(protocol))
        estimate = proportion(followed, len(landed))
        per_instruction[text] = {
            "expected_box": protocol.box_for(index),
            "n_rollouts": len(rows),
            "n_landed": len(landed),
            "n_followed": followed,
            "follow_acc": _round(estimate.point, 4),
            "wilson": [round(v, 4) for v in estimate.wilson],
            "itt": _round(followed / len(rows), 4) if rows else None,
        }

    for rollout in rollouts:
        if not rollout.landed(protocol):
            outcome_counts[NOT_LANDED] += 1
            if rollout.reason:
                outcome_counts[f"{NOT_LANDED}:{rollout.reason}"] += 1
        else:
            landed_box_counts[rollout.landed_box] += 1
            outcome_counts[FOLLOWED if rollout.followed(protocol) else WRONG_BOX] += 1
    if not protocol.strict_boxes:
        unknowns = {box: c for box, c in landed_box_counts.items() if box not in known_boxes}
        outcome_counts[UNKNOWN_BOX] = sum(unknowns.values())

    pairs: dict[tuple[int, int], list[Rollout]] = defaultdict(list)
    for rollout in rollouts:
        pairs[rollout.pair_key].append(rollout)

    complete = [rows for rows in pairs.values() if len(rows) == len(protocol.instructions)]
    incomplete = len(pairs) - len(complete)
    both = any_landed = neither = 0
    pattern_counts: Counter[str] = Counter()
    for rows in complete:
        ordered = sorted(rows, key=lambda r: r.instruction)
        followed_vector = tuple(r.followed(protocol) for r in ordered)
        landed_vector = tuple(r.landed(protocol) for r in ordered)
        pattern_counts["".join("T" if v else "F" for v in followed_vector)] += 1
        if all(followed_vector):
            both += 1
        if any(landed_vector):
            any_landed += 1
        else:
            neither += 1

    n_landed = sum(landed_box_counts.values())
    n_followed = sum(1 for r in rollouts if r.landed(protocol) and r.followed(protocol))

    # A policy that ignores the instruction collapses onto one box, so the landing
    # counts themselves carry evidence: test the most-used box against a fair split.
    side_bias_p = None
    distinct_boxes = sorted(set(protocol.boxes))
    if n_landed and len(distinct_boxes) > 1:
        dominant = max(landed_box_counts.get(box, 0) for box in distinct_boxes)
        side_bias_p = binom_two_sided(dominant, n_landed, 1.0 / len(distinct_boxes))

    discordant_p = None
    if len(protocol.instructions) == 2 and complete:
        only_first = pattern_counts.get("TF", 0)
        only_second = pattern_counts.get("FT", 0)
        discordant_p = mcnemar_exact(only_first, only_second)

    box_share = {}
    for index, text in enumerate(protocol.instructions):
        box = protocol.box_for(index)
        box_share[box] = box_share.get(box, 0) + sum(
            1 for r in rollouts if r.instruction == index
        )
    always_single_box = max(box_share.values()) / len(rollouts) if rollouts else 0.0

    chance_baselines = {
        "follow_acc_uniform_guess": round(protocol.chance, 4),
        "follow_acc_always_single_box": round(always_single_box, 4),
        "pairs_both_follow_always_single_box": 0.0,
    }

    notes: list[str] = []
    if incomplete:
        notes.append(
            f"{incomplete} of {len(pairs)} pairs are incomplete (missing instruction rollouts) "
            "and are excluded from the pair metrics"
        )
    if n_landed and n_landed < len(rollouts):
        notes.append(
            f"{len(rollouts) - n_landed} rollouts never landed in any box; they are excluded "
            "from follow_acc but counted in the ITT estimate"
        )

    report = Report(
        protocol=protocol,
        n_rollouts=len(rollouts),
        n_landed=n_landed,
        n_followed=n_followed,
        n_pairs=len(pairs),
        complete_pairs=len(complete),
        pairs_both_follow=both,
        pairs_any_landed=any_landed,
        pairs_neither_landed=neither,
        pattern_counts=dict(pattern_counts),
        per_instruction=per_instruction,
        outcome_counts=dict(outcome_counts),
        landed_box_counts=dict(landed_box_counts),
        side_bias_p=side_bias_p,
        discordant_p=discordant_p,
        chance_baselines=chance_baselines,
        verdict=Verdict(label="", passed=False, checks={}),
        notes=tuple(notes),
    )
    report.verdict = _pre_registered_verdict(report)
    return report


def _pre_registered_verdict(report: Report) -> Verdict:
    protocol = report.protocol
    acc = report.follow_acc or 0.0
    lower = report.follow_wilson[0]
    checks = {
        f"follow_acc >= {protocol.min_follow_acc}": acc >= protocol.min_follow_acc,
        f"wilson_lower > {protocol.min_wilson_lower}": lower > protocol.min_wilson_lower,
        f"pairs_both_follow >= {protocol.min_pairs_both_follow}": (
            report.pairs_both_follow >= protocol.min_pairs_both_follow
        ),
    }
    reasons = [name for name, ok in checks.items() if not ok]

    if all(checks.values()):
        return Verdict("instruction followed", True, checks, ())
    if checks[f"follow_acc >= {protocol.min_follow_acc}"] and checks[
        f"wilson_lower > {protocol.min_wilson_lower}"
    ]:
        return Verdict(
            "landing tracks the instruction, but no pair is fully correct "
            "(the counterfactual pairing is broken or the scenes differ)",
            False,
            checks,
            tuple(reasons),
        )
    return Verdict(
        "instruction ignored (landing is indistinguishable from guessing)",
        False,
        checks,
        tuple(reasons),
    )


def analyze_rows(rows: Iterable[Mapping], protocol: Protocol, **kwargs) -> Report:
    """``analyze`` for plain dicts (the shape every rollout log uses)."""
    return analyze((Rollout.from_dict(row, **kwargs) for row in rows), protocol)


# --------------------------------------------------------------------------------------
# legacy dict-shaped API (kept so older scripts and notebooks keep working)
# --------------------------------------------------------------------------------------
def aggregate(
    rows: Sequence[Mapping],
    task_order: Sequence[str],
    box_of: Mapping[int, str] | None = None,
    expect_map: Mapping[str, str] | None = None,
) -> dict:
    """Dict-returning wrapper around :func:`analyze`.

    ``box_of`` was the original way to map instruction index -> box; ``expect_map``
    superseded it because it keys on the instruction text.  Both are accepted.
    """
    expected = dict(expect_map) if expect_map else {
        task_order[int(index)]: box for index, box in (box_of or {}).items()
    }
    protocol = Protocol(instructions=tuple(task_order), expected_box=expected, strict_boxes=False)
    report = analyze_rows(rows, protocol)
    return {
        "n_rollouts": report.n_rollouts,
        "n_landed_in_some_box": report.n_landed,
        "follow_correct": report.n_followed,
        "follow_acc": None if report.follow_acc is None else round(report.follow_acc, 4),
        "follow_acc_itt": None if report.follow_acc_itt is None else round(report.follow_acc_itt, 4),
        "follow_wilson": [round(v, 4) for v in report.follow_wilson],
        "binom_p": None if report.binom_p is None else round(report.binom_p, 6),
    }


def aggregate_pairs(
    rows: Sequence[Mapping], task_order: Sequence[str], expect_map: Mapping[str, str]
) -> dict:
    """Pair-level counts only (see :class:`Report` for the full picture)."""
    protocol = Protocol(
        instructions=tuple(task_order), expected_box=dict(expect_map), strict_boxes=False
    )
    report = analyze_rows(rows, protocol)
    return {
        "n_pairs": report.n_pairs,
        "pairs_both_follow": report.pairs_both_follow,
        "pairs_any_landed": report.pairs_any_landed,
    }


def verdict(summary: Mapping, acc_min: float = 0.80, wilson_lo_min: float = 0.50) -> str:
    """Pre-registered decision from a summary dict (dict-shaped equivalent of ``Verdict``)."""
    acc = summary.get("follow_acc") or 0.0
    lower = (summary.get("follow_wilson") or [0.0])[0]
    pairs = summary.get("pairs_both_follow") or 0
    if acc >= acc_min and lower > wilson_lo_min and pairs > 0:
        return "instruction followed"
    if acc >= acc_min and lower > wilson_lo_min:
        return "landing tracks the instruction, but no pair is fully correct"
    return "instruction ignored"