"""Turn "the grasp succeeded" into a handful of countable gates.

Why this module exists
----------------------
Scoring a rollout as "the box went up 50 mm" rewards a policy that squashes the box
deeper into the table and then lifts: the lift trace looks great while the physical
interaction is garbage.  In this project that exact reward hole moved a residual-RL
policy from 32.2% to 13.3% once tilt and penetration were added to the criteria.

So a verdict here is a *vector* of gates, each with its own threshold, and the returned
dictionary always lists **every** failed gate rather than the first one.  A failure
distribution ("9 of 12 rollouts died on penetration") says far more than a success rate.

The judge is deliberately independent of the aggregator in :mod:`eeval.counterfactual`:
this module answers "was this single rollout a clean success?", the other answers "does
the policy use the instruction?".  Swapping in your own gates does not touch the stats.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

__all__ = ["CleanCfg", "Episode", "GateResult", "judge_episode", "judge_many"]

# Cheat patterns that a naive success criterion would reward.
CHEAT_LIFT_THEN_DIG = "lifted only after digging into the object"
CHEAT_INSTANT_LIFT = "lift peaked but was never held"
CHEAT_TILTED_GRASP = "held at an angle"


@dataclass(frozen=True)
class CleanCfg:
    """Thresholds for the clean-success criteria (millimetres and degrees)."""

    lift_min_mm: float = 50.0
    lift_consecutive: int = 5
    tilt_max_deg: float = 5.0
    penetration_max_mm: float = 0.5
    require_release: bool = False

    def __post_init__(self) -> None:
        if self.lift_min_mm < 0 or self.tilt_max_deg < 0 or self.penetration_max_mm < 0:
            raise ValueError("thresholds must be non-negative")
        if self.lift_consecutive < 1:
            raise ValueError(f"lift_consecutive must be >= 1, got {self.lift_consecutive}")


@dataclass(frozen=True)
class GateResult:
    """One criterion, evaluated."""

    name: str
    passed: bool
    observed: float
    threshold: float
    kind: str  # "min" (must reach) or "max" (must stay below)

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        arrow = ">=" if self.kind == "min" else "<="
        return f"{self.name}: {self.observed:.3f} {arrow} {self.threshold}"


@dataclass
class Episode:
    """One rollout's physical traces, sampled at the same rate and the same length."""

    lift_mm: Sequence[float] = field(default_factory=list)
    tilt_deg: Sequence[float] = field(default_factory=list)
    penetration_mm: Sequence[float] = field(default_factory=list)
    landed_box: str | None = None
    episode_id: int | None = None

    def __post_init__(self) -> None:
        lengths = {len(self.lift_mm), len(self.tilt_deg), len(self.penetration_mm)}
        non_empty = {length for length in lengths if length}
        if len(non_empty) > 1:
            raise ValueError(
                f"traces must have the same length, got lift={len(self.lift_mm)}, "
                f"tilt={len(self.tilt_deg)}, penetration={len(self.penetration_mm)}"
            )
        for name in ("lift_mm", "tilt_deg", "penetration_mm"):
            for value in getattr(self, name):
                if not math.isfinite(value):
                    raise ValueError(f"{name} contains a non-finite value: {value!r}")


def _longest_run(values: Sequence[float], minimum: float) -> int:
    best = current = 0
    for value in values:
        current = current + 1 if value >= minimum else 0
        best = max(best, current)
    return best


def judge_episode(episode: Episode, cfg: CleanCfg | None = None) -> dict:
    """Evaluate every gate; return the verdict plus the raw extremes.

    The returned dict keeps the historical keys (``ok``, ``reasons``, ``max_lift_mm``,
    ``max_tilt_deg``, ``max_penetration_mm``, ``consecutive_ok``) and adds ``gates``
    (per-criterion detail) and ``cheat_flags``.
    """
    cfg = cfg or CleanCfg()
    lift = list(episode.lift_mm) or [0.0]
    tilt = list(episode.tilt_deg) or [0.0]
    penetration = list(episode.penetration_mm) or [0.0]

    max_lift = max(lift)
    max_tilt = max(tilt)
    max_penetration = max(penetration)
    longest_run = _longest_run(lift, cfg.lift_min_mm)

    gates = [
        GateResult("max_lift_mm", max_lift >= cfg.lift_min_mm, max_lift, cfg.lift_min_mm, "min"),
        GateResult(
            "lift_consecutive",
            longest_run >= cfg.lift_consecutive,
            float(longest_run),
            float(cfg.lift_consecutive),
            "min",
        ),
        GateResult("max_tilt_deg", max_tilt <= cfg.tilt_max_deg, max_tilt, cfg.tilt_max_deg, "max"),
        GateResult(
            "max_penetration_mm",
            max_penetration <= cfg.penetration_max_mm,
            max_penetration,
            cfg.penetration_max_mm,
            "max",
        ),
    ]
    if cfg.require_release:
        gates.append(
            GateResult(
                "landed_in_box",
                1.0 if episode.landed_box else 0.0,
                1.0 if episode.landed_box else 0.0,
                1.0,
                "min",
            )
        )

    failed = [gate for gate in gates if not gate.passed]
    cheat_flags: list[str] = []
    if max_lift >= cfg.lift_min_mm and max_penetration > cfg.penetration_max_mm:
        cheat_flags.append(CHEAT_LIFT_THEN_DIG)
    if max_lift >= cfg.lift_min_mm and longest_run < cfg.lift_consecutive:
        cheat_flags.append(CHEAT_INSTANT_LIFT)
    if max_tilt > cfg.tilt_max_deg:
        cheat_flags.append(CHEAT_TILTED_GRASP)

    return {
        "ok": not failed,
        "reasons": [gate.describe() for gate in failed],
        "failed_gates": [gate.name for gate in failed],
        "gates": [gate.as_dict() for gate in gates],
        "cheat_flags": cheat_flags,
        "max_lift_mm": round(max_lift, 3),
        "max_tilt_deg": round(max_tilt, 3),
        "max_penetration_mm": round(max_penetration, 3),
        "consecutive_ok": longest_run >= cfg.lift_consecutive,
        "lift_consecutive_max": longest_run,
        "landed_box": episode.landed_box,
        "n_samples": len(lift),
    }


def judge_many(episodes: Iterable[Episode], cfg: CleanCfg | None = None) -> dict:
    """Judge a batch and summarise *why* the failures failed.

    Returns ``{"n", "n_ok", "success_rate", "by_gate": {...}, "by_cheat": {...},
    "per_episode": [...]}``; ``by_gate`` counts how many episodes failed each criterion,
    which is the number to look at before touching a policy.
    """
    cfg = cfg or CleanCfg()
    verdicts = [judge_episode(episode, cfg) for episode in episodes]
    by_gate: dict[str, int] = {gate: 0 for gate in ("max_lift_mm", "lift_consecutive",
                                                    "max_tilt_deg", "max_penetration_mm")}
    if cfg.require_release:
        by_gate["landed_in_box"] = 0
    by_cheat: dict[str, int] = {}
    for verdict in verdicts:
        for gate in verdict["failed_gates"]:
            by_gate[gate] = by_gate.get(gate, 0) + 1
        for flag in verdict["cheat_flags"]:
            by_cheat[flag] = by_cheat.get(flag, 0) + 1
    n = len(verdicts)
    n_ok = sum(1 for verdict in verdicts if verdict["ok"])
    return {
        "n": n,
        "n_ok": n_ok,
        "success_rate": round(n_ok / n, 4) if n else None,
        "by_gate": by_gate,
        "by_cheat": by_cheat,
        "per_episode": verdicts,
    }