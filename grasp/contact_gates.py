"""The six contact gates: turn "the grasp looked fine" into six numbers with thresholds.

Why this module exists
----------------------
Fake grasping is easy and it looks convincing in a video: press the fingertips onto the
top of the box, squeeze, lift. The box comes up. Nothing about that is a grasp — the
load path is "push down", not "squeeze from both sides", and the moment the fingers open
the box is gone. A user watching the recording said exactly this in 2024-11, and every
claim in the report had to be re-derived from measurements afterwards.

So the criterion is not "did it lift" but six independent quantities, each with a
threshold that a physically honest grasp passes and a fake one fails:

===========  ==========================================  ==============  ==============================
gate         what it measures                            threshold       what a failure looks like
===========  ==========================================  ==============  ==============================
G1           finger↔box contacts during the descent      ``== 0``        the fingertip lands on the box lid
G2           worst sink of the box below the support     ``<= 0.5 mm``   the box is pressed into the table
G3           box tilt at the moment the grip closes      ``<= 5 deg``    the box is knocked over by the approach
G4           vertical rise while held                     ``>= 50 mm``    nothing was actually gripped
G5           real clearance between finger and box       ``|gap| <= 1.2`` a gap (loose) or a finger inside the box
G6           deepest finger-mesh vertex inside the box   ``<= 1.6 mm``   the finger shell spikes through the box
===========  ==========================================  ==============  ==============================

Two properties make this a criterion rather than a wish:

* **G4 exists to stop G1–G3 from being gamed by not touching the box at all.** Moving the
  grasp point 10 mm sideways gives a perfectly clean descent (0 contacts, 0 press) and a
  rise of 0 mm; without G4 that rollout scores four green lights.
* **Every threshold is a measured boundary, not a round number.** They sit where the
  physics changes behaviour: at ``gz = 0.020`` the finger pads graze the lid during the
  descent, at ``wrist_roll = -95 deg`` there are 6 contacts in 400 steps, and slowing the
  descent to 1150 steps presses the structure onto the lid long enough to topple the box.

The thresholds are duplicated in ``tools/probe_contact_audit.py``'s history and in
``docs/PHYSICS_FACTS.md``; this module is the single executable copy, and
``tests/test_contact_gates.py`` pins the recorded values from the accepted run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = [
    "GATES",
    "LEGACY_FLAGS",
    "AuditReport",
    "Gate",
    "GateVerdict",
    "evaluate",
    "threshold_of",
]

#: ``comparator`` semantics, applied over every key of the gate:
#: ``max`` -- every value must stay at or below the threshold
#: ``min`` -- every value must reach the threshold
#: ``abs_max`` -- the largest magnitude must stay at or below the threshold
COMPARATORS = ("max", "min", "abs_max")


@dataclass(frozen=True)
class Gate:
    """One criterion: which measurements it reads, where the line is, and what it catches."""

    key: str
    label: str
    measurements: tuple[str, ...]
    comparator: str
    threshold: float
    unit: str
    protects: str
    #: why *this* threshold and not a rounder number (kept next to the number on purpose)
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.comparator not in COMPARATORS:
            raise ValueError(f"comparator must be one of {COMPARATORS}, got {self.comparator!r}")
        if not self.measurements:
            raise ValueError(f"gate {self.key} reads no measurements")

    @property
    def symbol(self) -> str:
        return {"max": "<=", "min": ">=", "abs_max": "<="}[self.comparator]

    def compare(self, observed: float) -> bool:
        if self.comparator == "min":
            return observed >= self.threshold
        return observed <= self.threshold


GATES: tuple[Gate, ...] = (
    Gate(
        key="descend_contact_steps",
        label="G1 descent touches nothing",
        measurements=("descend_contact_steps",),
        comparator="max",
        threshold=0.0,
        unit="steps",
        protects="the fingertip landing on the box lid instead of sliding past it",
        rationale="zero is the semantic requirement ('not once'), not 'a little'",
    ),
    Gate(
        key="press_mm",
        label="G2 does not press into the support surface",
        measurements=("press_mm",),
        comparator="max",
        threshold=0.50,
        unit="mm",
        protects="the box being shoved down into the table and lifted out of soft contact",
        rationale="a resting pad already sinks ~0.2 mm into its contact; 0.5 mm is the "
                  "soft-contact budget, past it the box is being pushed",
    ),
    Gate(
        key="tilt_at_grasp_end_deg",
        label="G3 not knocked over by the approach",
        measurements=("tilt_at_grasp_end_deg",),
        comparator="max",
        threshold=5.0,
        unit="deg",
        protects="an approach that topples the box before it is gripped",
        rationale="a real gripper on a 15 mm edge shows 1-2 deg of natural skew; 5 deg is "
                  "the point where it is visible in the video. Measured at the moment the "
                  "grip closes, not during transport -- this arm's parallel lift/elbow/flex "
                  "axes tilt the tool 0.22 deg per mm of TCP height by construction",
    ),
    Gate(
        key="rise_mm",
        label="G4 really lifted it",
        measurements=("rise_mm",),
        comparator="min",
        threshold=50.0,
        unit="mm",
        protects="G1-G3 being satisfied by never touching the box (rise = 0)",
        rationale="moved the grasp point 10 mm sideways once: 0 contacts, 0 press, 0 rise "
                  "-- four green lights for not grasping at all",
    ),
    Gate(
        key="gap_mm",
        label="G5 no gap and no finger inside the box",
        measurements=("gap_fixed_mm", "gap_jaw_mm"),
        comparator="abs_max",
        threshold=1.2,
        unit="mm",
        protects="a loose grip (positive gap) or a fingertip inserted into the box (negative)",
        rationale="the pads are 2 mm TPU and compress ~1 mm when gripping; beyond that it is "
                  "either a gap the box can slide out of, or a finger through the wall",
    ),
    Gate(
        key="pen_depth_mm",
        label="G6 no deep mesh penetration",
        measurements=("pen_depth_mm",),
        comparator="max",
        threshold=1.6,
        unit="mm",
        protects="the finger *shell* (not the pad) spiking through the box volume",
        rationale="measured in millimetres of depth, not by counting vertices: the pads "
                  "legitimately sit ~1 mm inside the surface, and the previous shell design "
                  "reached 7.5 mm, which is what 'penetration' actually looks like",
    ),
)

#: Legacy flag names used by ``tools/probe_contact_audit.py``'s json output.
LEGACY_FLAGS: dict[str, str] = {
    "descend_contact_steps": "gate_descend_clean",
    "press_mm": "gate_press",
    "tilt_at_grasp_end_deg": "gate_tilt",
    "rise_mm": "gate_lift",
    "gap_mm": "gate_gap",
    "pen_depth_mm": "gate_pen",
}


@dataclass(frozen=True)
class GateVerdict:
    """A gate plus what was actually measured, and whether it passed."""

    gate: Gate
    observed: float | None
    passed: bool
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "key": self.gate.key,
            "label": self.gate.label,
            "observed": self.observed,
            "threshold": self.gate.threshold,
            "unit": self.gate.unit,
            "comparator": self.gate.symbol,
            "passed": self.passed,
            "note": self.note,
        }

    def describe(self) -> str:
        if self.observed is None:
            return f"{self.gate.label}: not measured ({self.note or 'missing value'})"
        unit = f" {self.gate.unit}" if self.gate.unit else ""
        return (
            f"{self.gate.label}: {self.observed:+.{3}f}{unit} {self.gate.symbol} "
            f"{self.gate.threshold}{unit}"
        )


@dataclass(frozen=True)
class AuditReport:
    """The six verdicts for one rollout, plus the roll-up the pipeline gates on."""

    verdicts: tuple[GateVerdict, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return all(verdict.passed for verdict in self.verdicts)

    @property
    def failed(self) -> tuple[str, ...]:
        return tuple(verdict.gate.key for verdict in self.verdicts if not verdict.passed)

    def by_key(self) -> dict[str, GateVerdict]:
        return {verdict.gate.key: verdict for verdict in self.verdicts}

    def flag_dict(self) -> dict[str, bool]:
        """The ``gate_*`` booleans the audit json has always carried."""
        return {LEGACY_FLAGS[key]: verdict.passed for key, verdict in self.by_key().items()}

    def as_dict(self) -> dict:
        return {
            "ok": self.ok,
            "failed": list(self.failed),
            "gates": [verdict.as_dict() for verdict in self.verdicts],
        }

    def to_markdown(self) -> str:
        lines = [
            "| gate | measured | verdict | what a failure means |",
            "|---|---|---|---|",
        ]
        for verdict in self.verdicts:
            observed = "—" if verdict.observed is None else (
                f"{verdict.observed:+.3f} {verdict.gate.unit}".strip()
            )
            outcome = "pass" if verdict.passed else "**fail**"
            if not verdict.passed and verdict.note:
                outcome += f" ({verdict.note})"
            lines.append(
                f"| **{verdict.gate.label}** | {observed} "
                f"({verdict.gate.symbol} {verdict.gate.threshold} {verdict.gate.unit}) "
                f"| {outcome} | {verdict.gate.protects} |"
            )
        lines.append(f"\noverall: **{'ok' if self.ok else 'fail'}**")
        return "\n".join(lines)


def threshold_of(key: str) -> float:
    """Threshold for a gate key -- used by the CLI's argument defaults and help text."""
    for gate in GATES:
        if gate.key == key:
            return gate.threshold
    raise KeyError(f"unknown gate {key!r}; known: {[gate.key for gate in GATES]}")


def _observed(gate: Gate, measurements: Mapping[str, float | None]) -> float | None:
    """Reduce a gate's measurements to the single number the comparator sees."""
    values = []
    for name in gate.measurements:
        if name not in measurements:
            raise KeyError(
                f"gate {gate.key} needs measurement {name!r}; got {sorted(measurements)}"
            )
        value = measurements[name]
        if value is None:
            return None
        values.append(abs(float(value)) if gate.comparator == "abs_max" else float(value))
    return max(values) if gate.comparator in ("max", "abs_max") else min(values)


def evaluate(measurements: Mapping[str, float | None]) -> AuditReport:
    """Judge one rollout from its measured numbers.

    ``measurements`` maps the names in :attr:`Gate.measurements` to values; ``None`` means
    "this run did not measure it", which fails the gate instead of being skipped. A missing
    key is a programming error and raises, because silently dropping a criterion is how a
    fake grasp becomes a pass.
    """
    verdicts = []
    for gate in GATES:
        observed = _observed(gate, measurements)
        if observed is None:
            verdicts.append(GateVerdict(gate, None, False, note="not measured"))
            continue
        verdicts.append(GateVerdict(gate, observed, gate.compare(observed)))
    return AuditReport(verdicts=tuple(verdicts))


def summarise(reports: Sequence[AuditReport]) -> dict:
    """Per-gate failure counts over a batch -- which criterion is actually costing you."""
    by_gate = {gate.key: 0 for gate in GATES}
    for report in reports:
        for key in report.failed:
            by_gate[key] = by_gate.get(key, 0) + 1
    return {
        "n": len(reports),
        "n_ok": sum(1 for report in reports if report.ok),
        "by_gate": by_gate,
    }