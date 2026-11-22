"""Three-state receipts: ``ok`` / ``warn`` / ``fail``, and never promote a warn to ok.

The failure this prevents is mundane and fatal: a stage prints "grasp succeeded" while
the same log line says the box was shoved 6.85 mm.  The stage is a ``warn`` at best, and
a pipeline that reports ``ok`` there has manufactured a positive result.  Making the
three states a type, with the worst state rolling up automatically, removes the
opportunity to mess it up by hand.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

__all__ = ["STATUS", "Receipt", "Stage", "exit_code"]

STATUS = ("ok", "warn", "fail")
_SEVERITY = {"ok": 0, "warn": 1, "fail": 2}
_EXIT = {"ok": 0, "warn": 1, "fail": 2}


@dataclass
class Stage:
    """One named step with a status and the numbers that justify it."""

    name: str
    status: str = "ok"
    facts: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUS:
            raise ValueError(f"status must be one of {STATUS}, got {self.status!r}")

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "facts": self.facts, "note": self.note}


@dataclass
class Receipt:
    """An ordered list of stages plus the roll-up rule."""

    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, status: str, note: str = "", **facts) -> "Receipt":
        self.stages.append(Stage(name=name, status=status, facts=facts, note=note))
        return self

    def extend(self, stages: list[Stage]) -> "Receipt":
        self.stages.extend(stages)
        return self

    def merge(self, other: "Receipt") -> "Receipt":
        return self.extend(other.stages)

    def worst(self) -> str:
        return max((stage.status for stage in self.stages), key=_SEVERITY.__getitem__, default="ok")

    def failed(self) -> list[Stage]:
        return [stage for stage in self.stages if stage.status == "fail"]

    def warned(self) -> list[Stage]:
        return [stage for stage in self.stages if stage.status == "warn"]

    def require_ok(self) -> None:
        """Raise if anything is not ok -- for scripts that must not continue silently."""
        if self.worst() != "ok":
            bad = ", ".join(f"{stage.name}={stage.status}" for stage in self.stages
                            if stage.status != "ok")
            raise RuntimeError(f"receipt is not ok ({self.worst()}): {bad}")

    def to_dict(self) -> dict:
        return {"overall": self.worst(), "stages": [stage.as_dict() for stage in self.stages]}

    def to_json(self, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        lines = ["| stage | status | facts | note |", "|---|---|---|---|"]
        for stage in self.stages:
            facts = ", ".join(f"{key}={value}" for key, value in stage.facts.items())
            lines.append(f"| {stage.name} | **{stage.status}** | {facts} | {stage.note} |")
        lines.append(f"\noverall: **{self.worst()}**")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> "Receipt":
        receipt = cls()
        for stage in data.get("stages", []):
            receipt.stages.append(
                Stage(
                    name=stage["name"],
                    status=stage.get("status", "ok"),
                    facts=dict(stage.get("facts", {})),
                    note=stage.get("note", ""),
                )
            )
        return receipt


def exit_code(receipt: Receipt) -> int:
    """``0`` all green, ``1`` at least one warn, ``2`` at least one fail."""
    return _EXIT[receipt.worst()]