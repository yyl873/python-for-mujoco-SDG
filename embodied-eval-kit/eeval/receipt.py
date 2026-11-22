"""三态回执：ok / warn / fail —— 不许把 warn 写成 ok。

真实教训（本项目）：某流水线脚本自判"抓取成功"，同时打印"约束1 ❌ 药盒被撞动 6.85 mm"。
正确回执是 warn 并指向那条约束，而不是 ok。把 warn 压成 ok 等于伪造结论。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

STATUS = ("ok", "warn", "fail")


@dataclass
class Stage:
    name: str
    status: str = "ok"
    facts: dict = field(default_factory=dict)
    note: str = ""

    def __post_init__(self):
        if self.status not in STATUS:
            raise ValueError(f"status 只能是 {STATUS}，收到 {self.status!r}")


@dataclass
class Receipt:
    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, status: str, **facts) -> "Receipt":
        self.stages.append(Stage(name=name, status=status, facts=facts))
        return self

    def worst(self) -> str:
        order = {"ok": 0, "warn": 1, "fail": 2}
        return max((s.status for s in self.stages), key=lambda x: order[x], default="ok")

    def to_json(self) -> str:
        return json.dumps({"overall": self.worst(),
                           "stages": [s.__dict__ for s in self.stages]},
                          ensure_ascii=False, indent=1)

    def to_markdown(self) -> str:
        lines = ["| 段 | 状态 | 关键量 |", "|---|---|---|"]
        for s in self.stages:
            facts = ", ".join(f"{k}={v}" for k, v in s.facts.items())
            lines.append(f"| {s.name} | **{s.status}** | {facts} |")
        lines.append(f"\n总体：**{self.worst()}**")
        return "\n".join(lines)


def exit_code(receipt: Receipt) -> int:
    """0=全绿 / 1=有 warn / 2=有 fail（与 shell 约定一致）。"""
    return {"ok": 0, "warn": 1, "fail": 2}[receipt.worst()]
