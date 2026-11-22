#!/usr/bin/env python3
"""从已落盘的 json 生成 `docs/RESULTS.md`（不重跑任何训练/评测）。

README 里的数字必须能追溯到文件。手抄会出错，而且真出过一次：v1 基线的
`follow_acc` 其实是 56.4%，但口头只报了 ITT 28.9%，读起来像"接近一半"。
所以这张表由代码生成，`--check` 会在表落后于产物时直接失败。

用法::

    /usr/bin/python3 tools/make_tables.py            # 重写 docs/RESULTS.md
    /usr/bin/python3 tools/make_tables.py --check    # 只校验：文件齐不齐、字段全不全、表新不新

数据来源（全部是仓库里已有的产物）：

=============================  ==================================================
`rl/eval/lang_counterfactual*`  v1 双指令数据（指令与场景共线）
`rl/eval/lang_pair_*`           v2_pair 配对数据（同场景只换指令）
`rl/eval/lang_weight_curve.json` 语言通道权重曲线（rl/probe_lang_weight.py）
`rl/eval/residual_rl_summary.json` 残差 SAC 汇总（转自 docs/REPORT_RESIDUAL_RL.md）
=============================  ==================================================
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "rl" / "eval"
OUTPUT = ROOT / "docs" / "RESULTS.md"

BOX_A, BOX_B = "target_box", "target_box_b"

#: summary 字段缺一不可 —— 表格的每一列都要有据可依
REQUIRED_SUMMARY_KEYS = (
    "n_rollouts",
    "n_landed_in_some_box",
    "follow_correct",
    "follow_acc",
    "follow_acc_itt",
    "follow_wilson",
    "pairs_both_follow",
    "pairs_any_landed",
    "n_pairs",
    "neither_rate",
    "verdict",
)


@dataclass(frozen=True)
class LanguageRun:
    """一次语言条件评测产物：表格里的标签 + 它来自哪个文件。"""

    label: str
    filename: str

    @property
    def path(self) -> Path:
        return EVAL_DIR / self.filename


LANGUAGE_RUNS = (
    LanguageRun("v1 双指令数据 @1250 步", "lang_counterfactual_ckpt1250.json"),
    LanguageRun("v1 双指令数据 @5000 步", "lang_counterfactual.json"),
    LanguageRun("**v2_pair 配对 @3000 步**", "lang_pair_3000_interim.json"),
    LanguageRun("**v2_pair 配对 @5000 步（最终）**", "lang_pair_5000.json"),
)

WEIGHT_CURVE_FILE = "lang_weight_curve.json"
RESIDUAL_FILE = "residual_rl_summary.json"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_landings(rows: Iterable[dict]) -> dict:
    """每个箱子落了几次；`None` 键表示"没落进任何箱子"。"""
    counts: dict = {BOX_A: 0, BOX_B: 0, None: 0}
    for row in rows:
        landed_box = row.get("landed_box")
        counts[landed_box] = counts.get(landed_box, 0) + 1
    return counts


def as_percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _wilson_edges(summary: dict) -> tuple[float | None, float | None]:
    edges = summary.get("follow_wilson") or [None, None]
    return (edges[0], edges[1] if len(edges) > 1 else None)


# ---------------------------------------------------------------------------
# table 1 · counterfactual evaluation
# ---------------------------------------------------------------------------
def render_language_table() -> list[str]:
    lines = [
        "## 表 1 · 语言条件反事实评测"
        "（留出场景 2400-2409,2411-2419 × 2 seed × 2 指令 = 76 rollout）\n",
        "| 模型 | 落点 A / B / 未落箱 | follow_acc | Wilson 95% | ITT | 配对全对 | 判定 |",
        "|---|---|---|---|---|---|---|",
    ]
    for run in LANGUAGE_RUNS:
        if not run.path.exists():
            lines.append(f"| {run.label} | _缺 {run.filename}_ | — | — | — | — | — |")
            continue
        data = load_json(run.path)
        summary = data.get("summary") or {}
        landings = count_landings(data.get("rows") or [])
        lower, upper = _wilson_edges(summary)
        lines.append(
            "| {label} | {a} / {b} / {none} | {acc} ({correct}/{landed}) | "
            "[{lower}, {upper}] | {itt} | **{both} / {pairs}** | {verdict} |".format(
                label=run.label,
                a=landings.get(BOX_A, 0),
                b=landings.get(BOX_B, 0),
                none=landings.get(None, 0),
                acc=as_percent(summary.get("follow_acc")),
                correct=summary.get("follow_correct"),
                landed=summary.get("n_landed_in_some_box"),
                lower=as_percent(lower),
                upper=as_percent(upper),
                itt=as_percent(summary.get("follow_acc_itt")),
                both=summary.get("pairs_both_follow"),
                pairs=summary.get("n_pairs"),
                verdict=summary.get("verdict"),
            )
        )
    lines += [
        "",
        "**三个口径必须同时看，别只挑好看的**：",
        "- `follow_acc`：分母是**落进箱子的** rollout（任务失败的不算），衡量\"落对了没\"",
        "- `ITT`：分母是**全部** rollout（含没落箱的），衡量\"整体干成了没\"",
        "- `配对全对`：分母是**全部配对**（38 对），要求同场景两条指令都落对"
        "——这是\"语言真的起作用\"最硬的证据",
        "",
        "⚠️ v1 @5000 的 `follow_acc` 是 56.4%，看起来\"不算差\"，但它的落点是 **1 箱A / 38 箱B**——",
        "模型只是\"总往右放\"，凡指令是右的都算对，所以这个数没有意义；**配对全对 0/38** 才是真相。",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# table 2 · language-channel weight curve
# ---------------------------------------------------------------------------
def render_weight_table() -> list[str]:
    lines = [
        "## 表 2 · 语言通道权重曲线（`per_col`）\n",
        "`per_col` = 语言 2 列每列 RMS ÷ 其它 6 列每列 RMS。基座 `act_sim156_lang8` 是 6→8 补零加宽，",
        "所以这两列**初始恰好为 0**——它长不长大，直接回答\"语言通道有没有被用起来\"。\n",
    ]
    path = EVAL_DIR / WEIGHT_CURVE_FILE
    if not path.exists():
        return lines + [f"_缺 `rl/eval/{WEIGHT_CURVE_FILE}`，跑 `make weights` 生成_\n"]
    curve = load_json(path)
    lines += ["| 步数 | encoder | vae_encoder |", "|---|---|---|"]
    for row in curve.get("rows", []):
        lines.append(f"| {row['step']} | {row['encoder']:.5f} | {row['vae_encoder']:.5f} |")
    baseline = curve.get("baseline") or {}
    if baseline:
        lines.append(
            f"| 基座（0 步） | {baseline.get('encoder', 0):.5f} | "
            f"{baseline.get('vae_encoder', 0):.5f} |"
        )
    lines.append("")
    if curve.get("note"):
        lines.append(f"> {curve['note']}\n")
    return lines


# ---------------------------------------------------------------------------
# table 3 · residual RL (a negative result)
# ---------------------------------------------------------------------------
def render_residual_table() -> list[str]:
    lines = ["## 表 3 · 残差 SAC（负结果，同样是成果）\n"]
    path = EVAL_DIR / RESIDUAL_FILE
    if not path.exists():
        return lines + [f"_缺 `rl/eval/{RESIDUAL_FILE}`_\n"]
    summary = load_json(path)
    lines.append(
        f"> 数据来源：`{summary.get('source', 'docs/REPORT_RESIDUAL_RL.md')}`"
        "（held-out ep 100–129，训练池 0–49，不重叠）\n"
    )
    lines += ["| 配置 | 洁净成功率 | 旧口径成功率 | 备注 |", "|---|---|---|---|"]
    for name, row in (summary.get("rows") or {}).items():
        lines.append(
            f"| {name} | {as_percent(row.get('clean'))} | {as_percent(row.get('legacy'))} "
            f"| {row.get('note', '')} |"
        )
    lines.append("")
    for note in summary.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    return lines


def build_report() -> str:
    lines = [
        "# 结果表（自动生成，勿手改）\n",
        "> 由 `tools/make_tables.py` 从 `rl/eval/*.json` 生成。",
        "> 想更新：重新跑评测 → `make tables`。所有数字都能追到具体 json。\n",
    ]
    lines += render_language_table()
    lines += render_weight_table()
    lines += render_residual_table()
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# verification and entry point
# ---------------------------------------------------------------------------
def verify() -> int:
    problems = 0
    for run in LANGUAGE_RUNS:
        if not run.path.exists():
            print(f"✗ 缺文件 {run.filename}")
            problems += 1
            continue
        summary = load_json(run.path).get("summary") or {}
        missing = [key for key in REQUIRED_SUMMARY_KEYS if key not in summary]
        if missing:
            print(f"✗ {run.filename} 缺字段 {missing}")
            problems += 1
    for filename in (WEIGHT_CURVE_FILE, RESIDUAL_FILE):
        if not (EVAL_DIR / filename).exists():
            print(f"✗ 缺文件 {filename}")
            problems += 1
    fresh = OUTPUT.exists() and OUTPUT.read_text(encoding="utf-8") == build_report()
    if not fresh:
        print("✗ docs/RESULTS.md 不是最新（跑 make tables）")
        problems += 1
    print("✓ 结果表校验通过" if problems == 0 else f"✗ {problems} 处问题")
    return 0 if problems == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从 rl/eval/*.json 生成 docs/RESULTS.md")
    parser.add_argument("--check", action="store_true", help="只校验，不写文件")
    args = parser.parse_args(argv)

    if args.check:
        return verify()

    text = build_report()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ 已写入 {OUTPUT.relative_to(ROOT)}（{len(text)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())