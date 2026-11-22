#!/usr/bin/env python3
"""tools/make_tables.py —— 从已落盘的 json 自动生成结果表（不重跑任何训练/评测）

为什么要有它：README 里的数字必须能**追溯到文件**。手抄会出错，出过一次——
v1 基线的 `follow_acc` 实际是 56.4%，但口头只报了 ITT 28.9%，容易被当成"接近一半"。

用法：
    /usr/bin/python3 tools/make_tables.py            # 生成 docs/RESULTS.md 并打印摘要
    /usr/bin/python3 tools/make_tables.py --check    # 只校验：文件齐不齐、字段全不全、表是否落后

数据来源（全部是仓库里已有的产物）：
    rl/eval/lang_counterfactual.json           v1 双指令数据 5000 步
    rl/eval/lang_counterfactual_ckpt1250.json  v1 1250 步
    rl/eval/lang_pair_3000_interim.json        v2_pair 3000 步
    rl/eval/lang_pair_5000.json                v2_pair 5000 步（最终判定）
    rl/eval/lang_weight_curve.json             语言通道权重曲线（rl/probe_lang_weight.py 产物）
    rl/eval/residual_rl_summary.json           残差 SAC 的汇总（转自 docs/REPORT_RESIDUAL_RL.md）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EV = ROOT / "rl" / "eval"
OUT = ROOT / "docs" / "RESULTS.md"

LANG_RUNS = [
    ("v1 双指令数据 @1250 步", "lang_counterfactual_ckpt1250.json"),
    ("v1 双指令数据 @5000 步", "lang_counterfactual.json"),
    ("**v2_pair 配对 @3000 步**", "lang_pair_3000_interim.json"),
    ("**v2_pair 配对 @5000 步（最终）**", "lang_pair_5000.json"),
]
NEED_SUMMARY = ("n_rollouts", "n_landed_in_some_box", "follow_correct", "follow_acc",
                "follow_acc_itt", "follow_wilson", "pairs_both_follow",
                "pairs_any_landed", "n_pairs", "neither_rate", "verdict")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def landed_counts(rows: list[dict]) -> dict[str, int]:
    c = {"target_box": 0, "target_box_b": 0, None: 0}
    for r in rows:
        c[r.get("landed_box")] = c.get(r.get("landed_box"), 0) + 1
    return c


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def build() -> str:
    L: list[str] = []
    L.append("# 结果表（自动生成，勿手改）\n")
    L.append("> 由 `tools/make_tables.py` 从 `rl/eval/*.json` 生成。")
    L.append("> 想更新：重新跑评测 → `make tables`。所有数字都能追到具体 json。\n")

    # ---------------- 表 1：语言条件反事实评测 ----------------
    L.append("## 表 1 · 语言条件反事实评测（留出场景 2400-2409,2411-2419 × 2 seed × 2 指令 = 76 rollout）\n")
    L.append("| 模型 | 落点 A / B / 未落箱 | follow_acc | Wilson 95% | ITT | 配对全对 | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for name, fn in LANG_RUNS:
        p = EV / fn
        if not p.exists():
            L.append(f"| {name} | _缺 {fn}_ | — | — | — | — | — |")
            continue
        d = load(p)
        s = d.get("summary") or {}
        lc = landed_counts(d.get("rows") or [])
        L.append("| {n} | {a} / {b} / {c} | {fa} ({fc}/{nl}) | [{lo}, {hi}] | {itt} | **{pb} / {np_}** | {v} |".format(
            n=name, a=lc.get("target_box", 0), b=lc.get("target_box_b", 0), c=lc.get(None, 0),
            fa=pct(s.get("follow_acc")), fc=s.get("follow_correct"), nl=s.get("n_landed_in_some_box"),
            lo=pct((s.get("follow_wilson") or [None])[0]), hi=pct((s.get("follow_wilson") or [None, None])[1]),
            itt=pct(s.get("follow_acc_itt")),
            pb=s.get("pairs_both_follow"), np_=s.get("n_pairs"), v=s.get("verdict")))
    L.append("")
    L.append("**三个口径必须同时看，别只挑好看的**：")
    L.append("- `follow_acc`：分母是**落进箱子的** rollout（任务失败的不算），衡量\"落对了没\"")
    L.append("- `ITT`：分母是**全部** rollout（含没落箱的），衡量\"整体干成了没\"")
    L.append("- `配对全对`：分母是**全部配对**（38 对），要求同场景两条指令都落对——这是\"语言真的起作用\"最硬的证据")
    L.append("")
    L.append("⚠️ v1 @5000 的 `follow_acc` 是 56.4%，看起来\"不算差\"，但它的落点是 **1 箱A / 38 箱B**——")
    L.append("模型只是\"总往右放\"，凡指令是右的都算对，所以这个数没有意义；**配对全对 0/38** 才是真相。\n")

    # ---------------- 表 2：语言通道权重曲线 ----------------
    L.append("## 表 2 · 语言通道权重曲线（`per_col`）\n")
    L.append("`per_col` = 语言 2 列每列 RMS ÷ 其它 6 列每列 RMS。基座 `act_sim156_lang8` 是 6→8 补零加宽，")
    L.append("所以这两列**初始恰好为 0**——它长不长大，直接回答\"语言通道有没有被用起来\"。\n")
    wc = EV / "lang_weight_curve.json"
    if wc.exists():
        w = load(wc)
        L.append("| 步数 | encoder | vae_encoder |")
        L.append("|---|---|---|")
        for r in w.get("rows", []):
            L.append(f"| {r['step']} | {r['encoder']:.5f} | {r['vae_encoder']:.5f} |")
        base = w.get("baseline") or {}
        if base:
            L.append(f"| 基座（0 步） | {base.get('encoder', 0):.5f} | {base.get('vae_encoder', 0):.5f} |")
        L.append("")
        if w.get("note"):
            L.append(f"> {w['note']}\n")
    else:
        L.append("_缺 `rl/eval/lang_weight_curve.json`，跑 `make weights` 生成_\n")

    # ---------------- 表 3：残差 SAC ----------------
    L.append("## 表 3 · 残差 SAC（负结果，同样是成果）\n")
    rr = EV / "residual_rl_summary.json"
    if rr.exists():
        r = load(rr)
        L.append(f"> 数据来源：`{r.get('source', 'docs/REPORT_RESIDUAL_RL.md')}`（held-out ep 100–129，训练池 0–49，不重叠）\n")
        L.append("| 配置 | 洁净成功率 | 旧口径成功率 | 备注 |")
        L.append("|---|---|---|---|")
        for k, v in (r.get("rows") or {}).items():
            L.append(f"| {k} | {pct(v.get('clean'))} | {pct(v.get('legacy'))} | {v.get('note', '')} |")
        L.append("")
        for line in (r.get("notes") or []):
            L.append(f"- {line}")
        L.append("")
    else:
        L.append("_缺 `rl/eval/residual_rl_summary.json`_\n")
    return "\n".join(L)


def check() -> int:
    bad = 0
    for name, fn in LANG_RUNS:
        p = EV / fn
        if not p.exists():
            print(f"✗ 缺文件 {fn}")
            bad += 1
            continue
        s = (load(p).get("summary") or {})
        miss = [k for k in NEED_SUMMARY if k not in s]
        if miss:
            print(f"✗ {fn} 缺字段 {miss}")
            bad += 1
    for f in ("lang_weight_curve.json", "residual_rl_summary.json"):
        if not (EV / f).exists():
            print(f"✗ 缺文件 {f}")
            bad += 1
    fresh = OUT.exists() and OUT.read_text(encoding="utf-8") == build()
    if not fresh:
        print("✗ docs/RESULTS.md 不是最新（跑 make tables）")
        bad += 1
    print("✓ 结果表校验通过" if bad == 0 else f"✗ {bad} 处问题")
    return 0 if bad == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n→ 已写入 {OUT.relative_to(ROOT)}（{len(text)} 字节）")
    return 0


if __name__ == "__main__":
    sys.exit(main())