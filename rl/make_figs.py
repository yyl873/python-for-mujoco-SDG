#!/usr/bin/env python3
"""rl/make_figs.py —— 交付图表（spec §9 S4）：成功率 A/B 柱状图 + 训练奖励曲线

两张图，数字全部来自磁盘上的 JSON/JSONL，不含手写常数：
  1. `rl/curves/success_ab.png`  —— base vs base+R 成功率，Wilson 95% 区间，
     并把每个 seed 的点散在上面（避免用区间掩盖 seed 间差异）
  2. `rl/curves/reward_curves.png` —— 3 个 seed 的 episode 奖励（滑动平均）+
     首/末四分位均值（S2 判据用的就是这两段）

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/make_figs.py \
    --ab 'rl/eval/ab_s*_30.json' --curves 'rl/curves/s*.jsonl'
```

口径诚实声明
------------
- 成功率图只画**点估计 + Wilson 区间**，不画"显著性"，样本 ≤ 90/臂 时不足以声称显著。
- 奖励曲线是**训练期探索策略**的回报（含随机探索的负分），不是策略评估表现。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                                           # 无显示器，必须先设后端
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def load_ab(patterns: list[str]):
    """读若干 ab_*.json，按臂聚合（每臂各自 sum(n_success)/sum(n)）。"""
    arms: dict[str, dict] = {}
    per_seed: dict[str, list[float]] = {"base": [], "base+R": []}
    files = []
    for pat in patterns:
        files.extend(sorted(glob.glob(str(ROOT / pat)) or glob.glob(pat)))
    for f in files:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        for arm in ("base", "base+R"):
            r = (d.get("results") or {}).get(arm)
            if not r:
                continue
            a = arms.setdefault(arm, dict(n=0, n_success=0))
            a["n"] += int(r.get("n", 0))
            a["n_success"] += int(r.get("n_success", 0))
            if r.get("n"):
                per_seed[arm].append(int(r["n_success"]) / int(r["n"]))
    return arms, per_seed, files


def fig_success(arms: dict, per_seed: dict, out: Path) -> None:
    names = ["base", "base+R"]
    xs = np.arange(len(names))
    vals, los, his = [], [], []
    for nm in names:
        a = arms.get(nm) or dict(n=0, n_success=0)
        p, lo, hi = wilson(a["n_success"], a["n"])
        vals.append(p * 100)
        los.append((p - lo) * 100)
        his.append((hi - p) * 100)

    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    colors = ["#7f8c8d", "#c0392b"]
    bars = ax.bar(xs, vals, yerr=[los, his], capsize=6, width=0.55,
                  color=colors, alpha=0.85, edgecolor="black", linewidth=0.6,
                  error_kw=dict(ecolor="black", lw=1.0))
    for x, nm, v in zip(xs, names, vals):
        a = arms.get(nm) or dict(n=0, n_success=0)
        ax.text(x, v + 2.5, f"{v:.1f}%\n({a['n_success']}/{a['n']})",
                ha="center", va="bottom", fontsize=9)
        pts = per_seed.get(nm, [])
        if pts:
            ax.scatter([x] * len(pts), [p * 100 for p in pts], s=18,
                       color="black", alpha=0.55, zorder=3, marker="o")
    delta = vals[1] - vals[0]
    ax.set_xticks(xs)
    ax.set_xticklabels(["pure ACT\n(base)", "ACT + residual RL\n(SAC)"], fontsize=10)
    ax.set_ylabel("闭环成功率 (%)", fontsize=10)
    ax.set_ylim(0, max(60, max(vals) + 20))
    ax.set_title(f"MuJoCo 闭环成功率 A/B（Δ = {delta:+.1f} pp，误差棒 = Wilson 95%）",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_rewards(curve_files: list[Path], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#d35400", "#16a085"]
    for i, f in enumerate(sorted(curve_files)):
        rs = [json.loads(x)["ep_reward"]
              for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rs:
            continue
        ep = np.arange(1, len(rs) + 1)
        c = colors[i % len(colors)]
        ax.plot(ep, rs, color=c, alpha=0.22, linewidth=0.8)
        w = max(3, len(rs) // 12)
        if len(rs) >= w:
            smooth = np.convolve(rs, np.ones(w) / w, mode="valid")
            ax.plot(ep[w - 1:], smooth, color=c, linewidth=1.8,
                    label=f"{f.stem}（{len(rs)} ep，滑动均值 w={w}）")
        q = max(1, len(rs) // 4)
        ax.hlines(np.mean(rs[:q]), 0, q, color=c, linestyle="--", linewidth=1.0)
        ax.hlines(np.mean(rs[-q:]), len(rs) - q, len(rs), color=c, linestyle="-",
                  linewidth=1.4)
    ax.set_xlabel("episode（训练过程，每条约 15 s 仿真）", fontsize=10)
    ax.set_ylabel("episode 奖励（负值越小越接近 0 越好）", fontsize=10)
    ax.set_title("残差 SAC 训练奖励曲线（虚线=首四分位均值，实线=末四分位均值）",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="生成交付图表（成功率 A/B + 奖励曲线）")
    ap.add_argument("--ab", nargs="+", default=["rl/eval/ab_s*_30.json"],
                    help="A/B 结果 JSON（可多个，按臂聚合）")
    ap.add_argument("--curves", nargs="+", default=["rl/curves/s*.jsonl"])
    ap.add_argument("--out-dir", default=str(ROOT / "rl" / "curves"))
    a = ap.parse_args(argv)

    arms, per_seed, files = load_ab(a.ab)
    if not arms:
        print(f"没有可用的 A/B 结果（模式 {a.ab}）", file=sys.stderr)
        return 1
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = out_dir / "success_ab.png"
    fig_success(arms, per_seed, p1)

    curve_files = []
    for pat in a.curves:
        curve_files.extend(sorted(Path(p) for p in glob.glob(str(ROOT / pat)) or glob.glob(pat)))
    p2 = out_dir / "reward_curves.png"
    if curve_files:
        fig_rewards(curve_files, p2)

    for arm in ("base", "base+R"):
        r = arms.get(arm)
        if r:
            p, lo, hi = wilson(r["n_success"], r["n"])
            print(f"  {arm:8s} {r['n_success']}/{r['n']} = {p*100:.1f}% "
                  f"[{lo*100:.1f}, {hi*100:.1f}]")
    print(f"  图 1 → {p1.relative_to(ROOT)}")
    if curve_files:
        print(f"  图 2 → {p2.relative_to(ROOT)}（{len(curve_files)} 个 seed）")
    return 0


if __name__ == "__main__":
    sys.exit(main())