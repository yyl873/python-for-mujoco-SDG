#!/usr/bin/env python3
"""rl/plot_amplitude.py —— "残差幅度 → 闭环成功率" 曲线（含基座步长参考线）

回答的问题：动作空间该给多大？
输入是 `rl/eval/ab_*scale*_s*.json`（每档 3 个 seed），横轴用**度**（不是 scale 系数），
并画出基座自身逐步动作幅度（0.428°，`rl/eval/pilot.json` 实测）作为参考线，
让"5×/13× 基座步长"这种说法一眼能看出来。

用法
----
```bash
MPLCONFIGDIR=$PWD/.mplcache env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python \
    rl/plot_amplitude.py --out rl/curves/amplitude_curve.png
```
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

LIMIT_DEG = 5.7296          # env_residual.RESIDUAL_DEG
BASE_STEP_DEG = 0.428       # sim156 raw 档实测 act_step_mean_abs_deg


def collect(patterns: list[str]):
    """返回 {幅度度数: [(seed, base_rate, plus_rate, n), ...]}"""
    out: dict[float, list[tuple]] = {}
    for pat in patterns:
        for f in sorted(glob.glob(str(ROOT / pat)) or glob.glob(pat)):
            d = json.loads(Path(f).read_text(encoding="utf-8"))
            sc = d.get("res_scale")
            if sc is None:
                continue
            r = d["results"]
            if "base+R" not in r or "base" not in r:
                continue
            deg = round(LIMIT_DEG * float(sc), 3)
            out.setdefault(deg, []).append(
                (Path(f).stem, r["base"]["success_rate"][0], r["base+R"]["success_rate"][0],
                 r["base"]["n"]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="幅度→成功率曲线")
    ap.add_argument("--held", nargs="+", default=["rl/eval/ab_held_scale*_s*.json"],
                    help="held-out 上的扫描结果（主图）")
    ap.add_argument("--leaked", nargs="+", default=["rl/eval/ab_s*_scale*.json"],
                    help="旧口径（episode 0-29，含训练泄漏）上的结果，作对照副图")
    ap.add_argument("--out", default=str(ROOT / "rl" / "curves" / "amplitude_curve.png"))
    a = ap.parse_args(argv)

    held, leaked = collect(a.held), collect(a.leaked)
    if not held:
        print("没有 held-out 扫描结果", file=sys.stderr)
        return 1

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=160,
                             gridspec_kw=dict(width_ratios=[1.15, 1]))
    for ax, data, title in ((axes[0], held, "held-out（episode 100-119，无泄漏）"),
                            (axes[1], leaked, "旧口径（episode 0-29，含训练泄漏，仅供参考）")):
        xs = sorted(data)
        means, base_rates = [], []
        for deg in xs:
            plus = [t[2] for t in data[deg]]
            base = [t[1] for t in data[deg]]
            means.append(np.mean(plus))
            base_rates.append(np.mean(base))
            ax.scatter([deg] * len(plus), [p * 100 for p in plus], s=22,
                       color="#c0392b", alpha=0.55, zorder=3)
        ax.plot(xs, [m * 100 for m in means], "-o", color="#c0392b", lw=1.8,
                label="base + 残差（seed 均值 + 每 seed 点）")
        ax.axhline(np.mean(base_rates) * 100, color="#7f8c8d", ls="--", lw=1.4,
                   label=f"纯 ACT 基座 {np.mean(base_rates)*100:.1f}%")
        ax.axvline(BASE_STEP_DEG, color="#2980b9", ls=":", lw=1.2)
        ax.text(BASE_STEP_DEG * 1.08, ax.get_ylim()[0] + 2, f"基座步长 {BASE_STEP_DEG}°",
                color="#2980b9", fontsize=7.5, rotation=90, va="bottom")
        ax.set_xscale("log")
        ax.set_xlabel("残差幅度上限（度，对数轴）", fontsize=9.5)
        ax.set_ylabel("闭环成功率 (%)", fontsize=9.5)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, linestyle=":")
        ax.legend(fontsize=7.5, loc="best")
    fig.suptitle("残差幅度 vs 闭环成功率：幅度越大越破坏基座", fontsize=12)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)

    print(f"{'幅度(°)':>9}{'=几倍基座步长':>14}{'seed 数':>8}{'base':>8}{'+残差':>8}")
    for deg in sorted(held):
        d = held[deg]
        print(f"{deg:>9.2f}{deg/BASE_STEP_DEG:>13.1f}×{len(d):>8}"
              f"{np.mean([t[1] for t in d])*100:>7.1f}%{np.mean([t[2] for t in d])*100:>7.1f}%")
    print(f"  图 → {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())