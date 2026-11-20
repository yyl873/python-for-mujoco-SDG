#!/usr/bin/env python3
"""rl/s2_verdict.py —— 用**存量**训练曲线重算 S2 判据（不重训）

为什么单独一个文件
------------------
`rl/curves/summary.json`（2026-11-20 13:34 产出）里存的是旧判据字段 `ratio = tail/head`。
reward 是**负值**（稠密距离惩罚主导），所以 `-246 / -299 = 0.82 < 1.2` 被误判成"训练失败"，
实际上奖励是**变好**了（从 -299 涨到 -246，绝对值缩小 47）。

`rl/train.py` 已经改成相对改善 `(tail − head) / |head| ≥ 0.20`，但旧 summary.json 不会自动更新。
本脚本从 `rl/curves/s*.jsonl` 原始 episode 奖励重算，输出 `rl/curves/summary_v2.json`，
让 REPORT 里的数字有据可查。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/s2_verdict.py
```

口径（写进 REPORT 时必须声明）
------------------------------
- 分段：episode 奖励按**四分位**切（episode 少时 head/tail 用前 1/4 与后 1/4，不重叠）。
- 判据：三个 seed 的 `improve` 均值 ≥ 0.20 记 `ok`，否则 `warn`（spec §9 S2）。
- 这是**训练期探索策略**的回报，不是策略评估成功率；成功率看 `rl/eval/ab_*.json`。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURVES = ROOT / "rl" / "curves"


def load(path: Path) -> list[float]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(float(json.loads(line)["ep_reward"]))
    return out


def quartile_stats(rewards: list[float]) -> dict:
    q = max(1, len(rewards) // 4)
    head, tail = rewards[:q], rewards[-q:]
    head_m = sum(head) / len(head)
    tail_m = sum(tail) / len(tail)
    improve = (tail_m - head_m) / abs(head_m) if head_m else None
    return dict(n_episodes=len(rewards), head_n=q, tail_n=q,
                reward_head_mean=round(head_m, 3),
                reward_tail_mean=round(tail_m, 3),
                delta=round(tail_m - head_m, 3),
                improve=None if improve is None else round(improve, 3),
                reward_best=round(max(rewards), 3),
                reward_worst=round(min(rewards), 3))


def main() -> int:
    runs = []
    for curve in sorted(CURVES.glob("s*.jsonl")):
        seed = curve.stem.lstrip("s")
        if not seed.isdigit():
            continue
        rewards = load(curve)
        if not rewards:
            continue
        st = quartile_stats(rewards)
        runs.append(dict(seed=int(seed), curve=str(curve.relative_to(ROOT)),
                         ckpt=str(Path("rl/ckpt") / f"s{seed}" / "model.zip"), **st))

    if not runs:
        print("没有找到 rl/curves/s*.jsonl")
        return 1

    print(f"{'seed':>4}  {'episodes':>8}  {'head(mean)':>11}  {'tail(mean)':>11}  "
          f"{'Δ':>8}  {'improve':>8}  {'best':>9}  {'worst':>9}")
    for r in runs:
        print(f"{r['seed']:>4}  {r['n_episodes']:>8}  {r['reward_head_mean']:>11.2f}  "
              f"{r['reward_tail_mean']:>11.2f}  {r['delta']:>+8.2f}  "
              f"{r['improve']:>8.3f}  {r['reward_best']:>9.2f}  {r['reward_worst']:>9.2f}")

    imps = [r["improve"] for r in runs if r["improve"] is not None]
    mean_imp = sum(imps) / len(imps) if imps else None
    ok = mean_imp is not None and mean_imp >= 0.20
    verdict = "ok" if ok else "warn"
    print(f"\n  S2 判据（相对改善均值 ≥ 0.20）：mean improve = "
          f"{'n/a' if mean_imp is None else f'{mean_imp:.3f}'}  → {verdict}")
    print("  注：旧 summary.json 的 ratio=tail/head 在负奖励上失效，本文件的 improve 才是判据。")

    out = CURVES / "summary_v2.json"
    out.write_text(json.dumps(dict(
        criterion="mean improve = mean((tail_mean - head_mean)/|head_mean|) >= 0.20",
        note="reward 为负值，禁止用 tail/head 比值判定",
        mean_improve=None if mean_imp is None else round(mean_imp, 3),
        verdict=verdict, runs=runs), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())