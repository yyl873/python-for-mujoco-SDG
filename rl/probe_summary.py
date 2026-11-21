#!/usr/bin/env python3
"""汇总 `rl/shooting_probe.py` 的 jsonl（按 episode 去重），并和"基座+钳爪"逐条配对。

为什么要单独写：探针进程会因 mj_step 的 SIGFPE 中断并 `--resume` 续跑，
所以 `shooting_probe.py` 写进 json 的 `n_decisions` / `oracle` 只是**最后一次进程**的局部统计。
本脚本只信 jsonl（每个 episode 一行），重算全部聚合量。

用法：
    env -u PYTHONHOME -u PYTHONPATH python3 rl/probe_summary.py \
        rl/eval/shooting_clamp_a143.jsonl rl/eval/shooting_clamp_b573.jsonl ...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load(nl: Path) -> dict[int, dict]:
    rec = {}
    if not nl.exists():
        return rec
    for line in nl.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except Exception:                                    # noqa: BLE001
            continue
        rec[int(r["episode_id"])] = r                        # 同一 episode 多次则取最后一次
    return rec


def base_seed0() -> dict[int, dict]:
    """基座+钳爪在 seed=0 上的逐条结果（探针内部 reset(seed=0)，所以只能对 seed0）。"""
    p = ROOT / "rl" / "eval" / "target_base_jawclamp_90.json"
    rows = json.loads(p.read_text(encoding="utf-8"))["detail"]["base"]["rows"]
    return {int(r["episode_id"]): r for r in rows if int(r["seed"]) == 0}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = [str(p) for p in sorted((ROOT / "rl" / "eval").glob("shooting_*.jsonl"))]
    base = base_seed0()
    print(f"对照：基座+钳爪 seed0 = {sum(r['success'] for r in base.values())}/{len(base)} 成功\n")
    for f in argv:
        p = Path(f)
        if not p.is_absolute():
            p = ROOT / p
        rec = load(p)
        if not rec:
            print(f"{p.name}: 空")
            continue
        gains = [g for r in rec.values() for g in r.get("gains", [])]
        g = np.asarray(gains, float)
        orc = {e: r for e, r in rec.items() if r.get("oracle")}
        n_ok = sum(1 for r in orc.values() if r.get("success"))
        b_ok = sum(1 for e in orc if base.get(e, {}).get("success"))
        # 配对：oracle 修好了哪些基座失败
        fixed = [e for e in sorted(orc) if orc[e].get("success") and not base.get(e, {}).get("success")]
        broke = [e for e in sorted(orc) if not orc[e].get("success") and base.get(e, {}).get("success")]
        meta = {}
        mj = Path(str(p).replace(".jsonl", ".json"))
        if mj.exists():
            try:
                meta = json.loads(mj.read_text(encoding="utf-8"))
            except Exception:                                # noqa: BLE001
                meta = {}
        # 早期 json 没有 score 字段（那时只有 lift 一种口径）
        score = meta.get("score") or ("lift" if meta else "?")
        deg = meta.get("residual_deg", "?")
        print(f"== {p.name}  打分={score}  幅度±{deg}°  episode 数 {len(rec)}（oracle 行 {len(orc)}）")
        if g.size:
            unit = "mm（抬升）" if score == "lift" else "mm（靠近）"
            print(f"   chunk 级 best_gain {unit}: 均值 {g.mean():+.2f} / 中位 {np.median(g):+.2f} / "
                  f"p90 {np.percentile(g, 90):+.2f}；>2mm {100*np.mean(g > 2):.1f}%；≤0 {100*np.mean(g <= 0):.1f}%  "
                  f"（决策 {g.size} 次）")
        print(f"   oracle 整条 episode: {n_ok}/{len(orc)} 成功；同批基座+钳爪 {b_ok}/{len(orc)}")
        print(f"   oracle 修好的基座失败: {fixed}")
        print(f"   oracle 弄坏的基座成功: {broke}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())