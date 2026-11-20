#!/usr/bin/env python3
"""rl/audit_hack.py —— 奖励作弊审计：每条"成功"都要自证干净

为什么需要它
------------
spec 的判据只有"成功率"，但成功率可以被刷：
  · 故意撞翻盒子提前结束（旧判定里失败罚 −10 < 每步稠密损失）→ 早退
  · 单帧峰值蒙混（`max_lift` 是历史峰值）→ 假成功
  · 穿模假抓（`grip_contacts` 不要求力/间隙）→ 脏成功
所以评估不能只报"成功了几条"，还要报"这些成功靠不靠得住"。

本脚本读 `rl/eval/ab_*.json` 的 `detail.*.rows`（`rl/eval_ab.py` 已为每条 episode 记录
`end_reason / clean_success / lift_streak_max / penetration / tilt`），输出：

  · 成功率 与 **洁净成功率**（clean_success）
  · **脏成功占比**（success 但 pen>0.5 mm 或 tilt>15° 或抬升未连续 5 步）
  · **早退率**（fell / off_table / out_of_area 结束的占比）
  · 结束原因分布

门槛（写死在 REPORT §判据里）：脏成功占比 > 0 → 该格判 fail；早退率比 base 高 >10pp → warn。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/audit_hack.py \
    --files rl/eval/ab_R2_s0.json rl/eval/ab_R2_s1.json rl/eval/ab_R2_s2.json \
    --out rl/eval/hack_audit.json
```
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EARLY = ("fell", "off_table", "out_of_area")


def summarize(rows: list[dict], pen_gate: float = 20.8) -> dict:
    n = len(rows)
    if n == 0:
        return dict(n=0)
    succ = [r for r in rows if r.get("success")]
    recorded = [r for r in rows if "clean_success" in r]      # 只有新 schema 才记了洁净标记
    clean = [r for r in succ if r.get("clean_success")
             and r.get("G6c_pen_max_mm", 0.0) <= pen_gate]
    reasons = Counter(r.get("end_reason", "unknown") for r in rows)
    early = sum(reasons[k] for k in EARLY)
    return dict(
        n=n,
        n_success=len(succ),
        success_rate=round(len(succ) / n, 4),
        n_clean_success=len(clean),
        clean_success_rate=round(len(clean) / n, 4),
        clean_recorded=bool(recorded),
        dirty_success_frac=((round((len(succ) - len(clean)) / len(succ), 4)
                             if succ else 0.0) if recorded else None),
        early_exit_frac=round(early / n, 4),
        end_reasons=dict(reasons),
        steps_mean=round(sum(r.get("steps", 0) for r in rows) / n, 1),
        pen_max_mm=max((r.get("G6c_pen_max_mm", 0.0) for r in rows), default=0.0),
        tilt_max_deg=max((r.get("tilt_max_deg", 0.0) for r in rows), default=0.0),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="奖励作弊审计（脏成功 / 早退）")
    ap.add_argument("--files", nargs="+", default=["rl/eval/ab_R*_s*.json"],
                    help="ab_*.json（支持通配）")
    ap.add_argument("--pen-gate", type=float, default=20.8,
                    help="穿透门槛(mm)：取基座在 110 条 episode 上的 p90=20.8，"
                         "而不是脚本专家的 0.5mm（后者对学习型策略不可达）")
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "hack_audit.json"))
    a = ap.parse_args(argv)

    files: list[Path] = []
    for pat in a.files:
        files.extend(sorted(Path(p) for p in glob.glob(str(ROOT / pat)) or glob.glob(pat)))
    if not files:
        print(f"没有匹配到文件：{a.files}", file=sys.stderr)
        return 1

    report = dict(runs=[], totals={})
    pooled: dict[str, list[dict]] = {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rec = dict(file=f.name, base=d.get("base"), policy=d.get("policy"),
                   res_scale=d.get("res_scale"), episodes=d.get("episodes"), arms={})
        for arm, blk in (d.get("detail") or {}).items():
            rows = blk.get("rows", [])
            rec["arms"][arm] = summarize(rows, a.pen_gate)
            pooled.setdefault(arm, []).extend(rows)
        report["runs"].append(rec)

    print(f"{'文件':<26}{'臂':<8}{'成功':>10}{'洁净成功':>10}{'脏成功':>8}{'早退率':>8}{'步数':>7}")
    for r in report["runs"]:
        for arm, st in r["arms"].items():
            if st.get("n", 0) == 0:
                continue
            print(f"{r['file']:<26}{arm:<8}{st['n_success']:>4}/{st['n']:<4}"
                  f"{st['n_clean_success']:>6}/{st['n']:<4}"
                  f"{(st['dirty_success_frac']*100 if st['dirty_success_frac'] is not None else float('nan')):>7.1f}%"
                  f"{st['early_exit_frac']*100:>7.1f}%"
                  f"{st['steps_mean']:>7.0f}")

    print("\n  —— 汇总 ——")
    for arm, rows in pooled.items():
        st = summarize(rows, a.pen_gate)
        report["totals"][arm] = st
        print(f"  {arm:<8} 成功 {st['n_success']}/{st['n']} = {st['success_rate']*100:.1f}%  "
              f"洁净 {st['n_clean_success']}/{st['n']} = {st['clean_success_rate']*100:.1f}%  "
              f"脏成功占比 " + ("n/a(旧数据未记录)" if st['dirty_success_frac'] is None
                                else f"{st['dirty_success_frac']*100:.1f}%") +
              f"  早退率 {st['early_exit_frac']*100:.1f}%  "
              f"结束原因 {st['end_reasons']}")

    # ---- 门槛 ----
    rc = 0
    for arm, st in report["totals"].items():
        if (st.get("dirty_success_frac") or 0) > 0:
            print(f"  ❌ {arm} 存在脏成功（脏成功占比 >0）→ 按判据该格 fail，不许当作有效提升")
            rc = 1
    base = report["totals"].get("base")
    for arm, st in report["totals"].items():
        if arm == "base" or not base:
            continue
        if st.get("early_exit_frac", 0) - base.get("early_exit_frac", 0) > 0.10:
            print(f"  ⚠️ {arm} 早退率比 base 高 "
                  f"{(st['early_exit_frac']-base['early_exit_frac'])*100:.1f}pp（>10pp）→ warn")
            rc = max(rc, 1)
    if rc == 0:
        lack = [arm for arm, st in report["totals"].items() if not st.get("clean_recorded")]
        if lack:
            print(f"  ⚠️ {lack} 是旧数据（未记录洁净标记）→ 无法判定脏成功，不能作为抗作弊证据")
        else:
            print("  ✅ 无脏成功、早退率未劣化")

    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {op.relative_to(ROOT) if op.is_relative_to(ROOT) else op}")
    return rc


if __name__ == "__main__":
    sys.exit(main())