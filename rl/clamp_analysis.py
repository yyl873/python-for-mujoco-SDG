#!/usr/bin/env python3
"""残差 9 臂在"夹爪钳制"下的重测汇总（§17.7）。

读 rl/eval/target_R{1,2,3}_s{0,1,2}_clamp_90.json，每份里：
  tag='base'   → 基座 + 夹爪钳制（90 次 rollout，应与 target_base_jawclamp_90.json 完全一致）
  tag='base+R' → 残差 + 夹爪钳制
clean 一律用 THRESHOLD 从 pen_finger_grasp_max_mm 重算（旧 json 里的 clean_success 是 10 mm 口径）。

用法：env -u PYTHONHOME -u PYTHONPATH python3 rl/clamp_analysis.py [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TH = 13.26           # clean 阈值 = 脚本专家 30 条回放 p90（报告 §17.1）
ARMS = [("R1", "s0"), ("R1", "s1"), ("R1", "s2"),
        ("R2", "s0"), ("R2", "s1"), ("R2", "s2"),
        ("R3", "s0"), ("R3", "s1"), ("R3", "s2")]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b: int, c: int) -> float:
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def load_rows(path: Path, tag: str, th: float = TH) -> dict[tuple[int, int], dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = ((d.get("detail") or {}).get(tag) or {}).get("rows") or []
    out = {}
    for r in rows:
        r = dict(r)
        r["clean"] = bool(r.get("success") and float(r.get("pen_finger_grasp_max_mm", 99.0)) <= th)
        out[(int(r["episode_id"]), int(r["seed"]))] = r
    return out


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.1f}% [{100 * lo:.0f},{100 * hi:.0f}]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pat", default="target_R*_clamp_90.json")
    ap.add_argument("--th", type=float, default=TH)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    files = sorted(glob.glob(str(ROOT / "rl" / "eval" / a.pat)))
    if not files:
        print(f"没有匹配 {a.pat} 的结果文件", file=sys.stderr)
        return 2

    out = {"threshold": a.th, "arms": {}}
    base_ref = None
    for f in files:
        p = Path(f)
        name = p.name.replace("target_", "").replace("_clamp_90.json", "")   # e.g. R1_s0
        b = load_rows(p, "base", a.th)
        r = load_rows(p, "base+R", a.th)
        if not b or not r:
            print(f"{name}: 行数不全 base={len(b)} base+R={len(r)}", file=sys.stderr)
            continue
        if base_ref is None:
            base_ref = b
        bs, bc = sum(x["success"] for x in b.values()), sum(x["clean"] for x in b.values())
        rs, rc = sum(x["success"] for x in r.values()), sum(x["clean"] for x in r.values())
        keys = sorted(set(b) & set(r))
        bS = [b[k]["success"] for k in keys]
        rS = [r[k]["success"] for k in keys]
        bC = [b[k]["clean"] for k in keys]
        rC = [r[k]["clean"] for k in keys]
        mS = (sum(1 for i in range(len(keys)) if rS[i] and not bS[i]),
              sum(1 for i in range(len(keys)) if bS[i] and not rS[i]))
        mC = (sum(1 for i in range(len(keys)) if rC[i] and not bC[i]),
              sum(1 for i in range(len(keys)) if bC[i] and not rC[i]))
        pen = [x.get("pen_finger_grasp_max_mm", 0.0) for x in r.values()]
        rec = dict(n=len(keys), base_success=bs, base_clean=bc,
                   succ=rs, clean=rc,
                   succ_pct=round(100 * rs / len(keys), 1), clean_pct=round(100 * rc / len(keys), 1),
                   pen_p50=round(statistics.median(pen), 2),
                   mcnemar_succ=dict(b=mS[0], c=mS[1], p=mcnemar_exact(*mS)),
                   mcnemar_clean=dict(b=mC[0], c=mC[1], p=mcnemar_exact(*mC)))
        out["arms"][name] = rec
        print(f"{name:6s} succ {pct(rs, len(keys)):32s} clean {pct(rc, len(keys)):30s} "
              f"pen_p50 {rec['pen_p50']:5.2f}  vs基座+钳: succ b={mS[0]}/c={mS[1]} p={mcnemar_exact(*mS):.2e} | "
              f"clean b={mC[0]}/c={mC[1]} p={mcnemar_exact(*mC):.2e}")

    if base_ref:
        bs = sum(x["success"] for x in base_ref.values())
        bc = sum(x["clean"] for x in base_ref.values())
        print(f"\n{'基座+钳':6s} succ {pct(bs, len(base_ref)):32s} clean {pct(bc, len(base_ref))}")
        out["base_clamp"] = dict(n=len(base_ref), success=bs, clean=bc)
    if a.json:
        Path(a.json).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"落盘 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())