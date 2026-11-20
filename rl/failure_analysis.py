#!/usr/bin/env python3
"""rl/failure_analysis.py —— A2：失败归因 + 配对分析（不训练、不跑仿真）

为什么需要它
------------
每格 30 条 episode、种子间成功率 10%–43%，方差远大于配置差异 —— 用"总成功率"比高低等于看噪声。
正确做法是**按 episode 配对**：同一批 30 条 held-out episode 上，看
  · b = base 失败但残差成功（残差救回来的）
  · c = base 成功但残差弄坏（残差搞砸的）
这才是 McNemar 检验该看的数（b 与 c 的差），比成功率差敏感得多。

同时把失败拆成模式：撞翻 / 推出工作区 / 掉桌 / 超时 / 没夹住，回答"残差到底在哪类失败上有救"。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/failure_analysis.py \
    --base rl/eval/ab_base_heldout_strict.json \
    --arms R1:rl/eval/ab_R1_s0.json,R1:rl/eval/ab_R1_s1.json,R1:rl/eval/ab_R1_s2.json \
    --out rl/eval/failure_analysis.json
```
（`--arms 名字:file1,file2,...` 或直接用 `--glob 'R?=rl/eval/ab_R?_s*.json'`）
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EARLY = ("fell", "off_table", "out_of_area")


def load_rows(path: str, arm: str = "base+R") -> dict[int, dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = (d.get("detail") or {}).get(arm, {}).get("rows", [])
    return {int(r["episode_id"]): r for r in rows}


def mcnemar_exact(b: int, c: int) -> float:
    """精确二项检验的双侧 p 值（b/c 都很小时用它，比卡方可靠）。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) * 2
    return min(1.0, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="失败归因 + 配对分析")
    ap.add_argument("--base", required=True, help="base 臂结果 json（base+R 行的 episode 集合不算数，只用 base 结果）")
    ap.add_argument("--arms", nargs="+", default=[], help="名字:file1,file2,... 可多组")
    ap.add_argument("--glob", nargs="+", default=None, help="自动发现：如 'R3=rl/eval/ab_R3_s*.json'")
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "failure_analysis.json"))
    a = ap.parse_args(argv)

    # base 的每 episode 结果（取 base 臂）
    base_map = {}
    bd = json.loads(Path(a.base).read_text(encoding="utf-8"))
    for r in bd["detail"]["base"]["rows"]:
        base_map[int(r["episode_id"])] = r
    print(f"base：{len(base_map)} 条 episode，成功 {sum(1 for r in base_map.values() if r['success'])}")

    groups: dict[str, list[str]] = {}
    if a.glob:
        for spec in a.glob:
            name, pat = spec.split("=", 1)
            groups[name] = sorted(globmod.glob(str(ROOT / pat)))
    for spec in a.arms:
        name, files = spec.split(":", 1)
        groups.setdefault(name, []).extend(files.split(","))

    report = dict(base=dict(n=len(base_map),
                            n_success=sum(1 for r in base_map.values() if r["success"]),
                            end_reasons=dict(Counter(r.get("end_reason", "?") for r in base_map.values()))),
                    groups={})

    print(f"\n{'组':<6}{'n':>5}{'成功':>7}{'洁净':>7}{'b(base败→残差成)':>18}{'c(base成→残差败)':>18}{'McNemar p':>11}")
    for name, files in groups.items():
        pooled_success = pooled_clean = pooled_n = 0
        b = c = 0
        per_ep: dict[int, list[dict]] = defaultdict(list)
        reasons = Counter()
        for f in files:
            rows = load_rows(f)
            for eid, r in rows.items():
                if eid not in base_map:
                    continue
                per_ep[eid].append(r)
                pooled_n += 1
                pooled_success += int(r["success"])
                pooled_clean += int(bool(r.get("clean_success")))
                reasons[r.get("end_reason", "?")] += 1
                bs, rs = bool(base_map[eid]["success"]), bool(r["success"])
                if (not bs) and rs:
                    b += 1
                if bs and (not rs):
                    c += 1
        p = mcnemar_exact(b, c)
        print(f"{name:<6}{pooled_n:>5}{pooled_success:>7}{pooled_clean:>7}{b:>18}{c:>18}{p:>11.3f}")
        report["groups"][name] = dict(
            files=[str(Path(f).name) for f in files], n=pooled_n,
            n_success=pooled_success, n_clean=pooled_clean,
            b_base_fail_arm_success=b, c_base_success_arm_fail=c,
            mcnemar_p=round(p, 4), end_reasons=dict(reasons))

    print("\n  失败模式（base vs 各组，占该组条数比）")
    print(f"  {'组':<6}" + "".join(f"{k:>13}" for k in ("fell", "out_of_area", "off_table", "timeout", "success")))
    def fmt(cnt: Counter, n: int) -> str:
        return "".join(f"{cnt.get(k, 0) / max(1, n) * 100:>12.0f}%" for k in ("fell", "out_of_area", "off_table", "timeout", "success"))
    print(f"  {'base':<6}" + fmt(Counter(r.get('end_reason', '?') for r in base_map.values()), len(base_map)))
    for name, g in report["groups"].items():
        print(f"  {name:<6}" + fmt(Counter(g["end_reasons"]), g["n"]))

    print("\n  解读：b 远大于 c 才有'救回来'的意思；p 值看 b/c 差是否可信（n 小时只看方向）")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())