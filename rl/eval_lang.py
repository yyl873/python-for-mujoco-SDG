#!/usr/bin/env python3
"""rl/eval_lang.py —— 语言条件的**反事实**评测：同一场景，只换指令。

要回答的问题
------------
"策略到底有没有用指令？" 唯一的检验方法是把指令单独换掉、其它一切不变：

    同一个 (episode_id, seed) × 同一个 ckpt
        ├─ 给 "…into the LEFT wooden box"  → 的药盒最终落在哪个箱子？
        └─ 给 "…into the RIGHT wooden box" → 的药盒最终落在哪个箱子？

主指标
------
    follow_acc = 落进"指令指定的那个箱子"的 rollout 数 / 落进**任一**箱子的 rollout 数
    chance = 50%（若策略完全忽略指令、只按视觉惯性推，两个箱子对称 → 约一半一半）

判据（**预先注册，看完数据不许改**）
------------------------------------
    语言生效  : follow_acc ≥ 0.80 且 Wilson 95% 下界 > 0.50
    语言未生效: follow_acc ≈ 0.50（等于抛硬币）→ 负结果，如实写，不许改判据

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/eval_lang.py \
    --ckpt /home/yyl/lerobot/outputs/train/act_lang_dual/checkpoints/last/pretrained_model \
    --episodes 2060-2079 --seeds 2 --ds rl/lang_ds/dual
```

依赖的约定
----------
- `--ds` 指向用 `collect/export_lerobot.py --lang-onehot` 导出的数据集；本脚本从
  `meta/sim_provenance.json` 读 `task_order`（指令→one-hot 下标），保证评测与训练同序。
- 两个木箱的世界坐标默认取 `target_box (0.290, -0.132)` 与 `target_box_b (0.206, 0.245)`；
  落点判据与采集验收同口径（xy 内腔 + 底部高度带）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 与 collect/collect.py 的验收判据同口径（木箱内腔 x±53.5 / y±67 mm，底板顶面 z=4mm）
BOXES = {
    "target_box":   (0.290, -0.132),   # 左箱（默认）
    "target_box_b": (0.206,  0.245),   # 右箱（VLA 新增）
}
BOX_INNER_X = 0.050
BOX_INNER_Y = 0.062
BOX_DROP_Z = (0.008, 0.045)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def binom_test_two_sided(k: int, n: int, p0: float = 0.5) -> float:
    """精确二项检验（双侧），用来和 chance=50% 比。"""
    if n == 0:
        return 1.0
    from math import comb
    probs = [comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(min(1.0, sum(p for p in probs if p <= obs + 1e-12)))


def classify_box(pos) -> str | None:
    """药盒最终落点属于哪个箱子；都不满足返回 None。"""
    pos = np.asarray(pos, float)
    for name, (cx, cy) in BOXES.items():
        if (abs(pos[0] - cx) <= BOX_INNER_X and abs(pos[1] - cy) <= BOX_INNER_Y
                and BOX_DROP_Z[0] <= pos[2] <= BOX_DROP_Z[1]):
            return name
    return None


def load_task_order(ds: Path | None, fallback: list[str]) -> list[str]:
    if ds is not None:
        prov = ds / "meta" / "sim_provenance.json"
        if prov.exists():
            d = json.loads(prov.read_text(encoding="utf-8"))
            order = d.get("task_order") or d.get("tasks")
            if order:
                return list(order)
    return fallback


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="语言条件 ACT 的反事实评测")
    ap.add_argument("--ckpt", required=True, help="ACT checkpoints/.../pretrained_model 目录")
    ap.add_argument("--ds", default=None, help="训练用数据集（读 task_order；可省）")
    ap.add_argument("--base", default="sim156", help="环境基座名（只用于构建环境）")
    ap.add_argument("--episodes", default="2060-2079", help="held-out 编号池，如 2060-2079")
    ap.add_argument("--seeds", type=int, default=2, help="每个 episode 跑几个 reset seed")
    ap.add_argument("--task-a", default="Put the medicine box into the left wooden box.")
    ap.add_argument("--task-b", default="Put the medicine box into the right wooden box.")
    ap.add_argument("--box-a", default="target_box")
    ap.add_argument("--box-b", default="target_box_b")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--jaw-min-deg", type=float, default=16.05,
                    help="夹爪下限（与采集/评测同口径；16.05=修掉夹爪 bug 的配置）")
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "lang_counterfactual.json"))
    ap.add_argument("--no-instr", action="store_true",
                    help="控制组：不给指令向量（state 仍 6 维，可用于验证评测框架本身，"
                         "例如用 sim156 基座跑一遍，预期它只会放进一个箱子 → follow≈50%）")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)

    from base_policy import ACTBase                              # noqa: PLC0415
    from env_residual import ResidualSO101                       # noqa: PLC0415

    pool = []
    for part in a.episodes.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            pool.extend(range(int(lo), int(hi) + 1))
        elif part:
            pool.append(int(part))

    order = load_task_order(Path(a.ds) if a.ds else None, [a.task_a, a.task_b])
    if a.task_a not in order or a.task_b not in order:
        print(f"⚠️  训练集里的指令顺序 {order} 不包含 {a.task_a!r}/{a.task_b!r}，用前两个", flush=True)
        order = [a.task_a, a.task_b]
    n_instr = len(order)
    ix_a, ix_b = order.index(a.task_a), order.index(a.task_b)

    def onehot(ix: int) -> np.ndarray:
        v = np.zeros(n_instr, dtype=np.float32)
        v[ix] = 1.0
        return v

    print("=" * 78)
    print(" 语言条件反事实评测")
    print("=" * 78)
    print(f"  ckpt    : {a.ckpt}")
    print(f"  指令顺序: {order}（one-hot {n_instr} 维）")
    print(f"  对照    : A={a.task_a!r}→{a.box_a}  vs  B={a.task_b!r}→{a.box_b}")
    print(f"  场景池  : {pool[0]}-{pool[-1]}（{len(pool)} 条）× {a.seeds} seed "
          f"= {len(pool) * a.seeds} 配对")
    print()

    out_path = Path(a.out)
    rows: list[dict] = []
    done = set()
    if a.resume and out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            rows = prev.get("rows") or []
            done = {(int(r["episode_id"]), int(r["seed"]), str(r["instr"])) for r in rows}
            print(f"  续跑：已有 {len(rows)} 行")
        except Exception:                                        # noqa: BLE001
            rows, done = [], set()

    def run_one(eid: int, seed: int, instr: str, ix: int) -> dict:
        env = ResidualSO101(base_name=a.base, episode_pool=(eid,),
                            max_seconds=a.max_seconds, judge="target",
                            jaw_min_deg=a.jaw_min_deg,
                            instr_onehot=(None if a.no_instr else onehot(ix)),
                            device=a.device, seed=seed)
        try:
            env.episode_pool = (eid,)
            obs, info = env.reset(seed=seed)
            n = 0
            for _ in range(int(round(env.max_seconds * env.control_hz))):
                obs, r, term, trunc, info = env.step(np.zeros(6, np.float64))
                n += 1
                if term or trunc:
                    break
            pos = np.asarray(env.d.qpos[env.arm.oadr:env.arm.oadr + 3], float)
            box = classify_box(pos)
            return dict(episode_id=eid, seed=seed, instr=instr, ix=ix,
                        landed_box=box, end_pos=[round(float(v), 5) for v in pos],
                        steps=n, wall_s=None)
        finally:
            env.close()

    t0 = time.time()
    for eid in pool:
        for sd in range(a.seeds):
            for instr, ix in ((a.task_a, ix_a), (a.task_b, ix_b)):
                if (eid, sd, instr) in done:
                    continue
                rec = run_one(eid, sd, instr, ix)
                rec["wall_s"] = round(time.time() - t0, 1)
                rows.append(rec)
                print(f"  ep{eid} s{sd}  {instr[-18:]!r:22s} → 落在 {rec['landed_box']}  "
                      f"({rec['steps']} 步)", flush=True)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(
                    dict(ckpt=a.ckpt, task_order=order, task_a=a.task_a, task_b=a.task_b,
                         box_a=a.box_a, box_b=a.box_b, rows=rows),
                    ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------------- 汇总 ----------------
    expect = {a.task_a: a.box_a, a.task_b: a.box_b}
    landed_any = [r for r in rows if r["landed_box"] in BOXES]
    correct = [r for r in landed_any if r["landed_box"] == expect[r["instr"]]]
    k, n = len(correct), len(landed_any)
    lo, hi = wilson(k, n)
    p_binom = binom_test_two_sided(k, n, 0.5)

    # 配对层面：同一 (episode,seed) 两条 rollout 都跟随指令
    pairs: dict[tuple[int, int], dict] = {}
    for r in rows:
        pairs.setdefault((r["episode_id"], r["seed"]), {})[r["instr"]] = r
    both_ok = sum(1 for d in pairs.values()
                  if d.get(a.task_a, {}).get("landed_box") == a.box_a
                  and d.get(a.task_b, {}).get("landed_box") == a.box_b)
    any_landed = sum(1 for d in pairs.values()
                     if any(r.get("landed_box") in BOXES for r in d.values()))

    summary = dict(
        n_rollouts=len(rows), n_landed_in_some_box=n,
        follow_correct=k, follow_acc=round(k / n, 4) if n else None,
        follow_wilson=[round(lo, 4), round(hi, 4)],
        binom_p_vs_chance=round(p_binom, 5),
        n_pairs=len(pairs), pairs_both_follow=both_ok,
        pairs_any_landed=any_landed,
        neither_rate=round(1 - n / max(1, len(rows)), 4),
        verdict=("语言生效" if (n and k / n >= 0.80 and lo > 0.50)
                 else ("语言未被使用（≈抛硬币）" if n and abs(k / n - 0.5) < 0.15
                       else "证据不足")),
        wall_s=round(time.time() - t0, 1),
    )
    print()
    print("  结果")
    print(f"    配对场景      : {len(pairs)}（每条跑 A/B 两次，共 {len(rows)} 次 rollout）")
    print(f"    落进箱子      : {n}/{len(rows)}（其余既不在 A 也不在 B → 任务失败）")
    print(f"    跟随指令      : {k}/{n} = "
          f"{100 * k / n if n else 0:.1f}%  Wilson [{100 * lo:.0f}%, {100 * hi:.0f}%]  "
          f"chance=50% 二项检验 p={p_binom:.4g}")
    print(f"    配对全对      : {both_ok}/{any_landed}（同场景两条都跟随才有意义）")
    print(f"    判定          : {summary['verdict']}")
    print(f"    （判据：follow_acc ≥80% 且 Wilson 下界 >50%）")

    payload = dict(ckpt=a.ckpt, task_order=order, task_a=a.task_a, task_b=a.task_b,
                   box_a=a.box_a, box_b=a.box_b, summary=summary, rows=rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {out_path}（{out_path.stat().st_size / 1024:.0f} KB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())