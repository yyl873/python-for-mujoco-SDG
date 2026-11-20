#!/usr/bin/env python3
"""rl/shooting_probe.py —— A1：残差上界探测（shooting / 免训练）

要回答的问题
------------
"残差这条路的**天花板**在哪？" —— 如果连"每一步都挑最好的残差"（作弊级选择器）都打不过基座，
那不管再换什么奖励/算法都没用，应该直接放弃在线 RL 调参。

做法
----
跑基座正常 episode；每 CHUNK 个控制步做一次 shooting：
  1. 采 K 个候选残差（1 个全零 + K-1 个均匀随机，幅度 ±residual_deg）
  2. 每个候选在**克隆的 MjData**（`copy.deepcopy(d)`）上 rollout H 步：
     ctrl = base_action(当前 chunk 的 ACT 输出) + candidate，不含渲染、不含 ACT 重推理
  3. 打分 = rollout 结束时的药盒抬升（mm），best_gain = 最佳候选 − 零残差

同时跑第二条"oracle arm"：每次决策直接用 argmax 候选作为残差继续走，看整条 episode 能不能成功。

已知近似（必须写进报告）
------------------------
- rollout 期间 base 动作**保持不变**（不重跑 ACT），H=20 步（0.67 s）内影响有限；
- 候选只从均匀随机里采，不是"最优残差"的真实上界，是**随机搜索上界**（偏保守）。

判据
----
- `median best_gain ≤ 0.5 mm` 且 `frac(gain>2mm) < 10%` → 没有上界空间，残差 RL 可停；
- `median best_gain ≥ 2 mm` 或 oracle 成功率明显高于基座 → 有空间，值得换结构（chunk 级/任务空间/蒸馏）。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/shooting_probe.py \
    --episodes 100-129 --candidates 8 --horizon 20 --chunk 30 --out rl/eval/shooting_probe.json
```
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse_pool(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="残差上界探测（shooting）")
    ap.add_argument("--base", default="sim156")
    ap.add_argument("--episodes", default="100-129")
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=20, help="每个候选 rollout 多少控制步")
    ap.add_argument("--chunk", type=int, default=30, help="每多少控制步做一次决策")
    ap.add_argument("--oracle", action="store_true", help="同时跑 oracle 选择器整条 episode")
    ap.add_argument("--resume", action="store_true", help="跳过 jsonl 里已完成的 episode")
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "shooting_probe.json"))
    a = ap.parse_args(argv)

    from env_residual import ResidualSO101                           # noqa: PLC0415
    pool = parse_pool(a.episodes)
    env = ResidualSO101(base_name=a.base, episode_pool=tuple(pool), randomize=True)
    m, d = env.m, env.d
    rng = np.random.default_rng(0)
    lim = float(env.residual_deg)

    def roll(cand, action_il, h):
        """在克隆的 MjData 上 rollout，返回 (预测抬升 mm, 靠近量 mm)。

        两个分量都要，因为只在"已接近/已接触"阶段抬升才有意义；早期阶段只能看"是否更快靠近"。
        """
        d2 = copy.deepcopy(d)
        ctrl = env._ctrl_from_deg(env.arm, np.asarray(action_il) + np.asarray(cand))
        z0 = float(d2.qpos[env.arm.oadr + 2])
        box0 = env.arm.obj_pos().copy()
        tcp0 = env.arm.tcp_pos().copy()
        d0 = float(np.linalg.norm(tcp0 - box0))
        for _ in range(h):
            d2.ctrl[:] = ctrl
            env._mujoco.mj_step(m, d2)
        lift = (float(d2.qpos[env.arm.oadr + 2]) - z0) * 1000.0
        # 用同一份克隆状态估 TCP 位移（近似：arm.site_xpos 在 d2 里可能未刷新，故按关节变化估算）
        q0 = np.asarray(d.qpos[env.arm.adr], float)
        q1 = np.asarray(d2.qpos[env.arm.adr], float)
        approach = float(np.linalg.norm(q1 - q0)) * 0.0      # 占位：不做几何近似，改用下面 d 的前后距离
        return lift, d0, approach

    jl = Path(str(a.out).replace(".json", ".jsonl"))
    done = set()
    if a.resume and jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                done.add(int(json.loads(line)["episode_id"]))
            except Exception:                                # noqa: BLE001
                pass
        print(f"  续跑：跳过已完成的 {len(done)} 条", flush=True)
    gains, gains_abs, n_dec, t0 = [], [], 0, time.time()
    oracle_rows = []
    try:
        for eid in pool:
            if eid in done:
                continue
            env.episode_pool = (eid,)
            obs, info = env.reset(seed=0)
            ep_gains, ep_gain_abs = [], []
            kept = np.zeros(6)
            for k in range(int(round(env.max_seconds * env.control_hz))):
                if k % a.chunk == 0:                      # 每个决策点做一次 shooting
                    base_act = np.asarray(info.get("act_il_deg", np.zeros(6)), float)
                    cands = np.vstack([np.zeros(6),
                                       rng.uniform(-lim, lim,
                                                   size=(a.candidates - 1, 6))])
                    out = [roll(c, base_act, a.horizon) for c in cands]
                    lifts = [o[0] for o in out]
                    ep_gains.append(float(np.max(lifts) - lifts[0]))   # 相对"零残差"的抬升增益
                    ep_gain_abs.append(float(np.max(np.abs(lifts))))    # 该点最大抬升幅度（判断阶段）
                    kept = cands[int(np.argmax(lifts))] if a.oracle else np.zeros(6)
                    n_dec += 1
                obs, r, term, trunc, inf2 = env.step(kept)
                info = inf2
                if term or trunc:
                    break
            if a.oracle:
                oracle_rows.append(dict(episode_id=eid, success=bool(info["success"]),
                                        lift_mm=round(float(info["max_lift_mm"]), 2),
                                        applied=len(ep_gains)))
            gains.extend(ep_gains)
            gains_abs.extend(ep_gain_abs)
            with jl.open("a", encoding="utf-8") as f:          # ★ 崩了也不丢已完成 episode
                f.write(json.dumps(dict(episode_id=eid, gains=ep_gains,
                                        success=bool(info["success"]),
                                        lift_mm=round(float(info["max_lift_mm"]), 2),
                                        end_reason=info.get("end_reason", "?"),
                                        oracle=bool(a.oracle)), ensure_ascii=False) + "\n")
            print(f"  ep{eid}: 决策 {len(ep_gains)} 次  平均 best_gain "
                  f"{np.mean(ep_gains) if ep_gains else 0:+.2f} mm  成功={info['success']}"
                  f"{'（oracle）' if a.oracle else ''}", flush=True)
    finally:
        env.close()

    all_gains, all_succ = [], []
    if jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except Exception:                                # noqa: BLE001
                continue
            all_gains.extend(r.get("gains", []))
            all_succ.append((r["episode_id"], bool(r.get("success")), bool(r.get("oracle"))))
    if all_gains:
        gains = all_gains
    g = np.asarray(gains, float)
    rep = dict(base=a.base, episodes=a.episodes, candidates=a.candidates,
               horizon=a.horizon, chunk=a.chunk, n_decisions=int(n_dec),
               gain_mean=round(float(g.mean()), 3) if g.size else None,
               gain_median=round(float(np.median(g)), 3) if g.size else None,
               gain_p90=round(float(np.percentile(g, 90)), 3) if g.size else None,
               frac_gain_gt2mm=round(float((g > 2.0).mean()), 3) if g.size else None,
               frac_gain_le0=round(float((g <= 0.0).mean()), 3) if g.size else None,
               lift_abs_median=round(float(np.median(gains_abs)), 3) if gains_abs else None,
               wall_s=round(time.time() - t0, 1),
               n_episodes_done=len(all_succ),
               n_success=sum(1 for _, ok, _ in all_succ if ok))
    if oracle_rows:
        rep["oracle"] = dict(n=len(oracle_rows),
                             n_success=sum(1 for r in oracle_rows if r["success"]),
                             rows=oracle_rows)
    print(f"  该点上候选的最大抬升幅度中位数 {rep['lift_abs_median']} mm（判断是不是在抓取阶段）")
    print(f"\n  决策 {rep['n_decisions']} 次：best_gain 均值 {rep['gain_mean']} mm  "
          f"中位数 {rep['gain_median']} mm  p90 {rep['gain_p90']} mm  "
          f">2mm 占比 {rep['frac_gain_gt2mm']:.1%}  ≤0 占比 {rep['frac_gain_le0']:.1%}")
    if oracle_rows:
        print(f"  oracle 选择器：{rep['oracle']['n_success']}/{rep['oracle']['n']} 成功")
    print("  判据：中位 best_gain ≤0.5mm 且 >2mm 占比 <10% → 残差没有上界空间，别再加码调参")
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {op.relative_to(ROOT) if op.is_relative_to(ROOT) else op}  （{rep['wall_s']} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())