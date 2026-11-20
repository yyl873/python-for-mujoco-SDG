#!/usr/bin/env python3
"""rl/probe_residual.py —— 诊断：残差策略到底往基座上叠加了多大的动作？

为什么需要它
------------
S3 实测：`base+R` 的闭环成功率从 26.7% 掉到 0.0%，抬升均值从 15.3 mm 掉到 0.96 mm。
在写"残差有害"这个结论之前，必须先回答一个更基本的问题：
**策略输出的残差幅度有多大？** 如果它长期贴着 ±5.73° 的上限，那"有害"的原因就不是
"策略方向错"，而是"动作空间给得太大，等于给基座叠了全幅噪声"。

本脚本不改任何东西，只跑一条 episode 并统计：
  · 每关节 |残差| 均值 / 最大值
  · 饱和比例（|a| > 0.9 × 上限）
  · 与基座自身逐步动作幅度（pilot 实测 act_step_mean_abs_deg ≈ 1.17°）的对比

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/probe_residual.py \
    --policy rl/ckpt/s1/model.zip --episodes 0 --steps 150
```
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="诊断残差幅度（是否饱和/是否过大）")
    ap.add_argument("--policy", default=str(ROOT / "rl" / "ckpt" / "s1" / "model.zip"))
    ap.add_argument("--base", default="sim156")
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "residual_probe.json"))
    a = ap.parse_args(argv)

    from stable_baselines3 import SAC                              # noqa: PLC0415
    from env_residual import ResidualSO101, RESIDUAL_DEG           # noqa: PLC0415

    pool = tuple(int(x) for x in a.episodes.split(",") if x.strip())
    env = ResidualSO101(base_name=a.base, episode_pool=pool)
    pol = SAC.load(a.policy, device="cpu")
    acts, rewards, lifts = [], [], []
    obs, _ = env.reset(seed=0)
    try:
        for _ in range(a.steps):
            act, _ = pol.predict(obs, deterministic=True)
            acts.append(np.asarray(act, dtype=float))
            obs, r, term, trunc, info = env.step(act)
            rewards.append(float(r))
            lifts.append(float(info["max_lift_mm"]))
            if term or trunc:
                obs, _ = env.reset()
    finally:
        env.close()

    arr = np.abs(np.asarray(acts))                                  # (N, 6)
    sat = float(np.mean(arr > 0.9 * RESIDUAL_DEG))
    per_joint = [dict(joint=JOINTS[i],
                      abs_mean_deg=round(float(arr[:, i].mean()), 3),
                      abs_max_deg=round(float(arr[:, i].max()), 3),
                      sat_frac=round(float(np.mean(arr[:, i] > 0.9 * RESIDUAL_DEG)), 3))
                 for i in range(arr.shape[1])]
    rep = dict(policy=a.policy, base=a.base, steps=len(arr),
               residual_limit_deg=round(RESIDUAL_DEG, 3),
               abs_mean_deg=round(float(arr.mean()), 3),
               abs_p95_deg=round(float(np.percentile(arr, 95)), 3),
               abs_max_deg=round(float(arr.max()), 3),
               saturation_frac=sat,
               base_step_ref_deg=1.17,     # pilot 实测 act_step_mean_abs_deg（raw 档）
               per_joint=per_joint,
               reward_sum=round(sum(rewards), 2),
               max_lift_mm=round(max(lifts), 2) if lifts else None)

    print(f"残差幅度（上限 ±{RESIDUAL_DEG:.2f}°）  均值 {rep['abs_mean_deg']:.2f}°  "
          f"p95 {rep['abs_p95_deg']:.2f}°  最大 {rep['abs_max_deg']:.2f}°  "
          f"饱和比例 {sat:.1%}")
    print(f"  对照：基座自身逐步动作幅度 ≈ 1.17°（pilot raw 档实测）")
    print(f"  {'关节':<14}{'|a|均值':>9}{'|a|最大':>9}{'饱和':>8}")
    for r in per_joint:
        print(f"  {r['joint']:<14}{r['abs_mean_deg']:>9.2f}{r['abs_max_deg']:>9.2f}"
              f"{r['sat_frac']:>8.0%}")
    print(f"  本段 reward_sum={rep['reward_sum']}  max_lift={rep['max_lift_mm']} mm")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {Path(a.out).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())