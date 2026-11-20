#!/usr/bin/env python3
"""rl/train.py —— 残差 SAC 训练（spec §9 S2）

为什么是 SAC
------------
1. 观测是**连续低维**（14 维）、动作连续 6 维、每步都要跑一次 ACT 前向 + 渲染，
   env step 比纯 MuJoCo 贵得多 → 需要**样本效率**，off-policy 的 SAC 比 on-policy 的 PPO 合适。
2. 残差是"在基座附近做小幅纠偏"，SAC 的最大熵目标对"保持接近基座"天然友好。

两个硬约束
----------
1. **进程级重试**：`mj_step` 有约 7% 概率触发 SIGFPE，进程会被直接杀掉。
   进程内 try/except 兜不住 → 本脚本支持 `--resume`，崩了由外层循环重启接着跑。
2. **不许为了让曲线好看改奖励量级**（plan §阶段2 失败处理）。

用法
----
```bash
# 3 个 seed，各 3 万步，可中断续跑
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
  .venv-rl/bin/python rl/train.py --seeds 0,1,2 --steps 30000 --resume
```
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="残差 SAC 训练（ACT 冻结基座）")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--base", default="sim156",
                    help="lance/armA/armB/sim156（S0 pilot 结论：sim156 是唯一闭环成功的基座，故设默认）")
    ap.add_argument("--episodes", default="0-5", help="训练用母本编号池")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--swap-cams", action="store_true")
    ap.add_argument("--no-sensor", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ckpt-dir", default=str(ROOT / "rl" / "ckpt"))
    ap.add_argument("--curve-dir", default=str(ROOT / "rl" / "curves"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--log-every", type=int, default=1000)
    a = ap.parse_args(argv)

    from stable_baselines3 import SAC                            # noqa: PLC0415
    from stable_baselines3.common.callbacks import BaseCallback   # noqa: PLC0415
    from env_residual import ResidualSO101                        # noqa: PLC0415

    pool = []
    for part in a.episodes.split(","):
        if not part.strip():
            continue
        if "-" in part:
            lo, hi = part.split("-")
            pool.extend(range(int(lo), int(hi) + 1))
        else:
            pool.append(int(part))

    ck_dir, cv_dir = Path(a.ckpt_dir), Path(a.curve_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)

    class CurveCB(BaseCallback):
        def __init__(self, path: Path, every: int):
            super().__init__()
            self.path, self.every = path, every
            self.ep_rewards, self.ep_success = [], []

        def _on_step(self) -> bool:
            infos = self.locals.get("infos", []) or []
            dones = self.locals.get("dones", None)
            for i, info in enumerate(infos):
                if "episode" not in info:
                    continue
                ep_r = float(info["episode"]["r"])
                is_done = bool(dones[i]) if dones is not None else True
                ep_success = bool(info.get("success")) if is_done else None
                self.ep_rewards.append(ep_r)
                self.ep_success.append(ep_success)
                rec = dict(step=int(self.num_timesteps), ep_reward=ep_r,
                           ep_success=ep_success, episodes=len(self.ep_rewards),
                           mean_last20=round(float(sum(self.ep_rewards[-20:]) / len(self.ep_rewards[-20:])), 3))
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            return True

    summary = []
    for seed in [int(s) for s in a.seeds.split(",") if s.strip()]:
        env = ResidualSO101(base_name=a.base, device=a.device, episode_pool=tuple(pool),
                            swap_cams=a.swap_cams, use_sensor=not a.no_sensor,
                            max_seconds=a.max_seconds, seed=seed)
        model_path = ck_dir / f"s{seed}" / "model.zip"
        curve = cv_dir / f"s{seed}.jsonl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if a.resume and model_path.exists():
            model = SAC.load(str(model_path), env=env, device=a.device)
            done_steps = int(model.num_timesteps)
        else:
            model = SAC("MlpPolicy", env, verbose=0, seed=seed,
                        learning_rate=a.learning_rate, batch_size=a.batch_size,
                        device=a.device, buffer_size=200_000,
                        train_freq=1, gradient_steps=1,
                        policy_kwargs=dict(net_arch=[128, 128]))
            done_steps = 0
            if curve.exists():
                curve.unlink()
        remain = max(0, a.steps - done_steps)
        t0 = time.time()
        print(f"[s{seed}] base={a.base} 已有 {done_steps} 步，本次再跑 {remain} 步", flush=True)
        if remain > 0:
            model.learn(total_timesteps=remain, callback=CurveCB(curve, a.log_every),
                        reset_num_timesteps=False, progress_bar=False)
            model.save(str(model_path))
        env.close()

        rewards, succ_flags = [], []
        if curve.exists():
            for line in curve.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    rewards.append(d["ep_reward"])
                    succ_flags.append(d.get("ep_success"))
        # ★ episode 少时 head/tail 会重叠 → ratio 恒为 1.0，判据失效。
        #   改成按四分位切，且 n<8 时不给 ratio（只报绝对数）。
        q = max(1, len(rewards) // 4)
        head = rewards[:q]
        tail = rewards[-q:]
        rec = dict(seed=seed, base=a.base, steps=int(model.num_timesteps),
                   n_episodes=len(rewards),
                   reward_head_mean=round(float(sum(head) / max(1, len(head))), 3),
                   reward_tail_mean=round(float(sum(tail) / max(1, len(tail))), 3),
                   wall_s=round(time.time() - t0, 1),
                   ckpt=str(model_path), curve=str(curve))
        # ★ reward 是负值（稠密距离惩罚主导），"tail/head ≥ 1.2" 在负值上没意义。
        #   改成相对改善：(tail − head) / |head|，≥0.2 记为达标。
        has_ratio = len(rewards) >= 8
        rec["improve"] = (round((rec["reward_tail_mean"] - rec["reward_head_mean"])
                                / abs(rec["reward_head_mean"]), 3)
                          if (has_ratio and rec["reward_head_mean"]) else None)
        rec["head_n"] = rec["tail_n"] = q
        succ = [x for x in succ_flags if x is not None]
        rec["n_done"] = len(succ)
        rec["success_head"] = round(sum(succ[:q]) / max(1, len(succ[:q])), 3)
        rec["success_tail"] = round(sum(succ[-q:]) / max(1, len(succ[-q:])), 3)
        summary.append(rec)
        print(f"[s{seed}] {json.dumps(rec, ensure_ascii=False)}", flush=True)

    out = cv_dir / "summary.json"
    old = []
    if a.resume and out.exists():
        try:
            old = [r for r in json.loads(out.read_text(encoding="utf-8")).get("runs", [])
                   if r["seed"] not in {s["seed"] for s in summary}]
        except Exception:                                        # noqa: BLE001
            old = []
    out.write_text(json.dumps(dict(runs=old + summary), ensure_ascii=False, indent=1),
                   encoding="utf-8")

    # ---- S2 判据 ----
    imps = [r["improve"] for r in summary if r.get("improve") is not None]
    succs = [r["success_tail"] for r in summary if r.get("success_tail") is not None]
    ok = bool(imps) and float(sum(imps) / len(imps)) >= 0.2
    print(f"\n  S2 判据（负 reward 用相对改善，不用 tail/head 比）：")
    print(f"    相对改善 {imps}  均值 {sum(imps)/len(imps):.3f}  (需 ≥0.20)")
    print(f"    训练期成功率(末四分位) {succs}")
    print(f"    → {'ok' if ok else 'warn（未达 0.20，按 plan §阶段2 查 reward/缩放/episode 长度，不许调奖励量级）'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())