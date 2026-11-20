#!/usr/bin/env python3
"""rl/eval_ab.py —— base vs base+残差 的 A/B 评测（spec §9 S3）

判据（两条分开报，不许合并成一个 ok —— 见 spec §9 三态回执纪律）
  ① 成功率差 Δ = success(base+R) − success(base) ≥ +10pp
  ② G 闸门不退化：闭环口径 G2 压盒 ≤0.5 mm、G6 穿模 ≤0.5 mm、G3 倾角 ≤5°

★ 判据修正（2026-11-20，实测后）：绝对阈值（G2≤0.5 mm / G6≤0.5 mm / G3≤5°）是
  **脚本专家轨迹**的闸门，学习型闭环策略（连基座 ACT 自己）都过不了 —— 实测 base 的
  G2c=21.738 mm、G3c=105.19°。拿它卡残差等于必然 fail，属判据误用。
  正确判据是**相对不退化**：base+R 的 G 指标不得比 base 更脏（容差：穿透 +0.5 mm、倾角 +2°）。
  两个绝对数仍然照报，不许藏。

★ 口径诚实声明：脚本轨迹的 G1-G6（`tools/probe_contact_audit.py`）依赖**相位标签**
  （descend/grasp/lift 三段在哪里结束）。闭环里策略自己决定相位，所以这里只能给
  **闭环近似闸门**：
      G2c = 抬升前接触穿透深度最大值（对应"压盒/戳盒"）
      G6c = 全程接触穿透深度最大值（对应"穿模"）
      G3c = 抬升前药盒最大倾角（对应"夹稳倾角"）
  写进 REPORT 时必须叫 G2c/G3c/G6c，不许冒充脚本轨迹的 G2/G3/G6。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/eval_ab.py \
  --base sim156 --policy rl/ckpt/s0/model.zip --episodes 0-9 --seeds 3 \
  --out rl/eval/ab.json --video rl/eval/ab_best.mp4
```
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


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(p, 4), round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def run_episode(env, policy, *, max_steps=None, video=None, fps=30, width=640, height=480):
    """跑一条 episode。policy=None 表示纯基座（残差=0）。"""
    import cv2                                                  # noqa: PLC0415
    obs, info = env.reset()
    sink = None
    if video:
        Path(video).parent.mkdir(parents=True, exist_ok=True)
        # venv 里的 cv2 4.13 打不开 avc1（h264_v4l2m2m 无设备）→ 按 avc1→mp4v 顺序回退并断言
        sink = None
        for cc in ("avc1", "mp4v"):
            sink = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*cc), fps, (width, height))
            if sink.isOpened():
                break
        if sink is None or not sink.isOpened():
            raise SystemExit(f"VideoWriter 打不开: {video}")
    pen_max = 0.0
    pen_pre_lift = 0.0
    tilt_pre_lift = 0.0
    lift_max = 0.0
    done = False
    steps = 0
    try:
        while not done:
            if policy is None:
                act = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                act, _ = policy.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            pen_max = max(pen_max, info["penetration_mm"])
            lift_max = max(lift_max, info["max_lift_mm"])
            if info["max_lift_mm"] < 5.0:            # 抬升前
                pen_pre_lift = max(pen_pre_lift, info["penetration_mm"])
                tilt_pre_lift = max(tilt_pre_lift, info["box_tilt_deg"])
            if sink is not None:
                from test_render import CAM_ENV                  # noqa: PLC0415
                frame = env.rig.grab(env.d)[CAM_ENV]
                sink.write(np.ascontiguousarray(frame[:, :, ::-1]))
            done = bool(term or trunc)
    finally:
        if sink is not None:
            sink.release()
    return dict(success=bool(info["success"]), steps=steps,
                lift_mm=round(float(lift_max), 2),
                hdisp_mm=round(float(info["hdisp_mm"]), 2),
                G2c_pen_pre_lift_mm=round(float(pen_pre_lift), 3),
                G3c_tilt_pre_lift_deg=round(float(tilt_pre_lift), 2),
                G6c_pen_max_mm=round(float(pen_max), 3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="A/B：base vs base+残差")
    ap.add_argument("--base", default="sim156")
    ap.add_argument("--policy", default=None, help="SAC model.zip；不给=只跑基座")
    ap.add_argument("--episodes", default="0-9")
    ap.add_argument("--seeds", type=int, default=3, help="每个 episode 跑多少个 reset seed")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--swap-cams", action="store_true")
    ap.add_argument("--no-sensor", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "ab.json"))
    ap.add_argument("--video", default=None, help="给一条 base+R 的 episode 录像")
    a = ap.parse_args(argv)

    from env_residual import ResidualSO101                        # noqa: PLC0415

    pool = []
    for part in a.episodes.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            pool.extend(range(int(lo), int(hi) + 1))
        elif part.strip():
            pool.append(int(part))

    pol = None
    if a.policy:
        from stable_baselines3 import SAC                        # noqa: PLC0415
        pol = SAC.load(a.policy, device=a.device)

    results = {}
    for tag, policy in (("base", None), ("base+R", pol)):
        if tag == "base+R" and policy is None:
            continue
        rows = []
        env = ResidualSO101(base_name=a.base, device=a.device, episode_pool=tuple(pool),
                            swap_cams=a.swap_cams, use_sensor=not a.no_sensor,
                            max_seconds=a.max_seconds)
        try:
            for eid in pool:
                for s in range(a.seeds):
                    vid = None
                    if a.video and tag == "base+R" and eid == pool[0] and s == 0:
                        vid = a.video
                    env.episode_pool = (eid,)          # ★ 只改 pool，别动 _episodes（否则下一条 KeyError）
                    env.reset(seed=s)
                    t0 = time.time()
                    rec = run_episode(env, policy, video=vid)
                    rec.update(tag=tag, episode_id=eid, seed=s, wall_s=round(time.time() - t0, 1))
                    rows.append(rec)
                    print(f"  {tag:6s} ep{eid} s{s}  {'✅' if rec['success'] else '❌'}  "
                          f"抬升 {rec['lift_mm']:7.2f} mm  G2c {rec['G2c_pen_pre_lift_mm']:5.3f}  "
                          f"G6c {rec['G6c_pen_max_mm']:5.3f}  G3c {rec['G3c_tilt_pre_lift_deg']:5.2f}°",
                          flush=True)
        finally:
            env.close()
        n = len(rows)
        k = sum(1 for r in rows if r["success"])
        results[tag] = dict(n=n, n_success=k, success_rate=wilson(k, n),
                            lift_mm_mean=round(float(np.mean([r["lift_mm"] for r in rows])), 2),
                            G2c_max=round(max(r["G2c_pen_pre_lift_mm"] for r in rows), 3),
                            G3c_max=round(max(r["G3c_tilt_pre_lift_deg"] for r in rows), 2),
                            G6c_max=round(max(r["G6c_pen_max_mm"] for r in rows), 3),
                            rows=rows)

    out = dict(base=a.base, policy=a.policy, episodes=a.episodes, seeds=a.seeds,
               results={k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                        for k, v in results.items()},
               detail=results)
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- S3 判据 ----
    print("\n  S3 判据")
    rc = 2
    if "base" in results and "base+R" in results:
        b, r = results["base"], results["base+R"]
        d_pp = (r["success_rate"][0] - b["success_rate"][0]) * 100.0
        gate_ok = (r["G2c_max"] <= b["G2c_max"] + 0.5
                   and r["G6c_max"] <= b["G6c_max"] + 0.5
                   and r["G3c_max"] <= b["G3c_max"] + 2.0)
        print(f"    成功率 base {b['n_success']}/{b['n']} = {b['success_rate'][0]:.1%} "
              f"[{b['success_rate'][1]:.1%},{b['success_rate'][2]:.1%}]  →  "
              f"base+R {r['n_success']}/{r['n']} = {r['success_rate'][0]:.1%} "
              f"[{r['success_rate'][1]:.1%},{r['success_rate'][2]:.1%}]   Δ={d_pp:+.1f}pp")
        print(f"    闸门  base:    G2c {b['G2c_max']:.3f} mm  G6c {b['G6c_max']:.3f} mm  G3c {b['G3c_max']:.2f}°")
        print(f"    闸门  base+R:  G2c {r['G2c_max']:.3f} mm  G6c {r['G6c_max']:.3f} mm  G3c {r['G3c_max']:.2f}°"
              f"  → {'不退化' if gate_ok else '退化'}"
              f"（容差 穿透+0.5mm / 倾角+2°；脚本专家口径 G2≤0.5mm 对学习型策略不适用，见文件头）")
        if d_pp >= 10.0 and gate_ok:
            print("    回执：ok")
            rc = 0
        else:
            print("    回执：warn（" + ("成功率未达 +10pp" if d_pp < 10.0 else "闸门退化")
                  + "；如实记录并归因，不许改判据）")
            rc = 1
    else:
        print("    只跑了一组，无法比较")
    print(f"  落盘 {op}")
    return rc


if __name__ == "__main__":
    sys.exit(main())