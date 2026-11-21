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


def run_episode(env, policy, *, max_steps=None, video=None, fps=30, width=640, height=480,
                res_scale=1.0, judge="strict"):
    """跑一条 episode。policy=None 表示纯基座（残差=0）。

    res_scale：**消融用**的残差幅度缩放。不改权重、不改动作空间，只把 SAC 输出的残差
    乘一个系数再交给 env（env 自己还会按 ±5.73° 裁剪）。用途：判断"基座被残差打崩"
    是因为残差**幅度太大**，还是因为策略方向本身错。
    """
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
    tilt_max = 0.0
    streak_max = 0
    end_reason = "unknown"
    done = False
    steps = 0
    try:
        while not done:
            if policy is None:
                act = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                act, _ = policy.predict(obs, deterministic=True)
                if res_scale != 1.0:                     # ★ 消融：缩放残差幅度，不碰权重
                    act = np.asarray(act, dtype=np.float32) * float(res_scale)
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            pen_max = max(pen_max, info["penetration_mm"])
            lift_max = max(lift_max, info["max_lift_mm"])
            tilt_max = max(tilt_max, float(info.get("box_tilt_deg", 0.0)))
            streak_max = max(streak_max, int(info.get("lift_streak", 0)))
            if info["max_lift_mm"] < 5.0:            # 抬升前
                pen_pre_lift = max(pen_pre_lift, info["penetration_mm"])
                tilt_pre_lift = max(tilt_pre_lift, info["box_tilt_deg"])
            if sink is not None:
                from test_render import CAM_ENV                  # noqa: PLC0415
                frame = env.rig.grab(env.d)[CAM_ENV]
                sink.write(np.ascontiguousarray(frame[:, :, ::-1]))
            done = bool(term or trunc)
            if done:                                  # 审计用：这一条是怎么结束的
                if info.get("success"):
                    end_reason = "success"
                elif info.get("off_table"):
                    end_reason = "off_table"
                elif info.get("out_of_area"):
                    end_reason = "out_of_area"
                elif info.get("fell"):
                    end_reason = "fell"
                else:
                    end_reason = "timeout"
    finally:
        if sink is not None:
            sink.release()
    success = bool(info["success"])
    # 洁净成功：口径随 judge 走。
    #   strict/legacy：抬升/接触/倾角/持续性都过（历史口径；倾角 ≤15° 在抬升段物理不可达，见报告 §16）
    #   target       ：成功 且 **抓取段手指穿透 ≤10 mm**（专家母本实测 3.1/8.9 mm，ACT 8~23 mm）
    if judge == "target":
        clean_success = bool(success and float(info.get("pen_finger_grasp_max_mm", 99.0)) <= 10.0)
    else:
        clean_success = bool(success and tilt_max <= 15.0 and streak_max >= 5)
    return dict(success=success, steps=steps,
                lift_mm=round(float(lift_max), 2),
                hdisp_mm=round(float(info["hdisp_mm"]), 2),
                G2c_pen_pre_lift_mm=round(float(pen_pre_lift), 3),
                G3c_tilt_pre_lift_deg=round(float(tilt_pre_lift), 2),
                G6c_pen_max_mm=round(float(pen_max), 3),
                pen_finger_grasp_max_mm=round(float(info.get("pen_finger_grasp_max_mm", 0.0)), 3),
                dxy_target_mm=round(float(info.get("dxy_target_mm", -1.0)), 2),
                end_reason=end_reason, tilt_max_deg=round(float(tilt_max), 2),
                lift_streak_max=int(streak_max), clean_success=clean_success)


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
    ap.add_argument("--video", default=None, help="给一条 episode 录像")
    ap.add_argument("--video-tag", default="base+R", choices=["base", "base+R"],
                    help="录像录哪一臂（默认 base+R；录基座要配 --video-tag base 且不给 --policy）")
    ap.add_argument("--judge", default="legacy", choices=["legacy", "strict", "target"],
                    help="成败判定口径：legacy=旧；strict=连续5步洁净成功(历史口径，与任务矛盾，见报告 §16)；"
                         "target=抬升≥50mm 且把药盒放进木盒区域（推荐）")
    ap.add_argument("--jaw-min-deg", type=float, default=0.0,
                    help="夹爪最小指令（度）。0=只把负指令钳到 0（修 bug）；16.05=不许合过头"
                         "（夹 15 mm 盒需要 0.28 rad，标定 0.76 mm/度）")
    ap.add_argument("--reward", default="v1", choices=["v1", "v2", "v2b"],
                    help="奖励版本（只影响 env 内部回报，不影响成败判定；用于与训练配置一致）")
    ap.add_argument("--res-scale", type=float, default=1.0,
                    help="消融：残差幅度缩放（1.0=训练时的幅度；0.25/0.1 用于判断是否幅度过大）")
    ap.add_argument("--skip-base", action="store_true",
                    help="复用 --base-from 里的 base 臂结果（base 对固定 seed 是确定性的）")
    ap.add_argument("--base-from", default=None,
                    help="从该 ab_*.json 读取 base 臂行；按当前 pool 过滤后重算 Wilson")
    ap.add_argument("--resume", action="store_true",
                    help="续跑：跳过 --out 里已完成的 (tag,episode_id,seed)；mj_step 有约 7% 进程级 SIGFPE，崩了就重跑同一条命令")
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

    done_rows = {}
    op0 = Path(a.out)
    if a.resume and op0.exists():
        try:
            prev = json.loads(op0.read_text(encoding="utf-8"))
            for tag, blk in (prev.get("detail") or {}).items():
                for row in blk.get("rows", []):
                    done_rows[(tag, int(row["episode_id"]), int(row["seed"]))] = row
            print(f"  续跑：已有 {len(done_rows)} 条结果", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"  (续跑读取失败，重头跑: {e})", flush=True)

    results = {}
    pairs = [("base", None), ("base+R", pol)]
    if a.skip_base and pol is not None and a.base_from:
        prev = json.loads(Path(a.base_from).read_text(encoding="utf-8"))
        rows = [r for r in prev["detail"]["base"]["rows"]
                if int(r["episode_id"]) in set(pool)]
        if not rows:
            raise SystemExit(f"--base-from 与 --episodes 没有交集: {a.base_from}")
        k = sum(1 for r in rows if r["success"])
        results["base"] = dict(
            n=len(rows), n_success=k, success_rate=wilson(k, len(rows)),
            lift_mm_mean=round(float(np.mean([r["lift_mm"] for r in rows])), 2),
            G2c_max=round(max(r["G2c_pen_pre_lift_mm"] for r in rows), 3),
            G3c_max=round(max(r["G3c_tilt_pre_lift_deg"] for r in rows), 2),
            G6c_max=round(max(r["G6c_pen_max_mm"] for r in rows), 3),
            rows=rows, reused_from=str(a.base_from))
        expected = len(pool) * a.seeds
        if len(rows) != expected:
            print(f"  ⚠️ base 复用行数 {len(rows)} != 期望 {expected}（pool×seeds）；"
                  f"Δ 会按实际行数算，请核对 {a.base_from}", flush=True)
        pairs = [("base+R", pol)]
        print(f"  复用 base 臂：{len(rows)} 条（来自 {Path(a.base_from).name}）", flush=True)

    for tag, policy in pairs:
        if tag == "base+R" and policy is None:
            continue
        # 续跑：先把 --out 里该臂已有的结果全部装回来（包括不在本次 pool 里的 episode），
        # 否则分块跑会把之前的块丢掉。
        rows = [done_rows[k] for k in sorted(done_rows) if k[0] == tag]
        env = ResidualSO101(base_name=a.base, device=a.device, episode_pool=tuple(pool),
                            swap_cams=a.swap_cams, use_sensor=not a.no_sensor,
                            max_seconds=a.max_seconds,
                            judge=a.judge, reward_version=a.reward,
                            jaw_min_deg=a.jaw_min_deg)
        try:
            for eid in pool:
                for s in range(a.seeds):
                    if (tag, int(eid), int(s)) in done_rows:
                        continue        # 已在 rows 里预置，别再 append（否则续跑会让结果翻倍）
                    vid = None
                    if a.video and tag == a.video_tag and eid == pool[0] and s == 0:
                        vid = a.video
                    env.episode_pool = (eid,)          # ★ 只改 pool，别动 _episodes（否则下一条 KeyError）
                    env.reset(seed=s)
                    t0 = time.time()
                    rec = run_episode(env, policy, video=vid, res_scale=a.res_scale,
                                      judge=a.judge)
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
               judge=a.judge, reward=a.reward, base_from=a.base_from,
               res_scale=a.res_scale,
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