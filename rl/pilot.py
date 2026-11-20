#!/usr/bin/env python3
"""rl/pilot.py —— S0：纯真机 ACT 在 MuJoCo 闭环里到底能不能动？

为什么先做这一步
----------------
仓库里"ACT 闭环成功率"这个数**从来没有被测量过**（`logs/` 只有 `rollout_help.log`
和离线 L1）。而残差 RL 全部建立在"基座能动"之上。所以先用最小成本把这个不确定性
打掉：2 个纯真机 checkpoint × 4 种观测口径 × 1 条 episode，出一张对照表。

口径（四组）
------------
  raw         原样（当前 replay_policy 的行为）
  swap        只对调 env/wrist 两路图（补偿 yyl100_lance_fixed 的标签互换）
  sensor      只过 SensorModel（补偿"闭环没做传感器劣化"）
  swap+sensor 两者都做

判据（spec §9 S0）
------------------
  成功 = 全程最大抬升 ≥ 50 mm（与 test_render.evaluate_success / replay_policy 同口径）
  任一组 ≥ 1/3 → 采纳该组口径当主臂；全 0 → 按 spec §5 fallback 下移基座。

注意
----
- 本脚本**不落视频、不写 ctrl 以外的状态**：残差/基座只能通过 ctrl + mj_step 生效（红线 2）。
- `mj_step` 有约 7% 进程级 SIGFPE，进程会被直接杀掉 → 每跑完一组就落盘一次，
  外层用 `--resume` 续跑（不要试图在进程内 try/except 兜住 SIGFPE，兜不住）。
- 判据用"全程最大抬升"，不是"最后一帧还在空中"。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

COMBOS = {
    "raw": dict(swap_cams=False, use_sensor=False),
    "swap": dict(swap_cams=True, use_sensor=False),
    "sensor": dict(swap_cams=False, use_sensor=True),
    "swap+sensor": dict(swap_cams=True, use_sensor=True),
}


def load_episode(scripts: Path, eid: int):
    p = Path(scripts) / f"ep_{eid:06d}.npz"
    j = Path(scripts) / f"ep_{eid:06d}.json"
    if not p.exists():
        raise SystemExit(f"没有母本轨迹: {p}")
    with np.load(p, allow_pickle=True) as z:
        ep = {k: np.array(z[k]) for k in z.files if k != "meta_json"}
    meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
    return ep, meta


def run_one(arm, rig, base, ep, task, *, duration, control_hz, R, mujoco, width, height,
            state_from="ctrl"):
    """一条闭环 episode，返回指标 dict。"""
    m, d = arm.m, arm.d
    d.qpos[:] = ep["qpos"][0]
    d.qvel[:] = 0.0
    d.ctrl[:] = ep["ctrl"][0]
    mujoco.mj_forward(m, d)

    base.reset(task)
    obj0 = arm.obj_pos().copy()
    t0 = d.time
    step_dt = 1.0 / control_hz
    n_steps = int(round(duration * control_hz))

    max_lift = -1e9
    max_contact = 0
    first_contact = None
    q_travel = 0.0
    acts = []
    prev_q = np.array([d.qpos[i] for i in arm.adr], dtype=float)
    prev_uv = {c: None for c in R.CAMS}
    speeds = {c: 0.0 for c in R.CAMS}
    lift_end = 0.0
    hdisp = 0.0
    tcp_box_min = 1e9

    for k in range(n_steps):
        raw = rig.grab(d)

        # 运动模糊要跟着真实运动：用上一帧 TCP 在图像平面的像素位移算 speed_px
        uv_now = {}
        for c in R.CAMS:
            uv = R._project_px(rig, d, c, arm.tcp_pos())
            if uv is not None and prev_uv[c] is not None:
                speeds[c] = float(np.linalg.norm(uv - prev_uv[c]))
            prev_uv[c] = uv
            uv_now[c] = uv

        from replay_policy import _ctrl_from_deg, _deg_from_ctrl, _deg_from_qpos  # noqa: PLC0415
        # ★ 真机数据里的 observation.state 是电机实测位置（≈qpos），不是下发目标（ctrl）；
        #   sim 里两者有 PD 滞后差，选错就是给策略喂 OOD 观测。
        state = _deg_from_ctrl(arm, d.ctrl) if state_from == "ctrl" else _deg_from_qpos(arm)
        a_deg = base.act(state, raw, speeds)
        d.ctrl[:] = _ctrl_from_deg(arm, a_deg)

        while d.time - t0 < (k + 1) * step_dt:
            mujoco.mj_step(m, d)

        obj = arm.obj_pos()
        lift = (obj[2] - obj0[2]) * 1000.0
        max_lift = max(max_lift, lift)
        lift_end = lift
        hdisp = float(np.linalg.norm(obj[:2] - obj0[:2])) * 1000.0
        n_con = len(arm.grip_contacts())
        max_contact = max(max_contact, n_con)
        if n_con > 0 and first_contact is None:
            first_contact = k
        q_now = np.array([d.qpos[i] for i in arm.adr], dtype=float)
        q_travel += float(np.abs(np.degrees(q_now - prev_q)).sum())
        prev_q = q_now
        acts.append(np.asarray(a_deg, float))
        tcp_box_min = min(tcp_box_min, float(np.linalg.norm(arm.tcp_pos() - obj)))

    acts = np.asarray(acts, float) if acts else np.zeros((1, 6))
    tilt = float(np.degrees(np.arccos(np.clip(
        d.xmat[arm.obj].reshape(3, 3)[2, 2], -1.0, 1.0))))
    return dict(
        n_steps=n_steps, sim_s=round(float(d.time - t0), 2),
        lift_mm=round(float(max_lift), 2), lift_end_mm=round(float(lift_end), 2),
        hdisp_mm=round(float(hdisp), 2),
        success=bool(max_lift >= 50.0),
        max_contact=int(max_contact),
        first_contact_step=first_contact,
        box_tilt_deg=round(tilt, 2),
        tcp_box_min_mm=round(tcp_box_min * 1000.0, 2),
        q_travel_deg=round(q_travel, 1),
        act_abs_max_deg=round(float(np.abs(acts).max()), 2),
        act_std_deg=round(float(acts.std(axis=0).mean()), 3),
        act_step_mean_abs_deg=round(float(np.abs(np.diff(acts, axis=0)).mean()) if len(acts) > 1 else 0.0, 3),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S0 pilot：真机 ACT 的 MuJoCo 闭环对照表")
    ap.add_argument("--ckpts", default="lance,armA", help="逗号分隔（lance/armA/armB/sim156 或路径）")
    ap.add_argument("--episodes", default="0", help="母本编号，逗号分隔或 a-b")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--control-hz", type=float, default=30.0)
    ap.add_argument("--combos", default=",".join(COMBOS), help="raw,swap,sensor,swap+sensor")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--state-from", choices=["ctrl", "qpos"], default="ctrl",
                    help="喂给策略的关节角取 ctrl(下发目标) 还是 qpos(实测)；真机 data 更接近 qpos")
    ap.add_argument("--scripts", default=str(ROOT / "scripts"))
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "pilot.json"))
    ap.add_argument("--resume", action="store_true", help="跳过 out 里已有结果的组合")
    a = ap.parse_args(argv)

    import mujoco                                              # noqa: PLC0415
    import test_render as R                                    # noqa: PLC0415
    from base_policy import make_base                          # noqa: PLC0415

    eids = []
    for part in a.episodes.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            eids.extend(range(int(lo), int(hi) + 1))
        else:
            eids.append(int(part))
    combo_names = [c.strip() for c in a.combos.split(",") if c.strip()]
    ckpt_names = [c.strip() for c in a.ckpts.split(",") if c.strip()]

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if a.resume and out_path.exists():
        try:
            done = json.loads(out_path.read_text(encoding="utf-8")).get("runs", {})
        except Exception:
            done = {}
    runs = dict(done)

    arm = R.Arm(R.SCENE_XML)
    rig = R.CameraRig(arm.m, width=a.width, height=a.height)
    rc = 0
    try:
        for cname in ckpt_names:
            base = make_base(cname, device=a.device)   # 每个 ckpt 只加载一次，combo 只改 adapter
            for eid in eids:
                ep, meta = load_episode(Path(a.scripts), eid)
                task = meta.get("instruction") or "Pick up the medicine box and place it into the wooden box."
                for combo in combo_names:
                    key = f"{cname}|ep{eid}|{combo}|{a.state_from}"
                    if key in runs and runs[key].get("ok"):
                        print(f"  [skip] {key}")
                        continue
                    cfgr = COMBOS[combo]
                    base.adapter.swap_cams = cfgr["swap_cams"]
                    base.adapter.use_sensor = cfgr["use_sensor"]
                    t1 = time.time()
                    try:
                        m = run_one(arm, rig, base, ep, task, duration=a.duration,
                                    control_hz=a.control_hz, R=R, mujoco=mujoco,
                                    width=a.width, height=a.height,
                                    state_from=a.state_from)
                        m.update(ok=True, ckpt=cname, episode_id=eid, combo=combo,
                                 state_from=a.state_from,
                                 swap_cams=cfgr["swap_cams"], use_sensor=cfgr["use_sensor"],
                                 wall_s=round(time.time() - t1, 1))
                    except SystemExit:
                        raise
                    except Exception as e:                     # noqa: BLE001
                        m = dict(ok=False, ckpt=cname, episode_id=eid, combo=combo,
                                 error=f"{type(e).__name__}: {e}",
                                 wall_s=round(time.time() - t1, 1))
                    runs[key] = m
                    out_path.write_text(json.dumps(dict(runs=runs), ensure_ascii=False, indent=1),
                                        encoding="utf-8")
                    flag = "✅" if m.get("success") else ("❌" if m.get("ok") else "💥")
                    if m.get("ok"):
                        print(f"  {flag} {key:28s} 抬升 {m['lift_mm']:+8.2f} mm  "
                              f"接触峰 {m['max_contact']:2d}  关节行程 {m['q_travel_deg']:7.1f}°  "
                              f"动作std {m['act_std_deg']:.2f}°  {m['wall_s']}s")
                    else:
                        print(f"  {flag} {key:28s} {m['error']}")
    finally:
        rig.close()

    # ---- 汇总：谁满足 S0 判据 ----
    table = {}
    for k, v in runs.items():
        if not v.get("ok"):
            continue
        table.setdefault(v["ckpt"], {}).setdefault(v["combo"], []).append(v["success"])
    summary = {ck: {cb: f"{sum(s)}/{len(s)}" for cb, s in cbs.items()} for ck, cbs in table.items()}
    print("\n  汇总（成功 = 全程最大抬升 ≥ 50 mm）")
    for ck, cbs in summary.items():
        row = "  ".join(f"{cb}:{n}" for cb, n in cbs.items())
        print(f"    {ck:8s} {row}")
    out = json.loads(out_path.read_text(encoding="utf-8"))
    out["summary"] = summary
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  落盘 {out_path}")

    any_ok = any(s != "0" and not s.startswith("0/") for cbs in summary.values() for s in cbs.values())
    if not any_ok:
        print("  ⚠️ 全部 0 命中：按 spec §5 下移基座（L3 sim156 → L4 脚本专家）；"
              "这本身是一条可交付的负结论，别删。")
        rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())