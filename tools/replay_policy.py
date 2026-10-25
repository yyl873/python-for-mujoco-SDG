#!/usr/bin/env python3
"""
replay_policy.py —— 把训练出来的策略放回 MuJoCo 里闭环跑（开环重放 / 闭环回放）
==============================================================================

两种模式，别混：

  ① open-loop（开环重放）：直接播 npz 里录好的 ctrl。**不检验策略**，只检验"动作序列本身能不能
     在物理里复现"（对照组）。任何训练结果都跑出同一结果 → 说明它没用到策略。

  ② closed-loop（闭环回放）：控制器只给动作空间里的 6 个关节角目标，**每个控制周期**从渲染画面
     读当前状态喂给策略，取策略输出当下一个目标。这才是"验证策略可行性"。
     ★ 这正是 lerobot 的 `lerobot-replay` / `lerobot-eval` 在真机上走的同一条数据通路。

用法
----
```bash
# A. 先做对照组：动作序列在物理里重放，应该重现"抬升 ~122 mm"
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 tools/replay_policy.py \
    --episode scripts/ep_000000.npz --mode open-loop --rollout out/open.mp4

# B. 用训练好的 ACT 策略闭环跑（识别写进 info.json 的动作空间）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 tools/replay_policy.py \
    --episode scripts/ep_000000.npz --policy outputs/act_soarm101/checkpoints/last/pretrained_model \
    --mode closed-loop --rollout out/policy.mp4

# C. 只跑物理+判据，不录像（最快；像 0.3 s）
... --mode closed-loop --policy ... --no-render

# D. 批量看策略在 5 条不同初始位姿上的成功率
... --policy ... --episodes 0,1,2,3,4 --no-render
```

★ 与三条红线的关系
    · 仍然零 ROS；仍然不写 `data.qpos` 伪造握持 —— 握持来自 `mj_step` 的接触力与摩擦。
    · 策略输出只写 `data.ctrl`（位置执行器），与采集时下发的量完全同源。
    · 本工具会 `mj_step`（它本来就是闭环验证），因此**不属于** render 阶段；录像是把
      `mj_forward` 之后的帧抓下来的副产物。

★ 已知口径差（不改会误判）
    训练数据里的 `observation.state` / `action` 是**关节角（度）**，那是"目标角（ctrl）"；
    而仿真里实时能读到的是"实际角（qpos）"。两者有 PD 滞后（下探段实测约 11 mm 的 TCP 差）。
    为了与训练分布对齐，本工具喂给策略的 state 用 **当前 ctrl 换算成的关节角**（=策略自己上一步
    要它去的位置），而不是 qpos。想改成读 qpos 实测值，加 `--state-from qpos` 做 A/B。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "render"))
sys.path.insert(0, str(ROOT / "grasp"))

# 数据集里的特征名（lerobot 口径，带 .pos 后缀）
FEATURE_NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
                 "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
# MuJoCo 里的关节名（与上面一一对应、同序）
JOINT_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint",
               "wrist_flex_joint", "wrist_roll_joint", "gripper_joint"]


# ---------------------------------------------------------------------------
# 策略后端：lerobot 一个，其余是"没装 lerobot 也能验"的替身
# ---------------------------------------------------------------------------

class PolicyBase:
    def prepare(self, task: str):
        pass

    def act(self, state_deg: np.ndarray, images: dict) -> np.ndarray:  # 6 维，度
        raise NotImplementedError


class LerobotPolicy(PolicyBase):
    """
    读 lerobot 训练出的 checkpoint，走**官方推理管线**（与 `lerobot-eval` 同一条路径）：

        obs(未归一化) → preprocessor(Rename→Batch→Device→Normalize)
                      → policy.select_action()   （ACT 内部有 action chunk 队列，必须 reset）
                      → postprocessor(Unnormalize→CPU) → 6 维关节角（度）

    这样归一化参数直接来自 checkpoint 里的 `policy_preprocessor.json`，
    不需要我手搓 mean/std（手搓最容易在"度 vs 弧度"上翻车）。
    """

    def __init__(self, ckpt: Path, device: str = "cuda"):
        import torch                                              # noqa: PLC0415
        from lerobot.configs.policies import PreTrainedConfig      # noqa: PLC0415
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: PLC0415

        self.torch = torch
        self.device = device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        cfg.device = self.device
        cls = get_policy_class(cfg.type)
        self.policy = cls.from_pretrained(str(ckpt), config=cfg).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(ckpt))
        self.task = ""
        self.calls = 0

    def prepare(self, task: str):
        self.task = task or ""
        self.calls = 0
        self.policy.reset()          # ★ ACT 有 action chunk 队列，不 reset 会接着上一条的队列走

    def act(self, state_deg, images):
        t = self.torch
        obs = {"observation.state": t.from_numpy(np.asarray(state_deg, np.float32))}
        for k, v in images.items():
            # HWC uint8 RGB -> CHW float[0,1]（与 lerobot envs/utils.preprocess_observation 同口径）
            obs[k] = t.from_numpy(np.asarray(v, np.float32) / 255.0).permute(2, 0, 1)
        with t.inference_mode():
            a = self.post(self.policy.select_action(self.pre(obs)))
        self.calls += 1
        return a.detach().cpu().numpy().reshape(-1)[:6].astype(np.float64)


class PlaybackPolicy(PolicyBase):
    """不训练也能验通路：把示范轨迹的 action 当"策略输出"回放（开环对照组）。"""

    def __init__(self, ctrl_deg: np.ndarray):
        self._a = np.asarray(ctrl_deg, float)
        self.i = 0

    def prepare(self, task: str):
        pass

    def act(self, state_deg, images):
        a = self._a[min(self.i, len(self._a) - 1)]
        self.i += 1
        return a


class HoldPolicy(PolicyBase):
    """恒等保持：什么都不做。跑完抬升应为 0 —— 用来验证"判据真的在起作用"。"""

    def __init__(self, q0_deg):
        self.q0 = np.asarray(q0_deg, float)

    def prepare(self, task: str):
        pass

    def act(self, state_deg, images):
        return self.q0.copy()


# ---------------------------------------------------------------------------
# 闭环跑一条：用 arm 的现有封装，只走 mj_step
# ---------------------------------------------------------------------------

def _ctrl_from_deg(a, deg):
    """6 维关节角（度）-> 完整 ctrl（长度 nu，夹爪位置对应 gripper_joint 的地址）。"""
    ctrl = a.d.ctrl.copy()
    for i, nm in enumerate(JOINT_NAMES):
        jid = a.m.joint(nm).id
        ctrl[a.m.jnt_qposadr[jid]] = np.radians(deg[i])
    return ctrl


def _deg_from_ctrl(a, ctrl):
    out = []
    for nm in JOINT_NAMES:
        jid = a.m.joint(nm).id
        out.append(float(np.degrees(ctrl[a.m.jnt_qposadr[jid]])))
    return np.array(out, float)


def _deg_from_qpos(a):
    return np.array([float(np.degrees(a.d.qpos[a.m.jnt_qposadr[a.m.joint(nm).id]]))
                      for nm in JOINT_NAMES], float)


def run_closed_loop(a, policy, task, max_steps, control_hz=30.0, state_from="ctrl",
                    rollout=None, width=640, height=480):
    """
    闭环跑一条 episode，返回指标 dict（含抬升 / 水平位移 / 接触数 / 是否夹起来）。
    """
    import mujoco
    import test_render as R

    m, d = a.m, a.d
    rig = R.CameraRig(m, width=width, height=height) if rollout else None
    sink = None
    if rollout:
        rollout.parent.mkdir(parents=True, exist_ok=True)
        sink = cv2.VideoWriter(str(rollout), cv2.VideoWriter_fourcc(*"avc1"),
                               control_hz, (width, height))
        if not sink.isOpened():
            raise SystemExit(f"VideoWriter 打不开: {rollout}")

    policy.prepare(task)
    # 记录基线（与 EpisodeRecorder 同口径：lift = z - z0）
    mujoco.mj_forward(m, d)
    obj0 = a.obj_pos().copy()
    q0 = _deg_from_qpos(a)
    t0 = d.time
    hist = []
    step_dt = 1.0 / control_hz
    needs_images = not isinstance(policy, PlaybackPolicy)   # 回放对照不需要图
    rig_tmp = None
    max_contact = 0
    max_lift = -1e9

    n_steps = int(round(max_steps * control_hz))
    for k in range(n_steps):
        # ---- 观测：图像（真渲染，和训练数据同源）+ 关节角 ----
        images = {}
        if needs_images:
            if rig is not None:
                raw = rig.grab(d)
            else:                                  # 不录像也要图：临时开一个 rig
                if rig_tmp is None:
                    rig_tmp = R.CameraRig(m, width=width, height=height)
                raw = rig_tmp.grab(d)
            images["observation.images.env"] = raw[R.CAM_ENV]
            images["observation.images.wrist"] = raw[R.CAM_WRIST]

        state = _deg_from_ctrl(a, d.ctrl) if state_from == "ctrl" else _deg_from_qpos(a)
        a_deg = policy.act(state, images)
        d.ctrl[:] = _ctrl_from_deg(a, a_deg)

        # ---- 一个控制周期 = 若干个 mj_step，直到跨过 1/control_hz ----
        while d.time - t0 < (k + 1) * step_dt:
            mujoco.mj_step(m, d)

        obj = a.obj_pos()
        lift = (obj[2] - obj0[2]) * 1000.0
        hd = float(np.linalg.norm(obj[:2] - obj0[:2])) * 1000.0
        cs = a.grip_contacts()
        max_contact = max(max_contact, len(cs))
        max_lift = max(max_lift, lift)
        hist.append(dict(t=float(d.time - t0), lift_mm=float(lift), hdisp_mm=hd,
                         contact_n=len(cs)))
        if sink is not None:
            mujoco.mj_forward(m, d)
            frame = rig.grab(d)[R.CAM_ENV]
            sink.write(np.ascontiguousarray(frame[:, :, ::-1]))

    if sink is not None:
        sink.release()
    if rig is not None:
        rig.close()
    if rig_tmp is not None:
        rig_tmp.close()

    # ★判据口径与 render/test_render.evaluate_success 一致：
    #   抬升量 = 全程最大抬升（药盒被举起来的那一段），不是"最后一帧还在空中"。
    #   采集时的 lift = z(phase=lift 末) - z0，取 max 即可（transport 段药盒只平移不下落）。
    #   最后一帧的 lift 是"松爪掉落后是否还沾着"，只作为额外信息，不参与判据。
    lift_end = hist[-1]["lift_mm"] if hist else 0.0
    return dict(
        mode="closed-loop",
        steps=n_steps, sim_s=round(float(d.time - t0), 2),
        lift_mm=round(float(max_lift), 2),          # 判据用这个（与训练标签同口径）
        lift_end_mm=round(float(lift_end), 2),      # 收尾时是否还夹着
        hdisp_mm=round(float(hist[-1]["hdisp_mm"]), 2) if hist else 0.0,
        max_contact=max_contact,
        q0_deg=[round(v, 2) for v in q0],
        success=bool(max_lift >= 50.0),
        rollout=str(rollout) if rollout else None,
        history=hist[-5:],
    )


def run_open_loop(a, ep, rollout=None, width=640, height=480, fps=30.0):
    """对照组：直接播 ctrl，每帧 mj_forward 抓图（不额外改物理）。"""
    import mujoco
    import test_render as R

    m, d = a.m, a.d
    ctrl = np.asarray(ep["ctrl"], float)
    qpos0 = np.asarray(ep["qpos"], float)
    rig = R.CameraRig(m, width=width, height=height) if rollout else None
    sink = None
    if rollout:
        rollout.parent.mkdir(parents=True, exist_ok=True)
        sink = cv2.VideoWriter(str(rollout), cv2.VideoWriter_fourcc(*"avc1"),
                               fps, (width, height))
    obj0 = a.obj_pos().copy()
    hist = []
    t0 = d.time
    n = len(ctrl)
    for i in range(n):
        d.ctrl[:] = ctrl[i]
        while d.time - t0 < (i + 1) / fps:
            mujoco.mj_step(m, d)
        obj = a.obj_pos()
        hist.append(dict(t=float(d.time - t0),
                         lift_mm=float((obj[2] - obj0[2]) * 1000.0),
                         hdisp_mm=float(np.linalg.norm(obj[:2] - obj0[:2]) * 1000.0),
                         contact_n=len(a.grip_contacts())))
        if sink is not None:
            mujoco.mj_forward(m, d)
            sink.write(np.ascontiguousarray(rig.grab(d)[R.CAM_ENV][:, :, ::-1]))
    if sink is not None:
        sink.release()
        rig.close()
    lift_end = hist[-1]["lift_mm"] if hist else 0.0
    mx = float(max(h["lift_mm"] for h in hist)) if hist else 0.0
    return dict(mode="open-loop", steps=n, sim_s=round(float(d.time - t0), 2),
                lift_mm=round(mx, 2),           # 判据口径同上（全程最大抬升）
                lift_end_mm=round(float(lift_end), 2),
                hdisp_mm=round(float(hist[-1]["hdisp_mm"]), 2) if hist else 0.0,
                success=bool(mx >= 50.0),
                rollout=str(rollout) if rollout else None)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="把策略放回 MuJoCo 里跑（开环重放 / 闭环回放），输出抬升等判据",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    src = ap.add_argument_group("数据")
    src.add_argument("--episode", default=str(ROOT / "scripts" / "ep_000000.npz"),
                     help="母本轨迹 npz（提供 task 文本、初始姿态、出生位姿）")
    src.add_argument("--episodes", default=None,
                     help="批量：编号列表（如 0,1,2），用于多初值成功率")
    src.add_argument("--scripts", default=str(ROOT / "scripts"), help="母本目录")
    pol = ap.add_argument_group("策略")
    pol.add_argument("--policy", default=None,
                     help="lerobot checkpoint 目录（…/checkpoints/last/pretrained_model）；"
                          "不给则用替身策略")
    pol.add_argument("--stub", choices=["playback", "hold"], default="playback",
                     help="没给 --policy 时用哪个替身（默认 playback=开环对照）")
    pol.add_argument("--device", default="cuda", help="cuda / cpu（无 GPU 会自动降级）")
    run = ap.add_argument_group("运行")
    run.add_argument("--mode", choices=["closed-loop", "open-loop"], default="closed-loop")
    run.add_argument("--duration", type=float, default=15.0, help="闭环跑多久（秒，默认 15）")
    run.add_argument("--control-hz", type=float, default=30.0, help="控制频率（默认 30）")
    run.add_argument("--state-from", choices=["ctrl", "qpos"], default="ctrl",
                     help="喂给策略的关节角取'下发目标(ctrl)'还是'实际(qpos)'")
    run.add_argument("--rollout", default=None, help="录像输出 mp4（不给就只算指标）")
    run.add_argument("--no-render", action="store_true", help="不渲染不录像（最快）")
    run.add_argument("--width", type=int, default=640)
    run.add_argument("--height", type=int, default=480)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--json", action="store_true", help="stdout 只输出 JSON")
    args = ap.parse_args(argv)

    import mujoco                                             # noqa: PLC0415
    import test_render as R                                   # noqa: PLC0415

    # ---- 选 episode ----
    if args.episodes:
        eids = [int(x) for x in str(args.episodes).replace(" ", "").split(",") if x != ""]
    else:
        eids = None

    def _load(eid):
        p = Path(args.scripts) / f"ep_{eid:06d}.npz"
        js = Path(args.scripts) / f"ep_{eid:06d}.json"
        if not p.exists():
            raise SystemExit(f"没有轨迹: {p}")
        with np.load(p, allow_pickle=True) as z:
            ep = {k: np.array(z[k]) for k in z.files if k != "meta_json"}
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        if js.exists():
            meta.update(json.loads(js.read_text(encoding="utf-8")))
        return ep, meta

    targets = eids or []
    if not targets:
        ep0, meta0 = _load(int(Path(args.episode).stem.split("_")[-1]))
        targets = [int(Path(args.episode).stem.split("_")[-1])]

    task = None
    results = []
    arm = R.Arm(R.SCENE_XML)          # 模型只加载一次
    for eid in targets:
        ep, meta = _load(eid)
        task = task or meta.get("instruction") or "Pick up the medicine box and place it into the wooden box."

        # 每条 episode 都从它自己的初始姿态开始（qpos[0] 含药盒出生位姿）
        arm.d.qpos[:] = ep["qpos"][0]
        arm.d.qvel[:] = 0.0
        arm.d.ctrl[:] = ep["ctrl"][0]
        mujoco.mj_forward(arm.m, arm.d)

        if args.mode == "open-loop":
            roll = None
            if not args.no_render:
                roll = Path(args.rollout or (ROOT / "out" / f"open_{eid:06d}.mp4"))
            res = run_open_loop(arm, ep, rollout=roll,
                                width=args.width, height=args.height, fps=R.FPS)
        else:
            if args.policy:
                pol_obj = LerobotPolicy(Path(args.policy), device=args.device)
            elif args.stub == "playback":
                # 替身：把示范的 ctrl（弧度）转成度当"策略输出"
                pol_obj = PlaybackPolicy(np.degrees(np.asarray(ep["ctrl"][:, :6], float)))
            else:
                pol_obj = HoldPolicy(np.degrees(np.asarray(ep["qpos"][0, :6], float)))
            roll = None
            if not args.no_render:
                roll = Path(args.rollout or (ROOT / "out" / f"policy_{eid:06d}.mp4"))
            res = run_closed_loop(arm, pol_obj, task, max_steps=args.duration,
                                  control_hz=args.control_hz, state_from=args.state_from,
                                  rollout=roll, width=args.width, height=args.height)
        res["episode_id"] = eid
        results.append(res)
        if not args.json:
            print(f"  [{'✅' if res['success'] else '❌'}] ep_{eid:06d}  "
                  f"{res['mode']}  {res['sim_s']:.1f}s/{res['steps']}控制周期")
            print(f"       抬升峰值 {res['lift_mm']:+.2f} mm（收尾 {res.get('lift_end_mm', 0):+.2f}）  "
                  f"水平 {res['hdisp_mm']:.1f} mm"
                  + (f"  接触峰值 {res.get('max_contact')}" if "max_contact" in res else ""))
            if res.get("rollout"):
                print(f"       录像 {res['rollout']}")

    summary = dict(mode=args.mode, policy=args.policy or f"stub:{args.stub}",
                   n=len(results), n_success=sum(1 for r in results if r["success"]),
                   results=results)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=1))
    else:
        print()
        print(f"  汇总: {summary['n_success']}/{summary['n']} 成功"
              f"（判据 抬升 ≥ 50 mm）；策略={summary['policy']}")
        if not args.policy:
            print("  ⚠️ 当前用的是替身策略（回放/保持），只验证通路，不代表任何训练结果。")
    return 0 if summary["n_success"] == summary["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
