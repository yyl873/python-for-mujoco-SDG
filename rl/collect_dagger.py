#!/usr/bin/env python3
"""rl/collect_dagger.py —— 第 2 步：DAgger 纠正数据采集（ACT 在环 + 审计后的专家组合）

要解决的问题
------------
`docs/REPORT_RESIDUAL_RL.md` §10/§11 的结论：6 维残差没有可用信号，替代路线是
"修基座" —— 让 ACT 自己学会从**漂移状态**回到正轨。这需要的正是 DAgger 纠正数据：
`(ACT 实际走到的状态, 专家在这个状态下会做什么)`。

为什么不是"直接让脚本专家接管"
------------------------------
`collect/collect.py` 的脚本专家是**开环路点脚本**：它从固定起点执行三段接近 + 抓取 + 搬运，
从没被设计成"从歪掉的状态恢复"。而 §11 实测的失败状态是：穿透 18 mm、药盒倾斜 57°、
夹爪已经捅进盒子里 —— 硬切到脚本专家只会得到一条同样失败的轨迹，标签还是错的。

本脚本的做法：**审计后的专家组合**（audited expert portfolio）
------------------------------------------------------------
在每个需要标注的控制步 k，候选动作最多 4 个（按优先级）：

    ref      母本轨迹同一相位 k 的动作（`ctrl[k]`）——"正轨上的专家"
    blend    0.5·(当前关节角 + ref) ——"半步回正"
    retract  当前 TCP 上抬 10 mm 的 IK 解（保持夹爪指令）——"先退出来"
    hold     保持当前关节角（冻结）——"别动，等接触消解"

每个候选在 **克隆的 MjData** 上向前滚 `--audit-steps` 个控制步（默认 1 = 33 ms），
用三个物理判据验收（全部相对**当前**状态）：

    pen_after  <= max(pen_before, --pen-gate) + 1.0 mm     不把手指更深地捅进盒子
    tilt_after <= max(tilt_before, --trigger-tilt) + 1.0°  不把盒子推得更歪
    lift_after >= lift_before - 8.0 mm                      不把已抬起来的盒子丢掉

按优先级取**第一个通过**的候选；四个都不通过就取违反量最小的那个并标 `forced`。
这样每个标签都是"在这个状态下**实测**不会让物理恶化的动作"，而不是靠信念。

★ 这一步也是本步骤的科学产出：候选通过率本身就在回答
  "开环母本轨迹在漂移状态下还剩多少可用的纠正信号"。

采集策略
--------
- 触发器（§11 定标）：`pen > --trigger-pen`（默认 14 mm）或 `tilt > --trigger-tilt`（默认 25°）
- 触发后专家**接管 `--handover` 步**（默认 20 步 ≈ 0.67 s），窗口内逐步标注；窗口结束把
  ACT 的 action chunk 队列清空（重新推理），回到 ACT 闭环 —— 与真机"接管后重新规划"一致
- 全程用 `env.step()` 驱动，因此物理/判定/info 语义与 `rl/eval_ab.py` 完全一致
  （不重写一份 step，避免口径漂移）

输出
----
1. lerobot v3.0 数据集（`--out`），特征与 `collect/export_lerobot.py` 完全一致，
   可直接用 `lerobot-train --policy.type=act` 微调，也可用 lerobot 的 aggregate 与 sim156 混合
2. 逐 episode JSONL（`--log`）+ 汇总 JSON（`--report`）
3. 图像 = 过 `ObsAdapter`（SensorModel，与 sim156 训练集同口径）后的帧，即模型实际看到的观测

用法
----
```bash
# 先体检（不渲染、不写数据集，只跑审计，约 10 s / 30 条）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl LP_NUM_THREADS=8 .venv-rl/bin/python \
    rl/collect_dagger.py --episodes 0-29 --dry-run --report rl/eval/dagger_audit.json

# 正式采集（写数据集 + 渲染）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl LP_NUM_THREADS=8 .venv-rl/bin/python \
    rl/collect_dagger.py --episodes 0-49 --out rl/dagger_ds/dagger_v1 \
    --log rl/eval/dagger_collect.jsonl --report rl/eval/dagger_collect.json

# 崩了（mj_step 偶发 SIGFPE）就原地 --resume：已完成的 episode 不会重采
... --resume
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
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp", ROOT / "collect"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# lerobot 的 HF 缓存落在仓库内（~/.cache 对沙箱只读），与 export_lerobot 同口径
import os                                                    # noqa: E402
_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))

TASK_DEFAULT = "Pick up the medicine box and place it into the wooden box."
CANDIDATES = ("ref", "blend", "retract", "hold")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def parse_pool(spec: str) -> list[int]:
    """'0-49,7,9' -> [0..49, 7, 9]（去重、保序）。"""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    seen, uniq = set(), []
    for e in out:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return uniq


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def _pen_mm(m, d, obj: int) -> float:
    """药盒相关的最大穿透深度（mm）。与 env_residual._penetration_mm 同口径。"""
    worst = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        if m.geom_bodyid[c.geom1] != obj and m.geom_bodyid[c.geom2] != obj:
            continue
        worst = max(worst, -float(c.dist))
    return worst * 1000.0


def _tilt_deg(m, d, obj: int) -> float:
    z = d.xmat[obj].reshape(3, 3)[2, 2]
    return float(np.degrees(np.arccos(np.clip(z, -1.0, 1.0))))


def _grip_contacts(m, d, grip: int, jaw: int, obj: int) -> int:
    """药盒与夹爪两指（grip/jaw）的接触点个数。"""
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        if obj not in (b1, b2):
            continue
        if (grip in (b1, b2)) or (jaw in (b1, b2)):
            n += 1
    return n


# ---------------------------------------------------------------------------
# 审计后的专家组合
# ---------------------------------------------------------------------------

class ExpertPortfolio:
    """在漂移状态上给出"实测不恶化"的专家动作。"""

    def __init__(self, env, *, pen_gate_mm: float, tilt_gate_deg: float,
                 lift_slack_mm: float = 8.0, audit_steps: int = 1,
                 tol_margin_mm: float = 1.0):
        self.env = env
        self.arm = env.arm
        self.m, self.d = env.m, env.d
        self.obj = self.arm.obj
        self.grip, self.jaw = self.arm.grip, self.arm.jaw
        self.pen_gate = float(pen_gate_mm)
        self.tilt_gate = float(tilt_gate_deg)
        self.lift_slack = float(lift_slack_mm)
        self.steps = int(audit_steps)
        self.margin = float(tol_margin_mm)
        self.ctrl_from_deg = env._ctrl_from_deg
        self.deg_from_ctrl = env._deg_from_ctrl
        # 6 维关节范围：Arm.lo/hi 只有 5 个臂关节，第 6 列是 gripper_joint（rad）
        import mujoco                                        # noqa: PLC0415
        jid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
        jlo, jhi = self.m.jnt_range[jid]
        self.lo_deg = np.degrees(np.append(self.arm.lo, jlo))
        self.hi_deg = np.degrees(np.append(self.arm.hi, jhi))

    # ---- 状态量 ----
    def snapshot(self) -> dict:
        d, m, a = self.d, self.m, self.arm
        tcp = d.site_xpos[a.tcp]
        box = d.qpos[a.oadr:a.oadr + 3]
        return dict(pen=_pen_mm(m, d, self.obj), tilt=_tilt_deg(m, d, self.obj),
                    lift=(float(box[2]) - float(self.env._obj0[2])) * 1000.0,
                    dist=float(np.linalg.norm(tcp - box)))

    # ---- 候选动作 ----
    def candidates(self, k: int, ref_ctrl_deg: np.ndarray, cur_deg: np.ndarray) -> dict:
        ref = np.asarray(ref_ctrl_deg, float).copy()
        # 夹爪列按关节范围裁剪（母本的 jaw 指令可能超出当前关节范围）
        ref = np.clip(ref, self.lo_deg, self.hi_deg)
        out = {"ref": ref, "blend": 0.5 * (cur_deg + ref), "hold": cur_deg.copy()}
        # retract：当前 TCP 上抬 10 mm（抓取姿态不动，只退一点）
        tcp = self.d.site_xpos[self.arm.tcp].copy()
        q_cur = np.array([self.d.qpos[a] for a in self.arm.adr], float)
        try:
            q_up, err = self.arm.ik([tcp[0], tcp[1], tcp[2] + 0.010], q_cur, tol=1e-3)
        except Exception:                                     # noqa: BLE001
            q_up, err = None, 1e9
        if q_up is not None and float(err) <= 0.002:
            # Arm.ik 只解 5 个臂关节，第 6 列（夹爪指令）保持当前值
            ret = np.append(np.degrees(np.asarray(q_up, float)), cur_deg[5])
            out["retract"] = ret
        return out

    # ---- 单候选前视 ----
    def lookahead(self, deg: np.ndarray) -> dict:
        """在克隆的 MjData 上滚 audit_steps 个控制步，返回物理量。"""
        env = self.env
        dc = copy.deepcopy(self.d)
        dc.ctrl[:] = self.ctrl_from_deg(self.arm, deg)
        dt = 1.0 / float(env.control_hz)
        t_end = float(dc.time) + self.steps * dt
        while float(dc.time) < t_end:
            env._mujoco.mj_step(self.m, dc)
        tcp = dc.site_xpos[self.arm.tcp]
        box = dc.qpos[self.arm.oadr:self.arm.oadr + 3]
        return dict(pen=_pen_mm(self.m, dc, self.obj),
                    tilt=_tilt_deg(self.m, dc, self.obj),
                    lift=(float(box[2]) - float(env._obj0[2])) * 1000.0,
                    dist=float(np.linalg.norm(tcp - box)),
                    ncon=_grip_contacts(self.m, dc, self.grip, self.jaw, self.obj))

    # ---- 主入口 ----
    def choose(self, k: int, ref_ctrl_deg: np.ndarray, cur_deg: np.ndarray) -> dict:
        """返回 {name, action, forced, audit{...}}。"""
        now = self.snapshot()
        pens = max(now["pen"], self.pen_gate) + self.margin
        tilts = max(now["tilt"], self.tilt_gate) + self.margin
        lifts = now["lift"] - self.lift_slack
        cands = self.candidates(k, ref_ctrl_deg, cur_deg)
        rows, best, best_v = [], None, 1e18
        for name in CANDIDATES:
            if name not in cands:
                continue
            try:
                after = self.lookahead(cands[name])
            except Exception as exc:                          # noqa: BLE001
                rows.append(dict(name=name, error=f"{type(exc).__name__}: {exc}"))
                continue
            viol = (max(0.0, after["pen"] - pens)
                    + 2.0 * max(0.0, after["tilt"] - tilts)
                    + 0.05 * max(0.0, lifts - after["lift"]))
            ok = bool(viol <= 0.0)
            rows.append(dict(name=name, ok=ok, viol=round(float(viol), 3),
                             pen=round(after["pen"], 2), tilt=round(after["tilt"], 2),
                             lift=round(after["lift"], 2), ncon=int(after["ncon"]),
                             dpen=round(after["pen"] - now["pen"], 2),
                             dtilt=round(after["tilt"] - now["tilt"], 2)))
            if ok and best is None:
                best = name                                    # 按优先级取第一个通过
            if viol < best_v:
                best_v, best_fallback = float(viol), name
        if best is None:
            best = best_fallback if rows else "hold"
        return dict(name=best, action=np.asarray(cands[best], float),
                    forced=bool(best_v > 0.0),
                    before={k2: round(float(v), 2) for k2, v in now.items()},
                    audit=rows)


# ---------------------------------------------------------------------------
# 单 episode 采集
# ---------------------------------------------------------------------------

def run_episode(env, eid: int, pf: ExpertPortfolio, args, *,
                collect_frames: bool) -> tuple[dict, list, list]:
    """跑一条 episode，返回 (统计行, 帧列表, 审计行列表)。"""
    from base_policy import OBS_ENV, OBS_WRIST

    ep, meta = env._episodes[eid]
    ref_ctrl = np.asarray(ep["ctrl"], float)                   # (N,6) rad
    n_ref = len(ref_ctrl)

    obs, info = env.reset(seed=eid + args.seed)
    task = info.get("task") or meta.get("instruction") or TASK_DEFAULT

    base = env.base
    orig_act, orig_needs = base.act, base.needs_images
    mode = {"expert": None}

    def patched_act(state_deg, raw=None, speeds_px=None):
        if mode["expert"] is not None:
            return mode["expert"]
        return orig_act(state_deg, raw, speeds_px)

    base.act = patched_act
    frames: list[dict] = []
    hand_until = -1
    cooldown_until = -1
    events, chosen, forced = 0, {c: 0 for c in CANDIDATES}, 0
    trigger_steps = 0
    audit_rows: list[dict] = []
    k = -1

    try:
        for k in range(int(args.max_steps)):
            if mode["expert"] is not None and k >= hand_until:
                # 接管结束：清空 action chunk 队列，回到 ACT 闭环（等于"重新规划"）
                mode["expert"] = None
                q = getattr(base.policy.policy, "_action_queue", None)
                if q is not None:
                    q.clear()
                base._pre_cached = None

            s_now = pf.snapshot()
            fired = (s_now["pen"] > args.trigger_pen) or (s_now["tilt"] > args.trigger_tilt)
            # ★ 冷却期：接手窗口结束后必须把控制权还给 ACT 至少 cooldown 步，否则
            #   "穿透一直 > 阈值" 会让专家永久接管（实测占 31% 步，退化成一串 hold）。
            if fired and mode["expert"] is None and k >= 1 and k >= cooldown_until:
                events += 1
                hand_until = k + int(args.handover)
                cooldown_until = hand_until + int(args.cooldown)

            if fired or mode["expert"] is not None:
                trigger_steps += int(fired)
                cur_deg = pf.deg_from_ctrl(env.arm, env.d.ctrl)
                kk = min(k, n_ref - 1)
                pick = pf.choose(kk, np.degrees(ref_ctrl[kk]), cur_deg)
                mode["expert"] = pick["action"]
                chosen[pick["name"]] += int(not pick["forced"])
                forced += int(pick["forced"])
                audit_rows.append(dict(episode_id=eid, step=k, name=pick["name"],
                                       forced=pick["forced"], **pick["before"],
                                       audit=pick["audit"]))
                if collect_frames and (args.keep_forced or not pick["forced"]):
                    raw = env.rig.grab(env.d)
                    seen = base.adapter(raw)
                    frames.append(dict(state=cur_deg.astype(np.float32),
                                       action=pick["action"].astype(np.float32),
                                       env=seen[OBS_ENV], wrist=seen[OBS_WRIST],
                                       step=k, phase="handover",
                                       picked=pick["name"], forced=pick["forced"]))

            base.needs_images = (lambda: False) if mode["expert"] is not None else orig_needs
            obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
            if term or trunc:
                break
    finally:
        base.act, base.needs_images = orig_act, orig_needs

    row = dict(episode_id=eid, task=task, steps=int(k + 1),
               success=bool(info.get("success")), end=("success" if info.get("success")
                                                       else "fail" if info.get("fell")
                                                       or info.get("off_table")
                                                       or info.get("out_of_area") else "timeout"),
               lift_max_mm=round(float(info.get("max_lift_mm", 0.0)), 1),
               pen_max_mm=round(max([a["pen"] for a in audit_rows], default=0.0), 1),
               tilt_max_deg=round(max([a["tilt"] for a in audit_rows], default=0.0), 1),
               n_trigger_events=events, n_trigger_steps=int(trigger_steps),
               n_labeled=len(frames) if collect_frames else len(audit_rows),
               candidates={c: v for c, v in chosen.items() if v}, forced=int(forced),
               n_dropped_forced=0 if (args.keep_forced or not collect_frames) else int(forced))
    return row, frames, audit_rows


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="ACT 在环 DAgger 纠正数据采集（审计后的专家组合）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--episodes", default="0-49", help="母本 episode 池（默认 0-49，勿用 held-out 100-129）")
    ap.add_argument("--base", default="sim156", help="基座 checkpoint 名（见 rl/base_policy.py）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(ROOT / "rl" / "dagger_ds" / "dagger_v1"),
                    help="lerobot 数据集输出目录（--dry-run 时忽略）")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--log", default=str(ROOT / "rl" / "eval" / "dagger_collect.jsonl"))
    ap.add_argument("--report", default=str(ROOT / "rl" / "eval" / "dagger_collect.json"))
    ap.add_argument("--trigger-pen", type=float, default=14.0,
                    help="穿透触发阈值 mm（§11 定标：p50 成功 12.0 / 失败 18.2）")
    ap.add_argument("--trigger-tilt", type=float, default=25.0,
                    help="倾角触发阈值 度（§11 定标：成功 p50 15.5 / 失败 57.4）")
    ap.add_argument("--handover", type=int, default=20, help="专家接管步数（30 Hz，默认 20 ≈ 0.67 s）")
    ap.add_argument("--cooldown", type=int, default=15,
                    help="窗口结束后交还 ACT 的强制步数（默认 15，防止穿透持续时专家永久接管）")
    ap.add_argument("--keep-forced", action="store_true",
                    help="把 forced=True（四个候选都不合格）的帧也写进数据集。默认丢弃："
                         "这些标签本身违反物理门槛，不可信")
    ap.add_argument("--steps-log", default=None,
                    help="逐标注步审计明细 JSONL（诊断用，默认不写）")
    ap.add_argument("--pen-gate", type=float, default=20.8,
                    help="审计用的穿透门槛 mm（基座 p90，见 env_residual.pen_gate_mm）")
    ap.add_argument("--audit-steps", type=int, default=1, help="前视控制步数（默认 1）")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--max-steps", type=int, default=None, help="上限控制步（默认按 max-seconds 推）")
    ap.add_argument("--randomize", action="store_true",
                    help="开训练期随机化。★默认关：母本轨迹本身就是随机化过的，"
                         "再随机一次会让 ref 标签和状态错配 30 mm")
    ap.add_argument("--judge", default="strict", choices=("strict", "legacy"))
    ap.add_argument("--dry-run", action="store_true", help="只跑审计与统计，不渲染、不写数据集")
    ap.add_argument("--resume", action="store_true", help="跳过 --log 里已完成的 episode")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    control_hz = 30.0
    if args.max_steps is None:
        args.max_steps = int(round(args.max_seconds * control_hz))

    pool = parse_pool(args.episodes)
    out = Path(args.out).expanduser()
    log_p = Path(args.log)
    rep_p = Path(args.report)
    log_p.parent.mkdir(parents=True, exist_ok=True)
    rep_p.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    rows: list[dict] = []
    if args.resume and log_p.exists():
        for line in log_p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r0 = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r0.get("done"):
                done.add(int(r0["episode_id"]))
                rows.append(r0)
    todo = [e for e in pool if e not in done]

    print("=" * 78)
    print(" collect_dagger  ACT 在环纠正数据采集")
    print("=" * 78)
    print(f"  基座    : {args.base}（冻结，残差=0）")
    print(f"  池      : {args.episodes}  共 {len(pool)} 条，待采 {len(todo)} 条"
          + (f"（已完成 {len(done)}）" if done else ""))
    print(f"  触发器  : pen > {args.trigger_pen} mm 或 tilt > {args.trigger_tilt}°"
          f"   → 接管 {args.handover} 步")
    print(f"  审计    : 前视 {args.audit_steps} 控制步 / gate pen {args.pen_gate} mm")
    print(f"  模式    : {'dry-run（只审计）' if args.dry_run else '采集（渲染 + 写数据集）'}")
    print(f"  输出    : {'—' if args.dry_run else out}")
    print()

    from env_residual import ResidualSO101, load_episode

    ep0 = todo[0] if todo else pool[0]
    env = ResidualSO101(base_name=args.base, device=args.device, randomize=args.randomize,
                        episode_pool=[ep0], judge=args.judge, reward_version="v2b",
                        max_seconds=args.max_seconds, control_hz=control_hz,
                        residual_deg=1.0, seed=args.seed)
    pf = ExpertPortfolio(env, pen_gate_mm=args.pen_gate, tilt_gate_deg=args.trigger_tilt,
                         audit_steps=args.audit_steps)

    # ---- 数据集（正式采集时）----
    ds = None
    if not args.dry_run:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset        # noqa: PLC0415
        from export_lerobot import build_features                          # noqa: PLC0415
        feats = build_features(480, 640)
        repo_id = args.repo_id or f"local/{out.name}"
        if out.exists() and args.resume:
            ds = LeRobotDataset(repo_id=repo_id, root=out)
            print(f"  续采    : 已存在数据集 {out}，episodes={ds.meta.total_episodes} "
                  f"frames={ds.meta.total_frames}")
        else:
            if out.exists():
                import shutil                                        # noqa: PLC0415
                shutil.rmtree(out)
            ds = LeRobotDataset.create(repo_id=repo_id, fps=30, features=feats,
                                       root=out, robot_type="so_follower")
            print(f"  建库    : {repo_id}  640x480 @30fps")

    t_all = time.time()
    n_done = 0
    n_frames = 0
    audit_all: list[dict] = []
    for eid in todo:
        t_ep = time.time()
        env.episode_pool = [eid]
        env._episodes = {eid: load_episode(env.scripts, eid)}
        row, frames, audit_rows = run_episode(env, eid, pf, args,
                                             collect_frames=not args.dry_run)
        audit_all.extend(audit_rows)
        if args.steps_log:
            with Path(args.steps_log).open("a", encoding="utf-8") as fh:
                for a in audit_rows:
                    fh.write(json.dumps(a, ensure_ascii=False) + "\n")
        if ds is not None and frames:
            for f in frames:
                ds.add_frame({"observation.state": f["state"],
                              "action": f["action"],
                              "observation.images.env": f["env"],
                              "observation.images.wrist": f["wrist"],
                              "task": row["task"]})
            ds.save_episode()
        row["done"] = True
        row["wall_s"] = round(time.time() - t_ep, 1)
        rows.append(row)
        n_done += 1
        n_frames += len(frames)
        with log_p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  [{n_done:>3}/{len(todo)}] ep_{eid:06d}  steps={row['steps']:>3} "
              f"succ={int(row['success'])}  trig={row['n_trigger_events']}/"
              f"{row['n_trigger_steps']}  标注={row['n_labeled']:>3}  "
              f"{row['candidates']}{' forced=' + str(row['forced']) if row['forced'] else ''}  "
              f"{row['wall_s']}s")

    if ds is not None:
        ds.finalize()

    # ---- 汇总 ----
    def rate(key):
        vals = [r[key] for r in rows if key in r]
        return float(np.mean(vals)) if vals else 0.0

    n_ok = sum(1 for r in rows if r["success"])
    lo, hi = wilson(n_ok, len(rows))
    pick_counts = {c: sum(r.get("candidates", {}).get(c, 0) for r in rows) for c in CANDIDATES}
    n_pick = sum(pick_counts.values())
    forced_n = sum(int(r.get("forced", 0)) for r in rows)
    # 审计逐候选通过率 + 一步前视的穿透变化
    by_cand: dict[str, dict] = {}
    for a in audit_all:
        for c in a.get("audit", []):
            if "error" in c:
                continue
            d = by_cand.setdefault(c["name"], dict(n=0, ok=0, dpen=[], dtilt=[]))
            d["n"] += 1
            d["ok"] += int(bool(c["ok"]))
            d["dpen"].append(c["dpen"])
            d["dtilt"].append(c["dtilt"])
    # 分层：穿透/倾角已过 gate 的"深度漂移态" vs 浅漂移态（看纠正信号是否只在小漂移时有效）
    layers = {"shallow": {"n": 0, "ok": {c: 0 for c in CANDIDATES}},
              "deep": {"n": 0, "ok": {c: 0 for c in CANDIDATES}}}
    for a in audit_all:
        deep = (a.get("pen", 0.0) >= args.pen_gate) or (a.get("tilt", 0.0) >= 30.0)
        key = "deep" if deep else "shallow"
        layers[key]["n"] += 1
        for c in a.get("audit", []):
            if "error" in c:
                continue
            layers[key]["ok"][c["name"]] += int(bool(c["ok"]))
    layered = {k: dict(n=v["n"], pass_rate={c: round(o / max(1, v["n"]), 3)
                                            for c, o in v["ok"].items()})
               for k, v in layers.items()}

    audit_summary = {}
    for name, d in by_cand.items():
        audit_summary[name] = dict(
            n=d["n"], pass_rate=round(d["ok"] / max(1, d["n"]), 3),
            dpen_median=round(float(np.median(d["dpen"])), 2),
            dpen_p10=round(float(np.percentile(d["dpen"], 10)), 2),
            dtilt_median=round(float(np.median(d["dtilt"])), 2))

    report = dict(
        generated_at=str(np.datetime64("now")), base=args.base, pool=args.episodes,
        trigger=dict(pen_mm=args.trigger_pen, tilt_deg=args.trigger_tilt,
                     handover_steps=args.handover),
        audit=dict(steps=args.audit_steps, pen_gate_mm=args.pen_gate,
                   per_candidate=audit_summary, layered=layered),
        n_episodes=len(rows), n_frames=int(n_frames),
        n_trigger_events=int(sum(r.get("n_trigger_events", 0) for r in rows)),
        n_trigger_steps=int(sum(r.get("n_trigger_steps", 0) for r in rows)),
        n_labeled=int(sum(r.get("n_labeled", 0) for r in rows)),
        chosen=pick_counts, chosen_total=n_pick, forced=int(forced_n),
        success_rate=round(n_ok / max(1, len(rows)), 3),
        success_ci=dict(wilson=[round(lo, 3), round(hi, 3)], n=len(rows), k=n_ok),
        wall_s=round(time.time() - t_all, 1), rows=rows,
    )
    rep_p.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"  完成    : {len(rows)} 条 / 标注 {report['n_labeled']} 帧 / "
          f"触发 {report['n_trigger_events']} 次 / 耗时 {report['wall_s']}s")
    print(f"  候选    : {pick_counts}  forced {forced_n}")
    for name, d in audit_summary.items():
        print(f"    {name:<8} n={d['n']:>5}  通过率 {d['pass_rate']:.2f}  "
              f"Δpen 中位 {d['dpen_median']:+.2f} mm  Δtilt 中位 {d['dtilt_median']:+.2f}°")
    for key in ("shallow", "deep"):
        print(f"    [{key:^7}] n={layered[key]['n']:>5}  " +
              "  ".join(f"{c}={p:.2f}" for c, p in layered[key]["pass_rate"].items()))
    print(f"  成功率  : {n_ok}/{len(rows)} = {report['success_rate']:.1%} "
          f"[{lo:.1%}, {hi:.1%}]（含专家接管，不是纯基座指标）")
    print(f"  报告    : {rep_p}")
    if ds is not None:
        print(f"  数据集  : {out}  episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}")
    return 0


if __name__ == "__main__":
    sys.exit(main())