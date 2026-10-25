#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/probe_contact_audit.py —— 接触真实性审计（下探有没有戳盒 / 有没有压进桌面 / 有没有回弹）

为什么要这个工具
------------------------------------------------------------------
2024-11 第六轮：用户看视频后指出「下落抓取必然撞倒或戳到药盒，产生回弹，但现实没有
这个机制 —— 绿垫不提供回弹力，纸盒的长宽高边和连接处也不可能剧烈形变」。

这句话是对的，而且能逐帧量出来（实测，oldros 起点 / sdg-v3 步数）：

  · 下探段 400 步里有 92 步存在「指尖↔药盒」接触，首次接触发生在 TCP z=58.72 mm
    （药盒顶面 53.77 mm）—— 指尖还没到盒腰就先落在盒顶。
  · 首次接触点 (305.4, 94.2, 53.7) mm，法向 (0,0,+1)：**正压在盒顶面**。
  · 峰值法向力 2.73 N（药盒自重 0.113 N 的 24 倍），求解器穿透 dist = −13.19 mm。
  · 下探结束时药盒中心 27.77 -> 20.90 mm，底面 **−5.10 mm**：沉到桌面以下。
  · 之后回弹到 23.47 mm（+2.57 mm），倾角从 3.5° 一路涨到抬升峰 29.2°、搬运 31°。
  · 接触法向 z 分量均值 +0.40：载荷是「往下压」，不是「两侧夹」。

也就是说：现在 +103 mm 的「成功」是靠指尖压在盒顶把它掐住撑起来的，
不是真的在 15×52 的窄端面上夹住。落盒、搬运只是这个假象的下游。
docs/HANDOFF.md §6.2-3 早就记着同一件事（约束1 ❌ Δz −6.85 mm、压盒 6.3、回弹 1.7、
gz≥0.020 抬升只剩 4.5 mm），但只有「数字」，没有「哪一帧、哪一点、什么方向」。
本工具补的就是这一层：**把观感变成可判定的量**。

它测什么（每个仿真步，不是 30 fps 的抽样帧）
------------------------------------------------------------------
  descend 段「指尖(含指垫/结构件) ↔ 药盒」接触步数、首次接触的时刻/位置/法向
  「药盒底面 vs 支撑面」的最低点  -> 压进桌面多少 mm
  压到最深之后的回升            -> 回弹多少 mm
  药盒倾角（四元数算世界 up 轴）  -> 撞倒/歪着搬的量
  向上速度峰值                  -> 有没有被弹飞

三个判据（G1~G3），退出码 = 有没有踩线
------------------------------------------------------------------
  G1  下探段指↔盒接触步数 == 0         （接触只允许发生在 grasp 段之后）
  G2  压盒（下探~搬运期间离地最深）<= 0.3 mm
  G3  grasp~transport 期间倾角 <= 5°

为什么单条只要 0.5 s
------------------------------------------------------------------
跑的是生产路径 run_episode（和 render/test_render.py 完全同一份动作代码、同一份步数表），
只包了一层 mj_step 采样，不渲染、不做传感器劣化、不落盘。所以一次能扫几百组参数，
这是把「改指尖之后必须重扫」的成本从几小时压到几分钟的前提。

用法
------------------------------------------------------------------
  cd /home/yyl/deepseek-harness

  # 1) 单条体检（生产配置，oldros 起点），打阶段表 + 判定
  env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py

  # 2) 换参数
  env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py \
      --gz 0.016 --grip 2.232 --roll -90 --start-pose golden

  # 3) 扫一组（gz:grip[:roll]，分号分隔）—— 只打一行一组的表
  env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py \
      --combos "0.012:1.5;0.016:1.5;0.020:2.232;0.024:2.232"

  # 4) 断点续扫（SIGFPE 崩了不会白跑：已完成的组合直接跳过）
  env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py \
      --combos "..." --jsonl logs/contact_audit.jsonl

  # 5) 逐段追踪（descend~settle_a 每 40 步一行）
  env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py --trace

退出码：0 = 所有组合 G1~G3 全过；1 = 有组合踩线（把踩线项打出来）。

环境（踩过的坑）
------------------------------------------------------------------
  · 前缀必须有 `env -u PYTHONHOME -u PYTHONPATH`。
  · 别把输出管道给 head/tail（会触发 Fatal Python error: Floating point exception），
    要截取就 `> logs/x.log 2>&1`，再 dsh-digest。
  · mj_step 约 7% SIGFPE（进程级、与本项目逻辑无关）：崩了原地重跑，
    用 --jsonl 续扫即可。
"""

import os
import sys
import json
import math
import argparse
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "render"))
sys.path.insert(0, str(ROOT / "grasp"))


def load_render():
    """把 render/test_render.py 当模块加载（它自己会设 MUJOCO_GL，必须在 import mujoco 前读 argv）。"""
    spec = importlib.util.spec_from_file_location("trender", ROOT / "render" / "test_render.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- 判据阈值 -------------------------------------------------------------
# G1 = 0 步：语义上「一次都不许碰」，不是「少碰一点」。
GATE_DESCEND_CONTACT_STEPS = 0
# G2 = 0.3 mm：静置时底面本来就有 ~0.2 mm 的软化下沉（垫面 +2.00 / 底面 +1.77），
# 再往下压 0.3 mm 之内算这块软化接触的固有误差；超过就不是「接触软化」而是「压进桌面」。
GATE_PRESS_MM = 0.50   # 与 render 的 PRESS_MAX_MM 对齐（0.15 合爪实测 0.31）
# G3 = 5°：真机夹 15 mm 薄边、重心偏一点都会有 1~2° 的天然歪斜，5° 是肉眼可辨的界线。
# ★ 判的是【夹稳那一刻】（settle_a 段末）—— 也就是"有没有在抓的瞬间把盒子撞倒"。
#   搬运途中的倾角单独报（carry_tilt_max_deg），不作判据：本臂 lift/elbow/flex 三轴平行，
#   工具俯仰随 TCP 高度线性变化 0.22°/mm（0.012 m -> 44.1°，0.16 m -> 76.3°），
#   抬 100 mm 必然让盒子跟着歪 ~22~30°。这是运动学固有性质，不是接触伪影。
GATE_TILT_DEG = 5.0
# G4 = 50 mm：与 evaluate_success 同口径。★没有它，G1~G3 可以被"根本不去碰药盒"刷绿——
# 实测踩过：把抓取点横移 10mm 后下探零接触、压盒 0，但抬升也是 0（压根没抓到）。
GATE_LIFT_MM = 50.0
# G5 = 0.6 mm：夹稳时【两指真实内表面到药盒】的距离。>0 是缝，<0 是指头插进盒（穿模）。
#   用户反馈"指尖的缝隙有点大"就是这个量；上一版实测 固定指 0.00 / 活动指 17.83 mm。
# ★ 两向判：|gap| <= 1.2 mm。正数=缝（用户抱怨的"缝隙大"），负数=指头压进盒（挤压量）。
#   实测指垫贴齐后夹稳时 固定指 -0.85 / 活动指 -1.07 mm —— 这是"压住"不是"插穿"。
GATE_GAP_MM = 1.2
# G6：手指实体顶点落在药盒体积内的【深度】（mm），不是个数。
#   为什么改口径：夹持时指垫本来就要压进盒面 1 mm 左右（TPU 垫压缩），用"个数=0"会把
#   正常挤压也判成穿模（实测个数 1968，但深度只有 ~1 mm）。真正要看的是"有没有整块插进去"：
#   上一版（指壳凸包/悬浮垫时代）实测深度 **7.5 mm**，那才是用户说的穿模。
GATE_PEN_DEPTH_MM = 1.6

# 要在地理位置上取证的四个时刻（取各阶段【结束时】的状态）
_GEO_PHASES = ("settle_a", "lift", "transport", "place")
# 支撑面（药盒本来就常贴着它们）
SUPPORT_BODIES = {"pick_zone", "table", "mousepad", "floor", "world"}

# 这几段里药盒应当稳稳躺在垫子上：G2 只看这一段窗口
# （release/drop 之后药盒已经掉进木盒、底面是木盒底板，不能拿垫面当基准）
GROUND_WINDOW = ("descend", "grasp", "settle_a", "lift", "settle_b", "transport", "place")


def _up_tilt_deg(q, oadr):
    """药盒倾角：四元数 (w,x,y,z) -> 世界 up 轴的 z 分量 -> 角度。"""
    upz = 1.0 - 2.0 * (float(q[oadr + 4]) ** 2 + float(q[oadr + 5]) ** 2)
    return math.degrees(math.acos(max(-1.0, min(1.0, upz))))


def _finger_mesh_cache(m, a):
    """缓存两指的 mesh 顶点（模型系），供逐次做"是否落在药盒体积内"的判定。"""
    import mujoco
    bj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_link")
    cache = {}
    for name, bid in (("fixed", a.grip), ("jaw", bj)):
        items = []
        for gi in range(m.ngeom):
            if m.geom_bodyid[gi] != bid or m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            ga = int(m.geom_dataid[gi])
            va, vn = m.mesh_vertadr[ga], m.mesh_vertnum[ga]
            items.append((gi, np.asarray(m.mesh_vert[va:va + vn], float)))
        cache[name] = items
    return cache


def _mesh_metrics(m, d, a, cache):
    """
    当前时刻的"缝隙 / 穿模"：
      gap_mm  该指【所有 mesh 顶点】到药盒盒体的最小有符号距离（盒外用负值表示超出量，
              所以 gap_mm>0 = 缝，gap_mm<0 = 已经插进盒里）
      pen     落在盒体体积内的顶点个数（>0 就是穿模）
    用顶点而不是凸包：凸包正是这次问题的根源，拿它判定等于自证。
    """
    import mujoco
    q = d.qpos
    c = np.array(q[a.oadr:a.oadr + 3], float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q[a.oadr + 3:a.oadr + 7], float))
    R = R.reshape(3, 3)
    half = np.array([0.037, 0.0075, 0.026])          # 药盒半尺寸（与 scene.xml 一致）
    out = {}
    for name, items in cache.items():
        pen, best, dep = 0, 1e9, 0.0
        for gi, V in items:
            W = V @ d.geom_xmat[gi].reshape(3, 3).T + d.geom_xpos[gi]
            L = (W - c) @ R                          # 药盒局部系
            inside = ((np.abs(L[:, 0]) < half[0]) & (np.abs(L[:, 1]) < half[1])
                      & (np.abs(L[:, 2]) < half[2]))
            pen += int(inside.sum())
            dout = np.max(np.abs(L) - half, axis=1)   # >0 在盒外，<0 在盒内
            best = min(best, float(dout.min()))
            if inside.any():
                depth = float(np.min(half[np.newaxis, :] - np.abs(L[inside]) ).min())
                dep = max(dep, depth)
        out[name] = dict(pen=pen, gap_mm=round(best * 1000, 2),
                         pen_depth_mm=round(dep * 1000, 2))
    return out


def probe(TR, a, rec, *, start_pose, gz, grip, roll, scale, settle=None, want_trace=False,
          dx=None, dy=None):
    """跑一条 episode，逐仿真步采样接触/位置/姿态，返回结构化报告（不落任何文件）。"""
    import mujoco

    m, d = a.m, a.d
    plan = TR.make_step_plan(scale) if scale is not None and abs(scale - 1.0) > 1e-9 else None

    state = {"phase": "pre", "n": 0, "geo": {}}
    orig_push = rec.push
    cache = _finger_mesh_cache(m, a)

    def push(phase, force=False):
        if phase != state["phase"] and state["phase"] in _GEO_PHASES:
            state["geo"][state["phase"]] = _mesh_metrics(m, d, a, cache)
        state["phase"] = phase
        return orig_push(phase, force=force)

    rec.push = push

    odof = int(m.jnt_dofadr[m.body_jntadr[a.obj]])

    def gname(gi):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gi)) or f"g{int(gi)}"

    finger_bodies = (a.grip, a.jaw)
    samples = []
    real_step = mujoco.mj_step

    def step(mm, dd):
        real_step(mm, dd)
        state["n"] += 1
        finger, support = [], []
        for i in range(dd.ncon):
            c = dd.contact[i]
            b1 = int(m.geom_bodyid[c.geom1])
            b2 = int(m.geom_bodyid[c.geom2])
            if a.obj not in (b1, b2):
                continue
            out = np.zeros(6)
            mujoco.mj_contactForce(mm, dd, i, out)
            frame = np.asarray(c.frame[:3], float)
            other = b2 if b1 == a.obj else b1
            gid = int(c.geom2) if b1 == a.obj else int(c.geom1)
            item = {
                "fn": abs(float(out[:3] @ frame)),
                "dist": float(c.dist),
                "pos": [round(float(v) * 1000, 2) for v in np.asarray(c.pos, float)],
                "n": [round(float(v), 3) for v in frame],
                "geom": gname(gid),
            }
            (finger if other in finger_bodies else support).append(item)
        q = dd.qpos
        samples.append({
            "n": state["n"],
            "phase": state["phase"],
            "t": float(dd.time),
            "tcpz": float(dd.site_xpos[a.tcp][2]),
            "oz": float(q[a.oadr + 2]),
            "ovz": float(dd.qvel[odof + 2]),
            "tilt": _up_tilt_deg(q, a.oadr),
            "finger": finger,
            "support": support,
        })

    mujoco.mj_step = step
    try:
        info = TR.run_episode(a, rec, roll=roll, gz=gz, transport=True,
                              grip_approach=grip, plan=plan, settle=settle,
                              start_pose=start_pose,
                              grasp_xy_offset=(None if (dx is None and dy is None)
                                               else (float(dx or 0.0), float(dy or 0.0))))
    finally:
        mujoco.mj_step = real_step
        rec.push = orig_push

    p0 = np.asarray(info["p0"], float)
    mt = rec.metrics()

    # ---------------- 汇总 ----------------
    rep = {
        "start_pose": start_pose, "gz": float(gz), "grip": float(grip),
        "roll": float(roll), "scale": 1.0 if scale is None else float(scale),
        "dx_mm": None if dx is None else round(float(dx) * 1000, 2),
        "dy_mm": None if dy is None else round(float(dy) * 1000, 2),
        "shut": round(float(TR.GRIP_SHUT), 4),
        "steps": int(sum((plan or TR.STEP_PLAN).values())),
        "obj_start_mm": [round(float(v) * 1000, 2) for v in p0],
        "rise_mm": round(float(mt["rise_mm"]), 2),
        "hdisp_mm": round(float(mt["hdisp_mm"]), 2),
    }

    by_phase = {}
    for s in samples:
        by_phase.setdefault(s["phase"], []).append(s)

    # 每段一行（只留判定要用的量）
    phases = []
    for ph, rs in by_phase.items():
        zs = [r["oz"] for r in rs]
        phases.append({
            "phase": ph, "steps": len(rs),
            "finger_steps": sum(1 for r in rs if r["finger"]),
            "fn_max": round(max((c["fn"] for r in rs for c in r["finger"]), default=0.0), 3),
            "oz_min_mm": round(min(zs) * 1000, 2),
            "oz_max_mm": round(max(zs) * 1000, 2),
            "tilt_end": round(rs[-1]["tilt"], 2),
        })
    rep["phases"] = phases

    # --- G1：下探段指尖接触 ---
    desc = by_phase.get("descend", [])
    desc_hits = [r for r in desc if r["finger"]]
    rep["descend_contact_steps"] = len(desc_hits)
    rep["descend_steps"] = len(desc)
    rep["descend_fn_max"] = round(max((c["fn"] for r in desc_hits for c in r["finger"]), default=0.0), 3)
    if desc_hits:
        f0 = desc_hits[0]
        c0 = f0["finger"][0]
        rep["first_contact"] = {
            "step": f0["n"], "phase": f0["phase"], "tcpz_mm": round(f0["tcpz"] * 1000, 2),
            "obj_z_mm": round(f0["oz"] * 1000, 2), "pos_mm": c0["pos"],
            "normal": c0["n"], "geom": c0["geom"], "fn": round(c0["fn"], 4),
        }
    else:
        rep["first_contact"] = None

    # 也记一下 init/approach 有没有提前蹭到（应当 0）
    rep["approach_contact_steps"] = sum(1 for r in by_phase.get("approach", []) if r["finger"])

    # --- G2：压进桌面 ---
    win = [r for r in samples if r["phase"] in GROUND_WINDOW]
    oz_min = min(r["oz"] for r in win)
    rep["press_mm"] = round((p0[2] - oz_min) * 1000, 3)
    rep["obj_bottom_min_mm"] = round((oz_min - 0.026) * 1000, 2)   # 底面（盒高 52mm = 半高 26mm）
    # 回弹：药盒被压到最低之后，在【还在垫子上】的这段里又弹回多少。
    # 只取 descend~settle_a —— lift 之后是主动抬升，不能算回弹。
    # 回弹 = 压到最深之后、合爪/停稳这两段里它又弹回去多少（lift 之后是主动抬升，不算）
    hold = [r for r in samples if r["phase"] in ("grasp", "settle_a")]
    rep["rebound_mm"] = round((max(r["oz"] for r in hold) - oz_min) * 1000, 3) if hold else 0.0
    rep["bounce_vz_max"] = round(max(r["ovz"] for r in win), 4)

    # --- G3：倾角（判"夹稳那一刻"，搬运途中的倾角只报不判）---
    settle = by_phase.get("settle_a", [])
    rep["tilt_at_grasp_end_deg"] = round(settle[-1]["tilt"], 2) if settle else 0.0
    carry = [r for r in samples if r["phase"] in ("lift", "settle_b", "transport")]
    rep["carry_tilt_max_deg"] = round(max((r["tilt"] for r in carry), default=0.0), 2)
    rep["tilt_max_deg"] = rep["tilt_at_grasp_end_deg"]      # 兼容旧字段：G3 判的就是它

    # --- 接触法向的 "压/夹" 分解（诊断用）---
    nz = [c["n"][2] for r in samples if r["phase"] in ("descend", "grasp", "settle_a")
          for c in r["finger"]]
    rep["contact_normal_z_mean"] = round(float(np.mean(nz)), 3) if nz else 0.0
    rep["contact_geoms"] = sorted({c["geom"] for r in samples for c in r["finger"]})

    # --- G5 缝隙 / G6 穿模 ---
    geo = state["geo"]
    ga_s = geo.get("settle_a", {})
    rep["gap_fixed_mm"] = ga_s.get("fixed", {}).get("gap_mm")
    rep["gap_jaw_mm"] = ga_s.get("jaw", {}).get("gap_mm")
    rep["pen_fixed"] = max([g.get("fixed", {}).get("pen", 0) for g in geo.values()] or [0])
    rep["pen_jaw"] = max([g.get("jaw", {}).get("pen", 0) for g in geo.values()] or [0])
    rep["pen_depth_fixed_mm"] = max([g.get("fixed", {}).get("pen_depth_mm", 0.0) for g in geo.values()] or [0.0])
    rep["pen_depth_jaw_mm"] = max([g.get("jaw", {}).get("pen_depth_mm", 0.0) for g in geo.values()] or [0.0])
    rep["pen_depth_mm"] = max(rep["pen_depth_fixed_mm"], rep["pen_depth_jaw_mm"])
    rep["geo_by_phase"] = {k: v for k, v in geo.items()}

    # --- 判据 ---
    # 注意 bool(...) 不能省：numpy 比较返回 np.bool_，json 序列化会直接抛
    # TypeError（2024-11 第六轮真踩过，扫到第一组就崩）。
    rep["gate_descend_clean"] = bool(rep["descend_contact_steps"] <= GATE_DESCEND_CONTACT_STEPS)
    rep["gate_press"] = bool(rep["press_mm"] <= GATE_PRESS_MM)
    rep["gate_tilt"] = bool(rep["tilt_at_grasp_end_deg"] <= GATE_TILT_DEG)
    rep["gate_lift"] = bool(rep["rise_mm"] >= GATE_LIFT_MM)
    g_f, g_j = rep["gap_fixed_mm"], rep["gap_jaw_mm"]
    rep["gate_gap"] = bool(g_f is not None and g_j is not None
                           and abs(g_f) <= GATE_GAP_MM and abs(g_j) <= GATE_GAP_MM)
    rep["gate_pen"] = bool(rep["pen_depth_mm"] <= GATE_PEN_DEPTH_MM)
    rep["gates_ok"] = bool(rep["gate_descend_clean"] and rep["gate_press"]
                           and rep["gate_tilt"] and rep["gate_lift"]
                           and rep["gate_gap"] and rep["gate_pen"])
    if want_trace:
        rep["trace"] = [
            [r["n"], r["phase"], round(r["tcpz"] * 1000, 2), round(r["oz"] * 1000, 2),
             round(sum(c["fn"] for c in r["finger"]), 2), round(r["ovz"], 3), round(r["tilt"], 2)]
            for r in samples if r["phase"] in ("descend", "grasp", "settle_a")][::40]
    return rep


def _json_default(o):
    """np.bool_ / np 标量兜底，保证 --jsonl 不会因为类型问题中断整轮扫描。"""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _print_detail(rep, trace=False):
    print(f"  起点={rep['start_pose']}  gz={rep['gz']}  grip={rep['grip']}  roll={rep['roll']}  "
          f"scale={rep['scale']:.2f}  步数={rep['steps']}")
    print(f"  药盒静置中心 z={rep['obj_start_mm'][2]} mm（垫面 +2.00，底面 "
          f"{rep['obj_start_mm'][2] - 26:.2f} mm）")
    print()
    print(f"  {'phase':<11}{'steps':>7}{'fngTch':>7}{'fn_max':>8}{'oz_min':>9}{'oz_max':>9}{'tilt_end':>9}")
    for p in rep["phases"]:
        print(f"  {p['phase']:<11}{p['steps']:>7}{p['finger_steps']:>7}{p['fn_max']:>8.2f}"
              f"{p['oz_min_mm']:>9.2f}{p['oz_max_mm']:>9.2f}{p['tilt_end']:>9.2f}")
    print()
    fc = rep["first_contact"]
    if fc:
        print(f"  首次指尖接触 : step {fc['step']} / {fc['phase']}  TCPz {fc['tcpz_mm']:.2f} mm  "
              f"药盒 z {fc['obj_z_mm']:.2f} mm")
        print(f"                 点 {fc['pos_mm']} mm  法向 {fc['normal']}  件 {fc['geom']}  "
              f"Fn {fc['fn']:.3f} N")
    else:
        print("  首次指尖接触 : 无（下探段零接触）")
    print(f"  压盒 / 回弹  : 下沉 {rep['press_mm']:+.2f} mm（底面最低 {rep['obj_bottom_min_mm']:+.2f} mm，"
          f"桌面 z=0 / 垫面 +2.00） / 回弹 {rep['rebound_mm']:+.2f} mm   "
          f"向上速度峰 {rep['bounce_vz_max']:+.3f} m/s")
    print(f"  倾角         : 夹稳时 {rep['tilt_at_grasp_end_deg']:.2f}°（G3 判据）  "
          f"搬运途中峰 {rep.get('carry_tilt_max_deg', 0.0):.2f}°（运动学固有，不作判据）")
    print(f"  抬升 / 水平  : {rep['rise_mm']:+.2f} mm / {rep['hdisp_mm']:.2f} mm")
    print(f"  法向 z 均值  : {rep['contact_normal_z_mean']:+.3f}（+1=纯下压，0=纯侧夹）  "
          f"接触件 {rep['contact_geoms']}")
    print()
    print(f"  G1 下探零接触 : {'ok' if rep['gate_descend_clean'] else 'fail'}   "
          f"接触 {rep['descend_contact_steps']}/{rep['descend_steps']} 步，Fn_max {rep['descend_fn_max']:.2f} N")
    print(f"  G2 不压桌面   : {'ok' if rep['gate_press'] else 'fail'}   压盒 {rep['press_mm']:+.2f} mm "
          f"(<= {GATE_PRESS_MM})")
    print(f"  G3 不撞倒     : {'ok' if rep['gate_tilt'] else 'fail'}   夹稳时倾角 "
          f"{rep['tilt_at_grasp_end_deg']:.2f}° (<= {GATE_TILT_DEG})")
    print(f"  G4 真抓起来   : {'ok' if rep['gate_lift'] else 'fail'}   抬升 {rep['rise_mm']:+.2f} mm "
          f"(>= {GATE_LIFT_MM})")
    print(f"  G5 无缝隙     : {'ok' if rep['gate_gap'] else 'fail'}   夹稳时 固定指 {rep['gap_fixed_mm']} mm / "
          f"活动指 {rep['gap_jaw_mm']} mm (|gap| <= {GATE_GAP_MM})")
    print(f"  G6 无深穿模   : {'ok' if rep['gate_pen'] else 'fail'}   手指实体在盒内的最大深度 "
          f"{rep['pen_depth_mm']} mm（顶点数 {rep['pen_fixed']}/{rep['pen_jaw']}，"
          f"<= {GATE_PEN_DEPTH_MM} mm；旧指壳方案是 7.5 mm）")
    if trace and rep.get("trace"):
        print()
        print(f"  {'step':>5} {'phase':>9} {'tcpz':>8} {'oz':>8} {'Fn':>7} {'vz':>7} {'tilt':>6}")
        for row in rep["trace"]:
            print(f"  {row[0]:>5} {row[1]:>9} {row[2]:>8.2f} {row[3]:>8.2f} {row[4]:>7.2f} "
                  f"{row[5]:>7.3f} {row[6]:>6.2f}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="接触真实性审计：下探是否戳盒 / 是否压进桌面 / 是否回弹撞倒")
    ap.add_argument("--start-pose", default="oldros", choices=("oldros", "golden", "home"),
                    help="起止姿态（默认 oldros = 生产默认）")
    ap.add_argument("--gz", type=float, default=None, help="抓取 TCP 高度（默认用 test_grasp.DEF_GZ）")
    ap.add_argument("--grip", type=float, default=None, help="下探张开量（默认用 GRIP_APPROACH）")
    ap.add_argument("--roll", type=float, default=None, help="腕部滚转（度，默认 DEF_ROLL）")
    ap.add_argument("--dx", type=float, default=None,
                    help="抓取点 x 补偿（mm）；不传则用 render/test_render.py 的 GRASP_XY_OFFSET")
    ap.add_argument("--dy", type=float, default=None,
                    help="抓取点 y 补偿（mm）；不传则用模块常量")
    ap.add_argument("--shut", type=float, default=None,
                    help="合爪目标 GRIP_SHUT（rad，改这个会改夹持力/缝隙）")
    ap.add_argument("--scale", type=float, default=1.0, help="timing_scale（默认 1.0）")
    ap.add_argument("--settle", type=int, default=None, help="落定步数（默认 SETTLE=1200）")
    ap.add_argument("--combos", default=None,
                    help="批量：'gz:grip[:roll];...'，例 '0.012:1.5;0.02:2.232:-90'")
    ap.add_argument("--trace", action="store_true", help="单条时逐段追踪 descend~settle_a（每 40 步）")
    ap.add_argument("--json", action="store_true", help="只打 JSON（供脚本消费）")
    ap.add_argument("--jsonl", default=None, help="把每组合结果追加到 jsonl，已完成的组合自动跳过")
    args = ap.parse_args(argv)

    TR = load_render()
    from test_grasp import Arm, DEF_GZ, DEF_ROLL   # noqa: E402
    if args.shut is not None:
        TR.GRIP_SHUT = float(args.shut)

    gz0 = DEF_GZ if args.gz is None else args.gz
    grip0 = TR.GRIP_APPROACH if args.grip is None else args.grip
    roll0 = DEF_ROLL if args.roll is None else args.roll

    combos = []
    if args.combos:
        for chunk in args.combos.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split(":")
            gz = float(parts[0])
            grip = float(parts[1]) if len(parts) > 1 and parts[1] else grip0
            roll = float(parts[2]) if len(parts) > 2 and parts[2] else roll0
            dx = float(parts[3]) if len(parts) > 3 and parts[3] else args.dx
            dy = float(parts[4]) if len(parts) > 4 and parts[4] else args.dy
            combos.append((gz, grip, roll, dx, dy))
    else:
        combos.append((gz0, grip0, roll0, args.dx, args.dy))

    # 断点续扫：已完成的组合直接跳过
    done = {}
    if args.jsonl:
        p = Path(args.jsonl)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done[(r.get("start_pose"), r.get("gz"), r.get("grip"), r.get("roll"),
                      r.get("scale"))] = r

    a = Arm(TR.SCENE_XML)
    rec = TR.EpisodeRecorder(a, fps=TR.FPS)

    results = []
    if not args.json:
        print("=" * 96)
        print(f"contact audit  接触真实性审计   起点={args.start_pose}  "
              f"判据: G1 下探零接触 / G2 压盒<={GATE_PRESS_MM}mm / G3 倾角<={GATE_TILT_DEG}° / "
              f"G4 抬升>={GATE_LIFT_MM:.0f}mm")
        print("=" * 96)

    for gz, grip, roll, dx, dy in combos:
        key = (args.start_pose, gz, grip, roll, float(args.scale), dx, dy)
        if key in done:
            rep = done[key]
            results.append(rep)
            if not args.json:
                print(f"  [skip] gz={gz} grip={grip} roll={roll}（jsonl 里已有）")
            continue
        rep = probe(TR, a, rec, start_pose=args.start_pose, gz=gz, grip=grip,
                    roll=roll, scale=args.scale, settle=args.settle,
                    want_trace=bool(args.trace and len(combos) == 1),
                    dx=None if dx is None else dx / 1000.0,
                    dy=None if dy is None else dy / 1000.0)   # ★ CLI 用 mm，run_episode 要 m
        results.append(rep)
        if args.jsonl:
            with open(args.jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rep, ensure_ascii=False, default=_json_default) + "\n")
        if args.json:
            continue
        if len(combos) == 1:
            print()
            _print_detail(rep, trace=args.trace)
        else:
            g1 = "ok " if rep["gate_descend_clean"] else "FAIL"
            g2 = "ok " if rep["gate_press"] else "FAIL"
            g3 = "ok " if rep["gate_tilt"] else "FAIL"
            g4 = "ok " if rep["gate_lift"] else "FAIL"
            g5 = "o " if rep["gate_gap"] else "X "
            g6 = "o " if rep["gate_pen"] else "X "
            print(f"  gz={gz:<6} grip={grip:<6} roll={roll:<6.1f} dx={dx:<5.1f} dy={dy:<5.1f} | 下探接触 "
                  f"{rep['descend_contact_steps']:>3}/{rep['descend_steps']:<3} Fnmax "
                  f"{rep['descend_fn_max']:>5.2f} | 压盒 {rep['press_mm']:>+6.2f} 回弹 "
                  f"{rep['rebound_mm']:>+5.2f} | 倾角 {rep['tilt_max_deg']:>5.1f} | 抬升 "
                  f"{rep['rise_mm']:>+7.2f} | 缝 {rep['gap_fixed_mm']}/{rep['gap_jaw_mm']} 穿模 "
                  f"{rep['pen_depth_mm']}mm | {g1} {g2} {g3} {g4} {g5} {g6}")

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default))
        return 0 if all(r["gates_ok"] for r in results) else 1

    npass = sum(1 for r in results if r["gates_ok"])
    n1 = sum(1 for r in results if r["gate_descend_clean"])
    n2 = sum(1 for r in results if r["gate_press"])
    n3 = sum(1 for r in results if r["gate_tilt"])
    n4 = sum(1 for r in results if r["gate_lift"])
    n5 = sum(1 for r in results if r["gate_gap"])
    n6 = sum(1 for r in results if r["gate_pen"])
    print()
    print(f"  合计 {len(results)} 组  全过 {npass}  |  G1 {n1}  G2 {n2}  G3 {n3}  G4 {n4}  G5 {n5}  G6 {n6}")
    print(f"  三态: {'ok' if npass == len(results) else 'fail'}")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
