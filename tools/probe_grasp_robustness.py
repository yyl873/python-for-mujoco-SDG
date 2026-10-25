#!/usr/bin/env python3
"""
抓取鲁棒性探针：换起点 / 换步数配比 / 换节奏后，还抓得住吗？

为什么必须有这个工具（2024-11 第五轮的教训）：
    本项目的"抓得住"依赖下探结束时那点 PD 滞后，处在临界点上。实测同一套接触段，
    起点从 golden 换成 old_ros initial_pose、或者只改自由段步数配比，都可能让
    合爪把药盒横甩出去（抬升 0mm、药盒 z 27.8 -> 9.3）。修长的验证（demo 渲染 +
    20 条批量 + 体检 + 切分）要十几分钟，而这个探针几十秒就能告诉你"这条路通不通"。
    => 动 STEP_PLAN 的任何一段、动起止姿态、动 roll/gz 之前，先跑它。

用法:
    # 默认：当前起点（old_ros initial_pose）+ (700,800,500) + 10 档节奏
    python3 tools/probe_grasp_robustness.py

    # A/B：看 golden 起点在同样的节奏档位下还剩几档
    python3 tools/probe_grasp_robustness.py --start-pose golden

    # 扫多组步数配比（三段合计必须相等，默认 2200，其余段不动）
    python3 tools/probe_grasp_robustness.py --combos "700,800,500;800,700,500;700,1000,300"

    # 只测一档节奏、只看一条组合
    python3 tools/probe_grasp_robustness.py --scales 1.00 --combos "700,800,500"

输出: 每个 (组合, 节奏) 一行 ----  ok/抬升/水平/总步数；末尾给通过率。
      纯内存跑物理（不写任何文件、不渲染），单次约 0.1~0.3 s。
      SIGFPE（signal 8，本机 mj_step 偶发）会让整批中断 —— 直接重跑本命令即可，
      物理结果确定性可复现。

退出码: 0 = 全部通过；1 = 有失败档位（不该当成故障，只是"临界"的信号）；2 = 参数错误。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grasp"))
sys.path.insert(0, str(ROOT / "render"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import test_grasp as G  # noqa: E402  （Arm 封装 / 模型路径）
import test_render as R  # noqa: E402  （run_episode / StepPlan / 录制器）


def _load_collect():
    """collect/collect.py 不是包（没有 __init__.py），按文件加载，才能复用 judge()。"""
    spec = importlib.util.spec_from_file_location("collect_mod", ROOT / "collect" / "collect.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_collect()

# 三段自由空间步数合计（其余段用 STEP_PLAN 的默认值）。换配比时保持这个总和不变，
# 总步数才仍是 7500 = 15.0 s。
COMBO_SUM = R.N_INIT_LIFT + R.N_RETRACT + R.N_ABOVE
DEFAULT_COMBOS = f"{R.STEP_PLAN['init_lift']},{R.STEP_PLAN['approach_retract']},"
DEFAULT_COMBOS += f"{R.STEP_PLAN['approach_above']}"
DEFAULT_SCALES = "0.90,0.92,0.95,0.98,0.99,1.00,1.02,1.05,1.08,1.10"


def parse_combos(text: str):
    out = []
    for chunk in str(text).split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        il, ar, ab = (int(v) for v in chunk.split(","))
        out.append((il, ar, ab))
    return out


def parse_scales(text: str):
    return [float(v) for v in str(text).split(",") if v.strip()]


def run_one(arm, start_pose, combo, scale, roll=None, gz=None):
    """跑一条真物理 episode（不渲染、不落盘），返回 (ok, 抬升mm, 水平mm, 总步数, reasons)。"""
    il, ar, ab = combo
    saved = dict(R.STEP_PLAN)
    try:
        R.STEP_PLAN["init_lift"] = il
        R.STEP_PLAN["approach_retract"] = ar
        R.STEP_PLAN["approach_above"] = ab
        plan = R.make_step_plan(scale)
        if sum(plan.values()) != R.TOTAL_STEPS and abs(scale - 1.0) < 1e-9:
            raise SystemExit(f"步数配平错了：{sum(plan.values())} != {R.TOTAL_STEPS}")
        arm.d = mujoco.MjData(arm.m)          # 每条新 data：复用会触发 SIGFPE
        rec = R.EpisodeRecorder(arm, fps=R.FPS)
        info = R.run_episode(arm, rec, roll=G.DEF_ROLL if roll is None else roll,
                             gz=G.DEF_GZ if gz is None else gz, transport=True,
                             plan=plan, start_pose=start_pose)
        ok, reasons, met = C.judge(rec, info, {})
        return ok, met["rise_mm"], met["hdisp_mm"], sum(plan.values()), reasons
    finally:
        R.STEP_PLAN.clear()
        R.STEP_PLAN.update(saved)


def main(argv=None):
    ap = argparse.ArgumentParser(description="抓取鲁棒性探针（纯内存，不落盘）")
    ap.add_argument("--start-pose", default=R.START_POSE_DEFAULT,
                    choices=["oldros", "golden", "home"], help="起止姿态")
    ap.add_argument("--combos", default=DEFAULT_COMBOS,
                    help=f"init_lift,approach_retract,approach_above（分号分隔多组；"
                         f"合计应 = {COMBO_SUM}，否则总步数不再是 7500）")
    ap.add_argument("--scales", default=DEFAULT_SCALES, help="timing_scale 列表（逗号分隔）")
    ap.add_argument("--roll", type=float, default=None, help="覆盖腕部滚转（度）")
    ap.add_argument("--gz", type=float, default=None, help="覆盖抓取 TCP 高度（m）")
    args = ap.parse_args(argv)

    combos = parse_combos(args.combos)
    scales = parse_scales(args.scales)
    for il, ar, ab in combos:
        if il + ar + ab != COMBO_SUM:
            raise SystemExit(f"组合 {il},{ar},{ab} 合计 {il + ar + ab} != {COMBO_SUM} —— "
                             f"三段配比必须守恒，否则总步数/时长会变")

    arm = G.Arm()
    print(f"[probe] 起点={args.start_pose}  组合数={len(combos)}  节奏档={len(scales)}  "
          f"三段合计={COMBO_SUM}  roll={args.roll or G.DEF_ROLL}  gz={args.gz or G.DEF_GZ}")
    print(f"  {'init_lift,retract,above':<24}{'scale':>6}  {'ok':<5}{'抬升mm':>9}"
          f"{'水平mm':>9}{'总步数':>8}")
    n_total = n_ok = 0
    for combo in combos:
        for s in scales:
            ok, rise, hd, tot, reasons = run_one(arm, args.start_pose, combo, s,
                                                 roll=args.roll, gz=args.gz)
            n_total += 1
            n_ok += int(ok)
            print(f"  {combo[0]},{combo[1]},{combo[2]:<20}{s:>6.2f}  "
                  f"{'ok' if ok else 'X':<5}{rise:>9.2f}{hd:>9.1f}{tot:>8}"
                  + ("" if ok else f"  {reasons}"), flush=True)
    print(f"[probe] 通过 {n_ok}/{n_total}")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())