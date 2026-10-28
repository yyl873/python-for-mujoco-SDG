#!/usr/bin/env python3
"""对比同一位置 A / B 两次 rollout 的释放动作。

回答 HANDOFF §4 实验 2 的三个问题：
  1. A/B 的 gripper_req 峰值是否不同
  2. A/B 相对「合爪时刻」何时开始张开
  3. A/B 张开时的关节位置是否相同（末端位姿）

输入：tools/log_gripper_rollout.py 产出的两个 CSV。
时间一律换算成「相对合爪时刻」的秒数，避免两次录制的起始时刻不同造成假差异。

用法：
  python tools/compare_release_ab.py logs/gripper_A_samepos.csv logs/gripper_B_samepos.csv
  python tools/compare_release_ab.py A.csv B.csv --thr 20 --fps 30
"""
import argparse
import csv
import sys

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r["t_s"]),
                    "req": _f(r.get("gripper_req")),
                    "sent": _f(r.get("gripper_sent")),
                    "present": _f(r.get("present_after")),
                    **{k: _f(r.get(f"{k}_before")) for k in ARM},
                })
            except (TypeError, ValueError):
                continue
    return rows


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def find_release(rows, thr):
    """返回 dict(t_grasp, t_open, peak, pose, ok, why)。

    合爪时刻 = 先张开(>thr)之后第一次跌到 <=5；
    释放开始 = 合爪之后第一次重新 >= thr。
    """
    idx = 0
    n = len(rows)
    while idx < n and (rows[idx]["req"] is None or rows[idx]["req"] < thr):
        idx += 1
    if idx >= n:
        return {"ok": False, "why": f"整段没有 gripper_req >= {thr}（策略始终没张开）"}
    grasp = None
    for j in range(idx, n):
        r = rows[j]["req"]
        if r is not None and r <= 5.0:
            grasp = j
            break
    if grasp is None:
        return {"ok": False, "why": "第一次张开之后再没合拢（没抓到东西？）"}
    open_i = None
    for j in range(grasp, n):
        r = rows[j]["req"]
        if r is not None and r >= thr:
            open_i = j
            break
    if open_i is None:
        return {"ok": False, "why": f"合爪之后始终没再张开到 {thr}（放不开）"}
    peak = max((row["req"] or 0.0) for row in rows[open_i:])
    pose = {k: rows[open_i][k] for k in ARM}
    return {
        "ok": True,
        "t_grasp": rows[grasp]["t"],
        "t_open": rows[open_i]["t"],
        "dt_open": rows[open_i]["t"] - rows[grasp]["t"],
        "peak": peak,
        "peak_present": max((row["present"] or 0.0) for row in rows[open_i:]),
        "pose": pose,
    }


def fmt(v, spec="6.2f"):
    return "  n/a " if v is None else format(v, spec)


def report(name, res):
    print(f"\n===== {name}")
    if not res["ok"]:
        print(f"  ✗ {res['why']}")
        return
    print(f"  合爪时刻            : t={res['t_grasp']:.2f}s")
    print(f"  相对合爪 {res['dt_open']:.2f}s 后开始张开")
    print(f"  gripper_req 峰值    : {res['peak']:.2f}")
    print(f"  present 峰值(实测)  : {res['peak_present']:.2f}")
    print("  张开时的关节角      : " + "  ".join(f"{k}={fmt(res['pose'][k])}" for k in ARM))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_a")
    ap.add_argument("csv_b")
    ap.add_argument("--thr", type=float, default=20.0, help="判定「张开」的 gripper_req 阈值")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    A, B = load(args.csv_a), load(args.csv_b)
    print(f"读入: A={args.csv_a} ({len(A)} 帧)   B={args.csv_b} ({len(B)} 帧)")
    ra = find_release(A, args.thr)
    rb = find_release(B, args.thr)
    report("A", ra)
    report("B", rb)

    print("\n===== 差异（A vs B）")
    if not (ra["ok"] and rb["ok"]):
        print("  至少一侧没检测到完整释放，无法比对。逐侧结论见上。")
        return 1

    dp = rb["peak"] - ra["peak"]
    dt = rb["dt_open"] - ra["dt_open"]
    print(f"  req 峰值    : A={ra['peak']:.2f}  B={rb['peak']:.2f}  Δ={dp:+.2f}"
          f"  ({100*dp/ra['peak']:+.0f}%)")
    print(f"  张开时机    : A={ra['dt_open']:.2f}s  B={rb['dt_open']:.2f}s  Δ={dt:+.2f}s"
          f"  ({dt*args.fps:+.0f} 帧)")
    print("  关节角差    : " + "  ".join(
        f"{k}={fmt(rb['pose'][k] - ra['pose'][k] if (rb['pose'][k] is not None and ra['pose'][k] is not None) else None, '+6.2f')}"
        for k in ARM))
    deltas = [abs(rb["pose"][k] - ra["pose"][k]) for k in ARM
              if rb["pose"][k] is not None and ra["pose"][k] is not None]
    maxd = max(deltas) if deltas else None
    print(f"  最大关节差  : {fmt(maxd, '6.2f')}°")

    print("\n===== 判读")
    if abs(dp) >= 0.2 * max(ra["peak"], 1e-6):
        print(f"  → req 峰值差 {100*dp/ra['peak']:+.0f}%，达 20% 以上：**幅度不足**成立。")
    else:
        print("  → req 峰值接近（差 <20%）：幅度不是原因，查释放位置/时机或盒子卡住。")
    if abs(dt) >= 0.5:
        print(f"  → 张开时机差 {dt:+.2f}s（≥0.5s）：**时机**不同，重点查轨迹相位。")
    else:
        print("  → 张开时机接近（差 <0.5s）：时机不是原因。")
    if maxd is not None and maxd >= 15:
        print(f"  → 张开时关节差 {maxd:.1f}°（≥15°）：**释放位姿**明显不同，盒子可能没对准木盒。")
    else:
        print("  → 张开时关节差 <15°：位姿基本一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())