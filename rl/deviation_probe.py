#!/usr/bin/env python3
"""rl/deviation_probe.py —— 第 1 步：ACT 闭环偏离诊断（给 DAgger 定触发器）

要回答的三个问题
----------------
1. ACT 从**第几步 / 哪个阶段**开始偏离专家轨迹？
2. 偏离时物理已经多脏（穿透、倾角）？
3. DAgger 的接管触发器该怎么写、会触发多少次（代价）？

做法
----
对每条 held-out episode（默认 100-129）：
  · `randomize=False` 起，保证和专家母本 `scripts/ep_XXXXXX.npz` 的起始状态一致；
  · 基座闭环跑（残差恒 0），每控制步（30 Hz，与母本帧一一对应）记录
    TCP 位置 / 穿透 / 倾角 / 抬升 / 接触数；
  · 与母本 `tcp_pos[t]` 比对，得到逐帧偏离 `dev_mm`；
  · 母本自带 `phase` 标签，所以能直接报"偏离首次出现在哪个阶段"。

输出
----
- `onset_step`：首次 dev > --dev-thresh 的步
- `pen_onset_step`：首次穿透 > --pen-thresh 的步
- `trigger_rate`：按推荐规则会被触发的步占比（决定 DAgger 采集成本）
- 按阶段聚合的偏离统计

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/deviation_probe.py \
    --episodes 100-129 --dev-thresh 25 --pen-thresh 5 --out rl/eval/deviation_probe.json
```
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


def parse_pool(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACT 闭环偏离诊断（DAgger 触发器定标）")
    ap.add_argument("--base", default="sim156")
    ap.add_argument("--episodes", default="100-129")
    ap.add_argument("--dev-thresh", type=float, default=25.0, help="偏离阈值（mm）")
    ap.add_argument("--pen-thresh", type=float, default=5.0, help="穿透阈值（mm）")
    ap.add_argument("--tilt-thresh", type=float, default=25.0, help="药盒倾角阈值（度）——实测最能区分成败的特征")
    ap.add_argument("--resume", action="store_true", help="跳过 jsonl 里已完成的 episode（防 SIGFPE 丢结果）")
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "deviation_probe.json"))
    a = ap.parse_args(argv)

    from env_residual import ResidualSO101                          # noqa: PLC0415
    pool = parse_pool(a.episodes)
    env = ResidualSO101(base_name=a.base, episode_pool=tuple(pool), randomize=False)
    t0 = time.time()
    jl = Path(str(a.out).replace(".json", ".jsonl"))
    rows = []
    done = set()
    if a.resume and jl.exists():
        for line in jl.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                rows.append(r)
                done.add(int(r["episode_id"]))
            except Exception:                                    # noqa: BLE001
                pass
        print(f"  续跑：已有 {len(done)} 条", flush=True)
    series = []          # [(dev[], pen[], phase[], success)]
    try:
        for eid in pool:
            if eid in done:
                continue
            npz = np.load(ROOT / "scripts" / f"ep_{eid:06d}.npz", allow_pickle=True)
            tcp_ref = np.asarray(npz["tcp_pos"], float)             # (T,3) 米
            ph_ref = [str(x) for x in npz["phase"]]
            lift_ref = np.asarray(npz["lift_mm"], float)
            env.episode_pool = (eid,)
            obs, info = env.reset(seed=0)
            devs, pens, tilts, lifts, phases, ncons = [], [], [], [], [], []
            trig_steps = trig_pt = 0
            onset = pen_onset = None
            onset_phase = None
            n = min(len(tcp_ref), int(round(env.max_seconds * env.control_hz)))
            for k in range(n):
                obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
                tcp = env.arm.tcp_pos()
                dev = float(np.linalg.norm(tcp - tcp_ref[k])) * 1000.0
                pen = float(info["penetration_mm"])
                tilt = float(info["box_tilt_deg"])
                lift = float(info["max_lift_mm"])
                nc = int(info["contact_n"])
                ph = ph_ref[k] if k < len(ph_ref) else "?"
                devs.append(dev); pens.append(pen); tilts.append(tilt)
                lifts.append(lift); phases.append(ph); ncons.append(nc)
                if onset is None and dev > a.dev_thresh:
                    onset, onset_phase = k, ph
                if pen_onset is None and pen > a.pen_thresh:
                    pen_onset = k
                # 推荐触发器（DAgger takeover）
                if dev > a.dev_thresh or pen > a.pen_thresh:
                    trig_steps += 1
                if pen > 14.0 or tilt > a.tilt_thresh:      # 物理型触发器（推荐）
                    trig_pt += 1
                if term or trunc:
                    break
            rec = dict(episode_id=eid, steps=len(devs),
                       base_success=bool(info["success"]),
                       onset_step=onset, onset_phase=onset_phase,
                       pen_onset_step=pen_onset,
                       dev_max_mm=round(float(np.max(devs)), 1),
                       dev_median_mm=round(float(np.median(devs)), 1),
                       pen_max_mm=round(float(np.max(pens)), 1),
                       tilt_max_deg=round(float(np.max(tilts)), 1),
                       lift_max_mm=round(float(np.max(lifts)), 1),
                       expert_lift_mm=round(float(np.max(lift_ref)), 1),
                       trigger_steps=trig_steps,
                       trigger_rate=round(trig_steps / max(1, len(devs)), 3),
                       trigger_rate_physics=round(trig_pt / max(1, len(devs)), 3),
                       phase_dev={p: round(float(np.median([d for d, pp in zip(devs, phases) if pp == p])), 1)
                                  for p in sorted(set(phases))})
            rows.append(rec)
            series.append((np.asarray(devs), np.asarray(pens), phases, bool(info["success"])))
            with jl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  ep{eid}: 偏离 onset={onset}({onset_phase}) 穿透 onset={pen_onset} "
                  f"dev中位={rec['dev_median_mm']}mm 触发率={rec['trigger_rate']:.0%} "
                  f"成功={rec['base_success']}", flush=True)
    finally:
        env.close()

    onsets = [r["onset_step"] for r in rows if r["onset_step"] is not None]
    pen_on = [r["pen_onset_step"] for r in rows if r["pen_onset_step"] is not None]
    phase_hits = {}
    for r in rows:
        if r["onset_phase"]:
            phase_hits[r["onset_phase"]] = phase_hits.get(r["onset_phase"], 0) + 1
    rep = dict(base=a.base, episodes=a.episodes, dev_thresh=a.dev_thresh,
               pen_thresh=a.pen_thresh, n_episodes=len(rows),
               n_base_success=sum(1 for r in rows if r["base_success"]),
               n_with_dev_onset=len(onsets),
               onset_step_median=float(np.median(onsets)) if onsets else None,
               onset_step_p25=float(np.percentile(onsets, 25)) if onsets else None,
               onset_step_p75=float(np.percentile(onsets, 75)) if onsets else None,
               phase_onset_counts=phase_hits,
               pen_onset_median=float(np.median(pen_on)) if pen_on else None,
               trigger_rate_mean=round(float(np.mean([r["trigger_rate"] for r in rows])), 3),
               trigger_rate_physics_mean=round(float(np.mean([r.get("trigger_rate_physics", 0) for r in rows])), 3),
               trigger_rate_physics_p90=round(float(np.percentile([r.get("trigger_rate_physics", 0) for r in rows], 90)), 3),
               trigger_rate_p90=round(float(np.percentile([r["trigger_rate"] for r in rows], 90)), 3),
               wall_s=round(time.time() - t0, 1), rows=rows)
    print(f"\n  base 成功 {rep['n_base_success']}/{rep['n_episodes']}")
    print(f"  偏离 onset：中位第 {rep['onset_step_median']} 步（IQR "
          f"{rep['onset_step_p25']}–{rep['onset_step_p75']}），{rep['n_with_dev_onset']}/{rep['n_episodes']} 条有偏离")
    print(f"  偏离首次出现的阶段分布：{phase_hits}")
    print(f"  穿透 onset 中位：第 {rep['pen_onset_median']} 步（阈值 {a.pen_thresh} mm）")
    print(f"  物理型触发器（pen>14mm 或 tilt>{a.tilt_thresh}°）：平均触发 {rep['trigger_rate_physics_mean']:.0%} "
          f"的步（p90 {rep['trigger_rate_physics_p90']:.0%}）")
    print(f"  推荐触发器（dev>{a.dev_thresh}mm 或 pen>{a.pen_thresh}mm）会触发 "
          f"{rep['trigger_rate_mean']:.0%} 的步（p90 {rep['trigger_rate_p90']:.0%}）→ 这就是 DAgger 的数据量代价")
    # ---- 触发器阈值格点：不同 (dev, pen) 下的触发率 与 对"失败局面"的覆盖率 ----
    grid = []
    for dev_t in (25, 40, 60, 80):
        for pen_t in (5, 10, 15):
            rates, cover = [], []
            for devs, pens, _ph, ok in series:
                trig = (devs > dev_t) | (pens > pen_t)
                rates.append(float(trig.mean()))
                if not ok:                      # 只在失败 episode 上看"是否被触发覆盖"
                    cover.append(float(trig.any()))
            grid.append(dict(dev_thresh_mm=dev_t, pen_thresh_mm=pen_t,
                             trigger_rate_mean=round(float(np.mean(rates)), 3),
                             fail_coverage=(round(float(np.mean(cover)), 3) if cover else None)))
    print("\n  触发器阈值格点（触发率 = DAgger 采集代价；覆盖率 = 失败 episode 中被触发到的比例）")
    print(f"  {'dev>mm':>7}{'pen>mm':>8}{'触发率':>9}{'失败覆盖':>10}")
    for g in grid:
        cov = "n/a" if g["fail_coverage"] is None else f"{g['fail_coverage']:.0%}"
        print(f"  {g['dev_thresh_mm']:>7}{g['pen_thresh_mm']:>8}{g['trigger_rate_mean']:>8.0%}{cov:>10}")
    rep["trigger_grid"] = grid
    op = Path(a.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  落盘 {op.relative_to(ROOT) if op.is_relative_to(ROOT) else op}（{rep['wall_s']} s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())