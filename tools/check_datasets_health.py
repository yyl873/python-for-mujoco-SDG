#!/usr/bin/env python3
"""真机/仿真 lerobot 数据集体检（只读；只写 log/report）。

查什么
------
1. 结构：meta/info、meta/episodes、tasks 与数据行数是否自洽
2. 数据：维度、NaN/Inf、全局 index 连续性、每 episode 时间戳单调、每步跳变（teleport）
3. 视频：每个 episode 的窗口能否解出、帧数与 meta 长度是否一致、图像亮/对比/锐度
4. 语义：夹爪每 episode 起止值与峰值（判断开合方向与量纲）
5. 重复：动作序列指纹查重
6. 域差：各数据集图像统计量对比（仿真 vs 真机）

输出 logs/<name>.json + 终端摘要。退出码 0=ok / 1=warn / 2=fail。

用法
----
  env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
      tools/check_datasets_health.py --root <数据集根> [--root ...] \
      [--out logs/dataset_health.json] [--sample-every 45] [--no-video]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def jload(p):
    with open(p) as f:
        return json.load(f)


def read_episodes_meta(root):
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "meta/episodes/**/*.parquet"), recursive=True)):
        t = pq.read_table(f)
        cols = {c: t.column(c).to_pylist() for c in t.column_names}
        for i in range(t.num_rows):
            rows.append({c: cols[c][i] for c in t.column_names})
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def read_data(root):
    """按全局 index 排序后返回 action/state/timestamp/episode_index 的拼接数组。"""
    acc = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "data/**/*.parquet"), recursive=True)):
        t = pq.read_table(
            f, columns=["action", "observation.state", "timestamp", "frame_index",
                        "episode_index", "index"]
        )
        for c in ("action", "observation.state"):
            acc[c].append(np.asarray(t.column(c).to_pylist(), dtype=np.float32))
        for c in ("timestamp", "frame_index", "episode_index", "index"):
            acc[c].append(np.asarray(t.column(c).to_pylist(), dtype=np.int64 if c != "timestamp" else np.float64))
    if not acc["index"]:
        raise RuntimeError("没有找到 data/*.parquet")
    out = {k: np.concatenate(v) for k, v in acc.items()}
    order = np.argsort(out["index"], kind="stable")
    for k in out:
        out[k] = out[k][order]
    out["_n_files"] = len(acc["index"])
    return out


def video_scan(root, info, cam_key, ep_rows, sample_every=45):
    import av

    vinfo = info["features"][cam_key]["info"]
    vdir = os.path.join(root, "videos", cam_key)
    files = sorted(glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True))
    by_file = defaultdict(list)
    for r in ep_rows:
        ci = r.get(f"videos/{cam_key}/chunk_index")
        fi = r.get(f"videos/{cam_key}/file_index")
        if ci is None or fi is None:
            continue
        by_file[(int(ci), int(fi))].append(r)

    res = {"camera": cam_key, "n_files": len(files), "video_hw": [vinfo["video.height"], vinfo["video.width"]],
           "codec": vinfo.get("video.codec"), "decode_errors": [], "frame_count_mismatch": [],
           "n_episodes_checked": 0, "img_mean": None, "img_std": None, "sharp": None}
    means, stds, sharps = [], [], []
    for (ci, fi), eps in sorted(by_file.items()):
        path = os.path.join(vdir, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
        windows = [(float(r[f"videos/{cam_key}/from_timestamp"]), float(r[f"videos/{cam_key}/to_timestamp"]),
                    int(r["episode_index"]), int(r["length"])) for r in eps]
        if not os.path.exists(path):
            res["decode_errors"].append({"file": os.path.basename(path), "err": "missing"})
            continue
        counts = defaultdict(int)
        t_end = max(w[1] for w in windows)
        try:
            with av.open(path) as c:
                stream = c.streams.video[0]
                for i, frame in enumerate(c.decode(stream)):
                    ts = float(frame.pts * frame.time_base) if frame.pts is not None else i / 30.0
                    for f0, f1, epi, _ in windows:
                        if f0 <= ts < f1:
                            counts[epi] += 1
                            break
                    if i % sample_every == 0:
                        a = frame.to_ndarray(format="rgb24").astype(np.float32)
                        means.append(float(a.mean()))
                        stds.append(float(a.std()))
                        if i % (sample_every * 4) == 0:
                            g = a.mean(2)
                            lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
                            sharps.append(float(lap.var()))
                    if ts > t_end:
                        break
        except Exception as e:  # noqa: BLE001
            res["decode_errors"].append({"file": os.path.basename(path), "err": f"{type(e).__name__}: {e}"})
            continue
        for _f0, _f1, epi, length in windows:
            got = counts.get(epi, 0)
            res["n_episodes_checked"] += 1
            if abs(got - length) > 2:
                res["frame_count_mismatch"].append({"episode": epi, "meta": length, "video": got})
    if means:
        res["img_mean"] = round(float(np.mean(means)), 1)
        res["img_std"] = round(float(np.mean(stds)), 1)
        res["sharp"] = round(float(np.mean(sharps)), 1) if sharps else None
    return res


def fp_of(a, stride=4, n=400):
    seg = a[: n * stride: stride]
    if len(seg) == 0:
        return "empty"
    return hashlib.sha1(np.round(seg, 1).tobytes()).hexdigest()[:12]


def check_dataset(root, sample_every=45, do_video=True, do_data=True):
    rep = {"root": root, "name": os.path.basename(os.path.normpath(root)), "warn": [], "fail": [], "notes": []}
    info_p = os.path.join(root, "meta", "info.json")
    if not os.path.exists(info_p):
        rep["fail"].append("meta/info.json 缺失")
        return rep
    info = jload(info_p)
    rep.update({
        "codebase": info.get("codebase_version"), "fps": info.get("fps"),
        "robot_type": info.get("robot_type"),
        "info_total_episodes": info.get("total_episodes"), "info_total_frames": info.get("total_frames"),
        "features": {k: {"dtype": v.get("dtype"), "shape": v.get("shape"), "names": v.get("names")}
                     for k, v in info["features"].items()},
    })
    tasks_p = os.path.join(root, "meta", "tasks.parquet")
    rep["tasks"] = [r["task"] for r in pq.read_table(tasks_p).to_pylist()] if os.path.exists(tasks_p) else None

    ep_rows = read_episodes_meta(root)
    lengths = np.array([int(r["length"]) for r in ep_rows], dtype=np.int64)
    rep["episode_len"] = {"n": int(len(lengths)), "min": int(lengths.min()), "med": int(np.median(lengths)),
                          "max": int(lengths.max()), "sum": int(lengths.sum()),
                          "n_distinct": int(len(np.unique(lengths)))}
    if len(np.unique(lengths)) == 1:
        rep["warn"].append(f"全部 episode 等长 {int(lengths[0])} 帧 = 固定时长录制/重采样，节奏多样性低")

    if not do_data:
        rep["videos"] = {}
        for cam in [k for k, v in info["features"].items() if v.get("dtype") == "video"] if do_video else []:
            print(f"   video[{cam}] 解码中…", flush=True)
            vr = video_scan(root, info, cam, ep_rows, sample_every=sample_every)
            rep["videos"][cam] = vr
            if vr["decode_errors"]:
                rep["fail"].append(f"{cam} 解码失败 {len(vr['decode_errors'])} 个文件")
            if vr["frame_count_mismatch"]:
                rep["warn"].append(f"{cam} {len(vr['frame_count_mismatch'])} 个 episode 视频帧数与 meta 不符")
        rep["status"] = "fail" if rep["fail"] else ("warn" if rep["warn"] else "ok")
        return rep

    d = read_data(root)
    A, S, E, T, I = d["action"], d["observation.state"], d["episode_index"], d["timestamp"], d["index"]
    rep["n_rows"] = int(len(A))
    rep["data"] = {
        "n_files": int(d["_n_files"]), "action_shape": list(A.shape), "state_shape": list(S.shape),
        "nan": int(np.isnan(A).sum() + np.isnan(S).sum()),
        "inf": int(np.isinf(A).sum() + np.isinf(S).sum()),
        "action_min": [round(float(x), 2) for x in A.min(0)],
        "action_max": [round(float(x), 2) for x in A.max(0)],
        "action_span": [round(float(x), 1) for x in (A.max(0) - A.min(0))],
        "act_minus_state_absmean": [round(float(x), 2) for x in np.abs(A - S).mean(0)],
    }
    if rep["data"]["nan"] or rep["data"]["inf"]:
        rep["fail"].append("action/state 含 NaN/Inf")
    if rep["n_rows"] != int(lengths.sum()):
        rep["fail"].append(f"数据行数 {rep['n_rows']} != meta 长度合计 {int(lengths.sum())}")
    if info.get("total_frames") not in (None, rep["n_rows"]):
        rep["warn"].append(f"info.total_frames {info.get('total_frames')} != 实际 {rep['n_rows']}")
    if info.get("total_episodes") not in (None, len(ep_rows)):
        rep["warn"].append(f"info.total_episodes {info.get('total_episodes')} != meta 行数 {len(ep_rows)}")

    rep["index_contiguous"] = bool(np.array_equal(np.sort(I), np.arange(len(I))))
    if not rep["index_contiguous"]:
        rep["warn"].append(f"全局 index 不连续（min {int(I.min())} max {int(I.max())} n {len(I)}）")

    bad_ts, dts = [], []
    for e in np.unique(E):
        t = T[E == e]
        dts.append(np.diff(t))
        if not np.all(np.diff(t) > 0):
            bad_ts.append(int(e))
    all_dt = np.concatenate(dts) if dts else np.array([0.0])
    rep["dt"] = {"med": round(float(np.median(all_dt)), 5), "min": round(float(all_dt.min()), 5),
                 "max": round(float(all_dt.max()), 5),
                 "fps_implied": round(float(1 / np.median(all_dt)), 2) if np.median(all_dt) else None,
                 "n_episodes_non_monotonic": len(bad_ts)}
    if bad_ts:
        rep["fail"].append(f"{len(bad_ts)} 个 episode 时间戳非单调（如 {bad_ts[:5]}）")
    exp = 1.0 / float(info.get("fps") or 30)
    if abs(rep["dt"]["med"] - exp) > 1e-4:
        rep["warn"].append(f"时间戳中位步长 {rep['dt']['med']} != 1/fps {exp:.5f}")
    if rep["dt"]["max"] > 0.2:
        rep["warn"].append(f"最大步长 {rep['dt']['max']} s > 0.2 s（丢帧）")

    jumps = np.abs(np.diff(A, axis=0))
    rep["jump"] = {"p99": [round(float(x), 2) for x in np.percentile(jumps, 99, axis=0)],
                   "max": [round(float(x), 2) for x in jumps.max(0)],
                   "n_gt20deg": int((jumps > 20).sum())}

    g = A[:, 5]
    rows, fp = [], {}
    for e in np.unique(E):
        m = E == e
        gg = g[m]
        rows.append({"episode": int(e), "n": int(m.sum()),
                     "start": round(float(gg[:5].mean()), 2), "end": round(float(gg[-5:].mean()), 2),
                     "max": round(float(gg.max()), 2), "min": round(float(gg.min()), 2)})
        fp.setdefault(fp_of(A[m]), []).append(int(e))
    starts = np.array([r["start"] for r in rows])
    ends = np.array([r["end"] for r in rows])
    rep["gripper"] = {
        "global_min": round(float(g.min()), 2), "global_max": round(float(g.max()), 2),
        "start_mean": round(float(starts.mean()), 2), "end_mean": round(float(ends.mean()), 2),
        "n_start_eq_min": int((np.abs(starts - g.min()) < 0.5).sum()),
        "n_end_eq_min": int((np.abs(ends - g.min()) < 0.5).sum()),
        "n_ep_end_gt_start3": int((ends > starts + 3).sum()),
        "n_ep_end_lt_start3": int((ends < starts - 3).sum()),
        "per_episode_head": rows[:5],
    }
    rep["duplicates"] = {k: v for k, v in fp.items() if len(v) > 1}
    if rep["duplicates"]:
        rep["warn"].append(f"重复 episode 指纹 {len(rep['duplicates'])} 组: {list(rep['duplicates'].values())[:3]}")

    rep["joint_span"] = {j: round(float(A[:, i].max() - A[:, i].min()), 1) for i, j in enumerate(JOINTS)}
    rep["wrist_roll"] = {"min": round(float(A[:, 4].min()), 1), "max": round(float(A[:, 4].max()), 1),
                         "std": round(float(A[:, 4].std()), 1)}

    rep["videos"] = {}
    if do_video:
        for cam in [k for k, v in info["features"].items() if v.get("dtype") == "video"]:
            print(f"   video[{cam}] 解码中…", flush=True)
            vr = video_scan(root, info, cam, ep_rows, sample_every=sample_every)
            rep["videos"][cam] = vr
            if vr["decode_errors"]:
                rep["fail"].append(f"{cam} 解码失败 {len(vr['decode_errors'])} 个文件")
            if vr["frame_count_mismatch"]:
                rep["warn"].append(f"{cam} {len(vr['frame_count_mismatch'])} 个 episode 视频帧数与 meta 不符")
    rep["status"] = "fail" if rep["fail"] else ("warn" if rep["warn"] else "ok")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", dest="roots", action="append", required=True)
    ap.add_argument("--out", default="logs/dataset_health.json")
    ap.add_argument("--sample-every", type=int, default=45)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--mode", choices=("all", "data", "video"), default="all",
                    help="data=只查 parquet；video=只查视频（pyarrow 与 pyav 同进程偶发 SIGBUS，拆两趟跑）")
    args = ap.parse_args()

    reports = []
    for r in args.roots:
        print(f"[scan] {r}", flush=True)
        try:
            rep = check_dataset(r, args.sample_every,
                                do_video=(args.mode != "data") and not args.no_video,
                                do_data=(args.mode != "video"))
        except Exception as e:  # noqa: BLE001
            rep = {"root": r, "name": os.path.basename(os.path.normpath(r)),
                   "warn": [], "fail": [f"体检异常 {type(e).__name__}: {e}"], "status": "fail"}
        reports.append(rep)
        print(f"   -> {rep['status']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reports": reports}, f, ensure_ascii=False, indent=1)

    print("\n================ 摘要 ================")
    for rep in reports:
        print(f"\n[{rep['status']}] {rep['name']}")
        if "episode_len" in rep:
            e = rep["episode_len"]
            print(f"  eps={e['n']} frames={e['sum']} len {e['min']}/{e['med']}/{e['max']} "
                  f"distinct={e['n_distinct']}  idx_contig={rep.get('index_contiguous')}")
        if "data" in rep:
            d, j = rep["data"], rep.get("jump", {})
            print(f"  action{d['action_shape']} nan={d['nan']} inf={d['inf']} files={d['n_files']}")
            print(f"  span_deg={rep.get('joint_span')}")
            print(f"  wrist_roll={rep.get('wrist_roll')}")
            print(f"  |a-s|={d['act_minus_state_absmean']}  jump_p99={j.get('p99')} steps>20deg={j.get('n_gt20deg')}")
            print(f"  dt={rep.get('dt')}")
        g = rep.get("gripper")
        if g:
            print(f"  gripper min/max {g['global_min']}/{g['global_max']} start~{g['start_mean']} "
                  f"end~{g['end_mean']} n(start==min)={g['n_start_eq_min']} n(end==min)={g['n_end_eq_min']}")
        for cam, v in (rep.get("videos") or {}).items():
            print(f"  video[{cam}] {v['video_hw']} {v['codec']} files={v['n_files']} eps={v['n_episodes_checked']} "
                  f"err={len(v['decode_errors'])} mismatch={len(v['frame_count_mismatch'])} "
                  f"mean={v['img_mean']} std={v['img_std']} sharp={v['sharp']}")
        for w in rep.get("warn") or []:
            print(f"  warn: {w}")
        for f_ in rep.get("fail") or []:
            print(f"  FAIL: {f_}")
    print(f"\n报告: {args.out}")

    # 跨数据集对比
    ok = [r for r in reports if r.get("videos")]
    if len(ok) > 1:
        print("\n---- 跨数据集域差（env 相机）----")
        for r in ok:
            v = r["videos"].get("observation.images.env")
            if v:
                print(f"  {r['name']:24s} mean={v['img_mean']:6} std={v['img_std']:6} sharp={v['sharp']}")

    rc = 0
    if any(r["status"] == "warn" for r in reports):
        rc = 1
    if any(r["status"] == "fail" for r in reports):
        rc = 2
    return rc


if __name__ == "__main__":
    sys.exit(main())
