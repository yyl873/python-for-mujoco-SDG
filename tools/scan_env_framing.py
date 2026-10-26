#!/usr/bin/env python3
"""量真机"环境相机"的取景：绿垫在画面里到底在哪、有没有被裁掉、是不是全程都这样。

回答的问题：
  1) 环境相机的取景在所有 episode 里是否一致（-> 相机是固定的、朝向偏了）
  2) 还是只有一部分 episode 是这样（-> 录制过程中相机被动过 / 分批录制换了机位）
  3) 某条 episode 内部相机有没有动（逐帧的垫子位置漂移）

注意：lerobot v3.0 的 mp4 是**按 200 MB 打包**的，一个 mp4 里装着多个 episode 的时间段
（yyl100_dataset_fixed 是 16 个 mp4 装 151 个 episode；holdout_fixed 恰好 1:1）。
所以默认 `--unit file` 数的是"文件"，要按"每条实录"数请用 `--unit episode`：
它从 meta/episodes 读每条 episode 的 file_index + from_timestamp + to_timestamp，按段采样。

对每个 mp4 / 每条 episode 均匀采样 N 帧，用"最大绿色连通域"当拾取垫，记录：
    cx, cy        归一化中心 (0~1)
    area%         占全图百分比
    touches       是否贴到画面左/上/右/下边框（贴边 = 垫子被裁掉）
    diam          max(bbox 宽, 高) 作为垫子直径的粗估（px）
输出：
    env_framing.csv            逐文件
    summary_env_framing.txt    按数据集聚合 + 逐帧时间线摘要
    montage_env_framing.png    每个数据集取首文件的 早/中/晚 三帧，画出垫子包围盒

相机路数的判定沿用 tools/scan_real_cams.py 的 static% 主判据（本脚本只测环境相机那一路）。

用法：
    env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/scan_env_framing.py \
        [--root ~/lerobot_datasets] [--frames 24] [--workers 8] [--out inspect/env_framing]
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

CAM_ENV = "observation.images.env"
CAM_WRIST = "observation.images.wrist"
GREEN_LO, GREEN_HI = (35, 40, 30), (95, 255, 255)
MIN_AREA_PCT = 0.8      # 小于此面积视为"垫子不在画面里"
MAX_AREA_PCT = 60.0     # 大于此面积多半是背景被并进来了


def pad_of(frame: np.ndarray) -> dict:
    """最大绿色连通域 -> 垫子的位置/大小/贴边情况。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    out = {"cx": np.nan, "cy": np.nan, "area_pct": 0.0, "diam": np.nan,
           "touch": "", "areas": []}
    if n > 1:
        order = np.argsort(-st[1:, cv2.CC_STAT_AREA]) + 1
        out["areas"] = [int(st[k, cv2.CC_STAT_AREA]) for k in order[:3]]
        k = int(order[0])
        x0, y0, bw, bh, a = st[k]
        x1, y1 = x0 + bw - 1, y0 + bh - 1
        t = "".join(nm for nm, cond in (("L", x0 <= 1), ("T", y0 <= 1),
                                        ("R", x1 >= w - 2), ("B", y1 >= h - 2)) if cond)
        out.update(cx=float(cen[k][0]) / w, cy=float(cen[k][1]) / h,
                   area_pct=100.0 * float(a) / (h * w), diam=int(max(bw, bh)), touch=t)
    return out


def scan(job: tuple[str, str, int]) -> dict:
    path, dataset, n_samples = job
    cv2.setNumThreads(1)
    rec = {"path": path, "dataset": dataset, "file": Path(path).stem,
           "total_frames": 0, "fps": 0.0, "n_sampled": 0,
           "cx_med": np.nan, "cy_med": np.nan, "area_med": np.nan, "diam_med": np.nan,
           "inside_frac": np.nan, "clip_frac": np.nan, "missing_frac": np.nan,
           "cx_spread": np.nan, "cy_spread": np.nan, "touch_modes": "", "error": ""}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        rec["error"] = "打不开"
        return rec
    rec["fps"] = round(cap.get(cv2.CAP_PROP_FPS) or 0.0, 3)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rec["total_frames"] = total
    if total < 4:
        rec["error"] = "帧数不足"
        cap.release()
        return rec
    rows = []
    for i in (np.linspace(0.02, 0.98, max(4, n_samples)) * (total - 2)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            rows.append(pad_of(fr))
    cap.release()
    if len(rows) < 4:
        rec["error"] = "采样帧不足(%d)" % len(rows)
        return rec
    rec["n_sampled"] = len(rows)
    present = [r for r in rows if MIN_AREA_PCT <= r["area_pct"] <= MAX_AREA_PCT]
    if not present:
        present = [r for r in rows if r["area_pct"] > 0]
    rec["missing_frac"] = round(1.0 - len(present) / len(rows), 3)
    if present:
        cx = np.array([r["cx"] for r in present])
        cy = np.array([r["cy"] for r in present])
        rec.update(cx_med=round(float(np.median(cx)), 3), cy_med=round(float(np.median(cy)), 3),
                   area_med=round(float(np.median([r["area_pct"] for r in present])), 2),
                   diam_med=round(float(np.median([r["diam"] for r in present])), 1),
                   cx_spread=round(float(np.ptp(cx)), 3), cy_spread=round(float(np.ptp(cy)), 3),
                   inside_frac=round(sum(1 for r in present if not r["touch"]) / len(present), 3),
                   clip_frac=round(sum(1 for r in present if r["touch"]) / len(present), 3))
        modes = {}
        for r in present:
            key = r["touch"] or "inside"
            modes[key] = modes.get(key, 0) + 1
        rec["touch_modes"] = " ".join("%s:%d" % (k, v) for k, v in sorted(modes.items()))
    return rec


def episode_jobs(ds_dir: Path, cam: str = CAM_ENV) -> list[dict]:
    """从 meta/episodes 读出每条 episode 落在哪个 mp4、哪段时间。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("按 episode 扫描需要 pyarrow：先跑 tools/get_pyarrow_libs.py，"
                         "再用 PYTHONPATH=$PWD/.cache/pylibs env -u PYTHONHOME /usr/bin/python3 ...")
    specs = []
    fps = 30.0
    info = ds_dir / "meta" / "info.json"
    if info.exists():
        fps = float(__import__("json").loads(info.read_text()).get("fps") or 30.0)
    for p in sorted((ds_dir / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(p)
        cols = t.column_names
        need = ["episode_index", "length",
                "videos/%s/chunk_index" % cam, "videos/%s/file_index" % cam,
                "videos/%s/from_timestamp" % cam, "videos/%s/to_timestamp" % cam]
        if not all(c in cols for c in need):
            continue
        for i in range(t.num_rows):
            specs.append({"episode": int(t.column("episode_index")[i].as_py()),
                          "length": int(t.column("length")[i].as_py()),
                          "chunk": int(t.column(need[2])[i].as_py()),
                          "file": int(t.column(need[3])[i].as_py()),
                          "from_ts": float(t.column(need[4])[i].as_py()),
                          "to_ts": float(t.column(need[5])[i].as_py()),
                          "fps": fps})
    return specs


def scan_file_episodes(job: tuple[str, str, list[dict], int]) -> list[dict]:
    """打开一个 mp4，把落在里面的每条 episode 各采样 N 帧。"""
    path, dataset, specs, n_samples = job
    cv2.setNumThreads(1)
    outs = []
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0 if cap.isOpened() else 30.0
    for sp in specs:
        rec = {"dataset": dataset, "episode": sp["episode"], "file": Path(path).stem,
               "length": sp["length"], "from_ts": round(sp["from_ts"], 2),
               "to_ts": round(sp["to_ts"], 2), "n_sampled": 0,
               "cx_med": np.nan, "cy_med": np.nan, "area_med": np.nan, "diam_med": np.nan,
               "inside_frac": np.nan, "clip_frac": np.nan,
               "cx_spread": np.nan, "cy_spread": np.nan, "touch_modes": "", "error": ""}
        if not cap.isOpened():
            rec["error"] = "打不开"
            outs.append(rec)
            continue
        start = int(round(sp["from_ts"] * fps))
        idxs = (np.linspace(0.05, 0.95, max(4, n_samples)) * max(1, sp["length"] - 1)).astype(int) + start
        idxs = [i for i in idxs if 0 <= i < max(1, total)]
        rows = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                rows.append(pad_of(fr))
        if len(rows) < 4:
            rec["error"] = "采样帧不足(%d)" % len(rows)
            outs.append(rec)
            continue
        rec["n_sampled"] = len(rows)
        present = [r for r in rows if MIN_AREA_PCT <= r["area_pct"] <= MAX_AREA_PCT] or \
                  [r for r in rows if r["area_pct"] > 0]
        if present:
            cx = np.array([r["cx"] for r in present])
            cy = np.array([r["cy"] for r in present])
            rec.update(cx_med=round(float(np.median(cx)), 3), cy_med=round(float(np.median(cy)), 3),
                       area_med=round(float(np.median([r["area_pct"] for r in present])), 2),
                       diam_med=round(float(np.median([r["diam"] for r in present])), 1),
                       cx_spread=round(float(np.ptp(cx)), 3), cy_spread=round(float(np.ptp(cy)), 3),
                       inside_frac=round(sum(1 for r in present if not r["touch"]) / len(present), 3),
                       clip_frac=round(sum(1 for r in present if r["touch"]) / len(present), 3))
            modes = {}
            for r in present:
                modes[r["touch"] or "inside"] = modes.get(r["touch"] or "inside", 0) + 1
            rec["touch_modes"] = " ".join("%s:%d" % (k, v) for k, v in sorted(modes.items()))
        outs.append(rec)
    cap.release()
    return outs


def montage_per_dataset(root: Path, datasets: list[str], out: Path,
                        fr=(0.10, 0.50, 0.90), cell=(320, 240)) -> Path | None:
    rows = []
    for ds in datasets:
        files = sorted((root / ds / "videos" / CAM_ENV).rglob("*.mp4"))
        if not files:
            continue
        cells = []
        cap = cv2.VideoCapture(str(files[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for frac in fr:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, frame = cap.read()
            cells.append((frac, frame if ok else None))
        cap.release()
        rows.append((ds, cells))
    if not rows:
        return None
    cw, ch = cell
    canvas = np.full((ch * len(rows), cw * len(fr), 3), 28, np.uint8)
    for r, (ds, cells) in enumerate(rows):
        for c, (frac, frame) in enumerate(cells):
            if frame is None:
                continue
            vis = frame.copy()
            p = pad_of(frame)
            h, w = vis.shape[:2]
            if np.isfinite(p["cx"]):
                cv2.drawMarker(vis, (int(p["cx"] * w), int(p["cy"] * h)), (0, 0, 255),
                               cv2.MARKER_CROSS, 24, 2)
            t = cv2.resize(vis, cell, interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (cw, 18), (0, 0, 0), -1)
            cv2.putText(t, "%s t=%.0f%% c=(%.2f,%.2f) a=%.1f%% %s"
                        % (ds[:14], frac * 100, p["cx"], p["cy"], p["area_pct"],
                           p["touch"] or "inside"),
                        (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        (120, 255, 120) if not p["touch"] else (120, 200, 255), 1, cv2.LINE_AA)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = t
    path = out / "montage_env_framing.png"
    cv2.imwrite(str(path), canvas)
    return path


def run_episodes(root: Path, out: Path, n_frames: int, workers: int) -> int:
    """按 episode 扫描（lerobot 的 mp4 是打包的多段，这里按真实实录逐条数）。"""
    jobs, datasets = [], []
    for ds in sorted(p for p in root.iterdir() if (p / "videos").is_dir()):
        specs = episode_jobs(ds, CAM_ENV)
        if not specs:
            continue
        datasets.append(ds.name)
        by_file = {}
        for sp in specs:
            by_file.setdefault((sp["chunk"], sp["file"]), []).append(sp)
        for (chunk, findex), group in sorted(by_file.items()):
            path = ds / "videos" / CAM_ENV / ("chunk-%03d" % chunk) / ("file-%03d.mp4" % findex)
            if not path.exists():
                grp = sorted((ds / "videos" / CAM_ENV).rglob("file-%03d.mp4" % findex))
                if not grp:
                    continue
                path = grp[0]
            jobs.append((str(path), ds.name, group, n_frames))
    total_eps = sum(len(j[2]) for j in jobs)
    print("按 episode 扫描：%d 个数据集 / %d 条实录 / %d 个 mp4 容器 / 每条采样 %d 帧"
          % (len(datasets), total_eps, len(jobs), n_frames))

    recs = []
    with mp.Pool(max(1, workers)) as pool:
        for group in pool.imap_unordered(scan_file_episodes, jobs, chunksize=1):
            recs += group
    recs.sort(key=lambda r: (r["dataset"], r["episode"]))

    with (out / "env_framing_episodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    lines = ["# 真机环境相机取景：按 episode（真实实录条数）统计",
             "# inside_frac=垫子完整落在画面内的采样帧比例；clip_frac=贴住画面边框的比例",
             ""]
    lines.append("%-24s %5s %8s %8s %9s %9s %9s  %s"
                 % ("dataset", "条数", "cx", "cy", "完整帧均值", "贴边帧均值", "估计间距", "分布"))
    grand = {}
    for ds in datasets:
        rs = [r for r in recs if r["dataset"] == ds and not r["error"]]
        if not rs:
            continue
        clp = np.array([r["clip_frac"] for r in rs], dtype=float)
        ins = np.array([r["inside_frac"] for r in rs], dtype=float)
        n_all = int(np.sum(clp >= 0.999))
        n_some = int(np.sum((clp > 0) & (clp < 0.999)))
        n_none = int(np.sum(clp <= 0.001))
        grand[ds] = (len(rs), n_all, n_some, n_none)
        lines.append("%-24s %5d %8.3f %8.3f %9.3f %9.3f %9s  %s"
                     % (ds, len(rs), float(np.nanmedian([r["cx_med"] for r in rs])),
                        float(np.nanmedian([r["cy_med"] for r in rs])),
                        float(np.nanmean(ins)), float(np.nanmean(clp)), "",
                        "每帧都贴边=%d 部分贴边=%d 从不贴边=%d" % (n_all, n_some, n_none)))
    tot = [sum(v[i] for v in grand.values()) for i in range(4)]
    lines.append("")
    lines.append("合计 %d 条实录：每帧都贴边 %d 条 / 部分贴边 %d 条 / 从不贴边 %d 条" % tuple(tot))
    (out / "summary_env_framing_episodes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print("\n输出:", out / "env_framing_episodes.csv", "|", out / "summary_env_framing_episodes.txt")
    return 0


def scan_union(job: tuple[str, str, int]) -> dict:
    """把 N 帧的绿色掩码做时序并集：机械臂遮挡会在不同帧落在不同位置，
    并集能还原"垫子真实占多大、有没有越过画面边框"，比单帧判贴边稳。"""
    path, dataset, n = job
    cv2.setNumThreads(1)
    rec = {"dataset": dataset, "file": Path(path).stem, "n_frames": 0,
           "x0": -1, "y0": -1, "x1": -1, "y1": -1, "w": 0, "h": 0,
           "touch": "", "area_pct": np.nan, "fill": np.nan, "error": ""}
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        rec["error"] = "打不开"
        return rec
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = None
    for i in (np.linspace(0.03, 0.97, max(4, n)) * max(1, total - 2)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        acc = m if acc is None else cv2.bitwise_or(acc, m)
        rec["n_frames"] += 1
    cap.release()
    if acc is None:
        rec["error"] = "无有效帧"
        return rec
    h, w = acc.shape
    nlab, _lab, st, _c = cv2.connectedComponentsWithStats(acc, 8)
    if nlab < 2:
        rec["error"] = "并集无绿色"
        return rec
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x0, y0, bw, bh, a = st[k]
    rec.update(x0=int(x0), y0=int(y0), x1=int(x0 + bw - 1), y1=int(y0 + bh - 1),
               w=int(bw), h=int(bh), area_pct=round(100.0 * int(a) / (h * w), 2),
               fill=round(float(a) / float(bw * bh), 2),
               touch="".join(nm for nm, cd in (("L", x0 <= 1), ("T", y0 <= 1),
                                               ("R", x0 + bw >= w - 1), ("B", y0 + bh >= h - 1)) if cd))
    return rec


def run_union(root: Path, out: Path, n_frames: int, workers: int) -> int:
    jobs = []
    for ds in sorted(p for p in root.iterdir() if (p / "videos").is_dir()):
        for f in sorted((ds / "videos" / CAM_ENV).rglob("*.mp4")):
            jobs.append((str(f), ds.name, n_frames))
    print("按文件做绿色并集：%d 个文件，每个采样 %d 帧" % (len(jobs), n_frames))
    recs = []
    with mp.Pool(max(1, workers)) as pool:
        recs = list(pool.imap_unordered(scan_union, jobs, chunksize=1))
    recs.sort(key=lambda r: (r["dataset"], r["file"]))
    with (out / "env_union.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)
    lines = ["# 绿色掩码时序并集 = 垫子的真实投影范围（抗遮挡）",
             "# x1=639 表示垫子右缘压到右边框；y1=479 表示垫子下缘被画面切掉",
             "%-24s %-9s %5s %5s %5s %5s %5s %5s %7s %6s %s"
             % ("dataset", "file", "x0", "y0", "x1", "y1", "w", "h", "area%", "fill", "贴边")]
    for r in recs:
        if r["error"]:
            lines.append("%-24s %-9s !! %s" % (r["dataset"], r["file"], r["error"]))
            continue
        lines.append("%-24s %-9s %5d %5d %5d %5d %5d %5d %7.2f %6.2f %s"
                     % (r["dataset"], r["file"], r["x0"], r["y0"], r["x1"], r["y1"],
                        r["w"], r["h"], r["area_pct"], r["fill"], r["touch"] or "-"))
    (out / "summary_env_union.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[2:]))
    print("\n输出:", out / "env_union.csv", "|", out / "summary_env_union.txt")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "lerobot_datasets"))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="inspect/env_framing")
    ap.add_argument("--unit", choices=("file", "episode", "union"), default="file",
                    help="file=按 mp4 文件（默认）；episode=按每条实录（需 pyarrow）；union=N 帧绿色并集（抗遮挡，量垫子真实范围）")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

    if args.unit == "episode":
        return run_episodes(root, out, args.frames, args.workers)
    if args.unit == "union":
        return run_union(root, out, args.frames, args.workers)

    datasets, jobs = [], []
    for ds in sorted(p for p in root.iterdir() if (p / "videos").is_dir()):
        files = sorted((ds / "videos" / CAM_ENV).rglob("*.mp4"))
        if files:
            datasets.append(ds.name)
            jobs += [(str(f), ds.name, args.frames) for f in files]
    print("环境相机文件 = %d 个，数据集 = %d 个，每个采样 %d 帧" % (len(jobs), len(datasets), args.frames))

    recs = []
    with mp.Pool(max(1, args.workers)) as pool:
        for r in pool.imap_unordered(scan, jobs, chunksize=1):
            recs.append(r)
    recs.sort(key=lambda r: (r["dataset"], r["file"]))

    with (out / "env_framing.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    lines = []
    lines.append("# 真机环境相机取景测量（绿垫 = 最大绿色连通域）")
    lines.append("# inside_frac: 垫子完整落在画面内的采样帧比例；clip_frac: 贴住画面边框的比例；")
    lines.append("# cx_spread/cy_spread: 同一文件内垫子中心的极差（大 -> 相机在录制中被移动或臂遮挡严重）")
    lines.append("")
    lines.append("%-24s %-10s %3s %6s %6s %7s %6s %7s %7s %7s  %s"
                 % ("dataset", "file", "帧", "cx", "cy", "area%", "diam", "inside", "clip", "缺帧", "贴边模式"))
    for r in recs:
        if r["error"]:
            lines.append("%-24s %-10s  !! %s" % (r["dataset"], r["file"], r["error"]))
            continue
        lines.append("%-24s %-10s %3d %6.3f %6.3f %7.2f %6.0f %7.3f %7.3f %7.3f  %s"
                     % (r["dataset"], r["file"], r["n_sampled"], r["cx_med"], r["cy_med"],
                        r["area_med"], r["diam_med"], r["inside_frac"], r["clip_frac"],
                        r["missing_frac"], r["touch_modes"]))
    lines.append("")
    lines.append("## 按数据集聚合（判断是「全部这个视角」还是「只有一部分」）")
    lines.append("%-24s %4s %8s %8s %9s %9s %9s %9s  %s"
                 % ("dataset", "文件", "cx", "cy", "cx极差", "cy极差", "inside均值", "clip均值", "结论"))
    for ds in datasets:
        rs = [r for r in recs if r["dataset"] == ds and not r["error"]]
        if not rs:
            continue
        cx = np.array([r["cx_med"] for r in rs], dtype=float)
        cy = np.array([r["cy_med"] for r in rs], dtype=float)
        ins = np.array([r["inside_frac"] for r in rs], dtype=float)
        clp = np.array([r["clip_frac"] for r in rs], dtype=float)
        cx_sp, cy_sp = float(np.nanmax(cx) - np.nanmin(cx)), float(np.nanmax(cy) - np.nanmin(cy))
        if np.nanmean(clp) > 0.9:
            verdict = "所有 episode 都裁到垫子（相机朝向偏了）"
        elif np.nanmean(clp) < 0.1:
            verdict = "所有 episode 垫子完整（这一批机位正常）"
        else:
            verdict = "部分裁部分不裁（录制中换过机位）"
        lines.append("%-24s %4d %8.3f %8.3f %9.3f %9.3f %9.3f %9.3f  %s"
                     % (ds, len(rs), float(np.nanmedian(cx)), float(np.nanmedian(cy)),
                        cx_sp, cy_sp, float(np.nanmean(ins)), float(np.nanmean(clp)), verdict))
    (out / "summary_env_framing.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(datasets) - 3:]))

    p = montage_per_dataset(root, datasets, out)
    if p:
        print("\n取景拼图:", p)
    print("\n输出目录:", out)
    for f in sorted(out.glob("*")):
        print("  %-28s %9d B" % (f.name, f.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())