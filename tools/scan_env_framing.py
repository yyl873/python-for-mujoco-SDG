#!/usr/bin/env python3
"""量真机"环境相机"的取景：绿垫在画面里到底在哪、有没有被裁掉、是不是全程都这样。

回答的问题：
  1) 环境相机的取景在所有 episode 里是否一致（-> 相机是固定的、朝向偏了）
  2) 还是只有一部分 episode 是这样（-> 录制过程中相机被动过 / 分批录制换了机位）
  3) 某条 episode 内部相机有没有动（逐帧的垫子位置漂移）

对每个 mp4 均匀采样 N 帧，用"最大绿色连通域"当拾取垫，记录：
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "lerobot_datasets"))
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="inspect/env_framing")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()

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