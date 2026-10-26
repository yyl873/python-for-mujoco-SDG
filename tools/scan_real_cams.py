#!/usr/bin/env python3
"""真机数据集相机流排查：判定每个 mp4 到底是"固定环境相机"还是"装在夹爪上的腕部相机"。

背景（2026-10 实测，已由用户目视确认）：
    *_fixed 系列数据集的 `observation.images.env` 与 `.wrist` 两路内容互换了；
    *_corners 系列没有互换。所以不能靠文件夹名判断，必须按画面内容判定。

判据（两条独立指标，都要给）：
  A. 时序静态像素占比 static%：均匀采样 N 帧，逐像素灰度时序标准差 < 8 的像素占全图比例。
     相机固定在桌上 -> 桌子/背景大范围不动 -> 25%~50%；
     相机装在夹爪上 -> 整幅随臂移动      -> 1%~3%。
  B. 绿区(拾取垫)面积变异 pad_cv = std/mean：固定相机下垫子占地几乎恒定(<0.35)；
     腕部相机随接近/离开剧烈变化(>0.6)。

输出（默认写到 inspect/cam_scan/，已被 .gitignore 忽略）：
    cam_scan.csv            每个 mp4 一行
    summary.txt             按数据集/流聚合 + 判定 + 需要互换的清单
    montage_all_files.png   每个 mp4 一帧缩略图，带 static%/pad% 标注，供人眼复核
    montage_flagged.png     只放判定为"与文件夹名不符"或"同数据集内不一致"的文件

用法：
    env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/scan_real_cams.py \
        [--root ~/lerobot_datasets] [--frames 16] [--workers 8] [--out inspect/cam_scan]
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
from pathlib import Path

import cv2
import numpy as np

STATIC_PX = 8.0          # 逐像素时序 std 阈值（灰度级）
FIXED_STATIC = 12.0      # static% >= 此值 -> 固定相机
MOVING_STATIC = 6.0      # static% <= 此值 -> 移动(腕部)相机
PAD_CV_FIXED = 0.60      # pad_cv <= 此值 佐证固定相机

CAM_ENV = "observation.images.env"
CAM_WRIST = "observation.images.wrist"


def green_frac(frame: np.ndarray) -> float:
    """最大绿色连通域面积占全图百分比（拾取垫）。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (35, 40, 30), (95, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _lab, st, _c = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return 0.0
    return 100.0 * int(st[1:, cv2.CC_STAT_AREA].max()) / (frame.shape[0] * frame.shape[1])


def scan_file(job: tuple[str, str, int]) -> dict:
    """单个 mp4：采样 N 帧 -> static% / pad_cv / 亮度。"""
    path, dataset, n_samples = job
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(path)
    rec = {"path": path, "dataset": dataset, "stream": Path(path).parent.parent.name,
           "file": Path(path).stem, "total_frames": 0, "fps": 0.0, "static_pct": float("nan"),
           "pad_cv": float("nan"), "pad_mean_pct": float("nan"), "mean_v": float("nan"),
           "error": ""}
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
    idxs = (np.linspace(0.02, 0.98, max(4, n_samples)) * (total - 2)).astype(int)
    grays, pads, means = [], [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if not ok:
            continue
        grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        pads.append(green_frac(fr))
        means.append(float(fr.mean()))
    cap.release()
    if len(grays) < 4:
        rec["error"] = "采样帧不足(%d)" % len(grays)
        return rec
    sd = np.stack(grays).std(axis=0)
    rec["static_pct"] = round(100.0 * float(np.mean(sd < STATIC_PX)), 2)
    p = np.asarray(pads, dtype=np.float64)
    rec["pad_mean_pct"] = round(float(p.mean()), 2)
    rec["pad_cv"] = round(float(p.std() / p.mean()), 3) if p.mean() > 1e-6 else float("nan")
    rec["mean_v"] = round(float(np.mean(means)), 1)
    rec["sampled"] = len(grays)
    return rec


def classify(rec: dict) -> tuple[str, str]:
    """-> (判定, 说明)。判定: fixed / moving / ambiguous / error"""
    if rec["error"]:
        return "error", rec["error"]
    s, cv = rec["static_pct"], rec["pad_cv"]
    if not np.isfinite(s):
        return "error", "无有效采样"
    # 主判据 static%：实测两族差距极大（固定相机 34~50%，腕部相机 0.6~2.3%），
    # 中间区 6~12% 现实数据里没出现过；pad_cv 只做边界地带的辅助确认，不参与否决。
    if s <= MOVING_STATIC:
        return "moving", "static%%=%.1f pad_cv=%.2f" % (s, cv)
    if s >= FIXED_STATIC:
        return "fixed", "static%%=%.1f pad_cv=%.2f" % (s, cv)
    if np.isfinite(cv) and cv > PAD_CV_FIXED:
        return "moving", "边界带 static%%=%.1f pad_cv=%.2f" % (s, cv)
    return "ambiguous", "边界带 static%%=%.1f pad_cv=%.2f" % (s, cv)


def discover(root: Path) -> tuple[list[tuple[str, str]], list[dict]]:
    """-> (待扫描的 mp4 任务列表, 需报告的异常/跳过项)"""
    jobs, notes = [], []
    for ds in sorted(p for p in root.iterdir() if p.is_dir()):
        info = ds / "meta" / "info.json"
        if not info.exists():
            continue
        meta = json.loads(info.read_text())
        keys = [k for k in (meta.get("features") or {}) if k.startswith("observation.images.")]
        found_any = False
        for cam in (CAM_ENV, CAM_WRIST):
            vdir = ds / "videos" / cam
            files = sorted(vdir.rglob("*.mp4"))
            if files:
                found_any = True
                jobs += [(str(f), ds.name) for f in files]
            elif vdir.exists():
                notes.append({"dataset": ds.name, "kind": "空目录", "detail": "%s 存在但没有 mp4" % cam})
        if not found_any:
            lancedirs = [p.name for p in ds.iterdir() if p.name.endswith(".lance")]
            notes.append({"dataset": ds.name, "kind": "无 mp4 视频",
                          "detail": "仅有 %s；features=%s" % (",".join(lancedirs) or "无", ",".join(keys))})
        # info.json 里 features 的键顺序本身就是线索：录制脚本按设备枚举顺序写键
        if keys:
            notes.append({"dataset": ds.name, "kind": "features 键顺序",
                          "detail": " -> ".join(keys) + ("  [wrist 在前]" if keys[0] == CAM_WRIST else "  [env 在前]")})
    return jobs, notes


def montage(items: list[dict], cols: int, cell: tuple[int, int], path: Path, title_h: int = 16,
            all_frames: bool = True) -> None:
    """items: 每项含 frame(BGR) 与 label。"""
    if not items:
        return
    cw, ch = cell
    nrows = (len(items) + cols - 1) // cols
    canvas = np.full((ch * nrows, cw * cols, 3), 28, np.uint8)
    for k, it in enumerate(items):
        fr = it["frame"]
        if fr is None:
            continue
        t = cv2.resize(fr, cell, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (cw, title_h), (0, 0, 0), -1)
        cv2.putText(t, it["label"][:52], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                    it.get("color", (255, 255, 255)), 1, cv2.LINE_AA)
        r, c = divmod(k, cols)
        canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = t
    cv2.imwrite(str(path), canvas)


def thumb(path: str) -> np.ndarray | None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ok, fr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    cap.release()
    return fr if ok else None


def pairs_montage(root: Path, out: Path, fr=(0.30, 0.70), cell=(320, 240)) -> Path | None:
    """每个数据集一行：[env] 两个时间点 | [wrist] 两个时间点，供人眼区分两路相机。"""
    rows = []
    for ds in sorted(p for p in root.iterdir() if (p / "videos").is_dir()):
        cells = []
        for cam in (CAM_ENV, CAM_WRIST):
            files = sorted((ds / "videos" / cam).rglob("*.mp4"))
            for f in fr:
                cells.append((cam.replace("observation.images.", ""), f,
                              frame_at(files[0], f) if files else None))
        rows.append((ds.name, cells))
    rows = [r for r in rows if any(c[2] is not None for c in r[1])]
    if not rows:
        return None
    cw, ch = cell
    canvas = np.full((ch * len(rows), cw * 4, 3), 28, np.uint8)
    for r, (ds, cells) in enumerate(rows):
        for c, (cam, frac, frame) in enumerate(cells):
            if frame is None:
                continue
            t = cv2.resize(frame, cell, interpolation=cv2.INTER_AREA)
            col = (120, 255, 120) if cam == "env" else (120, 200, 255)
            cv2.rectangle(t, (0, 0), (cw, 18), (0, 0, 0), -1)
            cv2.putText(t, "%s | %s t=%.0f%%" % (ds[:16], cam, frac * 100), (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = t
    for c in (0, 2):
        cv2.line(canvas, (c * cw, 0), (c * cw, canvas.shape[0]), (255, 255, 255), 2)
    path = out / "montage_env_vs_wrist_pairs.png"
    cv2.imwrite(str(path), canvas)
    return path


def frame_at(path: Path, frac: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(n * frac)))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "lerobot_datasets"))
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="inspect/cam_scan")
    ap.add_argument("--no-pairs", action="store_true", help="不生成 env/wrist 对照拼图")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    out = Path(args.out)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        if old.is_file():
            old.unlink()

    paths, notes = discover(root)
    jobs = [(p, ds, args.frames) for p, ds in paths]
    print("待扫描 mp4 = %d 个，每个采样 %d 帧，root=%s" % (len(jobs), args.frames, root), flush=True)

    recs: list[dict] = []
    with mp.Pool(max(1, args.workers)) as pool:
        for k, rec in enumerate(pool.imap_unordered(scan_file, jobs, chunksize=1), 1):
            verdict, why = classify(rec)
            rec["verdict"] = verdict
            rec["why"] = why
            recs.append(rec)
            if k % 10 == 0 or k == len(jobs):
                print("  ...%d/%d" % (k, len(jobs)), flush=True)
    recs.sort(key=lambda r: (r["dataset"], r["stream"], r["file"]))

    # 每个 流文件夹 里所有文件是否一致
    per_stream: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        per_stream.setdefault((r["dataset"], r["stream"]), []).append(r)

    rows = []
    for (ds, stream), rs in sorted(per_stream.items()):
        votes = [r["verdict"] for r in rs]
        unanimous = len(set(votes)) == 1
        majority = max(set(votes), key=votes.count)
        actual = {"fixed": "env(固定)", "moving": "wrist(随臂)", "ambiguous": "?", "error": "err"}[majority]
        labelled = "env" if stream == CAM_ENV else "wrist"
        swapped = (unanimous and majority in ("fixed", "moving")
                   and ((labelled == "env") != (majority == "fixed")))
        rows.append({"dataset": ds, "stream": stream, "labelled": labelled, "files": len(rs),
                     "unanimous": unanimous, "majority": majority, "actual": actual,
                     "swapped": swapped,
                     "static_med": float(np.median([r["static_pct"] for r in rs if np.isfinite(r["static_pct"])])),
                     "pad_cv_med": float(np.nanmedian([r["pad_cv"] for r in rs]))})
    rows.sort(key=lambda r: (r["dataset"], r["stream"]))

    with (out / "cam_scan.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)

    sw_datasets = sorted({r["dataset"] for r in rows if r["swapped"]})
    consistent = sorted({r["dataset"] for r in rows} - set(sw_datasets))
    mixed = [r for r in rows if not r["unanimous"]]
    ambig = [r for r in recs if r["verdict"] in ("ambiguous", "error")]

    with (out / "summary.txt").open("w", encoding="utf-8") as fh:
        fh.write("# 真机数据集相机流排查  (%s)\n" % "2026-10-26")
        fh.write("# 判据 A: 时序静态像素占比 static%%  (固定相机 25~50, 腕部相机 1~3)\n")
        fh.write("# 判据 B: 绿区面积变异 pad_cv=std/mean  (固定 <=0.60, 腕部 >0.60)\n")
        fh.write("# %-24s %-28s %-6s %-5s %-9s %-6s %-10s %-8s\n"
                 % ("dataset", "stream", "标称", "文件", "一致", "static%", "pad_cv", "结论"))
        for r in rows:
            tail = ("需互换" if r["swapped"] else "-- 正确")
            if not r["unanimous"]:
                tail = "同数据集内不一致(需逐文件看)"
            fh.write("  %-24s %-28s %-6s %-5d %-9s %7.1f %9.2f  %s\n"
                     % (r["dataset"], r["stream"], r["labelled"], r["files"],
                        "是" if r["unanimous"] else "否", r["static_med"], r["pad_cv_med"], tail))
        fh.write("\n## 标签互换的数据集（两路内容与文件夹名相反）\n")
        for ds in sw_datasets:
            fh.write("  - %s\n" % ds)
        if not sw_datasets:
            fh.write("  (无)\n")
        fh.write("\n## 标签正确的数据集\n")
        for ds in consistent:
            if ds in sw_datasets:
                continue
            fh.write("  - %s\n" % ds)
        if not consistent:
            fh.write("  (无)\n")
        fh.write("\n## 同数据集内文件不一致（可能有中途重插/换相机）\n")
        for r in mixed:
            fh.write("  - %s / %s：%d 个文件判定不统一\n" % (r["dataset"], r["stream"], r["files"]))
        if not mixed:
            fh.write("  (无)\n")
        fh.write("\n## 判定失败/存疑的文件\n")
        for r in ambig:
            fh.write("  - %s/%s/%s -> %s (%s)\n" % (r["dataset"], r["stream"], r["file"], r["verdict"], r["why"]))
        if not ambig:
            fh.write("  (无)\n")
        fh.write("\n## 目录/元数据异常与线索\n")
        for n in notes:
            fh.write("  - %-24s %-14s %s\n" % (n["dataset"], n["kind"], n["detail"]))

    # 拼图：全部文件一帧
    items = []
    for r in recs:
        fr = thumb(r["path"])
        label = "%s/%s/%s s=%.0f%% cv=%.2f" % (r["dataset"].replace("_fixed", "·f").replace("_corners", "·c"),
                                               r["stream"].replace("observation.images.", ""),
                                               r["file"], r["static_pct"], r["pad_cv"])
        color = {"fixed": (120, 255, 120), "moving": (120, 200, 255),
                 "ambiguous": (0, 220, 255), "error": (80, 80, 255)}[r["verdict"]]
        items.append({"frame": fr, "label": label, "color": color})
    cols = 8
    montage(items, cols, (200, 150), out / "montage_all_files.png")
    flagged = [it for it, r in zip(items, recs)
               if r["verdict"] in ("ambiguous", "error") or r["dataset"] in sw_datasets]
    montage(flagged, 4, (320, 240), out / "montage_flagged.png")

    if not args.no_pairs:
        pm = pairs_montage(root, out)
        if pm:
            print("对照拼图:", pm)

    print("\n扫描目录:", out)
    for p in sorted(out.glob("*")):
        print("  %-30s %9d B" % (p.name, p.stat().st_size))
    print("\n汇总:")
    for r in rows:
        print("  %-24s %-28s %-6s %7.1f %8.2f  %s"
              % (r["dataset"], r["stream"].replace("observation.images.", ""), r["labelled"],
                 r["static_med"], r["pad_cv_med"],
                 "需互换" if r["swapped"] else ("不一致" if not r["unanimous"] else "正确")))
    return 0


if __name__ == "__main__":
    sys.exit(main())