#!/usr/bin/env python3
"""导出真机"环境相机"画面 20 张 + 仿真参考 3 张 + 拼图 + 索引 + 垫子统计。

相机来源不再写死文件夹名：真机数据集的 env/wrist 两路曾经标签互换过（已于 2026-10-26 修正，
见 tools/fix_cam_labels.py 与 docs/REAL_DATASET_CAMERA_AUDIT.md），所以这里按画面内容自动判定：
    抽样 N 帧 -> 逐像素时序 std < 8 的像素占比 static%
    固定环境相机 33%~56%；装在夹爪上的腕部相机 1%~2%
哪一路 static% 高就取哪一路，并在 index.txt 里记录实测值，避免以后再翻车。

只读 ~/lerobot_datasets 与 videos/，产物写进 inspect/env_cam_20/（.gitignore 已忽略）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/home/yyl/deepseek-harness")
OUT = ROOT / "inspect" / "env_cam_20"
DATASETS = Path("/home/yyl/lerobot_datasets")

CAM_ENV = "observation.images.env"
CAM_WRIST = "observation.images.wrist"
STATIC_FIXED = 12.0     # static% 高于此值判为固定相机

PICKS = [("real_train_fixed", 2), ("yyl100_dataset_fixed", 2), ("holdout_fixed", 1)]
FRACTIONS = (0.10, 0.35, 0.60, 0.85)
SIM_REFS = [("videos/ep_000000_env.mp4", 1.0, "ep000000"),
            ("videos/ep_000007_env.mp4", 1.0, "ep000007"),
            ("videos/ep_000013_env.mp4", 1.0, "ep000013")]
CELL = (320, 240)


def vids(dataset: str, cam: str) -> list[Path]:
    return sorted((DATASETS / dataset / "videos" / cam).rglob("*.mp4"))


def measure_static(cam_dir: Path, frames: int = 12) -> float:
    """抽该目录第一个 mp4 的 N 帧，算逐像素时序 std<8 的像素占比(%)。"""
    files = sorted(cam_dir.rglob("*.mp4"))
    if not files:
        return float("nan")
    cap = cv2.VideoCapture(str(files[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    acc = []
    for i in (np.linspace(0.02, 0.98, frames) * max(1, total - 2)).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            acc.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(acc) < 4:
        return float("nan")
    return 100.0 * float(np.mean(np.stack(acc).std(axis=0) < 8.0))


def pick_fixed_camera(dataset: str) -> tuple[str, float, float]:
    """返回 (环境相机所在文件夹名, 该目录 static%, 另一路 static%)。"""
    s_env = measure_static(DATASETS / dataset / "videos" / CAM_ENV)
    s_wrt = measure_static(DATASETS / dataset / "videos" / CAM_WRIST)
    if not (s_env == s_env) and not (s_wrt == s_wrt):
        raise SystemExit("%s 两路都测不出 static%%" % dataset)
    cam = CAM_ENV if (s_env == s_env and s_env >= s_wrt) else CAM_WRIST
    return cam, s_env, s_wrt


def grab(cap: cv2.VideoCapture, frame_index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if ok:
        return frame
    start = max(0, frame_index - 30)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(start, frame_index + 1):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def scene_stats(frame: np.ndarray) -> dict:
    """绿区(拾取垫)位置/大小 + 是否被画面边框裁掉 + 机械臂暗像素占比 + 亮度。"""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (35, 40, 30), (95, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n_lab, _labels, cstats, cents = cv2.connectedComponentsWithStats(green, connectivity=8)
    out = {"pad_pct": 0.0, "pad_px": 0, "pad_bbox": None, "pad_cx": None,
           "pad_cy": None, "pad_wh": None, "clipped": None}
    if n_lab > 1:
        k = 1 + int(np.argmax(cstats[1:, cv2.CC_STAT_AREA]))
        x0, y0, bw, bh, area = cstats[k]
        x1, y1 = x0 + bw - 1, y0 + bh - 1
        clipped = bool(x0 <= 1 or y0 <= 1 or x1 >= w - 2 or y1 >= h - 2)
        out = {"pad_px": int(area), "pad_pct": round(100.0 * int(area) / (h * w), 2),
               "pad_bbox": [int(x0), int(y0), int(x1), int(y1)],
               "pad_cx": round(float(cents[k][0]) / w, 3),
               "pad_cy": round(float(cents[k][1]) / h, 3),
               "pad_wh": [int(bw), int(bh)], "clipped": clipped}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out["dark_pct"] = round(100.0 * float(np.count_nonzero(gray < 45)) / (h * w), 2)
    out["mean_v"] = round(float(gray.mean()), 1)
    return out


def labelled(img: np.ndarray, text: str) -> np.ndarray:
    thumb = cv2.resize(img, CELL, interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (CELL[0], 18), (0, 0, 0), -1)
    cv2.putText(thumb, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return thumb


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*"):
        if old.is_file():
            old.unlink()

    rows, stats, cam_report = [], {}, {}
    for dataset, want in PICKS:
        cam, s_env, s_wrt = pick_fixed_camera(dataset)
        cam_report[dataset] = (cam, s_env, s_wrt)
        print("  %-22s 环境相机=%s  (static%% env=%.1f wrist=%.1f)" % (dataset, cam, s_env, s_wrt))
        files = vids(dataset, cam)
        if not files:
            print("!! %s 无 %s" % (dataset, cam))
            continue
        chosen = 0
        for vid in files:
            if chosen >= want:
                break
            cap = cv2.VideoCapture(str(vid))
            if not cap.isOpened():
                print("!! 打不开 %s" % vid)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 10:
                cap.release()
                continue
            got = 0
            for frac in FRACTIONS:
                idx = min(total - 2, max(0, int(total * frac)))
                frame = grab(cap, idx)
                if frame is None:
                    continue
                sec = idx / fps
                name = "env__%s__%s__t%05.1fs.png" % (dataset, vid.stem, sec)
                cv2.imwrite(str(OUT / name), frame)
                stats[name] = scene_stats(frame)
                rows.append((dataset, vid.stem, round(sec, 1), idx, name,
                             "%dx%d" % (frame.shape[1], frame.shape[0])))
                got += 1
            cap.release()
            if got:
                chosen += 1
    print("环境/静止相机帧数 = %d" % len(rows))

    sim_rows, sim_imgs = [], []
    for rel, sec, tag in SIM_REFS:
        src = ROOT / rel
        if not src.exists():
            print("!! 缺仿真参考 %s" % src)
            continue
        cap = cv2.VideoCapture(str(src))
        idx = int(round((cap.get(cv2.CAP_PROP_FPS) or 30.0) * sec))
        frame = grab(cap, idx)
        cap.release()
        if frame is None:
            continue
        name = "sim__%s__t%05.1fs.png" % (tag, sec)
        cv2.imwrite(str(OUT / name), frame)
        stats[name] = scene_stats(frame)
        sim_rows.append(name)
        sim_imgs.append((name, frame))

    real_imgs = [cv2.imread(str(OUT / r[4])) for r in rows]
    cells = [labelled(img, "%s %s" % (rows[k][0][:12], ("t=%.1fs" % rows[k][2])))
             for k, img in enumerate(real_imgs) if img is not None]
    cols = 4
    nrows = (len(cells) + cols - 1) // cols
    canvas = np.full((CELL[1] * (nrows + 1), CELL[0] * cols, 3), 30, np.uint8)
    for k, c in enumerate(cells):
        r, cc = divmod(k, cols)
        canvas[r * CELL[1]:(r + 1) * CELL[1], cc * CELL[0]:(cc + 1) * CELL[0]] = c
    for k, (name, img) in enumerate(sim_imgs):
        canvas[nrows * CELL[1]:(nrows + 1) * CELL[1], k * CELL[0]:(k + 1) * CELL[0]] = labelled(img, "SIM " + name)
    cv2.imwrite(str(OUT / "montage_env_vs_sim.png"), canvas)

    with (OUT / "index.txt").open("w", encoding="utf-8") as fh:
        fh.write("# 真机【环境相机】画面（每帧的源目录见下表 chosen_cam）\n")
        fh.write("# 相机来源按画面内容自动判定（时序静态像素占比），不依赖文件夹名。\n")
        for ds, (cam, s_env, s_wrt) in cam_report.items():
            fh.write("#   %-24s chosen=%s  static%%: env=%.1f wrist=%.1f\n"
                     % (ds, cam, s_env, s_wrt))
        fh.write("# %-22s %-10s %-10s %-8s %-8s %s\n" % ("dataset", "file", "秒", "帧号", "分辨率", "路径"))
        for dataset, stem, sec, idx, name, res in rows:
            fh.write("  %-22s %-10s t=%6.1f  frame=%-5d %-8s %s\n"
                     % (dataset, stem, sec, idx, res, str(OUT / name)))
        fh.write("\n# 仿真参考（本仓库 videos/*_env.mp4，真 MuJoCo env_camera）\n")
        for name in sim_rows:
            img = cv2.imread(str(OUT / name))
            fh.write("  %-30s %dx%d %s\n" % (name, img.shape[1], img.shape[0], str(OUT / name)))
        fh.write("\n# 相机标定：各数据集 meta/ 下无 cameras/*.json，取不到真机内参/外参\n")

    with (OUT / "pad_stats.txt").open("w", encoding="utf-8") as fh:
        fh.write("# 绿区(拾取垫)在画面中的位置/大小；cx/cy 为 0~1 归一化中心；clip=垫子被画面边框裁掉\n")
        fh.write("# %-46s %7s %8s %-22s %6s %6s %7s %7s %7s %6s\n"
                 % ("file", "pad%", "pad_px", "bbox(x0,y0,x1,y1)", "w", "h", "cx", "cy", "dark%", "clip"))
        for name in [r[4] for r in rows] + sim_rows:
            s = stats[name]
            bb = "%d,%d,%d,%d" % tuple(s["pad_bbox"]) if s["pad_bbox"] else "-"
            wh = s["pad_wh"] or ["-", "-"]
            fh.write("  %-46s %7.2f %8d %-22s %6s %6s %7s %7s %7.2f %6s\n"
                     % (name, s["pad_pct"], s["pad_px"], bb, wh[0], wh[1],
                        s["pad_cx"] if s["pad_cx"] is not None else "-",
                        s["pad_cy"] if s["pad_cy"] is not None else "-", s["dark_pct"],
                        s["clipped"]))
        real = [stats[r[4]] for r in rows]
        sims = [stats[n] for n in sim_rows]

        def med(vals):
            vals = [v for v in vals if v is not None]
            return float(np.median(vals)) if vals else float("nan")

        ok = [s for s in real if s["pad_bbox"]]
        nclip = sum(1 for s in ok if s["clipped"])
        fh.write("\n# 中位数对比（真机 %d 帧全部参与；其中 %d 帧的绿区被画面边框裁掉 -> 面积/中心偏小偏边）\n"
                 % (len(ok), nclip))
        for key, label in (("pad_pct", "绿区面积占比%"), ("pad_cx", "绿区中心x(0~1)"),
                           ("pad_cy", "绿区中心y(0~1)"), ("mean_v", "整幅平均亮度"),
                           ("dark_pct", "暗像素(机械臂)%")):
            fh.write("  %-16s 真机=%8.2f  仿真=%8.2f  仿真-真机=%+.2f\n"
                     % (label, med([s[key] for s in ok]), med([s[key] for s in sims]),
                        med([s[key] for s in sims]) - med([s[key] for s in ok])))
        for j, label in ((0, "绿区宽度px"), (1, "绿区高度px")):
            rv = med([s["pad_wh"][j] for s in ok if s["pad_wh"]])
            sv = med([s["pad_wh"][j] for s in sims if s["pad_wh"]])
            fh.write("  %-16s 真机=%8.1f  仿真=%8.1f  仿真-真机=%+.1f\n" % (label, rv, sv, sv - rv))

    print("输出目录:", OUT)
    for p in sorted(OUT.glob("*")):
        print("  %-56s %9d B" % (p.name, p.stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main())
