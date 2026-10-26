#!/usr/bin/env python3
"""修正真机数据集里 env / wrist 两路相机标签互换的问题。

已确认的事实（2026-10-26，tools/scan_real_cams.py 逐文件测 + 用户目视确认）：
    *_fixed 三个数据集（real_train_fixed / yyl100_dataset_fixed / holdout_fixed）
    的 `observation.images.env` 里存的是随臂移动的腕部画面，
    `observation.images.wrist` 里存的才是固定环境相机画面 —— 与文件夹名相反。
    *_corners 三个数据集标签正确。
    旁证：info.json 里 features 的键顺序，互换的三个都是 wrist 在前，正确的都是 env 在前（6/6 符合）。

"改对标签"不只是改文件夹名，以下四处都要一起动，否则统计量和视频对不上：
    1) videos/observation.images.env        <-> videos/observation.images.wrist   目录互换
    2) meta/info.json  features 键顺序      <-> 互换这两项的位置
    3) meta/stats.json 两键的整块数值        <-> 互换（像素均值/方差属于画面内容，跟着内容走）
    4) meta/episodes/**/*.parquet 里
       stats/observation.images.{env,wrist}/* 列  <-> 互换
       videos/observation.images.{env,wrist}/*    -> 两路若完全相同则不动，不同则一起互换

用法：
    # 只看计划（默认）
    env -u PYTHONHOME -u PYTHONPATH .cache/venv/bin/python tools/fix_cam_labels.py
    # 真正执行（需要写 ~/lerobot_datasets）
    ... tools/fix_cam_labels.py --apply
    # 回滚
    ... tools/fix_cam_labels.py --revert

安全性：--apply 前会把 meta/info.json、meta/stats.json、meta/episodes/**/*.parquet 备份到
        数据集内的 .cam_fix_backup/<时间戳>/；并在数据集根写 .cam_label_fix.json 清单供 --revert 使用。
        目录互换用同盘 rename，O(1)，不会复制 GB 级视频。
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

CAM_ENV = "observation.images.env"
CAM_WRIST = "observation.images.wrist"
STATIC_FIXED = 12.0   # 固定相机 static% 下限（实测 33.9~50.5）
STATIC_MOVING = 6.0   # 腕部相机 static% 上限（实测 1.3~2.3）
TMP_SWAP = ".cam_swap_tmp"
MANIFEST = ".cam_label_fix.json"
BACKUP = ".cam_fix_backup"


# ---------------------------------------------------------------- 判定
def load_scan(csv_path: Path) -> dict[tuple[str, str], dict]:
    """-> {(dataset, stream): {unanimous, majority, files, static_med}}"""
    if not csv_path.exists():
        raise SystemExit("缺少扫描结果 %s，先跑 tools/scan_real_cams.py" % csv_path)
    per: dict[tuple[str, str], list[dict]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            per.setdefault((r["dataset"], r["stream"]), []).append(r)
    out = {}
    for key, rs in per.items():
        votes = [r["verdict"] for r in rs]
        s = [float(r["static_pct"]) for r in rs if r["static_pct"] not in ("nan", "")]
        out[key] = {"unanimous": len(set(votes)) == 1, "majority": max(set(votes), key=votes.count),
                    "files": len(rs), "static_med": sorted(s)[len(s) // 2] if s else float("nan")}
    return out


def plan(scan: dict, root: Path, only: list[str] | None) -> list[dict]:
    """挑出需要互换的数据集，并逐项核对前提。"""
    datasets = sorted({ds for ds, _ in scan})
    todo = []
    for ds in datasets:
        if only and ds not in only:
            continue
        e = scan.get((ds, CAM_ENV))
        w = scan.get((ds, CAM_WRIST))
        if not e or not w:
            todo.append({"dataset": ds, "verdict": "skip", "why": "缺少一路扫描结果（可能没有 mp4）"})
            continue
        if not (e["unanimous"] and w["unanimous"]):
            todo.append({"dataset": ds, "verdict": "skip",
                         "why": "同数据集内文件判定不统一，需逐文件处理"})
            continue
        env_actual = "env" if e["majority"] == "fixed" else ("wrist" if e["majority"] == "moving" else "?")
        wrt_actual = "env" if w["majority"] == "fixed" else ("wrist" if w["majority"] == "moving" else "?")
        if env_actual == env_actual and (env_actual == "env" and wrt_actual == "wrist"):
            todo.append({"dataset": ds, "verdict": "ok", "why": "标签已经正确",
                         "env_static": e["static_med"], "wrist_static": w["static_med"]})
            continue
        if env_actual == "wrist" and wrt_actual == "env":
            todo.append({"dataset": ds, "verdict": "swap", "why": "两路内容与文件夹名相反",
                         "env_static": e["static_med"], "wrist_static": w["static_med"]})
            continue
        todo.append({"dataset": ds, "verdict": "skip",
                     "why": "两路判定不是互补关系（env->%s, wrist->%s），不敢自动改" % (env_actual, wrt_actual)})
    return todo


# ---------------------------------------------------------------- 检查 parquet
def video_refs_equal(parquet_paths: list[Path]) -> bool:
    """两路 videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp} 是否完全相同。"""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return True  # 无 pyarrow 时不做这项判断，下面会提示
    for p in parquet_paths:
        t = pq.read_table(p)
        names = set(t.column_names)
        for suffix in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            a = "videos/%s/%s" % (CAM_ENV, suffix)
            b = "videos/%s/%s" % (CAM_WRIST, suffix)
            if a in names and b in names:
                if t.column(a).to_pylist() != t.column(b).to_pylist():
                    return False
    return True


def swap_parquet_stats(p: Path, swap_video_refs: bool) -> int:
    """互换 parquet 里两路相机的 stats 列（必要时连 videos 引用列一起换）。返回改动的列数。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(p)
    names = list(t.column_names)
    snapshot = {n: t.column(n).combine_chunks() for n in names}
    changed = 0

    def partner(n: str) -> str | None:
        for a, b in ((CAM_ENV, CAM_WRIST), (CAM_WRIST, CAM_ENV)):
            if n.startswith("stats/%s/" % a):
                return n.replace("stats/%s/" % a, "stats/%s/" % b, 1)
            if swap_video_refs and n.startswith("videos/%s/" % a):
                return n.replace("videos/%s/" % a, "videos/%s/" % b, 1)
        return None

    cols = []
    for n in names:
        q = partner(n)
        if q is not None and q in snapshot:
            cols.append(snapshot[q])
            changed += 1
        else:
            cols.append(snapshot[n])
    fields = [pa.field(n, cols[i].type, metadata=t.schema.field(i).metadata)
              for i, n in enumerate(names)]
    new = pa.Table.from_arrays(cols, schema=pa.schema(fields, metadata=t.schema.metadata))
    pq.write_table(new, p, compression="snappy")
    return changed


# ---------------------------------------------------------------- 动作
def swap_dirs(ds_dir: Path) -> None:
    v = ds_dir / "videos"
    a, b, tmp = v / CAM_ENV, v / CAM_WRIST, v / TMP_SWAP
    for p in (a, b):
        if not p.is_dir():
            raise SystemExit("缺少目录 %s" % p)
    if tmp.exists():
        raise SystemExit("残留临时目录 %s，先人工确认" % tmp)
    a.rename(tmp)
    b.rename(a)
    tmp.rename(b)


def swap_json_keys(path: Path, mode: str) -> str:
    """mode=order 只互换 features 里两项的先后；mode=values 互换两键的整块数值。"""
    j = json.loads(path.read_text(encoding="utf-8"))
    if mode == "order":
        feats = j.get("features") or {}
        if CAM_ENV in feats and CAM_WRIST in feats:
            items = list(feats.items())
            i = [k for k, _ in items].index(CAM_ENV)
            k = [k for k, _ in items].index(CAM_WRIST)
            items[i], items[k] = items[k], items[i]
            j["features"] = dict(items)
            info = "features 键顺序已互换"
        else:
            info = "features 里没有这两路，跳过"
    else:
        if CAM_ENV in j and CAM_WRIST in j:
            j[CAM_ENV], j[CAM_WRIST] = j[CAM_WRIST], j[CAM_ENV]
            info = "stats 两键数值已互换"
        else:
            info = "stats 里没有这两路，跳过"
    path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return info


def measure_dir(cam_dir: Path, frames: int = 12) -> float:
    """抽一个文件测 static%（用于改完自检）。"""
    import cv2
    import numpy as np
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


def do_fix(ds_dir: Path, apply: bool) -> list[str]:
    log = []
    eps = sorted((ds_dir / "meta" / "episodes").rglob("*.parquet"))
    refs_equal = video_refs_equal(eps)
    log.append("episodes parquet: %d 个；两路 video 引用%s" % (len(eps), "相同（不必换）" if refs_equal else "不同（要一起换）"))
    if not apply:
        log.append("将执行: 目录互换 + info.json 键顺序互换 + stats.json 数值互换 + %d 个 parquet 的 stats 列互换" % len(eps))
        return log

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bdir = ds_dir / BACKUP / stamp
    bdir.mkdir(parents=True, exist_ok=True)
    for rel in ("meta/info.json", "meta/stats.json"):
        src = ds_dir / rel
        if src.exists():
            (bdir / rel.replace("/", "__")).write_bytes(src.read_bytes())
    for p in eps:
        shutil.copy2(p, bdir / ("episodes__" + p.name.replace("file-", "f")))

    swap_dirs(ds_dir)
    log.append("目录已互换: %s <-> %s" % (CAM_ENV, CAM_WRIST))
    log.append(swap_json_keys(ds_dir / "meta" / "info.json", "order"))
    log.append(swap_json_keys(ds_dir / "meta" / "stats.json", "values"))
    n_cols = 0
    for p in eps:
        n_cols += swap_parquet_stats(p, swap_video_refs=not refs_equal)
    log.append("parquet 列互换完成，共改动 %d 列" % n_cols)

    (ds_dir / MANIFEST).write_text(json.dumps(
        {"when": stamp, "backup": str(bdir.relative_to(ds_dir)), "episodes": len(eps),
         "video_refs_equal": refs_equal, "columns_swapped": n_cols}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    s_env = measure_dir(ds_dir / "videos" / CAM_ENV)
    s_wrt = measure_dir(ds_dir / "videos" / CAM_WRIST)
    log.append("自检: 新 env static%%=%.1f (固定相机应 >=%.0f)，新 wrist static%%=%.1f (腕部应 <=%.0f)"
               % (s_env, STATIC_FIXED, s_wrt, STATIC_MOVING))
    ok = (s_env >= STATIC_FIXED) and (s_wrt <= STATIC_MOVING)
    log.append("自检结论: %s" % ("OK" if ok else "FAIL 请立刻 --revert"))
    return log


def do_revert(ds_dir: Path) -> list[str]:
    man_p = ds_dir / MANIFEST
    if not man_p.exists():
        return ["没有 %s，无法回滚" % MANIFEST]
    man = json.loads(man_p.read_text(encoding="utf-8"))
    swap_dirs(ds_dir)
    bdir = ds_dir / man["backup"]
    for rel, name in (("meta/info.json", "meta__info.json"), ("meta/stats.json", "meta__stats.json")):
        src = bdir / name
        if src.exists():
            (ds_dir / rel).write_bytes(src.read_bytes())
    for p in sorted((ds_dir / "meta" / "episodes").rglob("*.parquet")):
        cand = bdir / ("episodes__" + p.name.replace("file-", "f"))
        if cand.exists():
            shutil.copy2(cand, p)
    man_p.unlink()
    return ["已回滚目录、info.json、stats.json、episodes parquet（备份留在 %s）" % man["backup"]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / "lerobot_datasets"))
    ap.add_argument("--scan-csv", default="inspect/cam_scan/cam_scan.csv")
    ap.add_argument("--datasets", nargs="*", default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--apply", action="store_true")
    g.add_argument("--revert", action="store_true")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    csv_path = Path(args.scan_csv)
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    scan = load_scan(csv_path)
    todo = plan(scan, root, args.datasets)

    print("%-24s %-8s %-8s %-8s %s" % ("dataset", "env静态%", "wrist静态%", "判定", "说明"))
    for t in todo:
        print("%-24s %8s %9s %-8s %s"
              % (t["dataset"], ("%.1f" % t["env_static"]) if "env_static" in t else "-",
                 ("%.1f" % t["wrist_static"]) if "wrist_static" in t else "-", t["verdict"], t["why"]))

    targets = [t for t in todo if t["verdict"] == "swap"]
    lance = [d for d in sorted(p.name for p in root.iterdir() if p.is_dir()) if (root / d / "videos.lance").exists()]
    if lance:
        print("\n注意：以下数据集视频是 Lance 格式（无 mp4），本工具不动，需重跑其转换脚本：")
        for d in lance:
            try:
                feats = list(json.loads((root / d / "meta" / "info.json").read_text())["features"])
                first = next((k for k in feats if k.startswith("observation.images.")), "?")
                print("  - %-24s features 首相机=%s -> %s" % (d, first, "疑似同样互换" if first == CAM_WRIST else "标签正确"))
            except Exception as e:
                print("  - %-24s 读取失败: %s" % (d, e))

    if not targets:
        print("\n没有需要互换的数据集。")
        return 0

    mode = "APPLY" if args.apply else ("REVERT" if args.revert else "DRY-RUN")
    print("\n模式=%s，目标 %d 个数据集" % (mode, len(targets)))
    bad = 0
    for t in targets:
        ds_dir = root / t["dataset"]
        print("\n== %s" % t["dataset"])
        if args.revert:
            for line in do_revert(ds_dir):
                print("   " + line)
            continue
        for line in do_fix(ds_dir, args.apply):
            print("   " + line)
            if "FAIL" in line:
                bad += 1
    if not args.apply and not args.revert:
        print("\n这只是计划。加 --apply 才会真正执行（会写 ~/lerobot_datasets）。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())