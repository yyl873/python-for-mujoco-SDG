#!/usr/bin/env python3
"""
inspect_episode.py —— 数据排查工具（按需出肉眼能看的东西）
==========================================================

为什么要有它
------------
采集(collect.py)只产【训练要的那一份】：npz + json + 两条 mp4。
不再顺手出 avi、不再转格式、不再每条都拼图 —— 那些只在"我要看一眼"的时候才有价值。
本工具就是那个"我要看一眼"的入口：指定几条、或按条件筛几条，才生成排查产物。

    # 看指定几条（拼图 + 预览 + 视频体检）
    python3 tools/inspect_episode.py --episodes 0,3,7

    # 先看负样本（success=false 的），最多 5 条
    python3 tools/inspect_episode.py --failed --limit 5

    # 每 10 条抽 1 条抽查
    python3 tools/inspect_episode.py --every 10

    # 只做体检（帧数/尺寸/fps 是否与 npz 对得上），不出图 —— 100 条全扫也很快
    python3 tools/inspect_episode.py --all --check-only

    # 这条没视频（采集时用了 --no-render），现在补渲一份再看
    python3 tools/inspect_episode.py --episodes 4 --render

    # demo 目录（render/test_render.py 的产物）
    python3 tools/inspect_episode.py --demo

产物（默认落在 inspect/，不进训练集，可随时删）
    inspect/ep_000000_env_strip.png    外部相机 8 帧拼图（带 t / phase 标签）
    inspect/ep_000000_wrist_strip.png  腕部相机 8 帧拼图
    inspect/ep_000000_preview.png      夹稳时刻 env+wrist 各一帧（大图）

★本工具不碰物理：出视频走 collect.render_episode（纯运动学回放，只 mj_forward，不 mj_step），
  其余全部是解码已有的 mp4。所以排查产物和录制结果一定同源。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
FRAME_TAGS = ("Env", "Wrist")


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _label(img, text, color=(0, 255, 255)):
    """左上角打标签（cv2 不支持中文，所以一律英文/数字）。"""
    out = img.copy()
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def _decode(path, want=None):
    """
    解码视频为 (N,H,W,3) RGB。

    want=None 表示全解码；want 是下标列表时只保留那几帧（拼图用，省内存）。
    返回 (frames, total_frames_reported_by_container)。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if want is None:
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr[:, :, ::-1].copy())
        cap.release()
        return (np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)), total
    want = sorted(set(int(i) for i in want))
    keep = {}
    i = 0
    while want:
        ok, fr = cap.read()
        if not ok:
            break
        if i == want[0]:
            keep[i] = fr[:, :, ::-1].copy()
            want.pop(0)
        i += 1
    cap.release()
    got = [keep[i] for i in sorted(keep)]
    return (np.stack(got) if got else np.zeros((0, 1, 1, 3), np.uint8)), total


def _hstrip(tiles, gap=4):
    """横向拼一行（等高等宽）。"""
    if not tiles:
        return None
    h = max(t.shape[0] for t in tiles)
    out = []
    for t in tiles:
        if t.shape[0] != h:
            scale = h / t.shape[0]
            t = cv2.resize(t, (int(round(t.shape[1] * scale)), h))
        out.append(t)
        out.append(np.full((h, gap, 3), 24, np.uint8))
    return np.hstack(out[:-1])


def _vstack(rows, gap=6):
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] != w:
            pad = np.full((r.shape[0], w - r.shape[1], 3), 24, np.uint8)
            r = np.hstack([r, pad])
        out.append(r)
        out.append(np.full((gap, w, 3), 24, np.uint8))
    return np.vstack(out[:-1])


def _write_img(path, rgb, quality=90):
    """写图：.jpg 用 quality 压（拼图几千像素宽，PNG 会到 MB 级；jpeg 只要百 KB）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if path.suffix.lower() in (".jpg", ".jpeg") else []
    cv2.imwrite(str(path), rgb[:, :, ::-1], params)      # RGB -> BGR
    return path


def _write_png(path, rgb, **kw):
    return _write_img(path, rgb, **kw)


# ---------------------------------------------------------------------------
# 数据集读取（只读 index + npz + mp4，不加载 MuJoCo）
# ---------------------------------------------------------------------------

def load_entries(src: Path):
    idx = src / "index.json"
    if not idx.exists():
        raise SystemExit(f"找不到清单: {idx}（先跑 collect/collect.py）")
    d = json.loads(idx.read_text(encoding="utf-8"))
    rows = []
    for key, ok in (("accepted", True), ("rejected", False)):
        for e in d.get(key) or []:
            rows.append(dict(e, success=bool(e.get("success", ok))))
    rows.sort(key=lambda r: int(r["episode_id"]))
    return rows, d


def entry_videos(e):
    p = e.get("paths") or {}
    return (p.get("video_env") or p.get("env_avi"),
            p.get("video_wrist") or p.get("wrist_avi"))


def check_one(e, src: Path, videos: Path):
    """体检一条：视频帧数/尺寸/fps 与 npz 是否对得上。只读容器头，不解码。"""
    env_rel, wrist_rel = entry_videos(e)
    out = dict(episode_id=int(e["episode_id"]), success=bool(e.get("success")),
               problems=[])
    npz_rel = (e.get("paths") or {}).get("npz")
    n_traj = None
    if npz_rel:
        try:
            with np.load(src.parent / npz_rel) as z:
                n_traj = int(z["qpos"].shape[0])
        except Exception as exc:                      # 数据坏了也要报，不要静默
            out["problems"].append(f"npz 读不了: {exc}")
    if not env_rel:
        out["problems"].append("no_video")
        return out
    for cam, rel in (("env", env_rel), ("wrist", wrist_rel)):
        if not rel:
            out["problems"].append(f"missing_path[{cam}]")
            continue
        path = src.parent / rel
        if not path.exists():
            out["problems"].append(f"missing_file[{cam}]")
            continue
        cap = cv2.VideoCapture(str(path))
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        out[f"{cam}_frames"] = cnt
        out[f"{cam}_size"] = f"{w}x{h}"
        out[f"{cam}_fps"] = round(fps, 3)
        out[f"{cam}_mb"] = round(path.stat().st_size / 1024 / 1024, 2)
        if n_traj is not None and cnt != n_traj:
            out["problems"].append(f"frames[{cam}] {cnt} != npz {n_traj}")
        if abs(fps - 30.0) > 0.5:
            out["problems"].append(f"fps[{cam}] {fps:.2f}")
        if cnt == 0:
            out["problems"].append(f"empty[{cam}]")
    return out


# ---------------------------------------------------------------------------
# 缺视频时补渲（复用采集那条纯运动学回放，保证同源）
# ---------------------------------------------------------------------------

def render_missing(eid: int, src: Path, videos: Path, width=640, height=480,
                   burn=False, force=False):
    """用 npz 里已有的轨迹重渲两条 mp4。不跑物理：collect.render_episode 只 mj_forward。"""
    npz = src / f"ep_{eid:06d}.npz"
    js = src / f"ep_{eid:06d}.json"
    if not npz.exists():
        raise SystemExit(f"没有轨迹 npz，无法重渲: {npz}")
    out_env = videos / f"ep_{eid:06d}_env.mp4"
    out_wrist = videos / f"ep_{eid:06d}_wrist.mp4"
    if out_env.exists() and out_wrist.exists() and not force:
        return False
    # 采集端要完整初始化（EGL / sys.path / 复用 test_render）—— 走子进程更干净，
    # 也顺手把 SIGFPE（约 7%）挡在外面：崩了就重试一次。
    cmd = [sys.executable, str(ROOT / "collect" / "collect.py"),
           "--render-only", "--episodes", str(eid), "--width", str(width),
           "--height", str(height)]
    if burn:
        cmd.append("--burn-counter")
    for attempt in (1, 2):
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode == 0:
            return True
        sys.stderr.write(f"  [重渲失败 attempt={attempt}] {r.stderr.strip()[-400:]}\n")
    raise SystemExit(f"ep_{eid:06d} 重渲失败（看上面的 stderr）")


# ---------------------------------------------------------------------------
# 出图
# ---------------------------------------------------------------------------

def make_strips(eid, src: Path, videos: Path, out: Path, strip_n=8,
                preview=True, tile_w=480):
    """给一条 episode 出 strip / preview，并打印一行体检结果。"""
    env_rel, wrist_rel = None, None
    for name in (f"ep_{eid:06d}_env.mp4", f"ep_{eid:06d}_env.avi"):
        p = videos / name
        if p.exists():
            env_rel = p
            break
    for name in (f"ep_{eid:06d}_wrist.mp4", f"ep_{eid:06d}_wrist.avi"):
        p = videos / name
        if p.exists():
            wrist_rel = p
            break
    if env_rel is None:
        print(f"  ep_{eid:06d} 没有视频（需要补渲就加 --render）")
        return None

    # 轨迹（只要 phase / time 做标签，不参与渲染）
    phase = time = None
    npz = src / f"ep_{eid:06d}.npz"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            phase = np.array([str(p) for p in z["phase"]]) if "phase" in z.files else None
            time = np.array(z["time"], float) if "time" in z.files else None

    # 先读容器报告的帧数，按它算抽帧下标（避免为了算下标先全解码）
    cap = cv2.VideoCapture(str(env_rel))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    n = total if total > 0 else 1
    pick = np.unique(np.linspace(0, n - 1, min(strip_n, n)).astype(int)).tolist()

    rows = []
    for tag, path in (("Env", env_rel), ("Wrist", wrist_rel)):
        if path is None:
            continue
        frames, rep = _decode(path, want=pick)
        if len(frames) == 0:
            print(f"  ! {path.name} 解不出帧")
            continue
        tiles = []
        for k, fr in zip(pick[:len(frames)], frames):
            t = float(time[k]) if (time is not None and k < len(time)) else None
            ph = phase[k] if (phase is not None and k < len(phase)) else ""
            txt = f"{tag} #{k}" + (f" t={t:.2f}s" if t is not None else "") + (f" {ph}" if ph else "")
            if tile_w and fr.shape[1] > tile_w:      # 缩格省磁盘（拼图只为肉眼排查，不必原尺寸）
                sc = tile_w / fr.shape[1]
                fr = cv2.resize(fr, (tile_w, int(round(fr.shape[0] * sc))),
                                interpolation=cv2.INTER_AREA)
            tiles.append(_label(fr, txt, (0, 255, 255) if tag == "Env" else (255, 220, 0)))
        rows.append(_hstrip(tiles))
    strip = _vstack(rows)
    p = _write_img(out / f"ep_{eid:06d}_strip.jpg", strip) if strip is not None else None

    if preview:
        # 夹稳时刻：phase == grasp 的最后 1/3 处（拿不到 phase 就取中点）
        k = n // 2
        if phase is not None and (phase == "grasp").any():
            gi = np.where(phase == "grasp")[0]
            k = int(gi[int(len(gi) * 2 / 3)])
        k = max(0, min(k, n - 1))
        pv = []
        for tag, path in (("Env", env_rel), ("Wrist", wrist_rel)):
            if path is None:
                continue
            frames, _ = _decode(path, want=[k])
            if len(frames):
                pv.append(_hstrip([_label(frames[0], f"{tag} #{k} (夹稳)", (0, 255, 0))]))
        pv_img = _vstack(pv)
        if pv_img is not None:
            _write_img(out / f"ep_{eid:06d}_preview.png", pv_img)

    return dict(episode_id=eid, strip=str(p) if p is not None else None, frames=total)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="按需出排查产物（拼图/预览/体检）；采集阶段不再生成这些",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("产物（默认落在")[0])
    ap.add_argument("--src", default=str(ROOT / "scripts"), help="数据集目录（默认 <root>/scripts）")
    ap.add_argument("--videos", default=None, help="视频目录（默认 <root>/videos）")
    ap.add_argument("--out", default=str(ROOT / "inspect"), help="排查产物目录（默认 <root>/inspect）")
    ap.add_argument("--episodes", default=None, help="指定编号，如 0,3,7")
    ap.add_argument("--failed", action="store_true", help="只选 success=false 的")
    ap.add_argument("--every", type=int, default=None, help="每 N 条抽 1 条")
    ap.add_argument("--all", action="store_true", help="全选")
    ap.add_argument("--limit", type=int, default=None, help="最多处理几条")
    ap.add_argument("--strip", type=int, default=6, help="拼图抽几帧（默认 6）")
    ap.add_argument("--tile-w", type=int, default=480, help="拼图单格宽度上限（默认 480，0=原尺寸）")
    ap.add_argument("--no-preview", action="store_true", help="不出 preview 大图")
    ap.add_argument("--check-only", action="store_true", help="只体检，不出图")
    ap.add_argument("--render", action="store_true", help="缺视频时先补渲（纯运动学回放，不跑物理）")
    ap.add_argument("--force", action="store_true", help="配合 --render：已有视频也重渲")
    ap.add_argument("--burn-counter", action="store_true", help="补渲时烧帧号（排查用）")
    ap.add_argument("--demo", action="store_true", help="改看 demo/ 目录（test_render.py 的产物）")
    ap.add_argument("--json", action="store_true", help="stdout 只输出 JSON 回执")
    args = ap.parse_args(argv)

    src = Path(args.src)
    videos = Path(args.videos) if args.videos else src.parent / "videos"
    out = Path(args.out)

    if args.demo:
        demo = ROOT / "demo"
        env_p, wrist_p = demo / "env_camera.mp4", demo / "wrist_camera.mp4"
        if not env_p.exists():
            raise SystemExit(f"demo 里没有 env_camera.mp4（先跑 render/test_render.py）")
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for tag, path in (("Env", env_p), ("Wrist", wrist_p)):
            if not path.exists():
                continue
            frames, total = _decode(path, want=np.unique(
                np.linspace(0, max(int(cv2.VideoCapture(str(path)).get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0),
                            min(args.strip, 8)).astype(int)).tolist())
            rows.append(_hstrip([_label(f, f"{tag} #{i}")
                                 for i, f in enumerate(frames)]))
        img = _vstack(rows)
        p = _write_img(out / "demo_strip.jpg", img) if img is not None else None
        print(json.dumps(dict(demo=str(p)), ensure_ascii=False) if args.json
              else f"  demo 拼图: {p}")
        return 0

    rows, index = load_entries(src)
    if args.episodes:
        want = [int(x) for x in str(args.episodes).replace(" ", "").split(",") if x != ""]
    else:
        want = [int(r["episode_id"]) for r in rows]
    by_id = {int(r["episode_id"]): r for r in rows}

    sel = []
    for eid in want:
        e = by_id.get(eid)
        if e is None:
            print(f"  ! 清单里没有 ep_{eid:06d}，跳过")
            continue
        if args.failed and not e.get("success"):
            continue
        if args.failed and e.get("success"):
            continue
        sel.append(e)
    if args.every:
        keep = [e for i, e in enumerate(sel) if i % max(1, args.every) == 0]
        sel = keep
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        print("  没有选中任何 episode（检查 --episodes/--failed/--every/--limit）")
        return 2

    receipts = []
    for e in sel:
        eid = int(e["episode_id"])
        env_rel, _ = entry_videos(e)
        if args.render and (args.force or not env_rel
                            or not (src.parent / env_rel).exists()):
            if args.json:
                pass
            elif not args.check_only:
                print(f"  · ep_{eid:06d} 补渲中（纯运动学回放）…")
            render_missing(eid, src, videos, burn=args.burn_counter, force=args.force)
            rows2, _ = load_entries(src)
            by_id = {int(r["episode_id"]): r for r in rows2}
            e = by_id.get(eid, e)

        chk = check_one(e, src, videos)
        if not args.check_only:
            art = make_strips(eid, src, videos, out, strip_n=args.strip,
                              preview=not args.no_preview, tile_w=args.tile_w)
            if art:
                chk["strip"] = art["strip"]
        receipts.append(chk)
        if not args.json:
            flag = "✅" if not chk["problems"] else "⚠️"
            env_mb = chk.get("env_mb", "-")
            print(f"  {flag} ep_{eid:06d}  success={str(chk['success']).lower():5s} "
                  f"帧 {chk.get('env_frames','?')}  {chk.get('env_size','?')}  "
                  f"{chk.get('env_fps','?')}fps  env {env_mb} MB"
                  + (f"  ← {','.join(chk['problems'])}" if chk["problems"] else ""))

    n_bad = sum(1 for r in receipts if r["problems"])
    if args.json:
        print(json.dumps(dict(src=str(src), n=len(receipts), n_bad=n_bad,
                              items=receipts), ensure_ascii=False, indent=1))
    else:
        print()
        print(f"  体检: {len(receipts)} 条，问题 {n_bad} 条"
              + ("" if args.check_only else f"；排查产物 在 {out}"))
        if args.check_only:
            print("  （--check-only 只读容器头，不解码；要图去掉这个参数）")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
