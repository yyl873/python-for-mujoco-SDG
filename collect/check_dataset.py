#!/usr/bin/env python3
"""
check_dataset.py —— 批量体检：把"落盘后一次都没校验过"这个洞补上
=================================================================

为什么需要它
------------
collect.py 只保证"这条跑通了才落盘"，不保证"落下去的形状/标签/外参是对的"。
刷 1000 条，中间混进几条 dtype 错、帧数对不上、外参 NaN 的，训练时才发现，
返工成本是重刷。所以：**刷完就体检，报告落成 JSON，非零退出码给 CI 用。**

检查项（每条 episode）
----------------------
  1) schema       复用 render/test_render.validate_trajectory（16 个键、形状、dtype 逐字节）
                   —— 顺手验证 meta 里必须有 nq/nv/nu/success
  2) 视频帧对齐    AVI 帧数 == npz 帧数，fps == 30，尺寸 == meta.width/height
  3) 外参合法性    全 finite、旋转块正交（max|RRᵀ−I| < 1e-3）、det(R) > 0
  4) 内参自洽      fx == fy > 0，cx == W/2，cy == H/2，fy 与 fovy 能互相推出来
  5) 标签一致性    accepted ⇒ success；success ⇔ evaluate_success()（两套口径必须一致）
  6) 轨迹合理性    time 严格递增且 dt == timestep；qpos/qvel/obj_pos 无 NaN；
                   obj_pos.z 在 (0, 0.3)m；accepted 样本 box_n 全 0（机械臂不许碰木盒）
  7) 阶段名        phase 的元素必须都在已知 15 个名字里；首帧 init、末帧 tail
  8) 清单自洽      index.json 条目数 == 片段数；目录里的孤儿产物 / 清单里的缺文件

可选（--cross-extrinsics）
--------------------------
  用 render/test_render.CameraRig 现算一帧外参，和 npz 里存的比。
  这是为了守住"无渲染路径(collect._cam_math) 与 渲染路径(CameraRig) 同源"——
  两条路一旦分叉，有视频和没视频的两批数据外参就会系统性错开。默认关（要建模型）。

用法
----
    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py            # 全量体检
    env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py --ids 0,1,100
    env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py --quiet    # 只回汇总
    env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py --cross-extrinsics
退出码：0 = 全过；1 = 有 fail；2 = 数据集本身不可读（缺 index / 空）
"""

import os
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
INDEX_DIR = SCRIPTS_DIR / "index"
os.environ.setdefault("MUJOCO_GL", "egl")       # 只为 import test_render（不建 renderer）

sys.path.insert(0, str(ROOT / "render"))
sys.path.insert(0, str(ROOT / "collect"))
sys.unraisablehook = lambda *a, **k: None       # EGL 析构噪音静音

import dataset as D                              # noqa: E402
import test_render as R                          # noqa: E402

TOL_DT = 1e-9
TOL_ORTH = 1e-3
KNOWN_PHASE = set(D.PHASES)


# ---------------------------------------------------------------- 单项检查

def _check_schema(ep, meta, problems):
    path = (ep.get("paths") or {}).get("npz")
    if not path:
        problems.append("no_npz")
        return
    for p in R.validate_trajectory(ROOT / path):
        problems.append(f"schema: {p}")


def _check_video(ep, meta, problems, root=ROOT):
    """
    视频帧数/尺寸/fps 与轨迹对齐。
    负样本/--no-render 的样本没有视频：这时只要求 meta 自己别撒谎（has_video=false）。
    """
    p = ep.get("paths") or {}
    # sdg-v6 起键名 video_env/video_wrist（mp4）；更早是 env_avi/wrist_avi，回退兼容。
    def _rel(cam):
        return p.get(f"video_{cam}") or p.get(f"{cam}_avi")
    has_video = bool(_rel("env"))
    if not has_video:
        if meta.get("has_video"):
            problems.append("meta.has_video=true 但 paths 里没有 video_env")
        return
    import cv2
    n = int(meta.get("n_frames") or 0)
    for cam in ("env", "wrist"):
        rel = _rel(cam)
        vp = root / (rel or "")
        if not rel or not vp.exists():
            problems.append(f"missing_video[{cam}]")
            continue
        cap = cv2.VideoCapture(str(vp))
        cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n and cnt != n:
            problems.append(f"frames[{cam}] {cnt} != {n}")
        if abs(fps - float(ep.get("fps") or meta.get("fps") or 30.0)) > 1e-6:
            problems.append(f"fps[{cam}] {fps:.3f}")
        if meta.get("width") and (w, h) != (int(meta["width"]), int(meta["height"])):
            problems.append(f"size[{cam}] {w}x{h}")
        vf = (meta.get("video_frames") or {}).get(cam)
        if vf is not None and int(vf) != cnt:
            problems.append(f"meta.video_frames[{cam}] {vf} != 视频 {cnt}")


def _check_extrinsics(traj, problems):
    T = traj.get("cam_T_cw")
    if T is None:
        return
    for cam, M in T.items():
        if not np.isfinite(M).all():
            problems.append(f"extr_nan[{cam}]")
            continue
        Rm = M[:, :3, :3]
        orth = np.abs(np.einsum("nij,nkj->nik", Rm, Rm) - np.eye(3)).max()
        if orth > TOL_ORTH:
            problems.append(f"extr_not_orthonormal[{cam}] {orth:.2e}")
        det = np.linalg.det(Rm)
        if np.abs(det - 1.0).max() > TOL_ORTH:
            problems.append(f"extr_det[{cam}] {np.abs(det - 1.0).max():.2e}")


def _check_intrinsics(traj, problems):
    intr = traj.get("intrinsics") or {}
    if not intr:
        problems.append("no_intrinsics")
        return
    for cam, k in intr.items():
        fx, fy, cx, cy = (float(k.get(x, np.nan)) for x in ("fx", "fy", "cx", "cy"))
        W, H = float(k.get("width", np.nan)), float(k.get("height", np.nan))
        if not (fx > 0 and fy > 0):
            problems.append(f"intr_fx[{cam}] {fx}")
        if abs(fx - fy) > 1e-6:
            problems.append(f"intr_fx!=fy[{cam}]")
        if abs(cx - W / 2.0) > 0.5 or abs(cy - H / 2.0) > 0.5:
            problems.append(f"intr_center[{cam}] cx={cx} cy={cy}")
        # fy 与 fovy 必须能互相推出来（否则内参与相机模型不同源）
        fy2 = H / (2.0 * np.tan(np.radians(float(k.get("fovy", np.nan))) / 2.0))
        if np.isfinite(fy2) and abs(fy - fy2) > 1e-3 * max(1.0, fy):
            problems.append(f"intr_fovy[{cam}] fy={fy:.2f} vs fovy->{fy2:.2f}")


def _check_labels(ep, meta, traj, problems):
    acc = bool(ep.get("accepted"))
    suc = bool(meta.get("success", ep.get("success")))
    if acc and not suc:
        problems.append("accepted_but_success_false")
    # 与 render/test_render 同一口径再算一遍：两套口径分叉 = 标签不可信
    again = R.evaluate_success(dict(metrics=meta.get("metrics") or {}),
                               {k: traj[k] for k in ("phase", "lift_mm") if k in traj})
    si = meta.get("success_info") or {}
    if bool(again["success"]) != suc:
        problems.append(f"label_mismatch[success={suc} vs evaluate_success={again['success']}]")
    if si and bool(si.get("success")) != bool(again["success"]):
        problems.append("success_info_stale")
    if not suc and not (meta.get("reject_reasons") or meta.get("success_info")):
        problems.append("negative_without_reason")


def _check_traj(traj, meta, problems, notes=None):
    """
    时间轴检查。

    ★口径（实测确认，不是想当然）：
      - `time` 是按【时间栅格 @fps】采样的：0.002s 步长下 30fps = 16.67 步，
        所以相邻帧的 gap 在 0.032 / 0.034 s 之间交替（16 步 / 17 步），
        平均值必须 ≈ 1/fps。**不是** dt == timestep。
      - 阶段末会 force 补一帧，如果那一帧和上一条常规帧落在同一个 sim step，
        会出现 **dt == 0 的重复帧**（同 qpos、同 phase）。实测 1~2 帧/条（0.2~0.5%），
        来自 EpisodeRecorder.push 的"强制落帧"设计，不影响状态正确性（两行完全相同），
        所以只容忍不超过 2%，超过就说明录制逻辑被改坏了。
    """
    t = np.asarray(traj.get("time", []), float)
    if t.size < 2:
        problems.append("too_few_frames")
        return
    dt = np.diff(t)
    fps = float(meta.get("fps") or 30.0)
    frame_dt = 1.0 / fps
    n_dup = int((dt <= 0).sum())
    if (dt < -1e-12).any():
        problems.append("time_decreasing")
    if n_dup > max(2, int(0.02 * len(dt))):
        problems.append(f"dup_frames[{n_dup}/{len(dt)}]")
    elif n_dup and notes is not None:
        notes.append(f"dup_frames={n_dup}")
    # 平均 gap 会【小于】1/fps（强制帧是"额外多插的帧"，不占时间格），
    # 所以不能用 mean(dt)==1/fps 判；正确的两个不变量是：
    #   ① 帧数 ≈ 时间格数（只多出少量强制帧）  ② 没有任何 gap 超过 2 个时间格
    grid = int(round((t[-1] - t[0]) * fps)) + 1
    ratio = len(t) / max(1, grid)
    if not (0.99 <= ratio <= 1.20):
        problems.append(f"frames/grid={ratio:.3f}（应≈1.0~1.2：时间格数 {grid}，帧数 {len(t)}）")
    if dt.max() > 2.5 * frame_dt:
        problems.append(f"gap_too_large {dt.max():.4f}s（丢帧？）")
    pos = dt[dt > 0]
    want = float(meta.get("timestep") or 0.002)
    if want <= 0:
        problems.append("timestep_invalid")
    elif pos.size and pos.min() < 0.99 * want:
        problems.append(f"min_dt {pos.min():.5f} < timestep {want}")
    for k in ("qpos", "qvel", "obj_pos", "tcp_pos", "ctrl"):
        a = np.asarray(traj.get(k, []), float)
        if a.size and not np.isfinite(a).all():
            problems.append(f"nan[{k}]")
    z = np.asarray(traj.get("obj_pos", np.zeros((0, 3))), float)[:, 2]
    if z.size and (z.min() < -0.01 or z.max() > 0.3):
        problems.append(f"obj_z_out_of_range[{z.min():.3f},{z.max():.3f}]")
    ph = np.asarray(traj.get("phase", []))
    if ph.size:
        bad = sorted(set(map(str, ph)) - KNOWN_PHASE)
        if bad:
            problems.append(f"unknown_phase{bad}")
        if str(ph[0]) != "init":
            problems.append(f"first_phase={ph[0]}")
        if str(ph[-1]) != "tail":
            problems.append(f"last_phase={ph[-1]}")
    if bool(meta.get("success")) and "box_n" in traj:
        n_hit = int((np.asarray(traj["box_n"]) > 0).sum())
        if n_hit:
            problems.append(f"arm_hit_box[{n_hit}帧]")


def check_episode(ds, i, cross_extrinsics=False, **_kw):
    """检查一条。返回 dict(episode_id, ok, problems)。可被 dataset.SDGDataset.check() 复用。"""
    ep = ds.entry(i)
    problems, notes = [], []
    try:
        traj = ds.read_traj(i)
        meta = traj["meta"]
    except Exception as exc:                                # noqa: BLE001
        return dict(episode_id=int(ep.get("episode_id", -1)), ok=False,
                    problems=[f"unreadable: {type(exc).__name__}: {exc}"])
    _check_schema(ep, meta, problems)
    _check_video(ep, meta, problems)
    _check_extrinsics(traj, problems)
    _check_intrinsics(traj, problems)
    _check_labels(ep, meta, traj, problems)
    _check_traj(traj, meta, problems, notes)
    if cross_extrinsics:
        problems += _cross_extrinsics(traj, meta)
    return dict(episode_id=int(ep["episode_id"]), ok=not problems,
                problems=problems, notes=notes)


# ---------------------------------------------------------------- 交叉验证（可选）

_RIG_CACHE = {}


def _cross_extrinsics(traj, meta):
    """
    用 CameraRig + 同一份随机量现算第 0 帧外参，比 npz 里存的。
    期望差 == 0（两条路径是同一套公式），>1e-6 就说明有人改动了一边。
    """
    import mujoco
    import collect as C
    try:
        cfg = (meta.get("randomization") or {})
        key = id(meta)
        if key not in _RIG_CACHE:
            model = mujoco.MjModel.from_xml_path(R.SCENE_XML)
            C.apply_light_and_material(model, cfg)
            C.apply_camera_jitter(model, cfg)
            _RIG_CACHE[key] = (model, mujoco.MjData(model),
                               R.CameraRig(model, width=int(meta.get("width") or 640),
                                           height=int(meta.get("height") or 480)))
        model, data, rig = _RIG_CACHE[key]
        data.qpos[:] = traj["qpos"][0]
        mujoco.mj_forward(model, data)
        out = []
        for cam, short in (("env_camera", "env"), ("wrist_camera", "wrist")):
            if traj.get("cam_T_cw") is None:
                continue
            got = rig.extrinsics(data, cam).ravel()
            want = traj["cam_T_cw"][short][0].ravel()
            d = float(np.abs(got - want).max())
            if d > 1e-6:
                out.append(f"cross_extr[{short}] 差 {d:.2e}（两条路径分叉了）")
        for cam in ("env_camera", "wrist_camera"):
            gi = rig.intrinsics(cam)
            wi = (traj.get("intrinsics") or {}).get(cam)
            if wi and abs(gi["fy"] - wi["fy"]) > 1e-6:
                out.append(f"cross_intr[{cam}] 差 {gi['fy'] - wi['fy']:.2e}")
        return out
    except Exception as exc:                                # noqa: BLE001
        return [f"cross_extrinsics_failed: {type(exc).__name__}: {exc}"]


# ---------------------------------------------------------------- 清单自洽

def check_index(entries_by_id, strict=True):
    """index.json 的条目 vs 片段 vs 磁盘产物。返回 (problems, stats)。"""
    problems = []
    frag = {}
    if INDEX_DIR.exists():
        for p in sorted(INDEX_DIR.glob("ep_*.json")):
            try:
                e = json.loads(p.read_text(encoding="utf-8"))
                frag[int(e["episode_id"])] = e
            except Exception as exc:                        # noqa: BLE001
                problems.append(f"片段损坏 {p.name}: {type(exc).__name__}")
    if frag and set(frag) != set(entries_by_id):
        problems.append(f"清单 {len(entries_by_id)} 条 != 片段 {len(frag)} 条"
                        f"（缺片段: {sorted(set(entries_by_id)-set(frag))[:5]}，"
                        f"多片段: {sorted(set(frag)-set(entries_by_id))[:5]}）")

    # 磁盘孤儿：有 npz 但清单里没有
    have_npz = {int(p.stem.split("_")[1]) for p in SCRIPTS_DIR.glob("ep_*.npz")}
    orphan = sorted(have_npz - set(entries_by_id))
    if orphan:
        problems.append(f"孤儿 npz（有文件没清单）: {orphan[:10]}"
                        f"{' …' if len(orphan) > 10 else ''}")

    # 清单里有、文件没了
    missing = []
    for eid, e in entries_by_id.items():
        p = e.get("paths") or {}
        for k in ("npz", "json", "video_env", "video_wrist", "env_avi", "wrist_avi"):
            if p.get(k) and not (ROOT / p[k]).exists():
                missing.append(f"{eid}:{k}")
    if missing:
        problems.append(f"清单指向的文件不存在: {missing[:8]}"
                        f"{' …' if len(missing) > 8 else ''}")

    stats = dict(n_entries=len(entries_by_id), n_fragments=len(frag),
                 n_npz_on_disk=len(have_npz), n_orphan=len(orphan), n_missing=len(missing))
    return problems, stats


# ---------------------------------------------------------------- 主流程

def main(argv=None):
    ap = argparse.ArgumentParser(description="SDG 数据集批量体检")
    ap.add_argument("--ids", default=None, help="只检查这些编号，逗号分隔")
    ap.add_argument("--report", default=str(SCRIPTS_DIR / "report.json"),
                    help="报告落盘路径（默认 scripts/report.json）")
    ap.add_argument("--index", default=None, help="换一份 index.json（默认 scripts/index.json）")
    ap.add_argument("--cross-extrinsics", action="store_true",
                    help="用 CameraRig 现算外参与 npz 比对（慢，守两条路径同源）")
    ap.add_argument("--no-video", action="store_true", help="跳过视频帧数检查（不解码头）")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--json", action="store_true", help="把报告打到 stdout")
    args = ap.parse_args(argv)

    idx = Path(args.index) if args.index else SCRIPTS_DIR / "index.json"
    if not idx.exists():
        print(f"❌ 找不到 {idx}：先跑 collect/collect.py 或 --rebuild-index")
        return 2
    index = json.loads(idx.read_text(encoding="utf-8"))
    rows = list(index.get("accepted") or []) + list(index.get("rejected") or [])
    if not rows:
        print("❌ 清单里一条样本都没有")
        return 2
    by_id = {int(e["episode_id"]): e for e in rows}
    raw_ids = [int(e["episode_id"]) for e in rows]
    dupes = sorted({v for v in raw_ids if raw_ids.count(v) > 1})

    if args.ids:
        want = [int(v) for v in args.ids.replace(" ", "").split(",") if v]
        missing = [v for v in want if v not in by_id]
        if missing:
            print(f"❌ 这些编号不在清单里: {missing}")
            return 2
        for eid in list(by_id):
            if eid not in want:
                by_id.pop(eid)

    ds = D.SDGDataset(ids=sorted(by_id), load_images=False, strict=False)
    if args.no_video:
        global _check_video
        _check_video = lambda *a, **k: None

    results = []
    for i in range(len(ds)):
        r = check_episode(ds, i, cross_extrinsics=args.cross_extrinsics)
        results.append(r)
        if not args.quiet:
            flag = "✅" if r["ok"] else "❌"
            print(f"  {flag} ep_{r['episode_id']:06d}"
                  + ("" if r["ok"] else "  <- " + " | ".join(r["problems"][:4])))

    all_by_id = {int(e["episode_id"]): e for e in rows}
    idx_problems, idx_stats = check_index(all_by_id)
    if dupes:
        idx_problems.append(f"清单里出现重复 episode_id: {dupes[:5]}")
    if index.get("mjcf_mixed"):
        idx_problems.append("这批样本混了多个 MJCF 指纹（mjcf_mixed=true）：不能混切/混训")
    n_fail = sum(1 for r in results if not r["ok"])
    reason_cnt = Counter(p.split("[")[0].split(" ")[0] for r in results for p in r["problems"])
    report = dict(
        checked_at=str(np.datetime64("now")),
        dataset_version=index.get("dataset_version"),
        index=idx_stats,
        n_checked=len(results),
        n_ok=len(results) - n_fail,
        n_fail=n_fail,
        index_problems=idx_problems,
        problem_kinds=dict(reason_cnt.most_common()),
        episodes=[r for r in results if not r["ok"]],
        note="只列失败条目（全量结果太大）；problem_kinds 是失败类型分布",
    )
    rp = Path(args.report)
    rp = rp if rp.is_absolute() else (ROOT / rp)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    ok_all = (n_fail == 0) and not idx_problems
    print()
    print(f"  数据版本  : {report['dataset_version']}   检查 {report['n_checked']} 条")
    print(f"  通过/失败 : {report['n_ok']} / {n_fail}")
    if reason_cnt:
        print("  失败类型  : " + ", ".join(f"{k}×{v}" for k, v in reason_cnt.most_common(8)))
    print(f"  清单自洽  : {'✅' if not idx_problems else '❌ ' + ' | '.join(idx_problems[:3])}")
    try:
        shown = rp.relative_to(ROOT)
    except ValueError:
        shown = rp
    print(f"  报告      : {shown}（{rp.stat().st_size/1024:.1f} KB）")
    print(f"  总判定    : {'✅ 全部通过' if ok_all else '❌ 有失败项'}")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
