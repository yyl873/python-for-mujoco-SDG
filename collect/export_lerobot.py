#!/usr/bin/env python3
"""
export_lerobot.py —— 把采集产物导出成【lerobot v3.0 数据集】，ACT 直接训练
==========================================================================

为什么单独一步、单独一个解释器
------------------------------
采集端（collect.py）跑在系统 python3.10 + mujoco 里，只产【母本】：
    scripts/ep_XXXXXX.npz   轨迹（qpos/ctrl/接触力/外参）
    scripts/ep_XXXXXX.json  标签 + 随机化参数
    videos/ep_XXXXXX_*.mp4  h264 视频（训练要的就是这份，不再转格式）
写 lerobot 数据集需要 pyarrow + lerobot，只有 conda 的 lerobot312 环境有。
所以导出是**独立一步**，跑那个解释器，只做数据搬运（不解码物理、不碰 MuJoCo）：

    ~/miniforge3/envs/lerobot312/bin/python collect/export_lerobot.py \
        --dst ~/lerobot_datasets/soarm101_sim_v1

产出目录（v3.0 布局，与真机采集的数据集逐字段同构）：
    meta/info.json  meta/stats.json  meta/tasks.parquet
    meta/episodes/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.env/chunk-000/file-000.mp4
    videos/observation.images.wrist/chunk-000/file-000.mp4

字段（单位=度，与 SO-101 真机数据集一致）
    observation.state  qpos[:6]  当前 6 关节角（pan/lift/elbow/flex/roll/gripper）
    action             ctrl[:6]  当帧下发的 6 关节目标角
    observation.images.env / .wrist   640x480x3 视频

★迁移时不需要改任何代码：换机器只改 --src/--videos/--dst 三个路径参数。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# lerobot 会在 ~/.cache/huggingface 下落 parquet 缓存与 .lock（回读校验时会写）。
# 本机那个目录对沙箱只读 → 默认把 HF 缓存指到仓库内；用户自己设了 HF_HOME 就尊重用户。
_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
IMG_KEYS = {"env": "observation.images.env", "wrist": "observation.images.wrist"}


# ---------------------------------------------------------------------------
# 读母本
# ---------------------------------------------------------------------------

def load_index(src: Path):
    p = src / "index.json"
    if not p.exists():
        raise SystemExit(f"找不到清单 {p}（先跑 collect/collect.py）")
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for key in ("accepted", "rejected"):
        for e in d.get(key) or []:
            rows.append(dict(e, success=bool(e.get("success", key == "accepted"))))
    rows.sort(key=lambda r: int(r["episode_id"]))
    return rows, d


def video_rel(e, cam):
    p = e.get("paths") or {}
    return p.get(f"video_{cam}") or p.get(f"{cam}_avi")      # 兼容 sdg-v5 的 avi


def decode_frames(path: Path):
    """解码成 RGB uint8 (N,H,W,3)。用 pyav（lerobot 环境自带）。"""
    import av
    out = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        for fr in c.decode(st):
            out.append(fr.to_ndarray(format="rgb24"))
    if not out:
        raise RuntimeError(f"解不出帧: {path}")
    return np.stack(out)


def remap_gripper(arr, space, jaw_open_deg=85.94, real_full=100.0):
    """把夹爪列（第 5 列）从"仿真 jaw 角（0=全闭）"换到真机夹爪量纲。

    space:
      sim         不动（jaw 角，0=全闭 / 85.94°=全张）
      real-open   真机 0=张开 → real = real_full * (1 - jaw/jaw_open_deg)
      real-closed 真机 0=闭合 → real = real_full * (jaw/jaw_open_deg)
    """
    if space == "sim":
        return arr
    a = arr.copy()
    jaw = a[:, 5]
    frac = np.clip(jaw / float(jaw_open_deg), 0.0, 1.0)
    a[:, 5] = real_full * (1.0 - frac) if space == "real-open" else real_full * frac
    return a


def episode_arrays(e, src: Path, units: str, gripper_space: str = "sim",
                   jaw_open_deg: float = 85.94, real_full: float = 100.0):
    """从 npz 取 state/action，并核对帧数/维度。"""
    npz_rel = (e.get("paths") or {}).get("npz")
    if not npz_rel:
        raise RuntimeError("这条没有 npz 轨迹")
    with np.load(src.parent / npz_rel, allow_pickle=False) as z:
        qpos = np.asarray(z["qpos"], np.float64)
        ctrl = np.asarray(z["ctrl"], np.float64)
        phase = np.array([str(p) for p in z["phase"]]) if "phase" in z.files else None
    if qpos.shape[1] < 6 or ctrl.shape[1] < 6:
        raise RuntimeError(f"维度不符：qpos{qpos.shape} ctrl{ctrl.shape}")
    state = qpos[:, :6]
    action = ctrl[:, :6]
    if units == "deg":
        state = np.degrees(state)
        action = np.degrees(action)
    state = remap_gripper(state, gripper_space, jaw_open_deg, real_full)
    action = remap_gripper(action, gripper_space, jaw_open_deg, real_full)
    return state, action, phase


# ---------------------------------------------------------------------------
# 写 lerobot 数据集
# ---------------------------------------------------------------------------

def build_features(h, w):
    return {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        IMG_KEYS["env"]: {"dtype": "video", "shape": (h, w, 3),
                          "names": ["height", "width", "channels"]},
        IMG_KEYS["wrist"]: {"dtype": "video", "shape": (h, w, 3),
                            "names": ["height", "width", "channels"]},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="scripts/ + videos/ -> lerobot v3.0 数据集（ACT 可直接训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--src", default=str(ROOT / "scripts"), help="母本目录（默认 <root>/scripts）")
    ap.add_argument("--videos", default=None, help="视频目录（默认 <root>/videos）")
    ap.add_argument("--dst", required=True, help="输出目录（lerobot 数据集根，必须不存在或 --overwrite）")
    ap.add_argument("--repo-id", default=None, help="数据集标识（默认 local/<dst 目录名>）")
    ap.add_argument("--task", default=None,
                    help="自然语言任务（默认取母本 index.json / ep json 的 instruction）")
    ap.add_argument("--robot-type", default="so_follower", help="写进 info.json 的 robot_type")
    ap.add_argument("--gripper-space", choices=["sim", "real-open", "real-closed"], default="sim",
                    help="夹爪列量纲：sim=jaw 角（0=全闭）；real-open=真机 0 表示张开；"
                         "real-closed=真机 0 表示闭合。与真机数据混合训练时必须换成真机量纲")
    ap.add_argument("--jaw-open-deg", type=float, default=85.94,
                    help="仿真 jaw 全张开角（度，默认 85.94 = 采集上限 1.5 rad）")
    ap.add_argument("--gripper-full", type=float, default=100.0,
                    help="真机夹爪满量程值（仅 real-open / real-closed 有意义）。"
                         "★别用默认 100：本机真机实测 fixed 批次 49.08、corners 批次 42.27。"
                         "混训前必须传目标真机数据集的实测值，否则仿真夹爪尺度会和真机差约 2 倍")
    ap.add_argument("--units", choices=["deg", "rad"], default="deg",
                    help="state/action 单位（默认 deg，与 SO-101 真机数据集一致）")
    ap.add_argument("--episodes", default=None, help="只导这些编号，如 0,3,7（默认全部正样本）")
    ap.add_argument("--limit", type=int, default=None, help="最多导几条")
    ap.add_argument("--include-failed", action="store_true",
                    help="连 success=false 的也导（默认只导正样本）")
    ap.add_argument("--drop-phases", default=None,
                    help="去掉这些阶段的行，如 init,home,tail（默认全保留）")
    ap.add_argument("--overwrite", action="store_true", help="目标目录已存在就删掉重建")
    ap.add_argument("--check", action="store_true", help="只体检母本（帧数/维度/视频对齐），不写数据集")
    ap.add_argument("--no-verify", action="store_true", help="导出后不做回读校验")
    args = ap.parse_args(argv)

    src = Path(args.src)
    videos = Path(args.videos) if args.videos else src.parent / "videos"
    dst = Path(args.dst).expanduser()
    rows, index = load_index(src)

    # ---- 选哪些条 ----
    if args.episodes:
        want = {int(x) for x in str(args.episodes).replace(" ", "").split(",") if x != ""}
        sel = [r for r in rows if int(r["episode_id"]) in want]
    else:
        sel = [r for r in rows if args.include_failed or r.get("success")]
    if args.limit:
        sel = sel[:args.limit]
    if not sel:
        raise SystemExit("没有可导出的条目（检查 --episodes / --include-failed）")

    task = args.task or next((r.get("instruction") for r in rows
                              if r.get("instruction")), None)
    if not task:
        for r in sel:
            js = src / f"ep_{int(r['episode_id']):06d}.json"
            if js.exists():
                task = json.loads(js.read_text(encoding="utf-8")).get("instruction")
                if task:
                    break
    task = task or "Pick up the medicine box and place it into the wooden box."
    drop = {x.strip() for x in str(args.drop_phases).split(",") if x.strip()} if args.drop_phases else set()

    print("=" * 78)
    print(" export_lerobot  母本 -> lerobot v3.0 数据集")
    print("=" * 78)
    print(f"  母本    : {src}（{index.get('dataset_version')} / schema {index.get('schema_version')}）")
    print(f"  视频    : {videos}")
    print(f"  目标    : {dst}" + ("" if args.check else ""))
    print(f"  条数    : {len(sel)}（正样本 {sum(1 for r in sel if r.get('success'))}"
          f" / 负样本 {sum(1 for r in sel if not r.get('success'))}）")
    print(f"  单位    : {args.units}   任务: {task!r}"
          + (f"   去掉阶段: {sorted(drop)}" if drop else ""))
    print()

    # ---- 逐条体检 + 导出准备（先全部过一遍，别写到一半才发现第 40 条坏了）----
    plan, problems, n_frames = [], [], 0
    t_scan = time.time()
    for r in sel:
        eid = int(r["episode_id"])
        try:
            state, action, phase = episode_arrays(
                r, src, args.units, args.gripper_space, args.jaw_open_deg, args.gripper_full)
            env_p, wrist_p = video_rel(r, "env"), video_rel(r, "wrist")
            if not env_p or not wrist_p:
                raise RuntimeError("没有 video_env/video_wrist（采集时用了 --no-render？）")
            env_abs, wrist_abs = src.parent / env_p, src.parent / wrist_p
            for p in (env_abs, wrist_abs):
                if not p.exists():
                    raise RuntimeError(f"视频不存在: {p}")
            keep = np.ones(len(state), bool)
            if drop and phase is not None:
                keep = ~np.isin(phase, list(drop))
                if keep.sum() == 0:
                    raise RuntimeError(f"--drop-phases {sorted(drop)} 把这条整条删空了")
            plan.append(dict(eid=eid, state=state[keep], action=action[keep],
                             env=env_abs, wrist=wrist_abs, keep=keep,
                             n=int(keep.sum())))
            n_frames += int(keep.sum())
        except Exception as exc:                       # noqa: BLE001
            problems.append(f"ep_{eid:06d}: {exc}")
    print(f"  体检    : {len(plan)} 条可用 / {len(problems)} 条有问题  "
          f"共 {n_frames} 帧（{n_frames / 30.0:.1f}s @30fps），扫描 {time.time() - t_scan:.1f}s")
    for p in problems[:10]:
        print(f"    ! {p}")
    if problems and not plan:
        raise SystemExit("没有一条能导出")
    if not plan:
        raise SystemExit("没有一条能导出")

    if args.check:
        print("  --check：只体检，不写数据集")
        return 0 if not problems else 1

    # ---- 建数据集 ----
    from lerobot.datasets.lerobot_dataset import LeRobotDataset    # 只在需要时 import

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} 已存在（换路径或加 --overwrite 删掉重建）")
        shutil.rmtree(dst)
    repo_id = args.repo_id or f"local/{dst.name}"

    # 分辨率从视频头取（与采集一致就是 640x480）
    import av
    with av.open(str(plan[0]["env"])) as c:
        h = int(c.streams.video[0].codec_context.height)
        w = int(c.streams.video[0].codec_context.width)
        fps = float(c.streams.video[0].average_rate) or 30.0
    features = build_features(h, w)

    ds = LeRobotDataset.create(repo_id=repo_id, fps=int(round(fps)),
                               features=features, root=dst,
                               robot_type=args.robot_type)
    print(f"  建库    : {repo_id}  {w}x{h} @ {fps:.0f}fps  特征 {list(features)}")

    t0 = time.time()
    done = 0
    for item in plan:
        t_ep = time.time()
        env_fr = decode_frames(item["env"])
        wrist_fr = decode_frames(item["wrist"])
        n = int(item["n"])
        if len(env_fr) != len(item["keep"]) or len(wrist_fr) != len(item["keep"]):
            raise SystemExit(f"ep_{item['eid']:06d} 视频帧数 env={len(env_fr)} "
                             f"wrist={len(wrist_fr)} 与轨迹 {len(item['keep'])} 不一致")
        idx = np.where(item["keep"])[0]
        for j, k in enumerate(idx):
            ds.add_frame({
                "observation.state": item["state"][j].astype(np.float32),
                "action": item["action"][j].astype(np.float32),
                IMG_KEYS["env"]: env_fr[k],
                IMG_KEYS["wrist"]: wrist_fr[k],
                "task": task,
            })
        ds.save_episode()
        done += 1
        print(f"  [{done:>3}/{len(plan)}] ep_{item['eid']:06d}  {n} 帧  "
              f"{time.time() - t_ep:.1f}s（累计 {time.time() - t0:.0f}s）")

    ds.finalize()

    # ---- 溯源：把"这批数据是怎么来的"钉在数据集里，迁移后还能对账 ----
    prov = dict(
        generator="collect/export_lerobot.py",
        exported_at=str(np.datetime64("now")),
        source_root=str(src.parent),
        source_dataset_version=index.get("dataset_version"),
        source_schema_version=index.get("schema_version"),
        source_index_sha=None,
        n_episodes=done, n_frames=int(n_frames),
        units=args.units, joints=JOINTS, task=task,
        gripper_space=args.gripper_space,
        jaw_open_deg=float(args.jaw_open_deg),
        gripper_full=float(args.gripper_full),
        gripper_note=("sim=jaw 角（0=全闭）；real-open/real-closed 时按 "
                      "gripper_full 线性映射越大越开。混训前必须与本机真机数据集实测满量程一致"
                      "（fixed 49.08 / corners 42.27），否则夹爪列尺度会差约 2 倍。"),
        drop_phases=sorted(drop),
        episode_ids=[int(i["eid"]) for i in plan],
        note=("observation.state=qpos[:6], action=ctrl[:6]（关节角命令）；"
              "视频=采集时的 h264 mp4 原帧；物理由 MuJoCo 真接触力产生，未写 qpos 伪造握持。"),
    )
    try:
        import hashlib
        prov["source_index_sha"] = hashlib.sha256(
            (src / "index.json").read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    (dst / "meta").mkdir(parents=True, exist_ok=True)
    (dst / "meta" / "sim_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")

    size_mb = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file()) / 1024 / 1024
    print()
    print(f"  ✅ 写完: {dst}  ({done} 条 / {n_frames} 帧 / {size_mb:.1f} MB)  "
          f"耗时 {time.time() - t0:.0f}s")

    # ---- 回读校验（不做就等于没保证：lerobot 自己读得出来才算数）----
    if not args.no_verify:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset as LRD
        print("  回读    : ", end="")
        try:
            back = LRD(repo_id=repo_id, root=dst)
            s0 = back[0]
            keys = sorted(s0.keys())
            # lerobot 回读给的是 CHW（3,H,W）；采集侧写的是 HWC，两种都认
            img_ok = all(tuple(s0[k].shape) in ((3, h, w), (h, w, 3))
                         or tuple(s0[k].shape[-3:]) == (h, w, 3)
                         for k in IMG_KEYS.values())
            st = back.meta.stats
            print(f"len={len(back)} episodes={back.meta.total_episodes} "
                  f"frames={back.meta.total_frames} "
                  f"state{tuple(s0['observation.state'].shape)} "
                  f"action{tuple(s0['action'].shape)} "
                  f"img={'ok' if img_ok else 'BAD'} "
                  f"stats={'ok' if st and st.get('action') else 'MISSING'}")
            print(f"            样本键: {keys}")
            # ---- 夹爪列自检：混训前唯一必须亲眼看的数字 ----
            try:
                gs = (st or {}).get("observation.state") or {}
                if "min" in gs and "max" in gs:
                    g_lo, g_hi = float(gs["min"][5]), float(gs["max"][5])
                    print(f"            夹爪列 min {g_lo:6.2f} / max {g_hi:6.2f}   "
                          f"(space={args.gripper_space} full={args.gripper_full:.2f} "
                          f"jaw_open={args.jaw_open_deg:.2f})")
                    if args.gripper_space != "sim":
                        if g_hi > 60.0:
                            print("  ⚠️  夹爪 max > 60：--gripper-full 很可能还是默认值 100，"
                                  "而真机实测是 fixed 49.08 / corners 42.27 → 混训前必须改")
                        elif g_hi < 30.0:
                            print("  ⚠️  夹爪 max < 30：满量程设小了，请核对该真机数据集的实测张开值")
            except Exception:                        # noqa: BLE001
                pass
            ok = (len(back) == n_frames and back.meta.total_episodes == done
                  and img_ok and st and st.get("action"))
            if not ok:
                print("  ❌ 回读校验未通过")
                return 1
        except Exception as exc:                       # noqa: BLE001
            print(f"❌ {type(exc).__name__}: {exc}")
            return 1

    print()
    print("  下一步（训练）:")
    print(f"    lerobot-train --dataset.repo_id={repo_id} --dataset.root={dst} \\")
    print("                   --policy.type=act --output_dir=outputs/act_" + dst.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
