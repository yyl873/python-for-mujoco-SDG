#!/usr/bin/env python3
"""
dataset.py —— SDG 数据集的【唯一读取入口】+ 数据契约（契约写在这里，不写在口口相传里）
================================================================================

为什么需要它
------------
`scripts/` 里一条数据是三份文件：`ep_XXXXXX.npz`（轨迹）+ `ep_XXXXXX.json`（标签）+
`videos/ep_XXXXXX_{env,wrist}.mp4`（视频）。谁都能自己 np.load + cv2.VideoCapture 拼，
但拼的时候有 6 条**隐式约定**，任何一条想错都会静默出错（训练不报错、指标变差）：

  ① 帧对齐：视频第 i 帧  ↔  npz 第 i 行。MJPG 编码不改帧数，但"没断言"就等于没保证。
     collect.py 落盘时用 `video_frames == n_frames` 断言，本模块读取时再核一次。
  ② 外参方向：`cam_env_T_cw` / `cam_wrist_T_cw` 是 **world -> camera**：
        p_cam = T[:3,:3] @ p_world + T[:3,3]
     npz 里存成**行主序展平的 (N*16,)** float32，用 `reshape(N, 4, 4)` 还原。
  ③ 内参：在 `meta["cameras"]["intrinsics_by_cam"]` 里**两台相机各一份**。
     针孔模型，方形像素（fx == fy），cx/cy = 画面中心。腕部相机的内参固定，
     但它是 body 挂载 -> **外参逐帧都不同**（所以别把腕部外参当成常量）。
  ④ 视频是 MJPG **有损**：解码出来的像素 ≠ 采集时的像素。要逐像素精确（比如做
     psnr/去噪标签）就用 `--keep-frames` 落的 `ep_XXXXXX_frames.npz`（体积约 800MB/条）。
  ⑤ `phase` 是阶段标签（15 个名字），`success` 是任务成败（抬升 ≥50mm 且没滑落），
     `accepted` 是 collect.py 的六条自动验收。正常数据两者一致，`check_dataset.py` 会守。
  ⑥ `dataset_version` 变了或 `mjcf_sha256` 变了，就是**另一批数据**：
     不能混在一起切分/训练（split.py 会直接报错）。

用法
----
    from collect.dataset import SDGDataset            # 或把 collect/ 加进 sys.path 后 import dataset
    ds = SDGDataset()                                  # 全部样本
    ds = SDGDataset(split="train")                     # 按 split.py 的划分取
    ds = SDGDataset(load_images=False)                 # 只要轨迹（训练机上不解码视频，快 100 倍）
    ep = ds[0]
    ep["qpos"], ep["phase"]
    ep["cam_T_cw"]["env"]           # (N,4,4) 键用短名 env / wrist
    ep["intrinsics"]["wrist_camera"]  # 键用模型里的相机全名（env_camera / wrist_camera）
    ep["images"]["env"]             # (N,H,W,3) uint8 RGB；load_images=False 时为 None
    for ep in ds.iter_episodes(): ...

    # 命令行自检（打印一条样本的形状 + 契约核对结果）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/dataset.py
    env -u PYTHONHOME -u PYTHONPATH python3 collect/dataset.py --split test --images

只依赖 numpy（+ cv2 仅在真解码视频时用），不 import mujoco —— 训练机不用装 MuJoCo。
"""

import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
VIDEOS_DIR = ROOT / "videos"
SPLITS_DIR = SCRIPTS_DIR / "splits"

CAMS = ("env_camera", "wrist_camera")
CAM_SHORT = {"env_camera": "env", "wrist_camera": "wrist"}

# 阶段名（与 render/test_render.make_step_plan 的 16 段一一对应；
# 注意 plan 里 approach 被拆成 approach_retract / approach_above 两段，
# 但 phase 标签统一写 "approach" —— 所以两边名字不是同一个集合，别拿 plan 的键去比。）
PHASES = ("init", "approach", "descend", "grasp", "settle_a", "lift", "settle_b",
          "transport", "place", "pre_open", "release", "drop", "retract", "home", "tail")

# 契约常量
T_KEY = "cam_{cam}_T_cw"          # npz 里的键名模板
SCHEMA_VERSION = 2                # npz/meta 结构版本（v1 没有 nq/nv/nu/success）


class DatasetError(RuntimeError):
    """数据集层面的错误（缺文件、版本混杂、划分不存在…）。"""


class FrameMismatch(DatasetError):
    """视频帧数与轨迹帧数不一致 —— 一旦发生，动作-画面对应关系就是错的，必须报出来。"""


def _atomic_write(path, obj, indent=1):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)
    return path


class SDGDataset:
    """
    一维数据集：第 i 条 = 一个 episode 的全部产物。

    参数
    ----
    root         仓库根目录（默认本文件上两级）
    split        取 scripts/splits/<split>.json 的划分（train / val / test）；None = 全部
    ids          显式指定 episode_id 列表（优先于 split）
    load_images  True 才解码 AVI（默认 True；只要轨迹时设 False，快很多）
    strict       True 时任何契约违背都抛异常；False 时降级（缺视频→images=None）
    """

    def __init__(self, root=None, split=None, ids=None, load_images=True, strict=True):
        self.root = Path(root) if root else ROOT
        self.scripts = self.root / "scripts"
        self.videos = self.root / "videos"
        self.splits_dir = self.scripts / "splits"
        self.load_images = bool(load_images)
        self.strict = bool(strict)
        self.split = split

        idx_path = self.scripts / "index.json"
        if not idx_path.exists():
            raise DatasetError(f"找不到 {idx_path}（先跑 collect/collect.py 或 --rebuild-index）")
        self.index = json.loads(idx_path.read_text(encoding="utf-8"))
        self.policy = self.index.get("randomization_policy")
        self.dataset_version = self.index.get("dataset_version")

        rows = list(self.index.get("accepted") or []) + list(self.index.get("rejected") or [])
        self._by_id = {int(e["episode_id"]): e for e in rows}
        self.all_ids = sorted(self._by_id)

        if ids is not None:
            want = [int(v) for v in ids]
            missing = [v for v in want if v not in self._by_id]
            if missing:
                raise DatasetError(f"这些 episode_id 不在清单里: {missing[:10]}"
                                   f"{' …' if len(missing) > 10 else ''}")
            self.ids = want
        elif split is not None:
            sp = self.splits_dir / f"{split}.json"
            if not sp.exists():
                raise DatasetError(f"找不到划分 {sp}（先跑 collect/split.py）")
            self.ids = [int(v) for v in json.loads(sp.read_text(encoding="utf-8"))["episode_ids"]]
        else:
            self.ids = list(self.all_ids)

        # 版本一致性：混版本 = 数据不可比，宁可现在就炸
        vers = {self._by_id[i].get("dataset_version", "legacy") for i in self.ids}
        if len(vers) > 1 and self.strict:
            raise DatasetError(f"数据集混了多个 dataset_version: {sorted(vers)}"
                               "（要么重刷，要么用 strict=False 自己承担后果）")

    # ---------------- 基本信息 ----------------
    def __len__(self):
        return len(self.ids)

    def __repr__(self):
        return (f"SDGDataset(n={len(self.ids)}, split={self.split!r}, "
                f"version={self.dataset_version}, images={self.load_images})")

    def entry(self, i):
        """索引条目（包含 accepted/success/reasons/metrics/keys/paths/run）。"""
        return self._by_id[self.ids[i]]

    def episode_id(self, i):
        return int(self.ids[i])

    @property
    def success(self):
        return np.array([bool(self._by_id[i].get("success", self._by_id[i].get("accepted")))
                         for i in self.ids], dtype=bool)

    def meta(self, i):
        """per-episode json（权威元数据：随机量、plan、指标、相机、mjcf 指纹）。"""
        e = self.entry(i)
        p = e.get("paths") or {}
        jp = p.get("json")
        if not jp:
            # 极端情况（--no-keep-failed 的负样本）：退回索引条目里的信息
            return {"episode_id": e["episode_id"], "success": e.get("success"),
                    "reject_reasons": e.get("reasons"), "metrics": e.get("metrics"),
                    "keys": e.get("keys"), "note": "无 per-episode json（仅有索引条目）"}
        return json.loads((self.root / jp).read_text(encoding="utf-8"))

    # ---------------- 轨迹 ----------------
    def read_traj(self, i):
        """
        读 npz → dict。外参按契约还原成 (N,4,4)；没有外参的样本（--no-extrinsics）返回 None。
        """
        e = self.entry(i)
        p = (e.get("paths") or {}).get("npz")
        if not p:
            raise DatasetError(f"episode {e['episode_id']} 没有 npz（负样本未落盘？）")
        z = np.load(self.root / p, allow_pickle=False)
        out = {k: z[k] for k in z.files if k != "meta_json"}
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
        n = int(out["time"].shape[0]) if "time" in out else None

        traj = {}
        for k, v in out.items():
            if k.startswith("cam_") and k.endswith("_T_cw"):
                cam = "env" if k.startswith("cam_env") else "wrist"
                if v.size != n * 16:
                    raise DatasetError(f"{p}: {k} 大小 {v.size} != {n}*16")
                traj.setdefault("cam_T_cw", {})[cam] = v.reshape(n, 4, 4).astype(np.float64)
            else:
                traj[k] = v
        traj.setdefault("cam_T_cw", None)
        traj["meta"] = meta
        traj["episode_id"] = int(e["episode_id"])
        traj["success"] = bool(meta.get("success", e.get("success", e.get("accepted"))))
        cams = (meta.get("cameras") or {})
        intr = cams.get("intrinsics_by_cam")
        if not intr and cams.get("intrinsics"):
            intr = {"env_camera": cams["intrinsics"]}
        traj["intrinsics"] = intr
        return traj

    # ---------------- 视频 ----------------
    # 键名换代：sdg-v6 起是 video_env/video_wrist（mp4）；
    # sdg-v5 及更早是 env_avi/wrist_avi（MJPG avi）。老数据集要能照读，所以回退。
    @staticmethod
    def _video_rel(entry, cam):
        p = (entry.get("paths") or {})
        return p.get(f"video_{cam}") or p.get(f"{cam}_avi")

    def has_video(self, i):
        return bool(self._video_rel(self.entry(i), "env"))

    def video_path(self, i, cam="env"):
        p = self._video_rel(self.entry(i), cam)
        if not p:
            return None
        return self.root / p

    def images(self, i, cam="env", expect_n=None):
        """
        解码一条视频成 (N,H,W,3) uint8 RGB。

        ★契约 ① 的强制执行点：解码帧数必须等于轨迹帧数，否则抛 FrameMismatch
          （宁可在读取时炸，也不要训练时"动作和画面错开一帧"）。
        """
        import cv2                                    # 只有真解码时才 import

        path = self.video_path(i, cam)
        if path is None:
            if self.strict:
                raise DatasetError(f"episode {self.episode_id(i)} 没有 {cam} 视频")
            return None
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise DatasetError(f"打不开视频: {path}")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, ::-1].copy())   # BGR -> RGB
        cap.release()
        arr = np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)
        if expect_n is not None and arr.shape[0] != expect_n:
            raise FrameMismatch(f"{path.name}: 视频 {arr.shape[0]} 帧 != 轨迹 {expect_n} 帧")
        return arr

    # ---------------- 样本 ----------------
    def __getitem__(self, i):
        if i < 0:
            i += len(self)
        if not (0 <= i < len(self)):
            raise IndexError(i)
        traj = self.read_traj(i)
        n = int(traj["time"].shape[0])
        images = None
        if self.load_images and self.has_video(i):
            # 键用短名（env / wrist），与 cam_T_cw 对齐；intrinsics 的键仍是模型里的相机全名
            images = {CAM_SHORT[c]: self.images(i, CAM_SHORT[c], expect_n=n) for c in CAMS}
        e = self.entry(i)
        return dict(
            episode_id=traj["episode_id"],
            success=traj["success"],
            accepted=bool(e.get("accepted")),
            meta=traj["meta"],
            qpos=traj["qpos"], qvel=traj["qvel"], ctrl=traj["ctrl"],
            phase=np.asarray(traj["phase"]),
            tcp_pos=traj["tcp_pos"], obj_pos=traj["obj_pos"], obj_quat=traj["obj_quat"],
            contact_n=traj["contact_n"], fn_jaw=traj["fn_jaw"],
            fn_grip=traj["fn_grip"], fn_sum=traj["fn_sum"],
            lift_mm=traj["lift_mm"], hdisp_mm=traj["hdisp_mm"],
            box_n=traj["box_n"], obj_speed=traj["obj_speed"],
            cam_T_cw=traj["cam_T_cw"], intrinsics=traj["intrinsics"],
            images=images,
        )

    def iter_episodes(self):
        for i in range(len(self)):
            yield self[i]

    # ---------------- 自检 ----------------
    def check(self, **kw):
        """转调 check_dataset.check_episode()（延迟 import，避免循环依赖）。"""
        from check_dataset import check_episode       # noqa: PLC0415
        out = []
        for i in range(len(self)):
            out.append(check_episode(self, i, **kw))
        return out


# ============================================================
# 命令行：契约自检（打印一条样本，核对 6 条约定）
# ============================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description="SDG 数据集读取器自检")
    ap.add_argument("--split", default=None)
    ap.add_argument("--images", action="store_true", help="连视频一起解码（慢，但能验帧对齐）")
    ap.add_argument("--index", type=int, default=0, help="看第几条（默认 0）")
    args = ap.parse_args(argv)

    ds = SDGDataset(split=args.split, load_images=args.images)
    print(f"{ds}")
    if not len(ds):
        print("（空数据集）")
        return 2
    i = min(args.index, len(ds) - 1)
    ep = ds[i]
    n = ep["qpos"].shape[0]
    print(f"  episode_id : {ep['episode_id']}  success={ep['success']}  accepted={ep['accepted']}")
    print(f"  帧数       : {n}   fps={ep['meta'].get('fps')}")
    print(f"  qpos/qvel  : {ep['qpos'].shape} / {ep['qvel'].shape}")
    print(f"  phase      : {len(set(ep['phase']))} 种，首末 = {ep['phase'][0]}/{ep['phase'][-1]}")
    T = ep["cam_T_cw"]
    if T is None:
        print("  cam_T_cw   : None（该条没有外参）")
    else:
        for k, v in T.items():
            Rc = v[:, :3, :3]
            orth = np.abs(np.einsum("nij,nkj->nik", Rc, Rc) - np.eye(3)).max()
            print(f"  cam_T_cw[{k}]: {v.shape}  R 正交性 max|RRᵀ-I|={orth:.2e}")
    if ep["intrinsics"]:
        for k, v in ep["intrinsics"].items():
            print(f"  intrinsics[{k}]: fx={v['fx']:.1f} cx={v['cx']:.1f} "
                  f"({v['width']}x{v['height']}, fovy={v['fovy']:.2f})")
    if ep["images"] is not None:
        for k, v in ep["images"].items():
            print(f"  images[{k}]  : {v.shape} {v.dtype}  == 轨迹帧数: {v.shape[0] == n}")
    print(f"  指令       : {ep['meta'].get('instruction')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
