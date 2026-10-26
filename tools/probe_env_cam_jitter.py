#!/usr/bin/env python3
"""量 collect 里 env_camera 抖动之后，绿垫取景变成什么样。

背景：`apply_camera_jitter` 以 model/so_arm101.xml 的名义位姿为基准，
叠加 ±1~3 cm 位置、±3° fovy、±2° yaw、±2° pitch 的抖动。
名义位姿已经改成"复刻真机 _fixed 视角"（垫子右缘压在右边框、下缘被切），
但抖动会让垫子有时整块进画面 —— 这个脚本就是量"抖完还剩多少帧是贴着边框的"。

用法：
    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
        /usr/bin/python3 tools/probe_env_cam_jitter.py [--n 12] [--seed0 1000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "collect"))

W, H = 640, 480
GREEN_LO, GREEN_HI = (35, 40, 30), (95, 255, 255)


def pad_bbox(bgr):
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _l, st, _c = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x0, y0, bw, bh, _a = st[k]
    t = "".join(nm for nm, cd in (("L", x0 <= 1), ("T", y0 <= 1),
                                  ("R", x0 + bw >= w - 1), ("B", y0 + bh >= h - 1)) if cd)
    return dict(x0=int(x0), y0=int(y0), x1=int(x0 + bw - 1), y1=int(y0 + bh - 1),
                w=int(bw), h=int(bh), touch=t or "-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    import mujoco
    import collect as C

    model = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene.xml"))
    pad = C.PadZone(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    data = mujoco.MjData(model)

    cid = model.cam("env_camera").id
    base = np.array(model.cam_pos[cid], dtype=float).copy()
    base_quat = np.array(model.cam_quat[cid], dtype=float).copy()   # 必须存原始朝向
    base_fovy = float(model.cam_fovy[cid])
    print("env_camera 名义 pos = %s   (from model/so_arm101.xml)" % np.round(base, 4))
    print()
    print("%-5s %-34s %-30s %s" % ("seed", "抖动后 pos", "垫子 bbox w×h", "贴边"))
    stats = []
    for i in range(args.n):
        rng = np.random.default_rng(args.seed0 + i)
        cfg = C.sample_episode(rng, i, pad, C.THETA_MAX_DEG)
        C.apply_camera_jitter(model, cfg)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="env_camera")
        bb = pad_bbox(np.asarray(renderer.render()))
        pos = np.round(model.cam_pos[model.cam("env_camera").id], 4)
        j = cfg["camera"]["env_pos_jitter_m"]
        if bb is None:
            print("%-5d %-34s 未检测到" % (args.seed0 + i, pos))
            continue
        print("%-5d %-34s (%4d,%4d,%4d,%4d) %3dx%-3d   %-4s  jit=%.3fm fovyΔ=%+.1f° yawΔ=%+.1f° pitchΔ=%+.1f°"
              % (args.seed0 + i, pos, bb["x0"], bb["y0"], bb["x1"], bb["y1"], bb["w"], bb["h"],
                 bb["touch"], j, cfg["camera"]["env_fovy_delta_deg"],
                 cfg["camera"]["env_yaw_delta_deg"], cfg["camera"]["env_pitch_delta_deg"]))
        stats.append(bb)
        # 下一轮要还原名义值，否则抖动会累积（cam_quat 必须还原成 XML 的原始朝向，
        # 写 zeros(4) 会让 quat2Mat 拿到非法四元数，相机朝向直接飞掉）
        model.cam_pos[cid] = base
        model.cam_quat[cid] = base_quat
        model.cam_fovy[cid] = base_fovy
        mujoco.mj_forward(model, data)

    renderer.close()
    if not stats:
        print("没有有效样本")
        return 1
    n = len(stats)
    nb = sum(1 for s in stats if "B" in s["touch"])
    nr = sum(1 for s in stats if "R" in s["touch"])
    ninside = sum(1 for s in stats if s["touch"] == "-")
    print()
    print("抖完 %d 帧：贴下边 %d 帧 (%.0f%%) / 贴右边 %d 帧 (%.0f%%) / 完全在画面内 %d 帧 (%.0f%%)"
          % (n, nb, 100 * nb / n, nr, 100 * nr / n, ninside, 100 * ninside / n))
    print("垫子左缘 x0 范围 %d~%d，上缘 y0 范围 %d~%d，宽 %d~%d"
          % (min(s["x0"] for s in stats), max(s["x0"] for s in stats),
             min(s["y0"] for s in stats), max(s["y0"] for s in stats),
             min(s["w"] for s in stats), max(s["w"] for s in stats)))
    print("对照真机 _fixed：贴下边 100%（302/302 条），左缘 370~420，上缘 284~291，宽 205~270")
    return 0


if __name__ == "__main__":
    sys.exit(main())