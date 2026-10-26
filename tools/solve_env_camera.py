#!/usr/bin/env python3
"""解 env_camera 的位姿，让仿真复刻真机 _fixed 批次的环境相机取景。

目标值来自真机实测（44 个文件 / 302 条实录的绿色掩码时序并集，见
docs/REAL_DATASET_CAMERA_AUDIT.md §9）：
    垫子像素直径     270 px
    垫子左缘 x0      370 px
    垫子右缘 x1      639 px（正好压在右边框）
    垫子上缘 y0      284~291 px（取 286）
    垫子下缘         超出画面底边 -> 可见高度 480-286 = 194 px（真机实测 189~196）
图像 640x480，相机 env_camera，保持 xyaxes 与 fovy 不变，只解 pos（3 个自由度）。

解算依据：针孔模型（fovy=30 固定 => f_px = 240/tan(15°)），
    w_px ∝ 1/深度        -> 深度用像素宽度反解
    u,v 由相机系坐标线性决定 -> 位置用反投影求出
迭代 3~4 次，每次重渲染并用与真机完全相同的绿色检测器复测，收敛到 ±2 px。

用法
----
    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
        /usr/bin/python3 tools/solve_env_camera.py            # 只解，不写文件
    ... --write                                               # 解完直接改写 model/so_arm101.xml
    ... --target-u 504.5 --target-top 286 --target-w 270       # 换目标
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
XML = ROOT / "model" / "so_arm101.xml"
W, H = 640, 480

# 真机 _fixed 批次实测目标
TARGET_U_LEFT = 370.0      # 垫子左缘
TARGET_U_RIGHT = 639.0     # 垫子右缘（压在右边框）
TARGET_V_TOP = 286.0       # 垫子上缘
GREEN_LO, GREEN_HI = (35, 40, 30), (95, 255, 255)


def pad_bbox(bgr: np.ndarray) -> dict:
    """与真机测量完全一致的绿色检测：最大连通域。"""
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_LO, GREEN_HI)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, _lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return {"x0": np.nan, "y0": np.nan, "x1": np.nan, "y1": np.nan,
                "w": np.nan, "h": np.nan, "cx": np.nan, "cy": np.nan}
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x0, y0, bw, bh, _a = st[k]
    return {"x0": float(x0), "y0": float(y0), "x1": float(x0 + bw - 1), "y1": float(y0 + bh - 1),
            "w": float(bw), "h": float(bh), "cx": float(cen[k][0]), "cy": float(cen[k][1])}


def parse_camera(xml: Path) -> tuple[np.ndarray, np.ndarray, float]:
    txt = xml.read_text()
    m = re.search(r'<camera name="env_camera"[^>]*/>', txt)
    if not m:
        raise SystemExit("在 %s 里找不到 env_camera" % xml)
    tag = m.group(0)
    pos = np.array([float(v) for v in re.search(r'pos="([^"]+)"', tag).group(1).split()])
    xa = np.array([float(v) for v in re.search(r'xyaxes="([^"]+)"', tag).group(1).split()])
    fovy = float(re.search(r'fovy="([^"]+)"', tag).group(1))
    return pos, xa, fovy


def cam_axes(xa: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MuJoCo 相机坐标系：x/y 由 xyaxes 给出，z = x×y，视线沿 -z。"""
    ex = xa[:3] / np.linalg.norm(xa[:3])
    ey = xa[3:] / np.linalg.norm(xa[3:])
    ez = np.cross(ex, ey)
    return ex, ey, ez


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-u-left", type=float, default=TARGET_U_LEFT)
    ap.add_argument("--target-u-right", type=float, default=TARGET_U_RIGHT)
    ap.add_argument("--target-top", type=float, default=TARGET_V_TOP)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--write", action="store_true", help="改写 model/so_arm101.xml")
    ap.add_argument("--debug-png", default="inspect/env_cam_solve/final_env.png")
    args = ap.parse_args()

    import mujoco

    target_w = args.target_u_right - args.target_u_left + 1.0     # 270 px
    target_uc = 0.5 * (args.target_u_left + args.target_u_right)

    m = mujoco.MjModel.from_xml_path(str(ROOT / "model" / "scene.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pad_id = m.body("pick_zone").id
    pad = d.xpos[pad_id].copy()

    xml_pos, xa, fovy = parse_camera(XML)
    ex, ey, ez = cam_axes(xa)
    f_px = (H / 2.0) / math.tan(math.radians(fovy) / 2.0)

    cam_id = m.camera("env_camera").id
    renderer = mujoco.Renderer(m, height=H, width=W)
    pos = xml_pos.copy()

    def render_and_measure(p: np.ndarray) -> dict:
        m.cam_pos[cam_id] = p
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera="env_camera")
        img = renderer.render()
        return pad_bbox(np.asarray(img))

    print("模型里 env_camera pos = %s  xyaxes = %s  fovy = %.1f" % (tuple(xml_pos), tuple(xa), fovy))
    print("垫子世界坐标 = %s   f_px = %.2f" % (tuple(np.round(pad, 4)), f_px))
    print("目标：左缘 %.1f  右缘 %.1f  上缘 %.1f  -> 宽 %.1f px，中心 u=%.1f，可见高 %.1f px"
          % (args.target_u_left, args.target_u_right, args.target_top, target_w,
             target_uc, H - args.target_top))
    print()
    print("%-3s %-34s %-34s %s" % ("it", "相机 pos", "实测 (x0,y0,x1,y1) w×h", "误差(左缘/上缘/宽)"))

    meas = None
    for it in range(args.iters):
        meas = render_and_measure(pos)
        e_left = meas["x0"] - args.target_u_left
        e_top = meas["y0"] - args.target_top
        e_w = meas["w"] - target_w
        print("%-3d %-34s %-34s 左%+7.1f 上%+7.1f 宽%+7.1f"
              % (it, np.round(pos, 4),
                 "(%4.0f,%4.0f,%4.0f,%4.0f) %3.0fx%3.0f"
                 % (meas["x0"], meas["y0"], meas["x1"], meas["y1"], meas["w"], meas["h"]),
                 e_left, e_top, e_w))
        if abs(e_left) <= 2 and abs(e_top) <= 2 and abs(e_w) <= 2:
            print("收敛。")
            break

        # 1) 用像素宽度反解深度：w ∝ 1/d
        depth = float(np.dot(pad - pos, -ez))
        depth_new = depth * meas["w"] / target_w
        # 2) 椭圆半高随深度同比例缩放
        half_h = 0.5 * meas["h"] * depth_new / depth
        # 3) 目标中心像素 -> 相机系坐标 -> 世界系里的相机位置
        v_center = args.target_top + half_h
        u_center = target_uc
        x_c = (u_center - W / 2.0) * depth_new / f_px
        y_c = -(v_center - H / 2.0) * depth_new / f_px
        z_c = -depth_new
        pos = pad - (x_c * ex + y_c * ey + z_c * ez)

    out = Path(args.debug_png)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    meas = render_and_measure(pos)
    renderer.update_scene(d, camera="env_camera")
    cv2.imwrite(str(out), cv2.cvtColor(np.asarray(renderer.render()), cv2.COLOR_RGB2BGR))
    renderer.close()

    print()
    print("解出 pos = \"%.4f %.4f %.4f\"" % tuple(pos))
    print("最终实测: 左缘 %.0f  右缘 %.0f  上缘 %.0f  宽 %.0f 高 %.0f  可见高 %.0f px"
          % (meas["x0"], meas["x1"], meas["y0"], meas["w"], meas["h"], H - meas["y0"]))
    ok = (abs(meas["x0"] - args.target_u_left) <= 3 and abs(meas["y0"] - args.target_top) <= 3
          and abs(meas["w"] - target_w) <= 3)
    print("判定: %s" % ("ok 命中真机目标" if ok else "warn 未完全命中，看上面的误差列"))
    print("渲染图:", out)

    if args.write:
        txt = XML.read_text()
        new = re.sub(r'(<camera name="env_camera"[^>]*?)pos="[^"]*"',
                     lambda mo: mo.group(1) + 'pos="%.4f %.4f %.4f"' % tuple(pos), txt, count=1)
        if new == txt:
            print("!! 改写失败：没匹配到 pos")
            return 1
        XML.write_text(new)
        print("已写入 %s" % XML)
    else:
        print("（未写文件；确认无误后加 --write）")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())