#!/usr/bin/env python3
"""rl/view_policy.py —— 在 MuJoCo 实时窗口里看策略跑（基座 / 微调 / 任意 ckpt）

为什么单独写：`rl/eval_ab.py` 只出数字和 mp4，看不了"现场"。这个脚本把
`env_residual.ResidualSO101`（物理 + ACT 推理 + 相机渲染）接到 `mujoco.viewer`，
逐步真实时间推进，终端实时打印抬升/穿透/倾角/干净成功判定。

用法（**必须在有显示器的终端里跑**，沙箱里没有 DISPLAY）
-------------------------------------------------------
```bash
cd /home/yyl/deepseek-harness

# 看微调后的模型（10000 步 ckpt），held-out 的 ep103（基座失手、微调修好的例子）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=glfw LP_NUM_THREADS=4 .venv-rl/bin/python \
    rl/view_policy.py --ckpt /home/yyl/lerobot/outputs/train/act_dagger_ft_1000/checkpoints/010000/pretrained_model \
    --episode 103 --speed 1.0

# 看基座（sim156）同一条 episode，做成对照
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=glfw LP_NUM_THREADS=4 .venv-rl/bin/python \
    rl/view_policy.py --ckpt sim156 --episode 103

# 看微调把基座弄坏的例子
... --ckpt <上面那个 ckpt> --episode 105
... --ckpt sim156 --episode 105
```

参数
----
- `--ckpt`：`sim156` / `lance` / `armA` / `armB` 等名字（见 `rl/base_policy.py` 的 CKPTS），
  或者直接给 `pretrained_model` 目录路径
- `--episode`：母本 episode 编号（held-out 评测集是 100-129；训练池 0-99 / 130-179）
- `--seed`：传感器噪声 seed（评测里的 seed 0/1/2）
- `--speed`：播放倍速（1.0 = 真实 30 Hz；0.5 = 慢放；2.0 = 快放）
- `--loop`：跑完自动重来
- `--camera`：默认自由视角；给 `env_camera` / `wrist_camera` 则贴到对应相机视角

★ 顺带说明：`MUJOCO_GL=glfw` 是必须的 —— viewer 要开真窗口，而 egl 只能离屏渲染。
  两条相机渲染（ACT 的观测）与窗口共用 glfw 后端，实测在同一进程里没问题。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# ★ 必须在 import mujoco 之前定后端：viewer 需要真窗口
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplcache"))

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MuJoCo 实时窗口里看策略跑",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ckpt", default="sim156", help="CKPTS 名字或 pretrained_model 目录")
    ap.add_argument("--episode", type=int, default=103, help="母本 episode 编号（默认 103）")
    ap.add_argument("--seed", type=int, default=0, help="传感器噪声 seed（评测口径 0/1/2）")
    ap.add_argument("--speed", type=float, default=1.0, help="播放倍速（1.0 = 30 Hz 实时）")
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--loop", action="store_true", help="跑完自动重来")
    ap.add_argument("--camera", default=None, help="env_camera / wrist_camera；不给=自由视角")
    ap.add_argument("--device", default="auto", help="auto（有 CUDA 就用 GPU）/ cpu / cuda")
    ap.add_argument("--swap-cams", action="store_true")
    ap.add_argument("--no-sensor", action="store_true", help="关掉 SensorModel（看原始渲染）")
    a = ap.parse_args(argv)

    import mujoco
    import mujoco.viewer

    from env_residual import ResidualSO101                     # noqa: PLC0415

    env = ResidualSO101(base_name=a.ckpt, device=a.device, episode_pool=(a.episode,),
                        swap_cams=a.swap_cams, use_sensor=not a.no_sensor,
                        max_seconds=a.max_seconds, judge="strict", reward_version="v2b")
    dt = 1.0 / env.control_hz

    cam = None
    if a.camera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cid = mujoco.mj_name2id(env.m, mujoco.mjtObj.mjOBJ_CAMERA, a.camera)
        if cid < 0:
            print(f"  ⚠️ 没有相机 {a.camera}，改用自由视角")
            cam = None
        else:
            cam.fixedcamid = cid

    print("=" * 70)
    print(f" view_policy  ckpt={a.ckpt}")
    print(f" episode={a.episode}  seed={a.seed}  speed={a.speed}x  max={a.max_seconds}s"
          f"  device={env.base.policy.device}")
    print(" 关窗口或 Ctrl-C 退出；--loop 会重来")
    print("=" * 70)

    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("  ❌ 没有 DISPLAY/WAYLAND_DISPLAY：这个脚本要开真窗口。")
        print("     在你自己桌面的终端里跑；只想看录像就用 rl/eval/videos/ 里的 mp4。")
        return 2
    try:
        viewer_ctx = mujoco.viewer.launch_passive(env.m, env.d)
    except Exception as exc:                                    # noqa: BLE001
        print(f"  ❌ 打不开 MuJoCo 窗口（{type(exc).__name__}: {exc}）")
        print("     常见原因：没有图形界面 / MUJOCO_GL 不是 glfw / 显卡驱动不支持 GLFW。")
        return 2

    ep_no = 0
    try:
        with viewer_ctx as viewer:
            if cam is not None:
                viewer.cam.type = cam.type
                viewer.cam.fixedcamid = cam.fixedcamid
            while True:
                ep_no += 1
                obs, info = env.reset(seed=a.seed)
                task = info.get("task", "")
                print(f"\n--- 第 {ep_no} 条  ep{a.episode} seed{a.seed}  task={task!r}")
                step = 0
                while viewer.is_running():
                    t0 = time.time()
                    obs, r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
                    step += 1
                    viewer.sync()
                    if step % 10 == 0 or term or trunc:
                        txt = (f"ep{a.episode} step {step:>3}  lift {info['lift_mm']:7.2f} mm  "
                               f"pen {info['penetration_mm']:5.2f} mm  tilt {info['box_tilt_deg']:5.1f}°  "
                               f"contact {info['contact_n']}  clean={int(info['clean'])}")
                        print(("  " + txt + ("   ✅SUCCESS" if info["success"] else "")))
                        try:
                            viewer.set_texts(((1, 1, txt),)) if hasattr(viewer, "set_texts") else None
                        except Exception:                       # noqa: BLE001
                            pass
                    if term or trunc:
                        res = ("✅ 成功" if info["success"] else
                               ("❌ 失败(" + ("掉桌/出区" if (info["fell"] or info["off_table"]
                                                        or info["out_of_area"]) else "超时") + ")")
                               if (info["fell"] or info["off_table"] or info["out_of_area"]
                                   or trunc) else "结束")
                        print(f"  === {res}  steps={step}  lift_max={info['max_lift_mm']:.1f} mm  "
                              f"return={info.get('ep_end_return', 0.0):.1f}")
                        break
                    # 真实时间推进（env.step 已按 1/30 s 步进物理，这里只补睡眠）
                    el = time.time() - t0
                    sleep = dt / max(1e-6, a.speed) - el
                    if sleep > 0:
                        time.sleep(sleep)
                if not a.loop or not viewer.is_running():
                    break
    except KeyboardInterrupt:
        print("\n  (Ctrl-C 退出)")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())