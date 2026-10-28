#!/usr/bin/env python3
"""Feetech 总线体检：哪些电机在线、哪个读写失败。

背景：rollout 在 configure() 阶段给 id=5(wrist_roll) 写 Lock 时报
"There is no status packet!"，机器人连不上。

用法：
  env -u PYTHONHOME -u PYTHONPATH \
  /home/yyl/miniforge3/envs/lerobot312/bin/python tools/probe_motor_bus.py

  # 指定端口
  ... tools/probe_motor_bus.py --port /dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00

判读要点（Feetech 是菊花链，信号逐个电机转发）：
  1~4 在线、5~6 失联  → 断点在 4→5 之间，或 5 号没电（最常见：排线松/脱落）
  只有 5 失联、6 在线 → 5 号本体异常（少见，但连接器仍可能是元凶）
  全都失联            → 电源没开 / USB 没通
  全部 ping 通但写失败 → 供电不足或总线时序问题
"""
import argparse
import sys

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

DEFAULT_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00"
CALIB_PATH = "/home/yyl/.cache/huggingface/lerobot/calibration/robots/so_follower/yyl_follower_arm.json"

EXPECTED = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baudrate", type=int, default=1_000_000)
    ap.add_argument("--calib", default=CALIB_PATH, help="标定文件（读归一化位置需要）")
    args = ap.parse_args()

    motors = {
        name: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100)
        for name, i in EXPECTED.items()
    }

    from pathlib import Path
    calib = None
    if Path(args.calib).is_file():
        import json
        raw = json.load(open(args.calib))
        calib = {k: {"id": v.get("id", EXPECTED.get(k)), "drive_mode": v.get("drive_mode", 0),
                     "homing_offset": v["homing_offset"],
                     "range_min": v["range_min"], "range_max": v["range_max"]}
                 for k, v in raw.items() if k in EXPECTED}
        print(f"标定: {args.calib}")
    else:
        print(f"⚠️ 标定文件不存在: {args.calib}（读位置会失败，属预期）")

    print(f"端口: {args.port}")
    bus = FeetechMotorsBus(port=args.port, motors=motors, calibration=calib)
    try:
        bus.connect(handshake=False)  # 只开串口，不 ping，避免 connect 里就抛
    except Exception as e:
        print(f"✗ 串口打不开: {type(e).__name__}: {e}")
        return 2
    print("串口已打开\n")

    # 1) 广播 ping：整条总线上有哪些 id 活着
    print("=== 1) 广播 ping ===")
    alive = {}
    try:
        alive = bus.broadcast_ping(num_retry=2, raise_on_error=False) or {}
        if alive:
            print("  有响应的 id:", sorted(alive.keys()))
            for i in sorted(alive):
                print(f"    id={i}  型号号={alive[i]}")
        else:
            print("  ✗ 一个都没响应 —— 先查电源和 USB")
    except Exception as e:
        print(f"  ✗ 广播 ping 异常: {type(e).__name__}: {e}")

    # 2) 逐个电机：ping / 读位置 / 开关扭矩
    print("\n=== 2) 逐个电机 ===")
    print(f"  {'电机':<14}{'id':>3}  {'ping':<8}{'读位置':<22}{'扭矩测试'}")
    rows = []
    for name, mid in EXPECTED.items():
        ping = "—"
        try:
            r = bus.ping(mid, num_retry=2, raise_on_error=False)
            ping = f"ok({r})" if r is not None else "✗ 无响应"
        except Exception as e:
            ping = f"✗{type(e).__name__}"

        pos = "—"
        try:
            v = bus.read("Present_Position", name)
            pos = f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
        except Exception as e:
            pos = f"✗{str(e)[:14]}"

        tq = "—"
        try:
            bus.enable_torque(name, num_retry=2)
            bus.disable_torque(name, num_retry=2)
            tq = "ok"
        except Exception as e:
            tq = f"✗{str(e)[:24]}"

        print(f"  {name:<14}{mid:>3}  {ping:<8}{pos:<22}{tq}")
        rows.append((name, mid, ping, tq))

    # 3) 结论
    print("\n=== 3) 结论 ===")
    bad = [r for r in rows if "✗" in r[2]]
    if not bad:
        print("  六个电机全部 ping 通。若 rollout 仍报 Lock 写失败，优先怀疑供电不足。")
    else:
        ids = [r[1] for r in bad]
        print(f"  失联电机: {[r[0] for r in bad]}  (id={ids})")
        if ids == [5, 6]:
            print("  → 断点在 id=4 与 id=5 之间，或 5 号没上电。先查这一段排线。")
        elif ids == [5]:
            print("  → 只有 5 号失联而 6 号在线：5 号本体或它的连接器有问题。")
        elif ids == [6]:
            print("  → 只有末端 6 号（夹爪）失联：查最末一段线。")
        else:
            print("  → 多个失联，优先查供电与总线入口。")

    try:
        bus.disconnect(disable_torque=True)
    except Exception:
        pass
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())