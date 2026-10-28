#!/usr/bin/env python3
"""带载夹爪测试：不使用 input()，用定时窗口让你把药盒放进夹爪。

流程：
  1. 张开夹爪 100
  2. 给你 10 秒把药盒放进夹爪
  3. 闭合到 0，读 present
  4. 再张到 100，读 present
  5. 判断带载能否张开

注意：机械臂放在桌面上方，盒子可能掉落。
"""
import argparse
import time

from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00")
    ap.add_argument("--id", default="yyl_follower_arm")
    ap.add_argument("--place-seconds", type=int, default=10)
    args = ap.parse_args()

    cfg = SO101FollowerConfig(port=args.port, id=args.id, cameras={})
    robot = SOFollower(cfg)
    robot.connect()

    def gp():
        return robot.get_observation().get("gripper.pos")

    try:
        print("initial:", gp())

        robot.send_action({"gripper.pos": 100})
        time.sleep(1)
        print("unloaded open:", gp())

        print(f"\n>>> 请在 {args.place_seconds} 秒内把药盒放进夹爪，盒子放稳后不要碰 <<<")
        for i in range(args.place_seconds, 0, -1):
            print(f"    {i}", flush=True)
            time.sleep(1)

        robot.send_action({"gripper.pos": 0})
        time.sleep(2)
        print("closed on box:", gp())

        print("\n>>> 现在测试带载张开 <<<")
        robot.send_action({"gripper.pos": 100})
        time.sleep(2)
        loaded = gp()
        print("loaded open:", loaded)

        if loaded is not None and loaded >= 80:
            print("RESULT: 带载可张开（>=80）")
        elif loaded is not None and loaded >= 50:
            print("RESULT: 带载部分张开（50~80），需要看盒子是否掉落")
        else:
            print("RESULT: 带载张不开（<50），优先查机械/扭矩")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
