#!/usr/bin/env python3
"""包装 lerobot-rollout，记录夹爪命令 + 实际位置 + 全部关节角。

为什么要记全部关节角：
  释放诊断要判断「A/B 是不是在同一个位姿张开夹爪」，只记夹爪列无法回答。
  一次 sync_read("Present_Position") 即可拿到全部 6 个电机，成本与只读夹爪相同。

用法：
  GRIPPER_LOG=logs/gripper_B.csv \
  env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
    tools/log_gripper_rollout.py <原来的 lerobot-rollout 参数...>

输出列：
  t_s, gripper_req, gripper_sent, present_before, present_after,
  pan_before, lift_before, elbow_before, flex_before, roll_before
  （*_before = 下发本条动作瞬间的实际关节角，也就是"张开/合上时的末端位姿"）
"""
import csv
import os
import sys
import time
from pathlib import Path

from lerobot.robots.so_follower.so_follower import SOFollower
import lerobot.scripts.lerobot_rollout as lr

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

out = Path(os.environ.get("GRIPPER_LOG", "logs/gripper_rollout.csv"))
out.parent.mkdir(parents=True, exist_ok=True)
f = open(out, "w", newline="")
w = csv.writer(f)
w.writerow(["t_s", "gripper_req", "gripper_sent", "present_before", "present_after"]
           + [f"{k}_before" for k in ARM])
t0 = time.time()

orig_send = SOFollower.send_action


def read_pos(robot):
    """一次读出全部 6 个电机位置；失败返回空 dict，绝不让 rollout 崩掉。"""
    try:
        return robot.bus.sync_read("Present_Position", num_retry=2) or {}
    except Exception:
        return {}


def send(self, action):
    req = action.get("gripper.pos")
    before = read_pos(self)
    sent = orig_send(self, action)
    after = read_pos(self)
    w.writerow(
        [round(time.time() - t0, 4), req, sent.get("gripper.pos"),
         before.get("gripper"), after.get("gripper")]
        + [before.get(k) for k in ARM]
    )
    f.flush()
    return sent


SOFollower.send_action = send

try:
    sys.argv = ["lerobot-rollout"] + sys.argv[1:]
    lr.main()
finally:
    f.close()