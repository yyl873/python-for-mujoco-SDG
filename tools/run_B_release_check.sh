#!/usr/bin/env bash
# B 释放诊断：恢复训练模式跑一次（关闭 temporal ensembling）
#
# 关键点：不要传 --policy.temporal_ensemble_coeff / --policy.n_action_steps。
#   checkpoint 的 config.json 自带 temporal_ensemble_coeff=None / n_action_steps=100，
#   不传即用训练时的执行方式。
#   注意 --policy.temporal_ensemble_coeff=None 会被 draccus 拒（None 转不了 float）。
#
# 用法：  bash tools/run_B_release_check.sh
# 换 A： 把下面 POLICY 一行改成 act_armA/checkpoints/017500/pretrained_model
# 换位置：直接在你终端里把机械臂摆到目标位置再跑，脚本不控制初始位姿。

set -euo pipefail

PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
LOGGER=/home/yyl/deepseek-harness/tools/log_gripper_rollout.py
POLICY=/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model
export GRIPPER_LOG=/home/yyl/deepseek-harness/logs/gripper_B_noens.csv

cd /home/yyl/lerobot

echo ">>> 设备自检"
ls -l /dev/serial/by-id/ 2>/dev/null | head -3 || echo "  ⚠️ 找不到 /dev/serial/by-id"
ls -l /dev/v4l/by-id/ 2>/dev/null | grep -c . | xargs -I{} echo "  摄像头 by-id 条目: {}"
echo ">>> 策略: $POLICY"
echo ">>> 日志: $GRIPPER_LOG"
echo ">>> 执行模式: 训练模式（temporal_ensemble_coeff=None, n_action_steps=100）"
echo

env -u PYTHONHOME -u PYTHONPATH "$PY" "$LOGGER" \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00 \
  --robot.id=yyl_follower_arm \
  --robot.cameras='{env: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0, width: 640, height: 480, fps: 60, fourcc: MJPG}}' \
  --policy.path="$POLICY" \
  --task="Pick up the medicine box and place it into the wooden box." \
  --duration=60 \
  --fps=30 \
  --display_data=false

echo
echo ">>> 跑完了。看这行判断夹爪有没有在木盒上方张开："
echo "    column -s, -t $GRIPPER_LOG | tail -60"
echo ">>> 或者把 CSV 给 AI 分析："
echo "    $PY /home/yyl/deepseek-harness/tools/compare_release_ab.py <另一份.csv> $GRIPPER_LOG"