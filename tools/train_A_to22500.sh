#!/usr/bin/env bash
# 把 A（纯真机）从 17500 步补到 22500 步，用于和 B@22500 做同步数对照
#
# 目的：补齐唯一缺的那一格
#   A@17500  = 4/12   （纯真机）
#   A@22500  = ?      ← 本脚本产出
#   B@22500  = 9/12   （真机+仿真）
#
# 判读：
#   A@22500 若也到 9/12  → 抓取靠训练量，仿真无增益
#   A@22500 若仍 4~5/12  → 仿真确有增益，结论成立
#
# 用法：bash tools/train_A_to22500.sh
# 预计：5000 步 ÷ 2 步/秒 ≈ 42 分钟（GPU 正常时）

set -euo pipefail

PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
LOG=/home/yyl/deepseek-harness/logs/train_A_ext22500.log
CKPT=/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500/pretrained_model

mkdir -p "$(dirname "$LOG")"

echo ">>> GPU 自检（必须是 True，否则会退化成 CPU，几小时都跑不完）"
env -u PYTHONHOME -u PYTHONPATH "$PY" -c "
import torch
ok = torch.cuda.is_available()
print('  cuda available:', ok)
if ok: print('  device:', torch.cuda.get_device_name(0))
else: raise SystemExit('  ✗ GPU 不可用，先重启机器再跑')
"

echo ">>> 起点权重: $CKPT"
ls -1 "$CKPT"/model.safetensors
echo ">>> 日志: $LOG"
echo
echo ">>> 后台启动（机器有过 kernel BUG/shmem 崩溃记录，所以用 nohup 防止终端断开）"

cd /home/yyl/lerobot
nohup env -u PYTHONHOME -u PYTHONPATH "$PY" -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path="$CKPT" \
  --steps=22500 \
  > "$LOG" 2>&1 &

PID=$!
echo "  PID = $PID"
echo
echo ">>> 看进度："
echo "    tail -f $LOG"
echo "    grep -oE 'step[: ]+[0-9]+' $LOG | tail -3"
echo
echo ">>> 跑完的产物应该是："
echo "    /home/yyl/lerobot/outputs/train/act_armA/checkpoints/022500/pretrained_model"
echo "    /home/yyl/lerobot/outputs/train/act_armA/checkpoints/020000/pretrained_model"
echo
echo ">>> 然后拿 022500 去跑 12 次九宫格（--policy.path 换成上面那个路径）"