#!/bin/bash
# armB 续训守护：崩了自动从最新 checkpoint 续，最多 20 次
set -u
cd /home/yyl/deepseek-harness
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
LOG=logs/train_v8_B_resume.log

for i in $(seq 1 20); do
    STEP=$(cat /home/yyl/lerobot/outputs/train/act_armB/checkpoints/last/training_state/training_step.json 2>/dev/null | grep -oE '"step":[ ]*[0-9]+' | grep -oE '[0-9]+')
    echo "=== 第 $i 次续训，当前 step=${STEP:-unknown} $(date) ===" >> "$LOG"
    if [ -n "${STEP:-}" ] && [ "$STEP" -ge 22500 ]; then
        echo "=== 已达 22500 步，结束 ===" >> "$LOG"
        exit 0
    fi
    env -u PYTHONHOME -u PYTHONPATH "$PY" tools/train_mix.py \
        --root /home/yyl/lerobot_datasets/armB --repo-id local/armB \
        --output /home/yyl/lerobot/outputs/train/act_armB --job-name act_armB \
        --policy-repo-id local/act_armB \
        --steps 22500 --batch-size 8 --num-workers 4 --save-freq 2500 \
        --resume >> "$LOG" 2>&1
    rc=$?
    echo "=== 退出码 $rc $(date) ===" >> "$LOG"
    [ $rc -eq 0 ] && exit 0
    # GPU 掉驱动就先等驱动恢复，最多等 10 分钟
    for w in $(seq 1 20); do
        if nvidia-smi >/dev/null 2>&1; then break; fi
        echo "  GPU 不可用，等 30s（$w/20）" >> "$LOG"
        sleep 30
    done
    sleep 10
done
echo "=== 超过 20 次重试，放弃 ===" >> "$LOG"
exit 1
