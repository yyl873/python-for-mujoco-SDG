#!/bin/bash
# A/B holdout 离线评测（CPU 版，不依赖 GPU/桌面）
set -u
cd /home/yyl/deepseek-harness || exit 1

PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
mkdir -p logs .cache/hf_datasets .cache/hf_home

export HF_DATASETS_CACHE=/home/yyl/deepseek-harness/.cache/hf_datasets
export HF_HOME=/home/yyl/deepseek-harness/.cache/hf_home
export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/home/yyl/.cache/pycache_eval

for pair in "armA:17500" "armB:22500"; do
    arm=${pair%%:*}
    step=${pair##*:}
    step6=$(printf '%06d' "$step")
    echo
    echo "########## $arm step=$step $(date) ##########"
    env -u PYTHONHOME -u PYTHONPATH "$PY" tools/eval_act_holdout.py \
        --ckpt "/home/yyl/lerobot/outputs/train/act_${arm}/checkpoints/${step6}/pretrained_model" \
        --device cpu --sample-every 10 --batch-size 8 \
        --out "logs/eval_holdout_${arm}.json"
    echo "退出码: $?"
done
echo
echo "###### 全部结束 $(date) ######"
