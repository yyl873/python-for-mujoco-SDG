#!/usr/bin/env bash
# 两阶段：纯仿真预训练 → 真机微调
#
#   Stage 1  纯仿真 sim156（156 集 / 70,824 帧）训 17,700 步（≈2 epoch，约 100 分钟）
#   Stage 2  从 Stage 1 末权重 warm start，用纯真机 real_train_fixed（139 集）
#            微调 8,000 步 / lr=1e-5（约 45 分钟）
#
# 为什么这么配：
#   - 仿真只贡献"抓取/定位"，真机微调负责把"释放"行为掰回真机量程
#   - 上次从混训权重微调 3000 步只把释放率从 38% 提到 56%，所以这次微调给到 8,000 步
#   - 仿真数据不要再加量：腕部约定冲突会随仿真占比加深，微调更难掰回来
#
# 用法：
#   bash tools/train_sim156_then_real.sh pre    # 只跑 Stage 1
#   bash tools/train_sim156_then_real.sh ft     # 只跑 Stage 2（要求 Stage 1 已存在）
#   bash tools/train_sim156_then_real.sh all    # 依次跑（默认）
#
# 判读（Stage 2 跑完拿真机测 12 次）：
#   抓取内释放率 > 56%  → 两阶段路线成立，仿真数据量就定 156 集
#   抓取内释放率 ≤ 56%  → 停止加仿真数据，问题在指垫几何，转去改 model/so_arm101.xml

set -euo pipefail

PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
LOGDIR=/home/yyl/deepseek-harness/logs
SIM_ROOT=/home/yyl/lerobot_datasets/sim156
REAL_ROOT=/home/yyl/lerobot_datasets/real_train_fixed
OUT_PRE=/home/yyl/lerobot/outputs/train/act_sim156_pre
OUT_FT=/home/yyl/lerobot/outputs/train/act_sim156_pre_ft
PRE_STEPS=17700
FT_STEPS=8000
FT_LR=1e-5
BATCH=8

STAGE="${1:-all}"
mkdir -p "$LOGDIR"

gpu_check() {
  echo ">>> GPU 自检（必须 True，否则会退化到 CPU）"
  env -u PYTHONHOME -u PYTHONPATH "$PY" -c "
import torch
ok = torch.cuda.is_available()
print('  cuda available:', ok)
if ok: print('  device:', torch.cuda.get_device_name(0))
else: raise SystemExit('  ✗ GPU 不可用，先重启机器再跑')
"
}

run_pre() {
  echo
  echo "======== Stage 1：纯仿真预训练 ========"
  echo "  数据: $SIM_ROOT  (156 集 / 70,824 帧)"
  echo "  步数: $PRE_STEPS   输出: $OUT_PRE"
  cd /home/yyl/lerobot
  nohup env -u PYTHONHOME -u PYTHONPATH "$PY" -m lerobot.scripts.lerobot_train \
    --policy.type=act \
    --policy.repo_id=local/act_sim156_pre \
    --policy.push_to_hub=false \
    --policy.device=cuda \
    --dataset.repo_id=local/sim156 \
    --dataset.root="$SIM_ROOT" \
    --steps="$PRE_STEPS" \
    --batch_size="$BATCH" \
    --save_freq=5000 \
    --output_dir="$OUT_PRE" \
    > "$LOGDIR/train_sim156_pre.log" 2>&1 &
  echo "  已启动 PID=$!  日志: $LOGDIR/train_sim156_pre.log"
  echo "  等它跑完（约 100 分钟）再执行: bash tools/train_sim156_then_real.sh ft"
}

run_ft() {
  echo
  echo "======== Stage 2：真机微调 ========"
  if [ ! -f "$OUT_PRE/checkpoints/last/pretrained_model/model.safetensors" ]; then
    echo "  ✗ 找不到 Stage 1 产物: $OUT_PRE/checkpoints/last/pretrained_model"
    echo "    先跑: bash tools/train_sim156_then_real.sh pre"
    exit 1
  fi
  echo "  起点: $OUT_PRE/checkpoints/last/pretrained_model"
  echo "  数据: $REAL_ROOT  (139 集 / 62,273 帧)"
  echo "  步数: $FT_STEPS   lr=$FT_LR   输出: $OUT_FT"
  cd /home/yyl/lerobot
  nohup env -u PYTHONHOME -u PYTHONPATH "$PY" -m lerobot.scripts.lerobot_train \
    --policy.path="$OUT_PRE/checkpoints/last/pretrained_model" \
    --dataset.repo_id=local/real_train_fixed \
    --dataset.root="$REAL_ROOT" \
    --steps="$FT_STEPS" \
    --batch_size="$BATCH" \
    --save_freq=2000 \
    --policy.optimizer_lr="$FT_LR" \
    --output_dir="$OUT_FT" \
    > "$LOGDIR/train_sim156_ft.log" 2>&1 &
  echo "  已启动 PID=$!  日志: $LOGDIR/train_sim156_ft.log"
  echo
  echo "  跑完拿这个权重去真机测 12 次："
  echo "    --policy.path=$OUT_FT/checkpoints/${FT_STEPS}/pretrained_model"
}

gpu_check

case "$STAGE" in
  pre) run_pre ;;
  ft)  run_ft ;;
  all) run_pre; echo; echo "⚠️  Stage 1 是后台任务，本脚本不会等它。"; \
       echo "   等 Stage 1 跑完后再执行: bash tools/train_sim156_then_real.sh ft" ;;
  *) echo "用法: bash tools/train_sim156_then_real.sh [pre|ft|all]"; exit 2 ;;
esac

echo
echo ">>> 看进度: tail -c 400 $LOGDIR/train_sim156_pre.log   （或 _ft.log）"