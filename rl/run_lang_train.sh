#!/usr/bin/env bash
# 语言条件 ACT 微调：基座 act_sim156_pre（6 维 state）-> 加宽到 8 维 -> 在双指令数据上微调
#
# 前置：
#   1) rl/lang_ds/v1 已导出（92 条 / 41922 帧 / state 8 维 / 两个 instruction）
#   2) rl/widen_state.py 已生成 /home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model
#      （state 的 6->8 是**补 0 列**，所以第 0 步的模型与被加宽前的基座逐位一致）
#
# ★ 必须在能看到 /dev/nvidia* 的权限下跑（沙箱默认看不到显卡，lerobot 会静默降级到 CPU，
#   batch 8 在 CPU 上约 4~5 s/step，5000 步要 6 小时）。GPU 上实测 ~0.2 s/step。
#
# 用法：
#   STEPS=5000 SEED=1000 bash rl/run_lang_train.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8
PY="/home/yyl/miniforge3/envs/lerobot312/bin/lerobot-train"

STEPS="${STEPS:-5000}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
OUT="${OUT:-/home/yyl/lerobot/outputs/train/act_lang_dual}"
DS_ROOT="${DS_ROOT:-$PWD/rl/lang_ds/v1}"
BASE="${BASE:-/home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model}"

for p in "$DS_ROOT" "$BASE"; do
  [ -e "$p" ] || { echo "缺依赖: $p" >&2; exit 2; }
done

echo "=== 语言条件 ACT 微调 ==="
echo "  数据集: $DS_ROOT"
echo "  基座  : $BASE"
echo "  步数  : $STEPS（batch $BATCH, seed $SEED）"
echo "  输出  : $OUT"
echo

env -u PYTHONHOME -u PYTHONPATH "$PY" \
  --dataset.repo_id=local/v1 \
  --dataset.root="$DS_ROOT" \
  --policy.type=act \
  --policy.repo_id=local/act_lang_dual \
  --policy.pretrained_path="$BASE" \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="${SAVE_FREQ:-1000}" \
  --output_dir="$OUT" --seed="$SEED" --num_workers=4
rc=$?
echo
echo "TRAIN_DONE rc=$rc  output=$OUT"
exit $rc