#!/usr/bin/env bash
# 同一数据集、同一评测口径下比较 ACT / Diffusion Policy / VQ-BeT
#
# 为什么这么比较才公平：三者吃**同一份** v2_pair 数据（state 8 维 = 6 关节角 + 2 指 one-hot），
# 差别只在"动作是怎么产生出来的"（回归 / 去噪 / 离散码本），所以同一个反事实评测
# （rl/run_lang_eval.sh）对三者都适用，指标可以直接并排。
#
# 用法（★ 需要能看见 /dev/nvidia* 的终端；本机沙箱看不到显卡）：
#   POLICY=diffusion STEPS=5000 bash rl/run_bc_compare.sh
#   POLICY=vqbet     STEPS=5000 bash rl/run_bc_compare.sh
#   POLICY=act       STEPS=5000 bash rl/run_bc_compare.sh   # 对照组（已有 005000，可跳过）
#
# 8GB 显存 / DP 的推荐跑法（先缩图，否则 batch8 OOM）：
#   POLICY=diffusion STEPS=3000 EXTRA='--policy.resize_shape "[240,320]" --policy.crop_ratio 0.9' \
#     bash rl/run_bc_compare.sh
# 显存还是不够就把 BATCH=4（注意：改 batch 会削弱与 ACT@8 的可比性，要在结论里注明）
#
# 跑完评测：
#   CKPT=$PWD/outputs/bc_diffusion/checkpoints/last/pretrained_model \
#     DS=$PWD/rl/lang_ds/v2_pair OUT=rl/eval/lang_bc_diffusion.json bash rl/run_lang_eval.sh
#   → 再跑 make tables 之前，把新 json 加进 tools/make_tables.py 的 LANG_RUNS
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS="${OMP:-8}"

POLICY="${POLICY:-diffusion}"
STEPS="${STEPS:-5000}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-250}"
DEVICE="${DEVICE:-cuda}"
DS="${DS:-$PWD/rl/lang_ds/v2_pair}"
OUT="${OUT:-$PWD/outputs/bc_${POLICY}}"
BASE="${BASE:-}"          # 空=从头训（DP/VQ-BeT 的正常用法）；ACT 想续用微调基座就指过去
# 额外透传参数（DP 在 8GB 显存上必须先缩图，否则显存爆）：
#   EXTRA='--policy.resize_shape "[240,320]" --policy.crop_ratio 0.9'
EXTRA="${EXTRA:-}"
# 推理步数只影响评测速度，不影响训练（DP 默认 100 步很慢，评测前降到 10）
NINF="${NINF:-}"
PY="/home/yyl/miniforge3/envs/lerobot312/bin/lerobot-train"

[ -e "$DS" ] || { echo "缺数据集: $DS" >&2; exit 2; }
[ -x "$PY" ] || { echo "缺 lerobot-train: $PY" >&2; exit 2; }

# 显卡闸门：lerobot 对 --policy.device=cuda 只会打一行 WARNING 就切回 cpu，必须先自己问
CUDA_OK=$(env -u PYTHONHOME -u PYTHONPATH "$(dirname "$PY")/python" -c "
import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null | tail -1)
case "$DEVICE" in
  cpu|CPU) : ;;
  *) [ "$CUDA_OK" = "yes" ] || {
       echo "✗ DEVICE=$DEVICE 但 torch.cuda.is_available()=${CUDA_OK:-unknown}" >&2
       echo "  换到能看见显卡的终端；或设 DEVICE=cpu 接受慢跑（DP 在 CPU 上非常慢）" >&2; exit 3; } ;;
esac

echo "=== 策略对照：$POLICY ==="
echo "  数据  : $DS"
echo "  步数  : $STEPS（batch $BATCH, save_freq $SAVE_FREQ）"
echo "  设备  : $DEVICE"
echo "  输出  : $OUT"
echo "  基座  : ${BASE:-（无，从头训）}"
echo

mkdir -p "$OUT"
LOG="$OUT/train.log"
env -u PYTHONHOME -u PYTHONPATH "$PY" \
  --dataset.repo_id="local/v2_pair" \
  --dataset.root="$DS" \
  --policy.type="$POLICY" \
  --policy.repo_id="local/bc_${POLICY}" \
  ${BASE:+--policy.pretrained_path="$BASE"} \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false \
  --policy.device="$DEVICE" \
  --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
  --num_workers="$WORKERS" --seed=1000 \
  ${NINF:+--policy.num_inference_steps="$NINF"} \
  --output_dir="$OUT" $EXTRA 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
if grep -qiE "Switching to 'cpu'" "$LOG"; then
  echo "  ⚠️ 日志里出现 Switching to 'cpu' —— 这次实际跑在 CPU 上，别当 GPU 结果用"
fi
echo "=== rc=$rc  输出=$OUT ==="
ls "$OUT/checkpoints" 2>/dev/null | tail -3
exit "$rc"
