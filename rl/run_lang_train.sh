#!/usr/bin/env bash
# 语言条件 ACT 微调：基座 act_sim156_lang8（state 已加宽 6→8）-> 在双指令数据上微调
#
# 前置：
#   1) rl/lang_ds/v1 已导出（92 条 / 41922 帧 / state 8 维 / 两个 instruction）
#   2) rl/widen_state.py 已生成 /home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model
#      （state 的 6->8 是**补 0 列**，新增维度在权重上的贡献初始为 0）
#
# ★ 必须在能看到 /dev/nvidia* 的终端跑（沙箱默认看不到显卡，lerobot 会静默降级到 CPU：
#   batch 8 在 CPU 上 6.5~8 s/step，5000 步要 9~11 小时）。GPU 实测 ~2.87 step/s。
#
# ★★ 2026-11-21 第一次正式跑的崩溃记录（别重复踩）：
#   GPU 上跑到 **839/5000 步（4:59，2.87 step/s）段错误 rc=139（核心已转储）**，
#   此前 loss 正常（l1 0.109 / kld 0.002），退出时 resource_tracker 报 20 个泄漏信号量
#   → 死在 dataloader worker 侧。而当时 `save_freq` 是 1000，**一步 ckpt 都没落**。
#   头号嫌疑：数据集视频是 **AV1(libdav1d)**，4 个 worker × prefetch 4 并发解码；
#   本机 AV1 栈已有前科——`collect/export_lerobot.py` 里记着 svt-av1 **编码** 92 条时
#   也段错误 rc=139，而历史上训练成功的 `dagger_v1` 是 **h264**。
#   对策（按顺序试）：
#     ① WORKERS=0                 —— 解码退回主进程，绕开 worker
#     ② DS_ROOT=rl/lang_ds/v1_h264 —— 换 h264 数据集（导出脚本加 --vcodec h264）
#     ③ SAVE_FREQ 默认从 1000 降到 250：崩了最多丢 1.5 分钟，且脚本自动 --resume 续跑
#
# 用法：
#   STEPS=5000 bash rl/run_lang_train.sh                              # 默认 4 worker
#   WORKERS=0  STEPS=5000 bash rl/run_lang_train.sh                    # 崩了先试这个
#   DS_ROOT=$PWD/rl/lang_ds/v1_h264 STEPS=5000 bash rl/run_lang_train.sh   # 还崩就换 h264
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8
PY="/home/yyl/miniforge3/envs/lerobot312/bin/lerobot-train"

STEPS="${STEPS:-5000}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-250}"
ATTEMPTS="${ATTEMPTS:-3}"
OUT="${OUT:-/home/yyl/lerobot/outputs/train/act_lang_dual}"
DS_ROOT="${DS_ROOT:-$PWD/rl/lang_ds/v1}"
BASE="${BASE:-/home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model}"
# 解码后端：lerobot 默认 torchcodec（本机 AV1 走 libdav1d）；BACKEND=pyav 可换 pyav 解码
BACKEND="${BACKEND:-}"

for p in "$DS_ROOT" "$BASE"; do
  [ -e "$p" ] || { echo "缺依赖: $p" >&2; exit 2; }
done

echo "=== 语言条件 ACT 微调 ==="
echo "  数据集: $DS_ROOT"
echo "  基座  : $BASE"
echo "  步数  : $STEPS（batch $BATCH, seed $SEED, workers $WORKERS, save_freq $SAVE_FREQ）"
echo "  输出  : $OUT"
echo

rc=1
for attempt in $(seq 1 "$ATTEMPTS"); do
  RESUME=()
  # 上一次已经落了 ckpt 就接着跑（lerobot 的 --resume 从 <output_dir>/checkpoints/last 续）
  if [ -e "$OUT/checkpoints/last" ]; then RESUME=(--resume=true); fi
  echo "=== attempt $attempt/$ATTEMPTS  $(date +%H:%M:%S)  resume=${RESUME[*]:-no} ==="

  env -u PYTHONHOME -u PYTHONPATH "$PY" \
    --dataset.repo_id=local/v1 \
    --dataset.root="$DS_ROOT" \
    --policy.type=act \
    --policy.repo_id=local/act_lang_dual \
    --policy.pretrained_path="$BASE" \
    --policy.push_to_hub=false --save_checkpoint_to_hub=false \
    --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
    --num_workers="$WORKERS" ${BACKEND:+--dataset.video_backend="$BACKEND"} \
    --output_dir="$OUT" --seed="$SEED" "${RESUME[@]}"
  rc=$?
  echo "=== attempt $attempt rc=$rc  $(date +%H:%M:%S) ==="
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -eq 139 ]; then
    echo "段错误（139）。若这是 attempt 1：改用 WORKERS=0 重跑；"
    echo "  仍是 139 就说明与 worker 解码无关 → 换 DS_ROOT=\$PWD/rl/lang_ds/v1_h264 再试。"
  fi
  sleep 5
done

echo
echo "TRAIN_DONE rc=$rc  output=$OUT  ($(date +%H:%M:%S))"
ls "$OUT/checkpoints" 2>/dev/null | tail -3
exit $rc