#!/usr/bin/env bash
# 语言条件 ACT 微调：基座 act_sim156_lang8（state 已加宽 6→8）-> 在双指令数据上微调
#
# 前置：
#   1) rl/lang_ds/v1 或 v1_h264 已导出（92 条 / 41922 帧 / state 8 维 / 两个 instruction）
#   2) rl/widen_state.py 已生成 /home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model
#      （state 的 6->8 是**补 0 列**；★ 它只能当权重初始化，不能直接推理：
#       policy_preprocessor.json 里 normalizer 仍是 6 维，直接跑会报 tensor 8 vs 6）
#
# ★ 必须在能看到 /dev/nvidia* 的终端跑（沙箱看不到显卡，lerobot 会静默降级到 CPU：
#   batch 8 在 CPU 上 6.0~6.5 s/step，5000 步 ≈ 8.5 小时）。GPU 实测 ~2.87 step/s ≈ 30 min。
#
# ★★ 崩溃记录（别重复踩）：
#   2026-11-21 第一次：AV1 数据集 + 4 worker，跑到 **839/5000 步** 段错误 rc=139，
#     当时 save_freq=1000 → 一步 ckpt 都没落。退出时 resource_tracker 报 20 个泄漏信号量
#     → 死在 dataloader worker 侧。
#   2026-11-21 第二次：h264 数据集 + 4 worker，仍然 rc=139，但这次 save_freq=250 →
#     三次 attempt 分别推进到 ~1000 / 1250 / **1473 步**，ckpt 落 001000、001250（不白跑）。
#     → 段错误**与视频编码无关**，是本机 dataloader/torch 栈的既有问题；结论：
#       **不指望它不崩，而是让它崩了能续、并且自动降档**。
#   对策（本脚本已自动执行，无需手改命令）：
#     ⓿ 续跑参数：`--resume=true` 必须配 `--config_path=<out>/checkpoints/last/pretrained_model/
#        train_config.json`（lerobot 0.6.2 的硬要求；只给 --resume 会 ValueError）。
#        本机验过 6 步 → 12 步真续跑。SAVE_FREQ 默认 250，崩了最多丢 ~1.5 分钟。
#     ① 同一档连续崩 3 次 → 自动切 `--num_workers=0`（彻底没有 worker 进程）
#     ② 再连续崩 3 次 → 自动再切 `--dataset.video_backend=pyav`
#     ATTEMPTS 默认 60：即使每 300 步崩一次，也能靠续跑磨到 5000 步。
#
# 用法：
#   DS_ROOT=$PWD/rl/lang_ds/v1_h264 STEPS=5000 bash rl/run_lang_train.sh
#   ATTEMPTS=200 ...                     # 想更硬地磨
#   WORKERS=0 BACKEND=pyav ...           # 想直接从最低档开始
set -u
cd "$(dirname "$0")/.." || exit 1
export HF_HOME=$PWD/.cache/huggingface OMP_NUM_THREADS=8
PY="/home/yyl/miniforge3/envs/lerobot312/bin/lerobot-train"

STEPS="${STEPS:-5000}"
SEED="${SEED:-1000}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
SAVE_FREQ="${SAVE_FREQ:-250}"
ATTEMPTS="${ATTEMPTS:-60}"
ESCALATE_AFTER="${ESCALATE_AFTER:-3}"
OUT="${OUT:-/home/yyl/lerobot/outputs/train/act_lang_dual}"
DS_ROOT="${DS_ROOT:-$PWD/rl/lang_ds/v1}"
BASE="${BASE:-/home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model}"
# 解码后端：lerobot 默认 torchcodec；BACKEND=pyav 可换 pyav 解码（降档第 2 级用）
BACKEND="${BACKEND:-}"

for p in "$DS_ROOT" "$BASE"; do
  [ -e "$p" ] || { echo "缺依赖: $p" >&2; exit 2; }
done

step_now() {  # 目前最新 ckpt 的步数（读不到就 0）
  env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 -c "
import json,glob
best=0
for p in glob.glob('$OUT/checkpoints/*/training_state/training_step.json'):
    try: best=max(best,int(json.load(open(p)).get('step',0)))
    except Exception: pass
print(best)" 2>/dev/null || echo 0
}

echo "=== 语言条件 ACT 微调 ==="
echo "  数据集: $DS_ROOT"
echo "  基座  : $BASE"
echo "  步数  : $STEPS（batch $BATCH, seed $SEED, save_freq $SAVE_FREQ）"
echo "  已训  : $(step_now) 步（有 ckpt 就自动续跑）"
echo "  输出  : $OUT"
echo "  重试  : 最多 $ATTEMPTS 次；每档连续崩 $ESCALATE_AFTER 次自动降档"
echo

stage=0            # 0=配置原样  1=num_workers=0  2=num_workers=0 + pyav
fails_here=0
rc=1
cuda_hits=0
for attempt in $(seq 1 "$ATTEMPTS"); do
  if [ "$(step_now)" -ge "$STEPS" ]; then
    echo "已训 $(step_now) 步，达到目标 $STEPS → 不用再跑"
    rc=0
    break
  fi
  # 档位阶梯：越往下越"低负载"，代价是慢一点
  #   档0 原样(4 worker) → 档1 worker=0 → 档2 +pyav → 档3 +CUDA_LAUNCH_BLOCKING=1
  case "$stage" in
    0) W="$WORKERS"; BK="$BACKEND"; CLB=0 ;;
    1) W=0;          BK="$BACKEND"; CLB=0 ;;
    2) W=0;          BK="pyav";     CLB=0 ;;
    *) W=0;          BK="pyav";     CLB=1 ;;
  esac

  RESUME=()
  if [ -e "$OUT/checkpoints/last/pretrained_model/train_config.json" ]; then
    RESUME=(--resume=true
            --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json")
  fi
  echo "=== attempt $attempt/$ATTEMPTS 档$stage(workers=$W backend=${BK:-默认} cuda_block=$CLB)" \
       "已训=$(step_now)步 $(date +%H:%M:%S) resume=${RESUME[*]:-no} ==="

  # ★ 记下本次 attempt 的日志，崩溃后好定位（尤其 CUDA 类错误）
  ALOG="$PWD/rl/logs/attempt_${attempt}_$(date +%H%M%S).log"
  mkdir -p "$PWD/rl/logs"
  env -u PYTHONHOME -u PYTHONPATH CUDA_LAUNCH_BLOCKING="$CLB" "$PY" \
    --dataset.repo_id=local/v1 \
    --dataset.root="$DS_ROOT" \
    --policy.type=act \
    --policy.repo_id=local/act_lang_dual \
    --policy.pretrained_path="$BASE" \
    --policy.push_to_hub=false --save_checkpoint_to_hub=false \
    --batch_size="$BATCH" --steps="$STEPS" --save_freq="$SAVE_FREQ" \
    --num_workers="$W" ${BK:+--dataset.video_backend="$BK"} \
    --output_dir="$OUT" --seed="$SEED" "${RESUME[@]}" 2>&1 | tee "$ALOG"
  rc=${PIPESTATUS[0]}
  echo "=== attempt $attempt rc=$rc 已训=$(step_now)步  $(date +%H:%M:%S) ==="
  [ "$rc" -eq 0 ] && break

  fails_here=$((fails_here + 1))
  echo "  失败 rc=$rc（第 $fails_here/$ESCALATE_AFTER 次），日志留档 $ALOG"
  # CUDA 侧错误（INTERNAL ASSERT / device-side assert / illegal memory access）：
  # 进程反复 SIGSEGV 后会留下**粘滞的 CUDA context**，重启（或 nvidia-smi --gpu-reset）才清
  if grep -qiE 'INTERNAL ASSERT|device-side assert|illegal memory access|CUDAGuardImpl' "$ALOG"; then
    echo "  ⚠️ 这是 CUDA 侧错误（不是数据/编码问题）。先做两件事再看结果："
    echo "     ① 重启机器（或 sudo nvidia-smi --gpu-reset）清掉粘滞的 CUDA context；"
    echo "     ② 下一档会关掉 worker 并开 CUDA_LAUNCH_BLOCKING=1（定位到具体 kernel）"
    cuda_hits=$((cuda_hits + 1))
  fi
  if [ "$fails_here" -ge "$ESCALATE_AFTER" ] && [ "$stage" -lt 3 ]; then
    stage=$((stage + 1))
    fails_here=0
    echo "  ⤵ 降档到 档$stage"
  fi
  # GPU 侧出问题超过 2 次就打印驱动面证据，省得你再去翻
  if [ "$cuda_hits" -ge 2 ]; then
    echo "  --- 驱动面证据（贴给我就行）---"
    nvidia-smi --query-gpu=name,driver_version,temperature.gpu,memory.used,utilization.gpu \
               --format=csv 2>&1 | head -3
    nvidia-smi -q 2>/dev/null | grep -iE 'Xid|Pending|Retired' | head -5
    (journalctl -k -n 40 2>/dev/null || dmesg 2>/dev/null) | grep -iE 'xid|nvrm|nvidia' | tail -8
  fi
  sleep 5
done

echo
echo "TRAIN_DONE rc=$rc  已训=$(step_now)/$STEPS 步  output=$OUT  ($(date +%H:%M:%S))"
if [ "$(step_now)" -lt "$STEPS" ]; then
  echo "★ 没跑满：再执行同一条命令会从最近 ckpt 继续（不会从头重训）。"
  echo "  若每次都停在同一位置，把那次的 attempt 日志尾部发我。"
fi
ls "$OUT/checkpoints" 2>/dev/null | tail -3
exit $rc