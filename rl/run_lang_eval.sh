#!/usr/bin/env bash
# 语言条件反事实评测（崩溃可续跑）
#
# 为什么要这个壳：`mj_step` 在本机**偶发 SIGFPE（rc=136，"浮点异常"）**，进程级、与逻辑无关。
# 76 条 rollout 一次跑完基本不可能不被它打断；而 `rl/eval_lang.py` 每跑完一条 rollout 就把
# 结果写进 `--out` 的 json，并支持 `--resume` 跳过已完成的行 → 崩了原地重跑，最多丢当前那一条。
#
# 判据（预注册，看完数据不许改）：follow_acc ≥ 0.80 且 Wilson 95% 下界 > 0.50。
#
# 用法：
#   bash rl/run_lang_eval.sh                       # 默认评测 act_lang_dual 的 last ckpt
#   CKPT=... EPISODES=2400-2409,2411-2419 SEEDS=2 bash rl/run_lang_eval.sh
#   CKPT=... DEVICE=auto bash rl/run_lang_eval.sh  # 有 GPU 的终端里可以指定设备
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL="${MUJOCO_GL:-egl}"

PY="${PY:-$PWD/.venv-rl/bin/python}"
CKPT="${CKPT:-/home/yyl/lerobot/outputs/train/act_lang_dual/checkpoints/last/pretrained_model}"
DS="${DS:-$PWD/rl/lang_ds/v1_h264}"
EPISODES="${EPISODES:-2400-2409,2411-2419}"
SEEDS="${SEEDS:-2}"
OUT="${OUT:-rl/eval/lang_counterfactual.json}"
ATTEMPTS="${ATTEMPTS:-8}"
DEVICE="${DEVICE:-cpu}"
# 额外参数透传（如 EXTRA=--no-instr：控制组，状态仍 6 维，用于验证评测框架本身）
EXTRA="${EXTRA:-}"

[ -e "$CKPT" ] || { echo "缺 ckpt: $CKPT（先把训练跑完）" >&2; exit 2; }

rows_now() {  # 已完成 rollout 条数（读 out json；不存在算 0）
  env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 -c "
import json,sys
try:
    print(len(json.load(open('$OUT')).get('rows') or []))
except Exception:
    print(0)" 2>/dev/null
}

echo "=== 语言条件反事实评测 ==="
echo "  ckpt : $CKPT"
echo "  数据 : $DS"
echo "  场景 : $EPISODES × $SEEDS seed × 2 指令"
echo "  落盘 : $OUT（每完成一条就写）"
echo

prev=0 stale=0
for attempt in $(seq 1 "$ATTEMPTS"); do
  RES=()
  [ -f "$OUT" ] && RES=(--resume)
  echo "=== attempt $attempt/$ATTEMPTS  $(date +%H:%M:%S)  已行=$(rows_now) resume=${RES[*]:-no} ==="
  timeout 3600 env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL="$MUJOCO_GL" "$PY" rl/eval_lang.py \
    --ckpt "$CKPT" --ds "$DS" --episodes "$EPISODES" --seeds "$SEEDS" \
    --out "$OUT" --device "$DEVICE" ${EXTRA:+$EXTRA} "${RES[@]}"
  rc=$?
  cur=$(rows_now)
  echo "=== attempt $attempt rc=$rc  已行=$cur  $(date +%H:%M:%S) ==="
  if [ "$rc" -eq 0 ]; then break; fi
  if [ "$rc" -eq 136 ]; then
    echo "  mj_step 浮点异常（SIGFPE）：这是本机已知的偶发问题，重跑即可（已完成的行不会重做）"
  fi
  if [ "$cur" -le "$prev" ]; then
    stale=$((stale + 1))
    echo "  ⚠️  本轮没有任何新行（连续 $stale 次）——若第 2 次仍为 0，多半是这个 (episode,seed) 场景"
    echo "      稳定触发 SIGFPE，不是偶发：把该条从 EPISODES 里排除并在报告里注明。"
    [ "$stale" -ge 2 ] && { echo "  放弃重试，结果保留在 $OUT"; break; }
  else
    stale=0
  fi
  prev=$cur
  sleep 3
done

echo
rows_now
echo "（若日志末尾已有判决行「跟随指令 / 判定」，本次即完整跑完）"