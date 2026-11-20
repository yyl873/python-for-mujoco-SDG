#!/bin/bash
# rl/_pilot_retry.sh —— S0 追加探针：段错误（mj_step/EGL 已知问题）会直接杀进程，
# 所以用 --resume 反复续跑，直到四组配置都出结果。用完可删。
set -u
cd /home/yyl/deepseek-harness || exit 1
LOG=logs/pilot_s0b.log
PY=.venv-rl/bin/python
: > "$LOG"

run() {
  env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl "$PY" rl/pilot.py \
      --episodes 0 --duration 15 --out rl/eval/pilot.json --resume "$@" >> "$LOG" 2>&1
  echo "  [rc=$?] $*" >> "$LOG"
}

declare -a CMDS=(
  "--ckpts sim156 --combos raw"
  "--ckpts sim156 --combos sensor"
  "--ckpts armA --combos raw --state-from qpos"
  "--ckpts armA --combos swap+sensor --state-from qpos"
)

for pass in 1 2 3 4; do
  echo "=== pass $pass ===" >> "$LOG"
  for c in "${CMDS[@]}"; do
    run $c
  done
done
echo "=== done ===" >> "$LOG"