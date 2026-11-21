#!/usr/bin/env bash
# 语言条件 pilot 采集：50 条左箱 + 50 条右箱（同一场景，两个指令）
# 编号段：2000-2049 = left（target_box），2050-2099 = right（target_box_b）
# 渲染是唯一贵的环节（~124 s/条），用 --jobs 并行；失败自动重试（SIGFPE）。
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface
PY=".venv-rl/bin/python"
JOBS="${JOBS:-8}"

run() {   # $1=start $2=n $3=target_box $4=instruction $5=log
  local start=$1 n=$2 tb=$3 ins=$4 log=$5
  for i in 1 2 3 4 5; do
    env -u PYTHONHOME -u PYTHONPATH $PY collect/collect.py \
        --n "$n" --start "$start" --jobs "$JOBS" \
        --target-box "$tb" --instruction "$ins" >> "$log" 2>&1
    local rc=$?
    local got
    got=$(ls scripts/ep_0*_*.npz 2>/dev/null | wc -l)
    echo "[start=$start] attempt=$i rc=$rc total_npz=$got" >> "$log"
    [ $rc -eq 0 ] && return 0
  done
  return 1
}

echo "=== 左箱 50 条（2000-2049）==="
run 2000 50 target_box "Put the medicine box into the left wooden box." .lang_left.log
echo "=== 右箱 50 条（2050-2099）==="
run 2050 50 target_box_b "Put the medicine box into the right wooden box." .lang_right.log
echo "LANG_COLLECT_DONE"