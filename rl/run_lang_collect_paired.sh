#!/usr/bin/env bash
# 反事实配对采集：同一场景 + 同一初始状态，左/右指令【各采一条】。
#
# 为什么必须这样采（2026-11-21 实测结论）：
#   旧脚本把 2000-2049 全给左指令、2050-2099 全给右指令，而场景按 episode 序号随机生成，
#   于是「指令」与「场景外观」完全共线（相关系数 −0.866）。模型只需看画面就能猜出该往哪个箱子送，
#   那 2 维 one-hot（state[6:8]）拿不到任何梯度 → 5000 步训完落点 38/39 全在右箱，语言贡献为 0。
#
# 机制（只改随机化种子派生，不碰任何物理参数）：
#   collect.py 的每条 rng = default_rng([base_seed, eid])；新增 --scene-seed-offset K 后变成
#   default_rng([base_seed, eid-K])。于是：
#     左段 eid=3000..3015            → 场景种子 3000..3015
#     右段 eid=4000..4015 + offset 1000 → 场景种子 3000..3015  ★同一个场景
#   两条 episode 的首帧 qpos 逐位相同（实测最大差 0.000e+00），唯一的差别是指令与目标箱。
#
# 用法：
#   SCENES=16 START_L=3000 START_R=4000 SEED=7788 bash rl/run_lang_collect_paired.sh
#   想复用旧批次编号段就改 START_L/START_R（两段必须间隔 >= SCENES）。
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface
PY="${PY:-.venv-rl/bin/python}"

SCENES="${SCENES:-16}"
START_L="${START_L:-3000}"
START_R="${START_R:-4000}"          # 场景偏移由 OFFSET 自动算出
SEED="${SEED:-7788}"
JOBS="${JOBS:-8}"
LEFT="Put the medicine box into the left wooden box."
RIGHT="Put the medicine box into the right wooden box."

OFFSET=$(( START_R - START_L ))
if [ "$OFFSET" -lt "$SCENES" ]; then
  echo "编号段重叠：START_R-START_L=$OFFSET < SCENES=$SCENES，会互相覆盖"; exit 2
fi

run_pass() {   # $1=start $2=target-box $3=instruction $4=offset $5=log
  local start=$1 tb=$2 ins=$3 off=$4 log=$5 rc=1
  for i in 1 2 3 4 5; do
    env -u PYTHONHOME -u PYTHONPATH $PY collect/collect.py \
        --n "$SCENES" --start "$start" --seed "$SEED" --scene-seed-offset "$off" \
        --jobs "$JOBS" --target-box "$tb" --instruction "$ins" >> "$log" 2>&1
    rc=$?
    echo "[start=$start tb=$tb offset=$off] attempt=$i rc=$rc" >> "$log"
    [ $rc -eq 0 ] && return 0
  done
  return 1
}

echo "=== 左指令 $SCENES 条（eid $START_L..$((START_L+SCENES-1))，场景种子 $START_L..$((START_L+SCENES-1))）==="
run_pass "$START_L" target_box "$LEFT" 0 .lang_pair_left.log || { echo "PAIR_COLLECT_FAIL_LEFT"; exit 1; }
echo "=== 右指令 $SCENES 条（eid $START_R..$((START_R+SCENES-1))，场景种子同上）==="
run_pass "$START_R" target_box_b "$RIGHT" "$OFFSET" .lang_pair_right.log || { echo "PAIR_COLLECT_FAIL_RIGHT"; exit 1; }

# 编号清单：成对写出，导入导出脚本用
python_ids() {
  local i
  for i in $(seq 0 $((SCENES-1))); do
    echo $((START_L+i)); echo $((START_R+i))
  done | paste -sd, -
}
python_ids > .lang_pair_ids.txt
echo "PAIR_COLLECT_DONE  ids=$(cat .lang_pair_ids.txt)"