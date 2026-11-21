#!/usr/bin/env bash
# 残差 9 臂在"夹爪钳制"下的重测（补齐 §17 最后一块空白）
# 每条 90 次 rollout（episodes 100-129, seeds 3），judge=target，jaw_min_deg=16.05
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl LP_NUM_THREADS=4 MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 单进程 1 核；并发靠进程，不靠线程（否则 3 进程抢 32 线程反而更慢）

run_arm() {   # $1=rewardtag(R1/R2/R3) $2=seedtag(s0/s1/s2) $3=rewardver
  local tag=$1 seed=$2 rw=$3
  local out IFS
  out="rl/eval/target_${tag}_${seed}_clamp_90.json"
  local log=".clamp_${tag}_${seed}.log"
  for attempt in 1 2 3 4 5 6; do
    env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/eval_ab.py \
      --base sim156 --policy "rl/ckpt/${tag}/${seed}/model.zip" --reward "$rw" \
      --episodes 100-129 --seeds 3 --judge target --jaw-min-deg 16.05 \
      --resume --out "$out" >> "$log" 2>&1
    local rc=$?
    # 判据：文件里 90 行齐全就算完成
    local n
    n=$(env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - "$out" <<'PY' 2>/dev/null || echo 0
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(sum(len(b.get("rows",[])) for b in (d.get("detail") or {}).values()))
except Exception:
    print(0)
PY
)
    echo "[$tag $seed] attempt=$attempt rc=$rc rows=$n" >> "$log"
    [ "$n" -ge 180 ] && { echo "DONE $tag $seed rows=$n"; return 0; }
  done
  echo "FAIL $tag $seed rows=$n"
  return 1
}

# 3 路并发，限制总负载
( run_arm R1 s0 v1 ) > .clamp_R1_s0.out 2>&1 &
( run_arm R1 s1 v1 ) > .clamp_R1_s1.out 2>&1 &
( run_arm R1 s2 v1 ) > .clamp_R1_s2.out 2>&1 &
wait
( run_arm R2 s0 v2 ) > .clamp_R2_s0.out 2>&1 &
( run_arm R2 s1 v2 ) > .clamp_R2_s1.out 2>&1 &
( run_arm R2 s2 v2 ) > .clamp_R2_s2.out 2>&1 &
wait
( run_arm R3 s0 v2b ) > .clamp_R3_s0.out 2>&1 &
( run_arm R3 s1 v2b ) > .clamp_R3_s1.out 2>&1 &
( run_arm R3 s2 v2b ) > .clamp_R3_s2.out 2>&1 &
wait
echo "ALL DONE"
cat .clamp_R*.out