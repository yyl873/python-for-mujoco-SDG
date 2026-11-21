#!/usr/bin/env bash
# 回答"放开残差幅度限制，RL 是否就有更多操作空间" —— oracle 上界探针，免训练
# 三条件同池同参数，只改候选幅度：1.43（训练值）/ 5.73（环境默认）/ 11.46（8 倍）
# mj_step 有进程级 SIGFPE（探针的 mj_step 次数是评测的 ~10 倍，所以更容易崩），
# 用 --resume + 多次重试把已完成的 episode 保住。
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl LP_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface

run() {   # $1=tag $2=residual_deg
  local tag=$1 deg=$2 out="rl/eval/shooting_clamp_${tag}.json" log=".sp_${tag}.log"
  local nl="$out"
  for attempt in $(seq 1 25); do
    env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/shooting_probe.py \
      --base sim156 --episodes 100-129 --candidates 8 --horizon 20 --chunk 30 \
      --oracle --judge target --jaw-min-deg 16.05 --residual-deg "$deg" \
      --resume --out "$out" >> "$log" 2>&1
    local rc=$?
    local got
    got=$(env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - "${nl%.json}.jsonl" <<'PY' 2>/dev/null || echo 0
import json,sys
try:
    eps={json.loads(l)["episode_id"] for l in open(sys.argv[1])}
    print(len(eps))
except Exception:
    print(0)
PY
)
    echo "[$tag deg=$deg] attempt=$attempt rc=$rc episodes=$got/30" >> "$log"
    [ "$got" -ge 30 ] && { echo "DONE $tag deg=$deg episodes=$got"; return 0; }
  done
  echo "PARTIAL $tag deg=$deg episodes=$got"
  return 1
}

for pair in "a143:1.43" "b573:5.73" "c1146:11.46"; do
  tag=${pair%%:*}; deg=${pair##*:}
  rm -f "rl/eval/shooting_clamp_${tag}.json" "rl/eval/shooting_clamp_${tag}.jsonl" ".sp_${tag}.log"
  run "$tag" "$deg" || true
done
echo "ALL DONE"