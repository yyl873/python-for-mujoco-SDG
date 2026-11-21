#!/usr/bin/env bash
# 补 §19.4 的盲区：把候选打分从"20 步后抬升"换成"20 步内 TCP-药盒距离下降"，
# 这样 approach 阶段（抬升恒为 0）才有区分度。三档幅度同池同参数。
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl LP_NUM_THREADS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface

run() {   # $1=tag $2=residual_deg
  local tag=$1 deg=$2 out="rl/eval/shooting_appr_${tag}.json" log=".sa_${tag}.log"
  for attempt in $(seq 1 25); do
    env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/shooting_probe.py \
      --base sim156 --episodes 100-129 --candidates 8 --horizon 20 --chunk 30 \
      --score approach --oracle --judge target --jaw-min-deg 16.05 --residual-deg "$deg" \
      --resume --out "$out" >> "$log" 2>&1
    local rc=$?
    local got
    got=$(env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - "${out%.json}.jsonl" <<'PY' 2>/dev/null || echo 0
import json,sys
try:
    print(len({json.loads(l)["episode_id"] for l in open(sys.argv[1])}))
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
  rm -f "rl/eval/shooting_appr_${tag}.json" "rl/eval/shooting_appr_${tag}.jsonl" ".sa_${tag}.log"
  run "$tag" "$deg" || true
done
echo "ALL DONE"
