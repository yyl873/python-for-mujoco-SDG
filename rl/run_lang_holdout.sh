#!/usr/bin/env bash
# 补采 held-out 评测段（spec 2026-11-21-dual-target-vla §10.4）
#
# 为什么需要：训练/导出用的是 2000–2099 段里验收通过的 92 条，
# `rl/eval_lang.py` 默认的 --episodes 2060-2079 **大部分已进训练集** →
# 拿它当 held-out 等于在训练场景上做反事实，跟准确率会被顶高。
# 这里另采 2400–2419（左 10 条 + 右 10 条），初始条件独立于训练集。
#
# 用法：
#   setsid nohup bash rl/run_lang_holdout.sh > .lang_holdout.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface
PY=".venv-rl/bin/python"
JOBS="${JOBS:-8}"

run() {   # $1=start $2=n $3=target_box $4=instruction $5=label
  local start=$1 n=$2 tb=$3 ins=$4 label=$5
  for i in 1 2 3 4 5; do
    env -u PYTHONHOME -u PYTHONPATH $PY collect/collect.py \
        --n "$n" --start "$start" --jobs "$JOBS" \
        --target-box "$tb" --instruction "$ins" >> .lang_holdout.log 2>&1
    local rc=$?
    echo "[$label start=$start] attempt=$i rc=$rc" >> .lang_holdout.log
    [ $rc -eq 0 ] && return 0
  done
  echo "[$label start=$start] FAILED after 5 attempts" >> .lang_holdout.log
  return 1
}

echo "=== held-out 左箱 10 条（2400-2409，target_box）$(date +%H:%M:%S) ==="
run 2400 10 target_box   "Put the medicine box into the left wooden box."  left
echo "=== held-out 右箱 10 条（2410-2419，target_box_b）$(date +%H:%M:%S) ==="
run 2410 10 target_box_b "Put the medicine box into the right wooden box." right

# 回执：验收通过的编号写进 .lang_holdout_ids.txt，供 eval_lang --episodes 用
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - <<'EOF' >> .lang_holdout.log 2>&1
import json, pathlib
ok = []
for i in list(range(2400, 2420)):
    p = pathlib.Path(f"scripts/ep_{i:06d}.json")
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("accepted"):
            ok.append(i)
pathlib.Path(".lang_holdout_ids.txt").write_text(",".join(map(str, ok)) + "\n")
print(f"HELDOUT_OK n={len(ok)} ids={','.join(map(str, ok))}")
EOF
echo "LANG_HOLDOUT_DONE $(date +%H:%M:%S)"