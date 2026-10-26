#!/bin/bash
# ============================================================================
# 1000 条 sdg-v8 母本生成（分 5 批 × 200，可断点续跑）
# ----------------------------------------------------------------------------
#   bash run_1000.sh            # 从头跑（同代同 seed 时已存在的编号会被逐位重算覆盖）
#   bash run_1000.sh 400        # 从第 400 条开始，跑剩下的 3 批
#   JOBS=6 bash run_1000.sh     # 显存/内存吃紧时降并发
#
# 与旧版的差别（旧版三个坑，都已补）：
#   1) 旧版不检查退出码：某批崩了（mj_step 有约 7% 的进程级 SIGFPE、EGL/显存挂掉）
#      脚本照样跑下一批，你最后拿到 800 条却以为 1000 条。→ 现在任一失败立即 exit。
#   2) 旧版 --jobs 4：有 GPU 的机器实测 8 进程 8 条 33.1 s、单条 19.1 s
#      → jobs 8 约 1~1.5 h，jobs 4 要 2~3 h。→ 现在默认 8，可用 JOBS 覆盖。
#   3) 旧版跑完没有核对。→ 现在每批核对 index.json 的 n_episodes，收尾再 rebuild-index。
#
# 内存：15 GB、每进程约 1.5 GB → jobs 8 是安全上限，别再加（用 6 更稳）。
# 磁盘：1000 条约 1.2 GB（videos ~1.0 GB + scripts ~170 MB），请留 ≥3 GB。
# ============================================================================
set -u

cd /home/yyl/deepseek-harness || exit 1
mkdir -p logs

PY="env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3"
JOBS="${JOBS:-8}"
SEED="${SEED:-3}"
WANT=1000
START_FROM="${1:-0}"

n_of() {
    $PY - <<'EOF' 2>/dev/null || echo -1
import json
try:
    print(json.load(open('scripts/index.json'))['n_episodes'])
except Exception:
    print(-1)
EOF
}

echo "=================================================================="
echo " sdg-v8 · 1000 条母本   jobs=$JOBS seed=$SEED  起始=$START_FROM"
echo "=================================================================="
echo " 开始前 index 里已有: $(n_of) 条"

for start in 0 200 400 600 800; do
    [ "$start" -lt "$START_FROM" ] && continue
    echo
    echo "------ 批次 $start / +200 ------"
    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
        collect/collect.py --start "$start" --n 200 --jobs "$JOBS" --seed "$SEED" \
        > "logs/collect_${start}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "❌ 批次 $start 退出码 $rc —— 立即停止，不影响后面的批次"
        echo "   看: tail -40 logs/collect_${start}.log"
        exit 1
    fi
    n=$(n_of)
    echo "✅ 批次 $start 完成，index 累计 $n 条"
    if [ "$n" -lt $((start + 200)) ]; then
        echo "⚠️  条数不足预期 $((start + 200))（差 $((start + 200 - n)) 条）——停止"
        echo "   多半有个别 episode 被拒（no_lift 等），查 logs/collect_${start}.log 里的 ❌ 行"
        exit 2
    fi
    sleep 2
done

echo
echo "------ 收尾：rebuild-index + DATA_CARD ------"
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --rebuild-index > logs/collect_rebuild.log 2>&1
if [ $? -ne 0 ]; then
    echo "❌ rebuild-index 失败"; tail -20 logs/collect_rebuild.log; exit 3
fi
$PY collect/stats.py --card > logs/stats_card.log 2>&1 \
    || echo "⚠️  DATA_CARD 生成失败（不影响数据本身）"

n=$(n_of)
echo
echo "=================================================================="
echo " 完成: index 里共 $n 条（目标 $WANT）"
$PY - <<'EOF'
import glob, json
import numpy as np
ver, succ, yaw, nz = set(), [], [], []
for f in sorted(glob.glob('scripts/ep_*.json')):
    d = json.load(open(f))
    ver.add(d['dataset_version'])
    succ.append(bool(d['success']))
    p = (d.get('randomization') or {}).get('posture') or {}
    if p.get('box_yaw_deg') is not None:
        yaw.append(p['box_yaw_deg'])
        if p.get('pad_nz') is not None:
            nz.append(p['pad_nz'])
print(f"  版本    : {sorted(ver)}")
print(f"  成功    : {sum(succ)}/{len(succ)}")
if yaw:
    y = np.array(yaw); s = np.sort(y)
    g = np.diff(np.r_[s, s[0] + 180])
    print(f"  盒朝向  : min {y.min():.1f} max {y.max():.1f} 中位 {np.median(y):.1f}"
          f"  连续覆盖 {180 - g.max():.1f} deg")
    if nz:
        print(f"  指垫 nz : 中位 {np.median(nz):.4f} max {max(nz):.4f}")
EOF
echo " 下一步: 训练前先过 docs/PLAN_TRAIN_V8.md 的 §0 三道闸门"
echo "=================================================================="