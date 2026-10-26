#!/bin/bash
# ============================================================================
# sdg-v8 的"导出 → 合并 → 训练"一键脚本（分阶段，每段都能单独跑）
# ----------------------------------------------------------------------------
#   bash run_train_v8.sh gate      只检查闸门，不动任何数据
#   bash run_train_v8.sh export    选 156 条 + 导出 sim156（含夹爪量纲对齐）+ 体检
#   bash run_train_v8.sh merge     建 sim39 / armA / armB / armC / armD
#   bash run_train_v8.sh train A   训练某一条臂（A|B|C|D|E）
#   bash run_train_v8.sh all       依次 gate → export → merge → train A B C D
#
# 前置：母本已生成（bash run_1000.sh）。本脚本不生成母本。
# 口径：P1（主实验只用 real_train_fixed 139 条，与 sdg-v8 取景严格同分布）。
#       想用 P2（fixed+corners 161 条）就设 REAL_TRAIN=... 见下。
#
# 可覆盖的环境变量：
#   PY         lerobot 解释器（默认 /home/yyl/miniforge3/envs/lerobot312/bin/python）
#   DP        数据集根目录（默认 /home/yyl/lerobot_datasets）
#   GRIP_FULL 真机夹爪满量程（默认 49.08 = fixed 批次实测；corners 批次是 42.27）
#   NSIM      仿真条数（默认 156）
#   REAL_TRAIN / REAL_NAME  训练用真机数据集（默认 fixed：139 条）
# ============================================================================
set -u

ROOT=/home/yyl/deepseek-harness
LR=/home/yyl/lerobot
PY="${PY:-/home/yyl/miniforge3/envs/lerobot312/bin/python}"
DP="${DP:-/home/yyl/lerobot_datasets}"
GRIP_FULL="${GRIP_FULL:-49.08}"
NSIM="${NSIM:-156}"
REAL_TRAIN="${REAL_TRAIN:-$DP/real_train_fixed}"
REAL_NAME="${REAL_NAME:-fixed}"
PYSYS="env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3"

export HF_HUB_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.cache/pycache_lerobot}"
export -n PYTHONHOME PYTHONPATH 2>/dev/null || true

die() { echo "❌ $*"; exit 1; }

# ----------------------------------------------------------------------------
gate() {
    echo "=============== 闸门检查 ==============="
    [ -f "$ROOT/scripts/index.json" ] || die "没有 scripts/index.json，先跑 bash run_1000.sh"

    $PYSYS - <<'EOF'
import glob, json, collections
idx = json.load(open('/home/yyl/deepseek-harness/scripts/index.json'))
print(f"  G0 母本   : {idx.get('n_episodes')} 条 / accepted {idx.get('n_accepted')} / "
      f"版本 {idx.get('dataset_version')} / schema {idx.get('schema_version')}")
if idx.get('n_episodes', 0) < 100:
    print("     ⚠️ 条数 < 100，先跑 bash run_1000.sh")
verts = collections.Counter()
for f in glob.glob('/home/yyl/deepseek-harness/scripts/ep_*.json'):
    verts[json.load(open(f))['dataset_version']] += 1
print(f"     逐条版本: {dict(verts)}")
EOF

    echo "  G1 夹爪量纲: 用 --gripper-space real-closed --gripper-full $GRIP_FULL"
    echo "     判定依据: 真机搬运段 gripper≈7~10(闭) / 接近段 28~49(开) → 0=闭合"
    echo "     目标真机族: $REAL_NAME  ($REAL_TRAIN)"

    if [ -f "$REAL_TRAIN/meta/stats.json" ]; then
        $PYSYS - "$REAL_TRAIN" <<'EOF'
import json, sys
st = json.load(open(sys.argv[1] + '/meta/stats.json'))['observation.state']
print(f"     真机实测 gripper: min {st['min'][5]:.2f} / max {st['max'][5]:.2f}")
EOF
    else
        echo "     ⚠️ 找不到 $REAL_TRAIN/meta/stats.json"
    fi

    av=$(df -Pk /home | awk 'NR==2{printf "%.0f", $4/1024/1024}')
    echo "  G2 磁盘   : /home 可用 ${av} GB（四个臂 + ckpt 需要 ≥15 GB）"
    [ "$av" -lt 15 ] && echo "     ⚠️ 空间偏紧"
    echo "========================================"
}

# ----------------------------------------------------------------------------
pick_episodes() {
    # 按盒朝向等距抽 NSIM 条（sdg-v8 的多样性是"盒朝向"，所以不能按编号取前 N 条）
    $PYSYS - "$NSIM" <<'EOF'
import glob, json, os, sys
n = int(sys.argv[1])
rows = []
for f in sorted(glob.glob('/home/yyl/deepseek-harness/scripts/ep_*.json')):
    d = json.load(open(f))
    if not d.get('success'):
        continue
    p = (d.get('randomization') or {}).get('posture') or {}
    y = p.get('box_yaw_deg')
    if y is None:                      # 退化样本（非反向生成）不入选
        continue
    rows.append((float(y), int(d['episode_id'])))
rows.sort()
if len(rows) < n:
    sys.exit(f"可用条数 {len(rows)} < {n}")
step = len(rows) / n
print(','.join(str(rows[int(i * step)][1]) for i in range(n)))
EOF
}

export_sim() {
    local EP
    EP=$(pick_episodes) || die "抽样失败"
    echo "=============== 导出 sim$NSIM ==============="
    echo "  前 20 个编号: $(echo "$EP" | cut -d, -f1-20)"
    rm -rf "$DP/sim$NSIM"
    $PY "$ROOT/collect/export_lerobot.py" \
        --dst "$DP/sim$NSIM" --repo-id "local/sim$NSIM" \
        --episodes "$EP" --units deg \
        --gripper-space real-closed --gripper-full "$GRIP_FULL" --overwrite \
        || die "导出失败"
    echo "  体检:"
    $PY "$ROOT/tools/check_datasets_health.py" --root "$DP/sim$NSIM" \
        --out "$ROOT/logs/health_sim$NSIM.json" 2>&1 | tail -12
    echo "  ⚠️ 上面必须看到 夹爪列 min≈0 / max≈$GRIP_FULL，且 frames ≈ $((NSIM*455))"
}

# ----------------------------------------------------------------------------
merge_all() {
    local SIM156="$DP/sim$NSIM"
    local SIM39="$DP/sim39"
    [ -d "$SIM156" ] || die "没有 $SIM156，先跑 export"
    echo "=============== 建 sim39 / armA~D ==============="

    local keep=$((39)) drop
    drop=$(seq -s, "$keep" $((NSIM-1)))
    [ -d "$SIM39" ] || $PY -m lerobot.scripts.lerobot_edit_dataset \
        --operation.type=delete_episodes --operation.episode_indices="[$drop]" \
        --repo_id="local/sim$NSIM" --root="$SIM156" \
        --new_repo_id=local/sim39 --new_root="$SIM39" || die "建 sim39 失败"

    _merge() {  # $1=臂名  $2=repo_ids python 列表  $3=roots python 列表
        local out="$DP/arm$1"
        [ -d "$out" ] && { echo "  arm$1 已存在，跳过"; return 0; }
        echo "  → arm$1"
        $PY -m lerobot.scripts.lerobot_edit_dataset --operation.type=merge \
            --new_repo_id="local/arm$1" --new_root="$out" \
            --operation.repo_ids="$2" --operation.roots="$3" --operation.concatenate_videos=false || die "arm$1 merge 失败"
    }

    _merge A "['local/$REAL_NAME']" "['$REAL_TRAIN']"
    _merge B "['local/$REAL_NAME','local/sim39']" "['$REAL_TRAIN','$SIM39']"
    _merge C "['local/$REAL_NAME','local/sim$NSIM']" "['$REAL_TRAIN','$SIM156']"
    _merge D "['local/sim$NSIM']" "['$SIM156']"

    du -sh "$DP"/arm* 2>/dev/null
}

# ----------------------------------------------------------------------------
train_one() {
    local arm="$1"
    local out="arm$arm"
    [ -d "$DP/$out" ] || die "没有 $DP/$out，先跑 merge"
    # 等 epoch 口径：steps = EPOCHS × 训练帧数 / batch（默认 2.25 epoch，与你那张表一致）
    local frames steps
    frames=$($PYSYS - "$DP/$out" <<'EOF'
import json, sys
print(json.load(open(sys.argv[1] + '/meta/info.json'))['total_frames'])
EOF
)
    steps=$($PYSYS -c "
f=$frames; bs=${BS:-8}; ep=${EPOCHS:-2.25}
print(int(round(f*ep/bs/500.0))*500)")
    echo "=============== 训练 arm$arm ==============="
    echo "  数据集 $DP/$out   帧数 $frames → steps $steps"
    echo "  口径 ${EPOCHS:-2.25} epoch @ batch ${BS:-8}（换 batch 16 时步数自动减半）"
    $PY "$ROOT/tools/train_mix.py" \
        --root "$DP/$out" --repo-id "local/arm$arm" \
        --output "$LR/outputs/train/act_arm$arm" --job-name "act_arm$arm" \
        --policy-repo-id "local/act_arm$arm" \
        --steps "$steps" --batch-size "${BS:-8}" --num-workers "${NW:-6}" \
        --save-freq "${SAVE_FREQ:-5000}" ${EXTRA_TRAIN_ARGS:-} \
        || die "训练 arm$arm 失败"
}

# ----------------------------------------------------------------------------
cmd="${1:-gate}"
case "$cmd" in
    gate)   gate ;;
    export) gate; export_sim ;;
    merge)  merge_all ;;
    train)  shift; for a in "$@"; do train_one "$a"; done ;;
    all)    gate; export_sim; merge_all; for a in A B C D; do train_one "$a"; done ;;
    *)      echo "用法: bash run_train_v8.sh {gate|export|merge|train A|all}"; exit 1 ;;
esac