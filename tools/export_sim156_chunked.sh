#!/bin/bash
# ============================================================================
# 分块导出 sim156：每块默认 10 条，已完成块自动跳过，最后 merge 成 sim156
# ----------------------------------------------------------------------------
# 用法：
#   rm -rf /home/yyl/lerobot_datasets/sim156 /home/yyl/lerobot_datasets/sim156_parts
#   nohup bash tools/export_sim156_chunked.sh all > logs/sim156_chunked.log 2>&1 &
#
# 断点续跑：再次执行同一条命令即可；已完整的 part_xxx 会跳过。
# 只导某一块：bash tools/export_sim156_chunked.sh part 3
# 只合并：    bash tools/export_sim156_chunked.sh merge
#
# 可覆盖：
#   CHUNK=10       每块条数
#   CPUS=0-7       限制使用的 CPU（默认只用 8 个逻辑核）
#   DS=...         数据集根
# ============================================================================
set -u

ROOT=/home/yyl/deepseek-harness
PY="${PY:-/home/yyl/miniforge3/envs/lerobot312/bin/python}"
PYSYS="env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3"
DS="${DS:-/home/yyl/lerobot_datasets}"
N="${N:-156}"
CHUNK="${CHUNK:-10}"
CPUS="${CPUS:-0-7}"
PARTS="$DS/sim156_parts"
EPLIST="$ROOT/logs/sim156_episodes.txt"
FINAL="$DS/sim156"

# 自己降到低优先级，避免把桌面拖死
if [ "${_CHUNKED_LOWPRIO:-0}" != "1" ]; then
    export _CHUNKED_LOWPRIO=1
    exec taskset -c "$CPUS" nice -n 15 ionice -c2 -n7 bash "$0" "$@"
fi

export HF_HUB_OFFLINE=1
export -n PYTHONHOME PYTHONPATH 2>/dev/null || true

die() { echo "❌ $*"; exit 1; }

# 独占锁；同时禁止和 run_train_v8.sh export 并发
exec 9>"${TMPDIR:-/tmp}/sim156_chunked.lock"
flock -n 9 || die "已有分块导出在跑"
if pgrep -f '[c]ollect/export_lerobot.py' >/dev/null 2>&1; then
    die "检测到其它 export_lerobot.py 进程；先全部 kill 再跑"
fi

pick_episodes() {
    $PYSYS - "$N" > "$EPLIST" <<'PY'
import glob, json, sys
n = int(sys.argv[1])
rows = []
for f in sorted(glob.glob('/home/yyl/deepseek-harness/scripts/ep_*.json')):
    d = json.load(open(f))
    if not d.get('success'):
        continue
    p = (d.get('randomization') or {}).get('posture') or {}
    y = p.get('box_yaw_deg')
    if y is None:
        continue
    rows.append((float(y), int(d['episode_id'])))
rows.sort()
if len(rows) < n:
    raise SystemExit(f"可用条数 {len(rows)} < {n}")
step = len(rows) / n
sel = [rows[int(i * step)][1] for i in range(n)]
if len(set(sel)) != n:
    raise SystemExit("抽样出现重复")
for eid in sel:
    print(eid)
PY
}

part_path() { printf '%s/part_%03d' "$PARTS" "$1"; }
part_repo() { printf 'local/sim156_part_%03d' "$1"; }

is_complete() {  # $1=part path  $2=expected episodes
    [ -f "$1/meta/info.json" ] || return 1
    [ -f "$1/meta/sim_provenance.json" ] || return 1
    local got
    got=$($PYSYS -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_episodes'))" \
          "$1/meta/info.json" 2>/dev/null) || return 1
    [ "$got" = "$2" ]
}

export_one() {  # $1=part index
    local idx="$1" start count ids part
    start=$((idx * CHUNK))
    count=$(( N - start ))
    [ "$count" -gt "$CHUNK" ] && count="$CHUNK"
    [ "$count" -le 0 ] && return 0
    part=$(part_path "$idx")

    if is_complete "$part" "$count"; then
        echo "⏭  part_$(printf '%03d' "$idx") 已完成（$count 条），跳过"
        return 0
    fi

    ids=$(sed -n "$((start + 1)),$((start + count))p" "$EPLIST" | paste -sd, -)
    echo
    echo "=============== part_$(printf '%03d' "$idx")  第 $((start + 1))~$((start + count)) 条 ==============="
    echo "  episodes: $ids"
    rm -rf "$part"
    mkdir -p "$PARTS"
    "$PY" "$ROOT/collect/export_lerobot.py" \
        --dst "$part" --repo-id "$(part_repo "$idx")" \
        --episodes "$ids" --units deg \
        --gripper-space real-closed --gripper-full "${GRIP_FULL:-49.08}" --overwrite \
        || die "part_$(printf '%03d' "$idx") 导出失败"
    is_complete "$part" "$count" || die "part_$(printf '%03d' "$idx") 导出后不完整"
    echo "✅ part_$(printf '%03d' "$idx") 完成（$count 条）"
}

merge_all() {
    [ -d "$FINAL" ] && die "$FINAL 已存在；先确认后 rm -rf 再 merge"
    [ -d "$PARTS" ] || die "没有 $PARTS"

    local parts=() repos="" roots="" p name first=1 nparts=0
    for p in "$PARTS"/part_*; do
        [ -d "$p" ] || continue
        parts+=("$p")
    done
    [ "${#parts[@]}" -gt 0 ] || die "没有 part_* 可合并"

    for p in "${parts[@]}"; do
        name=$(basename "$p")
        [ "$first" -eq 0 ] && { repos+=","; roots+=","; }
        repos+="'local/$name'"
        roots+="'$p'"
        first=0
        nparts=$((nparts + 1))
    done
    repos="[$repos]"
    roots="[$roots]"

    echo
    echo "=============== merge $nparts 块 -> $FINAL ==============="
    "$PY" -m lerobot.scripts.lerobot_edit_dataset \
        --operation.type=merge \
        --new_repo_id=local/sim156 --new_root="$FINAL" \
        --operation.repo_ids="$repos" \
        --operation.roots="$roots" \
        || die "merge 失败"
    echo "✅ merge 完成: $FINAL"
    echo "   下一步体检："
    echo "   $PY $ROOT/tools/check_datasets_health.py --root $FINAL --out $ROOT/logs/health_sim156.json"
}

ensure_episodes() {
    mkdir -p "$ROOT/logs"
    pick_episodes || die "抽样失败"
    local n
    n=$(wc -l < "$EPLIST")
    [ "$n" = "$N" ] || die "抽样条数 $n != $N"
    echo "抽样: $N 条，块大小 $CHUNK，CPU=$CPUS"
    echo "编号文件: $EPLIST"
}

cmd="${1:-all}"
case "$cmd" in
    all)
        ensure_episodes
        nparts=$(( (N + CHUNK - 1) / CHUNK ))
        for ((i=0; i<nparts; i++)); do export_one "$i"; done
        merge_all
        ;;
    part)
        ensure_episodes
        idx="${2:-0}"
        export_one "$idx"
        ;;
    merge)
        merge_all
        ;;
    *)
        echo "用法: bash tools/export_sim156_chunked.sh {all|part N|merge}"
        exit 1
        ;;
esac
