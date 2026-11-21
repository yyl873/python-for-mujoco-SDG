#!/usr/bin/env bash
# 导出双指令数据集（92 条 / state 8 维 / 两个 instruction）
#
# 踩过的坑（2026-11-21，按时间顺序）：
#  ① 工具调用超时会给**整个进程组**发 SIGTERM → 用 setsid 起独立会话。
#  ② 探活别用 `pgrep -f`:探测命令自己的 cmdline 就含那个 pattern，永远探不到"结束"。
#     改用 .lang_export.pid + kill -0。
#  ③ 两台导出同时写同一个 --dst 会互删 images/ 里的临时帧 → 开跑前查重。
#  ④ ★ 未定性的坑：全量 92 条在 **episode index 2** 附近会突然写不进帧
#     （`images/<cam>/episode-00000X/` 中途消失，报 317 条
#     `Error writing image ...: No such file or directory`，随后
#     compute_episode_stats 读 frame-000000.png 抛 FileNotFoundError）。
#     同样的命令导 5 条是干净的 rc=0。每次换一个**全新的 DST** 重试，
#     并统计 "Error writing image" 条数；条数必须是 0。
#  ⑤ ★ 2026-11-21 20:41 定性：上面 ④ 不是逻辑 bug，是**进程被会话带走**——
#     加 --streaming 后跑到 [57/92]、图像写入失败 **0 条**、无 traceback 就断了，
#     因为 GUI/工具会话崩溃时会 SIGTERM 掉整组。用 ①的 setsid + nohup + </dev/null
#     起独立会话后一次跑完：**92/92、rc=0、图像写入失败 0、5 分 4 秒**。
#     探活改用日志里的 `[ k/92]` 进度行，不要用 pgrep -f（见 ②）。
#
# 用法：
#   DST=rl/lang_ds/v1 setsid nohup bash rl/run_lang_export.sh > .lang_export.log 2>&1 < /dev/null &
#   慢不慢都别看进程：grep -oE "\[ *[0-9]+/92\]" .lang_export.log | tail -1
set -u
cd "$(dirname "$0")/.." || exit 1
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface

DST="${DST:-rl/lang_ds/v1}"
VCODEC="${VCODEC:-}"
ATTEMPTS="${ATTEMPTS:-2}"

# ---- 查重：同一份 --dst 只允许一个导出（排除查重进程自身）----
ME=$$
while read -r pid cmd; do
  [ "$pid" = "$ME" ] && continue
  case "$cmd" in *export_lerobot.py*"$DST"*)
    echo "已有导出进程(pid=$pid)在写 $DST，先停掉再跑"; exit 3 ;;
  esac
done <<< "$(ps -eo pid=,cmd=)"

IDS=$(cat .lang_ids.txt)
EXTRA=()
[ -n "$VCODEC" ] && EXTRA+=(--vcodec "$VCODEC")
# ★ 全量必须 --streaming：默认路径会在写帧中途丢掉 images/<cam>/episode-00000X/，
#   见文件头 ④。小批量（≤12 条）走默认路径是好的，别拿小批量的成功当作安全。
[ "${STREAMING:-1}" = "1" ] && EXTRA+=(--streaming)

for i in $(seq 1 "$ATTEMPTS"); do
  echo "=== 第 $i/$ATTEMPTS 次  开始 $(date +%H:%M:%S)  dst=$DST  条数=$(echo "$IDS" | tr ',' '\n' | wc -l)  vcodec=${VCODEC:-默认} ==="
  rm -rf "$DST"
  env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python collect/export_lerobot.py \
      --dst "$DST" --lang-onehot "${EXTRA[@]}" --episodes "$IDS" --overwrite
  rc=$?
  bad=$(grep -c "Error writing image" .lang_export.log 2>/dev/null || echo 0)
  echo "=== 第 $i/$ATTEMPTS 次  结束 $(date +%H:%M:%S)  rc=$rc  图像写入失败=${bad:-0} 条 ==="
  if [ $rc -eq 0 ]; then echo "LANG_EXPORT_OK"; exit 0; fi
done
echo "LANG_EXPORT_FAIL"
exit 1