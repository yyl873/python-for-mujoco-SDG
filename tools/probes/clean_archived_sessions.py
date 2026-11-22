#!/usr/bin/env python3
"""
clean_archived_sessions.py -- 清理 ~/.dsh 里"已归档"的会话 + 悬空缓存条目
==========================================================================

为什么必须单独跑（不能由 agent 代劳）
--------------------------------------
1) ~/.dsh 不在 agent 沙箱的写权限内，agent 写它会被拦成 Read-only file system；
2) harness 运行时每十几秒就覆写 workspace.json / session_projcache.json，
   运行中改元数据会被覆盖，甚至把当前会话写坏。
   => 所以脚本会先检测 harness 进程，活着就直接退出（--force 可强推）。

它做三件事
----------
  A. 把 workspace.json 里 archivedSessionIds 列出的会话目录【移到回收站】
     （默认移走可恢复；--purge 才是真删）
  B. 从 session_projcache.json 里删掉：归档会话 + 文件已不存在的悬空条目
  C. 从 workspace.json 里清掉 archivedSessionIds 和对应 sessionIds
  （B/C 之前都会备份成 *.bak-<时间戳>）

用法
----
  python3 tools/probes/clean_archived_sessions.py --dry-run     # 只看会动什么，不动手
  python3 tools/probes/clean_archived_sessions.py               # 移到 ~/.dsh/_trash_<ts>/
  python3 tools/probes/clean_archived_sessions.py --purge       # 直接删除（不可恢复）
  python3 tools/probes/clean_archived_sessions.py --keep <id>   # 指定的归档会话例外保留
  python3 tools/probes/clean_archived_sessions.py --force       # 跳过"harness 在运行"检查

回滚
----
  移走的会话在 ~/.dsh/_trash_<ts>/<session-id>/，挪回
  ~/.dsh/sessions/--home-yyl-deepseek-harness--/ 即可；
  元数据用 .bak-<ts> 覆盖回去。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import sys

DSH = os.path.expanduser("~/.dsh")
SESS_ROOT = os.path.join(DSH, "sessions")
STORAGES = os.path.join(DSH, "storages")
WORKSPACE_JSON = os.path.join(STORAGES, "workspace.json")
PROJCACHE_JSON = os.path.join(STORAGES, "session_projcache.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def harness_running() -> list[str]:
    """返回疑似 harness 进程的命令行片段（空列表 = 没在跑）。"""
    pat = "DeepSeek Harness/runtime/node"
    try:
        out = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True, timeout=5)
        return [ln for ln in out.stdout.splitlines() if pat in ln]
    except Exception:
        return []


def session_dirs(sid: str) -> list[str]:
    return glob.glob(os.path.join(SESS_ROOT, "*", sid))


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def backup(path: str, stamp: str) -> str | None:
    if not os.path.exists(path):
        return None
    dst = f"{path}.bak-{stamp}"
    shutil.copy2(path, dst)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="清理 ~/.dsh 已归档会话")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不动文件")
    ap.add_argument("--purge", action="store_true", help="真删除；默认移到回收站可恢复")
    ap.add_argument("--force", action="store_true", help="harness 在运行时也继续")
    ap.add_argument("--keep", action="append", default=[],
                    help="例外保留的 session id（可多次指定）")
    args = ap.parse_args(argv)

    for p in (WORKSPACE_JSON, PROJCACHE_JSON):
        if not os.path.exists(p):
            log(f"找不到 {p}")
            return 2

    procs = harness_running()
    if procs and not args.force and not args.dry_run:
        log("harness 似乎还在运行，现在改元数据会被它覆盖：")
        for p in procs[:5]:
            log("   " + p)
        log("\n先退出 harness（托盘退出 / 关窗口）再重跑；确认无碍可加 --force。")
        return 3

    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = json.load(open(WORKSPACE_JSON, encoding="utf-8"))
    projcache = json.load(open(PROJCACHE_JSON, encoding="utf-8"))

    archived = list(workspace.get("global", {}).get("archivedSessionIds") or [])
    keep = set(args.keep)
    targets = [s for s in archived if s not in keep]

    # ---- 盘点：归档会话 + 悬空条目 ----
    on_disk = {os.path.basename(d): d for d in glob.glob(os.path.join(SESS_ROOT, "*", "session-*"))}
    cache_ids = list(((projcache.get("tables") or {}).get("sessions") or {}).keys())
    dangling = [s for s in cache_ids if s not in on_disk]

    log(f"会话目录实际存在: {len(on_disk)} 条")
    log(f"projcache 记录:   {len(cache_ids)} 条（其中悬空 {len(dangling)} 条）")
    log(f"workspace 标记归档: {len(archived)} 条，本次处理 {len(targets)} 条"
        + (f"（--keep 保留 {len(keep)} 条）" if keep else ""))

    log("\n--- 要移走/删除的会话 ---")
    total = 0
    for sid in targets:
        ds = session_dirs(sid)
        if not ds:
            log(f"  {sid}  <文件已不存在，仅清元数据>")
            continue
        sz = sum(dir_size(d) for d in ds)
        total += sz
        log(f"  {sid}  {sz/1024/1024:6.2f} MB")
    log(f"  合计 {total/1024/1024:.1f} MB")

    if args.dry_run:
        log("\n[dry-run] 未做任何改动。")
        return 0

    # ---- 备份元数据 ----
    for src in (WORKSPACE_JSON, PROJCACHE_JSON):
        b = backup(src, stamp)
        if b:
            log(f"\n备份 {os.path.basename(src)} -> {os.path.basename(b)}")

    # ---- A. 处理会话目录 ----
    trash = os.path.join(DSH, f"_trash_{stamp}")
    done = 0
    for sid in targets:
        for d in session_dirs(sid):
            real = os.path.realpath(d)
            if not real.startswith(os.path.realpath(SESS_ROOT) + os.sep):
                log(f"  [跳过] 越界路径 {real}")
                continue
            if os.path.basename(real) != sid:
                log(f"  [跳过] 目录名与 id 不符 {real}")
                continue
            if args.purge:
                shutil.rmtree(real)
                log(f"  [删除] {sid}")
            else:
                os.makedirs(trash, exist_ok=True)
                shutil.move(real, os.path.join(trash, sid))
                log(f"  [移动] {sid} -> {trash}/{sid}")
            done += 1
    log(f"会话目录处理 {done} 条" + ("" if args.purge else f"（在 {trash}，可手动恢复）"))

    # ---- B. 清 projcache ----
    sess_tbl = projcache.setdefault("tables", {}).setdefault("sessions", {})
    removed_cache = [s for s in list(sess_tbl) if s in targets or s not in on_disk]
    for s in removed_cache:
        sess_tbl.pop(s, None)
    json.dump(projcache, open(PROJCACHE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"projcache: 删除 {len(removed_cache)} 条记录，剩 {len(sess_tbl)} 条")

    # ---- C. 清 workspace ----
    g = workspace.setdefault("global", {})
    g["archivedSessionIds"] = [s for s in archived if s not in targets and s in sess_tbl]
    for ws in (workspace.get("tables", {}).get("workspaces") or {}).values():
        ws["sessionIds"] = [s for s in (ws.get("sessionIds") or []) if s in sess_tbl]
    json.dump(workspace, open(WORKSPACE_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    log(f"workspace: archivedSessionIds 剩 {len(g['archivedSessionIds'])} 条")

    log("\n完成。重新打开 harness 即可。")
    if not args.purge:
        log(f"确认没问题后，可手动删掉回收站省空间: rm -rf {trash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())