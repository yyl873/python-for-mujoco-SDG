#!/usr/bin/env python3
"""
tools/digest.py —— 用摘要代替整读，把"读文件"的上下文成本压到可控。

解决的具体毛病：本仓库 test_render.py 有 70 KB、输出日志动辄上百行，
整篇塞进上下文既贵又没用（99% 的行当次读都不用）。
本工具只回三层信息之一：
  1. 目录 -> 文件数 / 体积 / 扩展名分布 / 最大的几个文件      （不列全部）
  2. 源文件 -> 定义骨架（def/class + 行号）+ 行数              （不贴正文）
  3. --grep -> 只回命中数量 + 首尾行号                        （不贴命中列表）
需要正文时再用 read 定向读 [起, 止] 行号区间。

用法：
  python3 tools/digest.py output                      # 目录摘要
  python3 tools/digest.py render/test_render.py  # 骨架
  python3 tools/digest.py logs/run_render.log --tail 8 # 只看尾部
  python3 tools/digest.py render/test_render.py --grep "def verify"
  python3 tools/digest.py . --json                    # 机器可读
  python3 tools/digest.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
TEXT_EXT = {".py", ".txt", ".md", ".xml", ".json", ".log", ".csv", ".sh", ".yaml", ".yml"}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def digest_dir(root: Path, max_files: int = 5) -> dict:
    files = [p for p in root.rglob("*")
             if p.is_file() and not (set(p.parts) & SKIP_DIRS)]
    by_ext, by_sub, sizes = {}, {}, []
    for p in files:
        ext = p.suffix or "(无扩展名)"
        e = by_ext.setdefault(ext, [0, 0])
        e[0] += 1
        e[1] += p.stat().st_size
        rel = p.relative_to(root)
        sub = rel.parts[0] if len(rel.parts) > 1 else "."
        s = by_sub.setdefault(sub, [0, 0])
        s[0] += 1
        s[1] += p.stat().st_size
        sizes.append((str(rel), p.stat().st_size))
    sizes.sort(key=lambda x: -x[1])
    return {
        "path": str(root),
        "files": len(files),
        "bytes": sum(s for _, s in sizes),
        "by_ext": dict(sorted(by_ext.items(), key=lambda kv: -kv[1][1])),
        "by_subdir": dict(sorted(by_sub.items(), key=lambda kv: -kv[1][1])),
        "largest": sizes[:max_files],
    }


DEF_RE = re.compile(r"^(?P<indent>\s*)(?:async\s+)?(?:def|class)\s+(?P<name>\w+)")
TOP_RE = re.compile(r"^(?P<name>[A-Z][A-Z0-9_]{2,})\s*=")


def digest_file(path: Path, head: int = 0, tail: int = 0, outline_max: int = 40) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    defs = []
    for i, line in enumerate(lines, 1):
        m = DEF_RE.match(line)
        if m:
            kind = "class" if line.lstrip().startswith("class") else "def"
            depth = len(m.group("indent")) // 4
            defs.append({"line": i, "depth": depth, "kind": kind, "name": m.group("name")})
    consts = [{"line": i, "name": m.group("name")} for i, line in enumerate(lines, 1)
              if (m := TOP_RE.match(line))]
    out = {
        "path": str(path),
        "lines": len(lines),
        "bytes": path.stat().st_size,
        "defs_count": len(defs),
        "defs": defs[:outline_max],
        "defs_truncated": max(0, len(defs) - outline_max),
        "top_consts": consts[:15],
    }
    if head:
        out["head"] = lines[:head]
    if tail:
        out["tail"] = lines[-tail:]
    return out


def grep_file(path: Path, pattern: str) -> dict:
    """只回数量与首尾行号，不回命中正文。"""
    rx = re.compile(pattern)
    hits = []
    for i, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if rx.search(line):
            hits.append(i)
    return {
        "path": str(path),
        "pattern": pattern,
        "count": len(hits),
        "first": hits[:5],
        "last": hits[-1:] if hits else [],
    }


def print_dir(d: dict) -> None:
    print(f"[dir] {d['path']}   {d['files']} 个文件 / {human(d['bytes'])}")
    print("  扩展名: " + ", ".join(f"{k} {v[0]}个/{human(v[1])}"
                                  for k, v in list(d["by_ext"].items())[:6]))
    print("  子目录: " + ", ".join(f"{k} {v[0]}个"
                                  for k, v in list(d["by_subdir"].items())[:8]))
    print("  最大文件: " + ", ".join(f"{k} {human(s)}" for k, s in d["largest"]))


def print_file(d: dict) -> None:
    print(f"[file] {d['path']}   {d['lines']} 行 / {human(d['bytes'])}"
          f"   {d['defs_count']} 个 def/class")
    if d["top_consts"]:
        print("  顶层常量: " + ", ".join(c["name"] for c in d["top_consts"]))
    for x in d["defs"]:
        print(f"    {x['line']:>5}  {'  ' * x['depth']}{x['kind']} {x['name']}")
    if d["defs_truncated"]:
        print(f"    ... 还有 {d['defs_truncated']} 个未列出（--outline 调大）")
    for key in ("head", "tail"):
        if key in d:
            print(f"  {key} {len(d[key])} 行:")
            for line in d[key]:
                print(f"    {line[:100]}")


def selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "m.py"
        p.write_text("MAX = 3\n\ndef foo():\n    pass\n\nclass Bar:\n"
                     "    def baz(self):\n        return 1\n", encoding="utf-8")
        d = digest_file(p)
        names = [x["name"] for x in d["defs"]]
        assert d["lines"] == 8 and d["defs_count"] == 3, d
        assert names == ["foo", "Bar", "baz"], names
        assert d["top_consts"][0]["name"] == "MAX", d["top_consts"]
        g = grep_file(p, r"def\s+\w+")
        assert g["count"] == 2 and g["first"] == [3, 7], g
        sub = Path(td) / "sub"
        sub.mkdir()
        (sub / "a.log").write_text("x\ny\n", encoding="utf-8")
        dd = digest_dir(Path(td))
        assert dd["files"] == 2 and dd["by_subdir"]["sub"][0] == 1, dd
    print("[selftest] ✅ digest 目录/骨架/计数三项判据符合预期")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="用摘要代替整读：目录回统计、源文件回骨架、--grep 只回数量",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", help="目录或文件路径")
    ap.add_argument("--head", type=int, default=0, help="附前 N 行正文")
    ap.add_argument("--tail", type=int, default=0, help="附后 N 行正文（看日志优先用它）")
    ap.add_argument("--outline", type=int, default=40, help="最多列多少个 def/class")
    ap.add_argument("--grep", metavar="PAT", help="只回命中数量与首尾行号")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--selftest", action="store_true", help="自检，不需要真实文件")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not args.path:
        ap.error("需要一个路径（或 --selftest）")

    p = Path(args.path)
    if not p.exists():
        print(f"[digest] ❌ 路径不存在: {p}")
        return 2

    if p.is_dir():
        d = digest_dir(p)
        if args.json:
            print(json.dumps(d, ensure_ascii=False, indent=2))
        else:
            print_dir(d)
        return 0

    if args.grep:
        g = grep_file(p, args.grep)
        if args.json:
            print(json.dumps(g, ensure_ascii=False, indent=2))
        else:
            print(f"[grep] {p.name}  /{args.grep}/  命中 {g['count']} 处"
                  f"   首: {g['first']}  末: {g['last']}")
        return 0

    d = digest_file(p, head=args.head, tail=args.tail, outline_max=args.outline)
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        print_file(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
