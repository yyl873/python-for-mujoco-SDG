"""给任意入口脚本出一张"仓库内函数成本表"——用来决定该读哪 20 行。

背景：render/test_render.py 136 KB、collect/collect.py 104 KB。逐行读不现实，
但一次运行实际走过的"本仓库函数"通常只有几十个。这张表就是那几十个。

用法（输出必须重定向到文件，别管道给 head/tail，本机会 SIGFPE）：

    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 tools/callmap.py ik/test_ik.py > logs/map_ik.txt 2>&1
    env -u PYTHONHOME -u PYTHONPATH python3 tools/callmap.py grasp/test_grasp.py > logs/map_grasp.txt 2>&1
    MUJOCO_GL=egl env -u PYTHONHOME -u PYTHONPATH python3 tools/callmap.py \\
        tools/probe_contact_audit.py > logs/map_audit.txt 2>&1

读法（重要）：
  - 按 cumtime 排序 = 真实主干（先读这些）；
  - ncalls 大 = 内层循环（通常是 mj_step / 逐帧循环）；
  - tottime 接近 cumtime = 该函数自己干重活，是真正值得读的实现；
  - cumtime 大但 tottime 小 = 只是调度壳子，可以直接跳过。

它只做观测：不改代码、不写数据、不发 ctrl。
"""
import cProfile
import os
import pstats
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _norm(fn: str) -> str:
    if not os.path.isabs(fn):
        fn = os.path.join(os.getcwd(), fn)
    return os.path.normpath(fn)


def _is_repo(fn: str) -> bool:
    if not fn or fn.startswith("<") or fn == "~":
        return False  # <frozen importlib...> / <built-in> 这类不是真文件
    n = _norm(fn)
    return (
        n.startswith(ROOT)
        and os.path.isfile(n)
        and "__pycache__" not in n
        and os.sep + ".cache" + os.sep not in n
    )


def report(prof: cProfile.Profile, top: int = 40) -> str:
    st = pstats.Stats(prof)
    rows = []
    for (fn, line, name), (_cc, nc, tt, ct, _callers) in st.stats.items():
        if not _is_repo(fn):
            continue
        rel = os.path.relpath(_norm(fn), ROOT)
        rows.append((ct, nc, tt, f"{rel}:{line}", name))
    rows.sort(key=lambda r: -r[0])
    out = [f"仓库内函数 {len(rows)} 个（按 cumtime 降序，前十是主干）", ""]
    out.append(f"{'cumtime_s':>10} {'ncalls':>8} {'tottime_s':>10}  location")
    for ct, nc, tt, loc, name in rows[:top]:
        out.append(f"{ct:10.3f} {nc:8d} {tt:10.3f}  {loc}  {name}()")
    return "\n".join(out)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    target, targs = argv[0], argv[1:]
    if not os.path.exists(target):
        print(f"找不到入口脚本: {target}", file=sys.stderr)
        return 2

    prof = cProfile.Profile()
    saved_argv = sys.argv[:]
    sys.argv = [target] + targs
    prof.enable()
    rc = 0
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as e:
        rc = int(e.code or 0)
    finally:
        prof.disable()
        sys.argv = saved_argv

    print("\n" + "=" * 78)
    print(report(prof))
    return rc


if __name__ == "__main__":
    sys.exit(main())