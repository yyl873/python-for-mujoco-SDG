#!/usr/bin/env python3
"""
tools/run_pipeline.py —— 一条命令跑完三步流水线，只回一份结构化回执。

为什么要它（不是"又一个脚本"，是省 Token 的执行器）：
  原来跑一轮要 4 次工具往返（ik / grasp / render / video），每次工具都把
  几十上百行控制台日志塞回模型上下文。本执行器把"执行"留在本地进程里，
  只把每个阶段的**结论字段**（收敛迭代数、抬升毫米数、自检绿/红、产物大小）
  回给模型；全文按阶段落盘到 logs/run_<stage>.log，需要细节时再定向去读。

用法：
  python3 tools/run_pipeline.py              # 跑全部 3 段（ik/grasp/render），回执出屏
  python3 tools/run_pipeline.py ik grasp     # 只跑指定阶段
  python3 tools/run_pipeline.py --status     # 不跑，只看当前产物状态
  python3 tools/run_pipeline.py --json       # 机器可读回执（stdout 只有 JSON）
  python3 tools/run_pipeline.py --keep-going # 某段失败不中断，继续跑后面的
  python3 tools/run_pipeline.py --tail 30    # 失败时回显日志尾部行数（默认 15）

退出码：0 = 全部通过；1 = 有阶段失败或判据未满足。
自检：python3 tools/run_pipeline.py --selftest
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
OUT_DIR = ROOT / "demo"

# 本机 PYTHONHOME / PYTHONPATH 被 AppImage 挂载污染，子进程必须清掉这两个变量，
# 等价于 README 里的 `env -u PYTHONHOME -u PYTHONPATH` 前缀。
def clean_env() -> dict:
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def py() -> str:
    return os.environ.get("PIPELINE_PYTHON") or shutil.which("python3") or "python3"


# ============================================================
# 各阶段：命令 + 判据解析 + 产物
#   parse(log) -> (facts: dict, ok: bool)
#   facts 只放"结论字段"，绝不放日志正文。
# ============================================================

def parse_ik(log: str):
    f = {}
    m = re.search(r"✅ 收敛于第 (\d+) 次迭代，误差 ([\d.]+) m", log)
    if m:
        f["iter"], f["err_m"] = int(m.group(1)), float(m.group(2))
    r = re.search(r"末端残留误差[:：]\s*([\d.]+) m", log)
    if r:
        f["residual_m"] = float(r.group(1))
    status = "ok" if (m and f.get("residual_m", 1.0) <= 1e-3) else "fail"
    f["summary"] = (f"收敛 {f.get('iter')} 次迭代, 残差 {f.get('residual_m')} m"
                    if m else "未收敛")
    return f, status


def parse_grasp(log: str):
    f = {}
    m = re.search(r"抬升 ([+-]?[\d.]+) mm\s+水平位移 ([\d.]+) mm", log)
    if m:
        f["rise_mm"] = float(m.group(1))
        f["move_mm"] = float(m.group(2))
    m = re.search(r"约束1 (✅|❌)\s+约束2 (✅|❌)", log)
    if m:
        f["constraint1"], f["constraint2"] = m.group(1) == "✅", m.group(2) == "✅"
    if "✅ 抓取成功" in log:
        verdict = "抓取成功"
    elif "部分成功" in log:
        verdict = "部分成功"
    else:
        verdict = "未成功"
    f["verdict"] = verdict
    cons_ok = bool(f.get("constraint1")) and bool(f.get("constraint2"))
    if verdict != "抓取成功" or f.get("rise_mm", 0) <= 50:
        status = "fail"
    elif not cons_ok:
        status = "warn"          # 抓取成功，但触犯了设计红线里的约束
    else:
        status = "ok"
    f["verdict"] = verdict
    f["summary"] = (f"{verdict}, 抬升 {f.get('rise_mm')} mm, 搬运 {f.get('move_mm')} mm, "
                    f"约束1 {'✅' if f.get('constraint1') else '❌'}/"
                    f"约束2 {'✅' if f.get('constraint2') else '❌'}")
    return f, status


def parse_contact_audit(log: str):
    """
    接触审计（tools/probe_contact_audit.py）的解析器 —— 第七轮起取代步骤二成为"抓取"阶段的闸门。

    为什么换：步骤二 `grasp/test_grasp.py` 是历史探针，它的动作序列与生产路径（步骤三）不同
    （没有 settle_a 停顿、下探开度/起点也不同）。第七轮把指垫改成"贴合手指真实内表面"之后，
    指垫位置是按生产路径的真实到达位姿反算的，步骤二那条路径下探段会压盒 21.85 mm
    （实测），于是它自己的约束1 报 ❌ —— 这已经是"探针与生产路径分叉"的信号，
    不再适合当流水线的闸门。改为直接审计生产路径。
    """
    f = {}
    for key, pat in (("descend_contact", r"G1 下探零接触 : (ok|fail)\s+接触 (\d+)/(\d+) 步"),
                     ("press_mm", r"G2 不压桌面\s+: (ok|fail)\s+压盒 ([+-][\d.]+) mm"),
                     ("tilt_deg", r"G3 不撞倒\s+: (ok|fail)\s+夹稳时倾角 ([\d.]+)°"),
                     ("rise_mm", r"G4 真抓起来\s+: (ok|fail)\s+抬升 ([+-][\d.]+) mm"),
                     ("gap", r"G5 无缝隙\s+: (ok|fail)"),
                     ("pen_depth", r"G6 无深穿模\s+: (ok|fail)")):
        m = re.search(pat, log)
        if m:
            f[key] = m.group(1) == "ok"
            if key == "descend_contact":
                f["descend_steps_hit"], f["descend_steps_all"] = int(m.group(2)), int(m.group(3))
            elif key in ("press_mm", "tilt_deg", "rise_mm"):
                f[key] = float(m.group(2))
                f[key + "_ok"] = m.group(1) == "ok"
    gates = ["descend_contact", "pen_depth", "gap"]
    ok_all = all(f.get(k) for k in gates) and f.get("press_mm_ok") and f.get("tilt_deg_ok") and f.get("rise_mm_ok")
    status = "ok" if ok_all else "fail"
    f["summary"] = (f"G1 {'✅' if f.get('descend_contact') else '❌'} / "
                    f"G2 {f.get('press_mm')} mm / G3 {f.get('tilt_deg')}° / "
                    f"G4 {f.get('rise_mm')} mm / G5 {'✅' if f.get('gap') else '❌'} / "
                    f"G6 {'✅' if f.get('pen_depth') else '❌'}")
    return f, status


def parse_render(log: str):
    f = {}
    head, _, tail = log.partition("verify  自检")
    m = re.search(r"step=(\d+) -> (\d+) 帧", log)
    if m:
        f["frames"] = int(m.group(2))
    m = re.search(r"分辨率\s*:\s*(\d+)x(\d+)\s+fps=([\d.]+)", log)
    if m:
        f["width"], f["height"], f["fps"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    checks = re.findall(r"^\s+\[(✅|❌)\]", tail, re.M)
    if checks:
        f["checks_pass"] = checks.count("✅")
        f["checks_fail"] = checks.count("❌")
    # success 标签：负样本是"数据导出了、但这条抓取没成功"，是 warn 不是 fail
    m = re.search(r"成功标签\s+success=(true|false)", log)
    if m:
        f["success"] = (m.group(1) == "true")
    f["schema_ok"] = ("[schema] ✅" in log) and ("[schema] ❌" not in log)
    f["verify_ok"] = "总判定: ✅ 全部通过" in log
    if (not checks) or f["checks_fail"] > 0 or not f["verify_ok"] or not f["schema_ok"]:
        status = "fail"
    elif f.get("success") is False:
        status = "warn"          # 负样本：产物齐了，但抓取没成功
    else:
        status = "ok"
    f["summary"] = (f"{f.get('frames')} 帧 x {f.get('width')}x{f.get('height')}"
                    if f.get("frames") else "无帧数")
    f["summary"] += f", 自检 {f.get('checks_pass', 0)}/{len(checks)} 绿"
    f["summary"] += f", success={str(f.get('success')).lower()}"
    return f, status


STAGES = [
    {
        "name": "ik",
        "desc": "步骤一  纯 MuJoCo DLS 逆运动学",
        "cmd": ["ik/test_ik.py"],
        "parse": parse_ik,
        "artifacts": [],
    },
    {
        "name": "grasp",
        "desc": "接触审计  下探零接触/压盒/倾角/抬升/缝隙/穿模（第七轮起，取代历史探针）",
        "cmd": ["tools/probe_contact_audit.py"],
        "parse": parse_contact_audit,
        "artifacts": [],
    },
    {
        "name": "render",
        "desc": "步骤三  录制 + 渲染 + 标注（mp4；排查图走 tools/inspect_episode.py）",
        "cmd": ["render/test_render.py"],
        "parse": parse_render,
        "artifacts": [
            "env_camera.mp4", "wrist_camera.mp4", "episode.npz", "meta.json",
        ],
    },
]
STAGE_BY_NAME = {s["name"]: s for s in STAGES}


# ============================================================
# 执行 + 回执
# ============================================================

def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


def artifact_sizes(names, extra=()):
    out = {}
    for name in list(names) + list(extra):
        p = OUT_DIR / name
        if p.exists():
            out[name] = p.stat().st_size
    return out


def run_stage(stage) -> dict:
    name = stage["name"]
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"run_{name}.log"
    cmd = [py()] + stage["cmd"]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=clean_env(),
                          capture_output=True, text=True, errors="replace")
    secs = time.time() - t0
    log = (proc.stdout or "") + (proc.stderr or "")
    log_path.write_text(log, encoding="utf-8")
    facts, status = stage["parse"](log)
    if proc.returncode != 0:
        status = "fail"
    return {
        "name": name,
        "desc": stage["desc"],
        "status": status,
        "exit": proc.returncode,
        "secs": round(secs, 1),
        "log": str(log_path.relative_to(ROOT)),
        "lines": log.count("\n"),
        "facts": facts,
        "artifacts": artifact_sizes(stage["artifacts"]),
    }


ICON = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}


def print_receipt(results, total_secs) -> None:
    n = {k: sum(1 for r in results if r["status"] == k) for k in ("ok", "warn", "fail")}
    head = "全部通过" if n["ok"] == len(results) else f"ok {n['ok']} / warn {n['warn']} / fail {n['fail']}"
    print(f"[pipeline] {ICON['ok' if n['ok'] + n['warn'] == len(results) else 'fail']} "
          f"{len(results)} 段 {head}   总耗时 {total_secs:.1f} s")
    for r in results:
        print(f"  {r['name']:<7}{ICON[r['status']]} {r['facts'].get('summary', ''):<44}"
              f"{r['secs']:>6.1f}s  log:{r['lines']}行")
    # 产物一行汇总（"200 个文件全部成功" 式回执，不逐条列）
    total_b = 0
    files = 0
    for r in results:
        for _, b in r["artifacts"].items():
            total_b += b
            files += 1
    print(f"  本次产物 demo/ 关键文件 {files} 个 / {human(total_b)}"
          f"   全文日志 logs/run_*.log")
    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        print(f"\n  非绿 {len(bad)} 段: "
              + ", ".join(f"{r['name']}({r['status']})" for r in bad)
              + f"  -> 定向读 {bad[0]['log']} 尾部即可")


def emit_json(results, total_secs) -> None:
    payload = {
        "ok": all(r["status"] == "ok" for r in results),
        "status_rank": max((["ok", "warn", "fail"].index(r["status"]) for r in results),
                           default=0),
        "total_secs": round(total_secs, 1),
        "stages": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def status_report() -> int:
    """不跑流水线，只回当前 demo/ 状态 + 上次回执。"""
    print(f"[status] {OUT_DIR}")
    if not OUT_DIR.exists():
        print("  (demo/ 不存在，先跑一次流水线)")
        return 1
    rows = []
    for r in STAGES:
        for name in r["artifacts"]:
            p = OUT_DIR / name
            if p.exists():
                age = time.time() - p.stat().st_mtime
                rows.append((name, p.stat().st_size, age))
    big = sorted(rows, key=lambda x: -x[1])[:6]
    print(f"  关键产物 {len(rows)} 个 / {human(sum(s for _, s, _ in rows))}")
    for name, size, age in big:
        print(f"    {name:<24}{human(size):>10}   {age/60:6.1f} 分钟前")
    cands = [p for p in (LOG_DIR / "run_last.json", LOG_DIR / "run_all.json") if p.exists()]
    if cands:
        p = max(cands, key=lambda x: x.stat().st_mtime)
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            label = {0: "✅ 全绿", 1: "⚠️  有 warn（见 HANDOFF §5.2）",
                     2: "❌ 有 fail"}.get(d.get("rank"), "?")
            names = "+".join(s["name"] for s in d.get("stages", []))
            print(f"  上次运行: {label}  {d.get('total_secs')} s  [{names}]")
        except Exception:
            pass
    return 0


def selftest() -> int:
    """用假的日志文本校验解析器：不跑仿真，秒级完成。"""
    ik_log = ("iter   err(m)   q(rad)\n"
              "  24    0.000900\n"
              "✅ 收敛于第 24 次迭代，误差 0.000923 m < 0.001 m\n"
              "  末端残留误差: 0.000923 m\n")
    f, st = parse_ik(ik_log)
    assert st == "ok" and f["iter"] == 24 and f["residual_m"] == 0.000923, (f, st)

    grasp_log = ("  抬升 +122.22 mm   水平位移 218.06 mm\n"
                 "  约束1 ✅   约束2 ✅\n"
                 "\n  ✅ 抓取成功: 药盒被抬离, 两指均有接触\n")
    f, st = parse_grasp(grasp_log)
    assert st == "ok" and f["rise_mm"] == 122.22 and f["move_mm"] == 218.06, (f, st)

    # 抓取成功但约束红 -> 必须是 warn，不能混成 ok 也不能压成 fail
    f, st = parse_grasp(grasp_log.replace("约束1 ✅", "约束1 ❌"))
    assert st == "warn" and f["constraint1"] is False, (f, st)

    # 没抓住 -> fail
    f, st = parse_grasp("  抬升 +2.00 mm   水平位移 5.0 mm\n  ⚠️  部分成功\n")
    assert st == "fail", (f, st)

    render_log = ("  轨迹      : demo/episode.npz  (13650 帧, step=30 -> 455 帧)\n"
                  "  分辨率    : 640x480  fps=30.00\n"
                  "verify  自检\n"
                  "  [✅] env_camera      455 帧  30.00 fps  640x480  5156 KB\n"
                  "  [✅] 木盒未受扰  臂↔木盒接触帧 0 (需 0)\n"
                  "  [schema] ✅ 轨迹 npz 形状/dtype 全部符合\n"
                  "  总判定: ✅ 全部通过\n")
    f, st = parse_render(render_log)
    assert st == "ok" and f["frames"] == 455 and f["checks_pass"] == 2, (f, st)

    bad_render = render_log.replace("[✅] 木盒未受扰", "[❌] 木盒未受扰") \
                            .replace("总判定: ✅ 全部通过", "总判定: ❌ 有项目未通过")
    f, st = parse_render(bad_render)
    assert st == "fail" and f["checks_fail"] == 1, (f, st)

    print("[selftest] ✅ 5 项解析判据全部符合预期（含三态 warn；video 段已退役）")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="一条命令跑完三步流水线，只回结构化回执（详细日志落 logs/run_*.log）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python3 tools/run_pipeline.py            # 全跑\n"
               "  python3 tools/run_pipeline.py ik grasp   # 只跑前两段\n"
               "  python3 tools/run_pipeline.py --status   # 只看产物状态\n"
               "  python3 tools/run_pipeline.py --json     # JSON 回执\n")
    ap.add_argument("stages", nargs="*", metavar="STAGE",
                    help=f"要跑的阶段: {', '.join(STAGE_BY_NAME)}（默认全部）")
    ap.add_argument("--json", action="store_true", help="stdout 只输出 JSON 回执")
    ap.add_argument("--status", action="store_true", help="不执行，只回产物状态")
    ap.add_argument("--keep-going", action="store_true", help="某段失败后继续跑后面的")
    ap.add_argument("--tail", type=int, default=15,
                    help="失败时回显该段日志尾部行数（0=不回显，默认 15）")
    ap.add_argument("--selftest", action="store_true", help="只校验解析判据，不跑仿真")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if args.status:
        return status_report()

    wanted = args.stages or [s["name"] for s in STAGES]
    unknown = [w for w in wanted if w not in STAGE_BY_NAME]
    if unknown:
        ap.error(f"未知阶段 {unknown}，可选: {list(STAGE_BY_NAME)}")

    results = []
    t0 = time.time()
    for name in wanted:
        r = run_stage(STAGE_BY_NAME[name])
        results.append(r)
        # warn = 脚本自判成功但触犯约束，流水线继续；fail 才中断（除非 --keep-going）
        if r["status"] == "fail" and not args.keep_going:
            break

    total = time.time() - t0
    rank = max((["ok", "warn", "fail"].index(x["status"]) for x in results), default=0)
    LOG_DIR.mkdir(exist_ok=True)
    # 只有"全跑"才覆盖 run_all.json；部分跑另存 run_last.json，
    # 免得上一次完整回执被单段调试覆盖掉、--status 误报"上次运行"。
    full = [s["name"] for s in STAGES] == wanted
    receipt_name = "run_all.json" if full else "run_last.json"
    (LOG_DIR / receipt_name).write_text(
        json.dumps({"ok": rank == 0, "rank": rank, "total_secs": round(total, 1),
                    "stages": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.json:
        emit_json(results, total)
    else:
        print_receipt(results, total)
        # 只回显第一段非绿日志的尾部，避免上下文里堆多份日志
        first_bad = next((r for r in results if r["status"] != "ok"), None)
        if first_bad and args.tail:
            tail = (ROOT / first_bad["log"]).read_text(encoding="utf-8", errors="replace")
            tail_lines = tail.rstrip().splitlines()[-args.tail:]
            print(f"\n  --- {first_bad['log']} 末尾 {len(tail_lines)} 行 ---")
            for line in tail_lines:
                print(f"  {line}")

    return 0 if rank == 0 else (1 if rank == 1 else 2)


if __name__ == "__main__":
    sys.exit(main())
