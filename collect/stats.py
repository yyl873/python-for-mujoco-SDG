#!/usr/bin/env python3
"""
stats.py —— 覆盖度统计：回答"SDG 到底刷够了没有"
================================================================

为什么需要它
------------
`index.json` 只有 `accept_rate` 和丢弃原因分布 —— 这两项都只说明"跑通了"，
不说明"**随机化的每一维都真的被覆盖到了**"。没有这个，`--n 1000` 就是盲刷：
可能 1000 条里 θ 全挤在 0 附近（采样器在某处退化），也可能成功率在 θ=±10° 掉了 20 个点
（同构解在几何极限处失效）—— 这两种情况都不影响 accept_rate 好看，但都直接决定模型泛化。

本脚本给出三样东西
  1) 每一维的直方图（默认 10 箱）+ 每箱样本数 → 一眼看出有没有"空箱/偏斜"；
  2) 成功率随关键维度的变化（带 n 与 Wilson 95% 区间）→ 看难度梯度；
  3) `timing_scale` → `sim_time_s` 的线性回归斜率 → 验证"节奏随机"真的体现在时长上
     （斜率≈0 就说明 timing_scale 是个空转的随机量）。

产物
----
    scripts/stats.json      机器可读（CI 里用 --fail-on-undercovered 卡住）
    scripts/stats.png       6 联图（matplotlib；缺库就跳过，不影响 json）
    scripts/DATA_CARD.md    数据卡：版本/规模/覆盖度/已知限制（自动生成，别手改）

用法
----
    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py
    env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py --min-per-bin 3   # 覆盖不足线
    env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py --fail-on-undercovered
    env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py --no-plot --card
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"

# 随机化维度：key 在索引条目的 keys 里，label 是给人看的名字
DIMS = [
    ("theta_deg", "场景旋转 θ", "°"),
    ("dr_mm", "径向抖动 dr", "mm"),
    ("yaw_extra_deg", "药盒额外偏航", "°"),
    ("timing_scale", "节奏系数", "×"),
    ("light_azim_deg", "光照方位角", "°"),
    ("light_tilt_deg", "光照俯仰", "°"),
    ("light_diffuse", "光照强度", "-"),
    ("material_gain", "材质增益", "-"),
    ("env_fovy_delta_deg", "相机 fovy 偏差", "°"),
    ("env_yaw_delta_deg", "相机 yaw 偏差", "°"),
]
SENSOR_DIMS = [
    ("noise_sigma", "传感器噪声 σ", "DN"),
    ("gain_jitter", "曝光抖动", "-"),
    ("motion_k", "运动虚影系数", "-"),
    ("vignette", "暗角", "-"),
    ("color_jitter", "色彩抖动", "-"),
]
OUTCOMES = [
    ("rise_mm", "抬升", "mm"),
    ("hdisp_mm", "水平位移", "mm"),
    ("sim_time_s", "时长", "s"),
    ("drop_mm", "落盒高度", "mm"),
    ("home_dq_max_deg", "归位误差", "°"),
    ("moving_ratio", "运动帧占比", "-"),
]


def _atomic_write(path, obj, indent=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)
    return path


def wilson(k, n, z=1.96):
    """二项比例的 Wilson 区间（小样本比正态近似靠谱得多）。n=0 返回 (0,1)。"""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def load_keys(index):
    """取每条样本的随机量投影（keys）；老数据没有 keys 时退回读 per-episode json。"""
    rows = list(index.get("accepted") or []) + list(index.get("rejected") or [])
    out = []
    for e in rows:
        k = dict(e.get("keys") or {})
        k["episode_id"] = int(e["episode_id"])
        k.setdefault("success", bool(e.get("success", e.get("accepted"))))
        k.setdefault("accepted", bool(e.get("accepted")))
        m = e.get("metrics") or {}
        for f in ("rise_mm", "hdisp_mm", "sim_time_s", "drop_mm",
                  "home_dq_max_deg", "moving_ratio", "frames"):
            k.setdefault(f, m.get(f))
        if k.get("theta_deg") is None:
            jp = (e.get("paths") or {}).get("json")
            if jp and (ROOT / jp).exists():
                try:
                    mm = json.loads((ROOT / jp).read_text(encoding="utf-8"))
                    sp = (mm.get("randomization") or {}).get("spawn") or {}
                    k["theta_deg"] = sp.get("theta_deg")
                    k["dr_mm"] = sp.get("dr_mm")
                    k["timing_scale"] = (mm.get("randomization") or {}).get("timing_scale")
                    k["legacy"] = True
                except Exception:                            # noqa: BLE001
                    pass
        out.append(k)
    return out


def hist(values, bins=10):
    vals = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if vals.size == 0:
        return dict(n=0, bins=[], counts=[])
    if bins == 1 or np.ptp(vals) == 0:
        edges = np.array([vals.min() - 1e-9, vals.max() + 1e-9])
    else:
        edges = np.linspace(vals.min(), vals.max(), bins + 1)
    cnt, _ = np.histogram(vals, bins=edges)
    return dict(n=int(vals.size), min=float(vals.min()), max=float(vals.max()),
                mean=float(vals.mean()), std=float(vals.std()),
                edges=[float(v) for v in edges], counts=[int(c) for c in cnt])


def success_by(rows, key, bins=4):
    """按某个维度分箱看成功率（带 Wilson 区间）。返回可直接 json 的列表。"""
    pairs = [(r.get(key), bool(r.get("success"))) for r in rows]
    pairs = [(v, s) for v, s in pairs if v is not None and np.isfinite(v)]
    if len(pairs) < 2:
        return []
    vals = np.array([p[0] for p in pairs], float)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-9:
        edges = np.array([lo - 1e-9, hi + 1e-9])
    else:
        edges = np.linspace(lo, hi, bins + 1)
    out = []
    for i in range(len(edges) - 1):
        sel = [s for v, s in pairs if edges[i] <= v < edges[i + 1] or
               (i == len(edges) - 2 and v == edges[-1])]
        n = len(sel)
        k = int(sum(sel))
        ci_lo, ci_hi = wilson(k, n)
        out.append(dict(lo=float(edges[i]), hi=float(edges[i + 1]), n=n, n_success=k,
                        rate=(k / n if n else None),
                        ci=[round(ci_lo, 3), round(ci_hi, 3)]))
    return out


def linfit(x, y):
    """最小二乘 y = a + b x（不引 scipy）。返回 (b, a, r, n)。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 3 or np.ptp(x) == 0:
        return None, None, None, n
    b, a = np.polyfit(x, y, 1)
    pred = a + b * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(b), float(a), float(r), n


def coverage_verdict(dim_stats, min_per_bin):
    """哪些箱样本太少 —— 这就是"覆盖不足"的可执行判据。"""
    bad = []
    for name, h in dim_stats.items():
        if not h.get("counts"):
            continue
        empty = sum(1 for c in h["counts"] if c == 0)
        thin = sum(1 for c in h["counts"] if 0 < c < min_per_bin)
        if empty or thin:
            bad.append(dict(dim=name, n_bins=len(h["counts"]), empty_bins=empty,
                            thin_bins=thin, min_per_bin=min_per_bin,
                            counts=h["counts"]))
    return bad


def make_plot(rep, path):
    """6 联图。matplotlib 不可用时静默跳过（json 里已经有全部数据）。"""
    cache = ROOT / ".mplcache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*missing from current font.*")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:                                 # noqa: BLE001
        print(f"  ⚠️ matplotlib 不可用（{type(exc).__name__}），跳过 PNG")
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()

    def _bar(a, h, title, unit):
        if not h.get("counts"):
            a.set_title(title + "（无数据）")
            return
        edges = h["edges"]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(len(edges) - 1)]
        a.bar(centers, h["counts"], width=(edges[1] - edges[0]) * 0.85, color="#4C78A8")
        a.set_title(f"{title}  (n={h['n']})")
        a.set_xlabel(unit)
        a.set_ylabel("episodes")

    h = rep["dims"].get("theta_deg", {})
    _bar(ax[0], h, "scene rotation theta", "deg")
    if rep["success_vs_theta"]:
        xs = [(b["lo"] + b["hi"]) / 2 for b in rep["success_vs_theta"]]
        ys = [b["rate"] if b["rate"] is not None else np.nan for b in rep["success_vs_theta"]]
        ns = [b["n"] for b in rep["success_vs_theta"]]
        ax[1].plot(xs, ys, "o-", color="#E45756")
        for x, y, n in zip(xs, ys, ns):
            ax[1].annotate(f"n={n}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
        ax[1].set_ylim(-0.05, 1.05)
        ax[1].set_title("success rate vs theta (edge check)")
        ax[1].set_xlabel("theta (deg)")
        ax[1].set_ylabel("success rate")
        ax[1].grid(alpha=0.3)
    else:
        ax[1].set_title("success vs theta (too few samples)")

    _bar(ax[2], rep["dims"].get("timing_scale", {}), "timing scale", "x")
    _bar(ax[3], rep["dims"].get("light_azim_deg", {}), "light azimuth", "deg")
    _bar(ax[4], rep["outcomes"].get("rise_mm", {}), "lift distribution", "mm")

    fit = rep["timing_vs_duration"]
    xs = [p["timing_scale"] for p in rep["scatter_timing"]]
    ys = [p["sim_time_s"] for p in rep["scatter_timing"]]
    ax[5].scatter(xs, ys, s=12, alpha=0.6, color="#54A24B")
    if fit.get("slope") is not None:
        xr = np.linspace(min(xs), max(xs), 10) if xs else []
        ax[5].plot(xr, fit["intercept"] + fit["slope"] * xr, "k--",
                   label=f"slope {fit['slope']:.2f} s/x (r={fit['r']:.2f})")
        ax[5].legend(fontsize=8)
    ax[5].set_title("timing_scale -> sim duration")
    ax[5].set_xlabel("timing_scale")
    ax[5].set_ylabel("sim_time_s")

    fig.suptitle(f"SDG coverage  {rep['dataset_version']}  n={rep['n']}  "
                 f"accept {rep['accept_rate']*100:.0f}%  (labels in EN: no CJK font)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def build_card(rep, under):
    """自动数据卡（DATA_CARD.md）。人读的版本，数字全部来自 stats.json。"""
    L = []
    L.append(f"# 数据卡 · {rep['dataset_version']}")
    L.append("")
    L.append("> 本文件由 `collect/stats.py --card` 自动生成，不要手改。")
    L.append(f"> 生成时间 {rep['generated_at']}，数据源 `scripts/index.json`。")
    L.append("")
    L.append("## 1. 规模")
    L.append("")
    L.append(f"| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 样本数 | {rep['n']} |")
    L.append(f"| 正样本（success） | {rep['n_success']} |")
    L.append(f"| 负样本（failure） | {rep['n'] - rep['n_success']} |")
    L.append(f"| 接受率 | {rep['accept_rate']*100:.1f}% |")
    L.append(f"| 有视频的样本 | {rep['n_video']} |")
    L.append(f"| schema_version | {rep.get('schema_version')} |")
    L.append(f"| 模型指纹 | `{json.dumps(rep.get('mjcf_sha256') or {}, ensure_ascii=False)}` |")
    L.append("")
    L.append("## 2. 覆盖度（每维 10 箱）")
    L.append("")
    L.append("| 维度 | n | min | max | 空箱 | 偏薄箱(<min_per_bin) |")
    L.append("|---|---|---|---|---|---|")
    bad = {b["dim"]: b for b in under}
    for key, label, unit in DIMS + SENSOR_DIMS:
        h = rep["dims"].get(key)
        if not h or not h.get("counts"):
            continue
        b = bad.get(key) or {}
        L.append(f"| {label} ({unit}) | {h['n']} | {h['min']:.3g} | {h['max']:.3g} | "
                 f"{b.get('empty_bins', 0)} | {b.get('thin_bins', 0)} |")
    L.append("")
    L.append("## 3. 成功率梯度")
    L.append("")
    for name in ("theta_deg", "dr_mm"):
        rows = rep.get(f"success_vs_{name.split('_')[0]}") or []
        if not rows:
            continue
        L.append(f"**{name}**")
        L.append("")
        L.append("| 区间 | n | 成功 | 成功率 | Wilson 95% |")
        L.append("|---|---|---|---|---|")
        for b in rows:
            rate = "-" if b["rate"] is None else f"{b['rate']*100:.0f}%"
            L.append(f"| [{b['lo']:.2f}, {b['hi']:.2f}) | {b['n']} | {b['n_success']} | "
                     f"{rate} | [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}] |")
        L.append("")
    fit = rep["timing_vs_duration"]
    L.append("## 4. 节奏随机是否真的生效")
    L.append("")
    if fit.get("slope") is None:
        L.append("样本不足，无法回归。")
    else:
        L.append(f"`sim_time_s ≈ {fit['intercept']:.2f} + {fit['slope']:.2f} × timing_scale`"
                 f"（r²={fit['r']:.3f}, n={fit['n']}）。")
        if abs(fit["slope"]) < 1.0:
            L.append("⚠️ 斜率 ≈ 0：`timing_scale` 是个空转的随机量，需要回头看步数分配。")
        else:
            L.append("斜率 > 1 s/×：节奏随机确实体现在时长上（自由段步数在变）。")
    L.append("")
    L.append("## 5. 已知限制")
    L.append("")
    L.append("- 物理/画面由 MuJoCo 软件光栅（EGL/mesa）生成，不是真机；域间隙未量化。")
    L.append("- 摩擦系数固定 μ=1.2（TPU 实测标定），未随机化 —— 等于假设同一台机器人、同一批指垫。")
    L.append("- 场景旋转上限 ±10° 由绿垫几何反算；超出会出垫，因此泛化轴到此为止。")
    L.append("- 抓取接近临界稳定（曲面指垫夹 15mm 薄边），失败样本是真实失败模式，不要当噪声删掉。")
    L.append("- 视频为 MJPG 有损；要逐像素精确用 `--keep-frames` 的 `frames.npz`。")
    L.append("")
    L.append("## 6. 划分")
    L.append("")
    sp = rep.get("splits")
    if not sp:
        L.append("尚未切分（跑 `collect/split.py`）。")
    else:
        L.append(f"策略 `{sp.get('policy')}`，seed={sp.get('seed')}，"
                 f"train {sp['n'].get('train')} / val {sp['n'].get('val')} / "
                 f"test {sp['n'].get('test')}，划分哈希 `{sp.get('files_sha256')}`。")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="SDG 覆盖度统计")
    ap.add_argument("--index", default=str(SCRIPTS_DIR / "index.json"))
    ap.add_argument("--out", default=str(SCRIPTS_DIR / "stats.json"))
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--min-per-bin", type=int, default=2,
                    help="每个箱子少于这么多条就算「偏薄」（默认 2）")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--card", action="store_true", help="额外生成 scripts/DATA_CARD.md")
    ap.add_argument("--fail-on-undercovered", action="store_true",
                    help="有空箱/偏薄箱时 exit 1（CI 用）")
    args = ap.parse_args(argv)

    idx_path = Path(args.index)
    if not idx_path.exists():
        print(f"❌ 找不到 {idx_path}：先跑 collect/collect.py 或 --rebuild-index")
        return 2
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    rows = load_keys(index)
    if not rows:
        print("⚠️ 数据集中没有样本：无可统计内容（先刷数据）")
        _atomic_write(args.out, dict(n=0, note="empty dataset",
                                     generated_at=str(np.datetime64("now"))))
        return 0

    dims = {}
    for key, label, unit in DIMS:
        dims[key] = hist([r.get(key) for r in rows], args.bins)
    for key, label, unit in SENSOR_DIMS:
        dims[key] = hist([(r.get("sensor") or {}).get(key) for r in rows], args.bins)
    outcomes = {}
    for key, label, unit in OUTCOMES:
        outcomes[key] = hist([r.get(key) for r in rows], args.bins)

    ts = [r.get("timing_scale") for r in rows]
    dur = [r.get("sim_time_s") for r in rows]
    slope, intercept, rr, nfit = linfit(ts, dur)
    scatter = [dict(timing_scale=float(a), sim_time_s=float(b))
               for a, b in zip(ts, dur) if a is not None and b is not None]

    n = len(rows)
    n_success = sum(1 for r in rows if r.get("success"))
    under = coverage_verdict(dims, args.min_per_bin)

    rep = dict(
        generated_at=str(np.datetime64("now")),
        dataset_version=index.get("dataset_version"),
        schema_version=index.get("schema_version"),
        n=n, n_success=n_success, n_failed=n - n_success,
        accept_rate=index.get("accept_rate", n_success / max(1, n)),
        n_video=sum(1 for r in rows if r.get("has_video")),
        mjcf_sha256=index.get("mjcf_sha256"),
        dims=dims, outcomes=outcomes,
        success_vs_theta=success_by(rows, "theta_deg", 4),
        success_vs_dr=success_by(rows, "dr_mm", 3),
        success_by_grasp_kind={k: dict(n=sum(1 for r in rows if r.get("grasp_kind") == k),
                                       n_success=sum(1 for r in rows
                                                     if r.get("grasp_kind") == k and r.get("success")))
                               for k in {r.get("grasp_kind") for r in rows} if k},
        timing_vs_duration=dict(slope=slope, intercept=intercept, r=rr, n=nfit),
        scatter_timing=scatter,
        undercovered=under,
        min_per_bin=args.min_per_bin,
        splits=index.get("splits"),
        reject_reasons={},
    )
    from collections import Counter
    rep["reject_reasons"] = dict(Counter(
        r0 for e in (index.get("rejected") or []) for r0 in (e.get("reasons") or [])).most_common())

    # ---- 文本表格 ----
    print("=" * 78)
    print("  SDG 覆盖度统计")
    print("=" * 78)
    print(f"  数据版本 : {rep['dataset_version']}   样本 {n}（成功 {n_success} / 失败 "
          f"{n-n_success}）  有视频 {rep['n_video']}  接受率 {rep['accept_rate']*100:.0f}%")
    print()
    print(f"  {'维度':<18}{'n':>5} {'min':>9} {'max':>9} {'mean':>9}   直方图（每箱条数）")
    for key, label, unit in DIMS + SENSOR_DIMS:
        h = dims[key]
        if not h.get("counts"):
            print(f"  {label:<18}{0:>5}  （无数据）")
            continue
        spark = "".join(str(min(c, 9)) if c < 10 else "+" for c in h["counts"])
        print(f"  {label:<16}({unit}){h['n']:>5} {h['min']:>9.3g} {h['max']:>9.3g}"
              f" {h['mean']:>9.3g}   {spark}")
    print()
    print(f"  {'结果指标':<18}{'n':>5} {'min':>9} {'max':>9} {'mean':>9}")
    for key, label, unit in OUTCOMES:
        h = outcomes[key]
        if not h.get("counts"):
            continue
        print(f"  {label:<16}({unit}){h['n']:>5} {h['min']:>9.3g} {h['max']:>9.3g} {h['mean']:>9.3g}")
    print()
    print("  成功率随 θ：")
    for b in rep["success_vs_theta"]:
        r = "-" if b["rate"] is None else f"{b['rate']*100:5.1f}%"
        print(f"    [{b['lo']:+6.2f},{b['hi']:+6.2f})°  n={b['n']:<4} 成功 {b['n_success']:<4} {r}"
              f"   Wilson[{b['ci'][0]:.2f},{b['ci'][1]:.2f}]")
    if slope is not None:
        print(f"  时长回归 : sim_time ≈ {intercept:.2f} + {slope:.2f}×timing_scale"
              f"  (r²={rr:.3f}, n={nfit})")
        if abs(slope) < 1.0:
            print("    ⚠️ 斜率 <1 s/×：节奏随机化基本没传到时长上，检查步数分配")
    if rep["reject_reasons"]:
        print("  丢弃原因 : " + ", ".join(f"{k}×{v}" for k, v in rep["reject_reasons"].items()))
    if under:
        print()
        print("  ⚠️ 覆盖不足（空箱/偏薄箱）：")
        for b in under:
            print(f"    {b['dim']:<18} 空箱 {b['empty_bins']}/{b['n_bins']}  "
                  f"偏薄 {b['thin_bins']}  counts={b['counts']}")
    else:
        print(f"\n  ✅ 覆盖度：每箱 ≥ {args.min_per_bin} 条，无空箱")

    out = _atomic_write(args.out, rep)
    print(f"\n  产物: {out.relative_to(ROOT)}（{out.stat().st_size/1024:.1f} KB）")
    if not args.no_plot:
        png = make_plot(rep, SCRIPTS_DIR / "stats.png")
        if png:
            print(f"        {png.relative_to(ROOT)}（{png.stat().st_size/1024:.0f} KB）")
    if args.card:
        card = SCRIPTS_DIR / "DATA_CARD.md"
        card.write_text(build_card(rep, under), encoding="utf-8")
        print(f"        {card.relative_to(ROOT)}（{card.stat().st_size/1024:.1f} KB）")
    if args.fail_on_undercovered and under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
