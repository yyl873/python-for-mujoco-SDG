#!/usr/bin/env python3
"""rl/ft_eval_sweep.py —— 微调扫描评测：多 seed × 多 checkpoint 步数 → 配对统计 + 曲线

为什么要单独一个脚本
--------------------
`docs/REPORT_RESIDUAL_RL.md` §14 标定出：held-out 100-129 是个"刀尖"指标，
**单 seed、30 次 rollout 的结论会被噪声地板（±15pp）淹掉**。所以第 3 步的判据必须是：

  1. 评测用 `--seeds 3`（90 次 rollout），按 (episode, seed) 与基座**配对**；
  2. 主指标是 **clean_success**（成功且不深压盒/不歪盒），不是 success —— §14.2 实测
     3 步微调能把 success 从 12/90 抬到 35/90，而 clean 只从 10/90 到 12/90；
  3. 训练至少 3 个 seed，报均值/区间，**不许挑最好的 seed**；
  4. 判定"有效"= 3 个 seed 的 clean_success 配对 McNemar p<0.05 且区间不跨基座（§13.3）。

本脚本把上面四件事固化下来，避免"先看数字再定判据"。

用法
----
```bash
# 全流程：对 3 个 seed × 4 个 checkpoint 跑评测，再汇总 + 画图
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl LP_NUM_THREADS=6 MPLCONFIGDIR=$PWD/.mplcache \
    .venv-rl/bin/python rl/ft_eval_sweep.py

# 只重算汇总/画图（不重跑评测）
... rl/ft_eval_sweep.py --aggregate-only

# 只评终点 ckpt
... rl/ft_eval_sweep.py --steps 10000
```
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))

CKPT_TMPL = ("/home/yyl/lerobot/outputs/train/act_dagger_ft_{seed}"
             "/checkpoints/{step:06d}/pretrained_model")


def parse_list(spec: str, cast=int) -> list:
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part and cast is int:
            lo, hi = part.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(cast(part))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(max(0.0, c - h)), float(min(1.0, c + h)))


def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验（双侧）。b/c = 仅在 A/B 里成功的配对数。"""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


def ckpt_ready(ck: Path, min_age_s: float = 30.0) -> bool:
    """checkpoint 是否已完整落盘（训练还在往里写时不能评）。

    判据：model.safetensors / config.json / policy_preprocessor.json 都在，
    且 model.safetensors 的 mtime 至少旧 min_age_s 秒。
    """
    need = ("model.safetensors", "config.json", "policy_preprocessor.json")
    if not ck.is_dir() or not all((ck / n).exists() for n in need):
        return False
    import time
    return (time.time() - (ck / "model.safetensors").stat().st_mtime) >= min_age_s


def load_rows(path: Path, arm: str | None = None) -> dict:
    """读 eval_ab 的输出，返回 {(episode_id, seed): row}。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    detail = d.get("detail") or {}
    if arm is None:
        arm = "base" if "base" in detail else next(iter(detail))
    rows = detail[arm]["rows"]
    return {(int(r["episode_id"]), int(r["seed"])): r for r in rows}


def collect(variants: list[dict], base_rows: dict) -> dict:
    out = []
    for v in variants:
        if not v["path"].exists():
            out.append(dict(v, missing=True))
            continue
        rows = load_rows(v["path"])
        keys = sorted(set(rows) & set(base_rows))
        if not keys:
            out.append(dict(v, missing=True, note="与基座没有交集"))
            continue
        n = len(keys)
        n_succ = sum(1 for k in keys if rows[k]["success"])
        n_clean = sum(1 for k in keys if rows[k].get("clean_success"))
        b_succ = sum(1 for k in keys if base_rows[k]["success"])
        b_clean = sum(1 for k in keys if base_rows[k].get("clean_success"))
        # 配对：只算方向不一致的
        bs = sum(1 for k in keys if rows[k]["success"] and not base_rows[k]["success"])
        cs = sum(1 for k in keys if base_rows[k]["success"] and not rows[k]["success"])
        bc = sum(1 for k in keys if rows[k].get("clean_success")
                 and not base_rows[k].get("clean_success"))
        cc = sum(1 for k in keys if base_rows[k].get("clean_success")
                 and not rows[k].get("clean_success"))
        lo, hi = wilson(n_clean, n)
        clo, chi = wilson(b_clean, n)
        out.append(dict(
            seed=v["seed"], step=v["step"], path=str(v["path"]), missing=False, n=n,
            success=n_succ, success_rate=round(n_succ / n, 4),
            clean=n_clean, clean_rate=round(n_clean / n, 4),
            clean_wilson=[round(100 * lo, 1), round(100 * hi, 1)],
            base_success=b_succ, base_clean=b_clean,
            base_clean_wilson=[round(100 * clo, 1), round(100 * chi, 1)],
            mcnemar_success=dict(ft_only=bs, base_only=cs,
                                 p=round(mcnemar_exact(bs, cs), 5)),
            mcnemar_clean=dict(ft_only=bc, base_only=cc,
                               p=round(mcnemar_exact(bc, cc), 5)),
        ))
    return dict(rows=out)


def summarize(res: dict, base_rows: dict) -> dict:
    ok = [r for r in res["rows"] if not r.get("missing")]
    n_base = len(base_rows)
    base_clean = sum(1 for r in base_rows.values() if r.get("clean_success"))
    base_succ = sum(1 for r in base_rows.values() if r["success"])
    if not ok:
        return dict(n_base=n_base, base_success=base_succ, base_clean=base_clean, verdict="no_data")
    # 终点（最大 step）按 seed 聚合
    last = max(r["step"] for r in ok)
    finals = [r for r in ok if r["step"] == last]
    rates = [r["clean_rate"] for r in finals]
    sig = [r["mcnemar_clean"]["p"] < 0.05 for r in finals]
    lo_w = [r["clean_wilson"][0] for r in finals]
    hi_w = [r["clean_wilson"][1] for r in finals]
    base_rate = base_clean / max(1, n_base)
    base_hi = wilson(base_clean, n_base)[1]
    beats = [r["clean_rate"] > base_rate for r in finals]
    # 判据（§13.3）：每个 seed 的 clean 配对 p<0.05 且 3 个 seed 的率都高于基座点估计
    verdict = "effective" if (all(sig) and all(beats) and len(finals) >= 3) else (
        "inconclusive" if len(finals) < 3 else "no_evidence")
    return dict(
        final_step=last, n_seeds=len(finals), n_base=n_base,
        base_success_rate=round(base_succ / max(1, n_base), 4),
        base_clean_rate=round(base_rate, 4),
        base_clean_wilson=[round(100 * wilson(base_clean, n_base)[0], 1),
                           round(100 * base_hi, 1)],
        final_clean_rate_mean=round(float(np.mean(rates)), 4),
        final_clean_rate_sd=round(float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0, 4),
        final_clean_wilson_union=[min(lo_w), max(hi_w)],
        per_seed=[dict(seed=r["seed"], clean=r["clean"], success=r["success"],
                       clean_rate=r["clean_rate"], p_clean=r["mcnemar_clean"]["p"])
                  for r in finals],
        all_seeds_significant=bool(all(sig)),
        all_seeds_above_base=bool(all(beats)),
        verdict=verdict,
    )


def plot(res: dict, sumry: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in res["rows"] if not r.get("missing")]
    if not ok:
        return
    seeds = sorted({r["seed"] for r in ok})
    base_rate = 100.0 * sumry.get("base_clean_rate", 0.0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for s in seeds:
        pts = sorted([r for r in ok if r["seed"] == s], key=lambda r: r["step"])
        ax.plot([p["step"] for p in pts], [100 * p["clean_rate"] for p in pts],
                marker="o", lw=1.6, label=f"fine-tune seed {s}")
    ax.axhline(base_rate, color="k", ls="--", lw=1.4,
               label=f"base sim156 ({base_rate:.1f}%)")
    lo, hi = sumry.get("base_clean_wilson", [0, 0])
    ax.axhspan(lo, hi, color="k", alpha=0.08)
    ax.set_xlabel("fine-tune steps")
    ax.set_ylabel("clean success % (held-out 100-129, 90 rollouts)")
    ax.set_title("ACT fine-tune on sim156+DAgger: clean success vs steps\n"
                 f"verdict: {sumry.get('verdict')}  (grey band = base 95% Wilson)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"  图  : {out_png}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="微调扫描评测（多 seed × 多 ckpt → 配对统计）",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--seeds", default="1000,1001,1002", help="训练 seed 列表")
    ap.add_argument("--steps", default="2500,5000,7500,10000", help="要评的 checkpoint 步数")
    ap.add_argument("--ckpt-template", default=CKPT_TMPL)
    ap.add_argument("--base-json", default=str(ROOT / "rl" / "eval" / "ab_base_90.json"),
                    help="基座对照（必须同为 --seeds-eval 的配置）")
    ap.add_argument("--episodes", default="100-129")
    ap.add_argument("--seeds-eval", type=int, default=3, help="评测 rollout 的 seed 数（默认 3 = 90 次）")
    ap.add_argument("--judge", default="strict")
    ap.add_argument("--reward", default="v2b")
    ap.add_argument("--out-dir", default=str(ROOT / "rl" / "eval" / "ft"))
    ap.add_argument("--summary", default=str(ROOT / "rl" / "eval" / "ft_sweep.json"))
    ap.add_argument("--plot", default=str(ROOT / "rl" / "curves" / "ft_clean_vs_steps.png"))
    ap.add_argument("--aggregate-only", action="store_true", help="只汇总/画图，不跑评测")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--timeout-s", type=int, default=1800, help="单次评测超时（秒）")
    a = ap.parse_args(argv)

    seeds = parse_list(a.seeds)
    steps = parse_list(a.steps)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_path = Path(a.base_json)
    if not base_path.exists():
        raise SystemExit(f"基座对照不存在: {base_path}（先跑 eval_ab --base sim156 --seeds {a.seeds_eval}）")
    base_rows = load_rows(base_path)

    variants = []
    for s in seeds:
        for st in steps:
            ck = Path(a.ckpt_template.format(seed=s, step=st))
            variants.append(dict(seed=s, step=st, path=out_dir / f"s{s}_{st}.json",
                                 ckpt=ck))

    print("=" * 78)
    print(" ft_eval_sweep  微调扫描评测")
    print("=" * 78)
    print(f"  基座对照: {base_path}  {len(base_rows)} 次 rollout（"
          f"success {sum(1 for r in base_rows.values() if r['success'])}, "
          f"clean {sum(1 for r in base_rows.values() if r.get('clean_success'))}）")
    print(f"  扫描    : seeds {seeds} × steps {steps} = {len(variants)} 个 ckpt")
    print(f"  评测    : episodes {a.episodes} / seeds {a.seeds_eval} / judge {a.judge}")
    print()

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("LP_NUM_THREADS", "6")
    env.setdefault("OMP_NUM_THREADS", "6")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))

    if not a.aggregate_only:
        for i, v in enumerate(variants, 1):
            if not ckpt_ready(v["ckpt"]):
                print(f"  [{i}/{len(variants)}] s{v['seed']} step{v['step']}: ckpt 未就绪，跳过")
                continue
            if a.skip_existing and v["path"].exists():
                print(f"  [{i}/{len(variants)}] s{v['seed']} step{v['step']}: 已有结果，跳过")
                continue
            print(f"  [{i}/{len(variants)}] s{v['seed']} step{v['step']}: 评测中…", flush=True)
            cmd = [sys.executable, str(ROOT / "rl" / "eval_ab.py"),
                   "--base", str(v["ckpt"]), "--episodes", a.episodes,
                   "--seeds", str(a.seeds_eval), "--judge", a.judge, "--reward", a.reward,
                   "--out", str(v["path"])]
            r = subprocess.run(cmd, capture_output=True, text=True, env=env,
                               cwd=str(ROOT), timeout=a.timeout_s)
            if r.returncode not in (0, 1, 2) or not v["path"].exists():
                print(f"      ⚠️  评测失败 rc={r.returncode}: {r.stderr.strip().splitlines()[-1:]}")
            else:
                rr = load_rows(v["path"])
                print(f"      success {sum(1 for x in rr.values() if x['success'])}/{len(rr)}  "
                      f"clean {sum(1 for x in rr.values() if x.get('clean_success'))}/{len(rr)}")

    res = collect(variants, base_rows)
    sumry = summarize(res, base_rows)
    payload = dict(base_json=str(base_path), episodes=a.episodes, eval_seeds=a.seeds_eval,
                   judge=a.judge, summary=sumry,
                   variants=[{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()}
                             for r in res["rows"]])
    Path(a.summary).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"  {'seed':>5} {'step':>6} {'success':>9} {'clean':>9}  {'配对(clean) b/c':>16} {'p':>8}")
    for r in res["rows"]:
        if r.get("missing"):
            print(f"  {r['seed']:>5} {r['step']:>6}        —（缺）")
            continue
        mc = r["mcnemar_clean"]
        print(f"  {r['seed']:>5} {r['step']:>6} {r['success']:>4}/{r['n']:<4} "
              f"{r['clean']:>4}/{r['n']:<4}  {mc['ft_only']:>7}/{mc['base_only']:<7} "
              f"{mc['p']:>8.4f}")
    print()
    print(f"  基座      : success {sumry['base_success_rate']:.1%}  "
          f"clean {sumry['base_clean_rate']:.1%}  "
          f"Wilson [{sumry['base_clean_wilson'][0]:.1f}, {sumry['base_clean_wilson'][1]:.1f}]")
    if sumry.get("final_step"):
        print(f"  终点({sumry['final_step']}): clean 均值 {sumry['final_clean_rate_mean']:.1%} "
              f"± {sumry['final_clean_rate_sd']:.1%}（{sumry['n_seeds']} seed）"
              f"  Wilson 并集 {sumry['final_clean_wilson_union']}")
        print("  逐 seed   : " + "  ".join(
            f"s{p['seed']} clean {p['clean']}/{sumry['n_base']} ({p['clean_rate']:.1%}, "
            f"p={p['p_clean']:.4f})" for p in sumry["per_seed"]))
        print(f"  判据      : 每个 seed 显著={sumry['all_seeds_significant']}  "
              f"每个 seed 高于基座={sumry['all_seeds_above_base']}  →  **{sumry['verdict']}**")
    print(f"  汇总      : {a.summary}")
    plot(res, sumry, Path(a.plot))
    return 0


if __name__ == "__main__":
    sys.exit(main())