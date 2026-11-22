#!/usr/bin/env python3
"""看语言通道权重到底长没长：ACT 的 state 输入投影里第 6/7 列（instr_left / instr_right）。

背景（实测）：基座 act_sim156_lang8 是把 6 维 state 补 0 列加宽到 8 维的，
所以 `encoder_robot_state_input_proj.weight[:, 6:8]` 初始 **恰好 0.0**，而且左右两列完全对称。
语言能不能被用上，第一看点就是这两列在训练里长不长大。

用法（必须用 lerobot312 解释器，它才有 safetensors）：
    P=/home/yyl/miniforge3/envs/lerobot312/bin/python
    $P rl/probe_lang_weight.py --ckpt outputs/act_lang_pair/checkpoints/001250
    $P rl/probe_lang_weight.py --ckpt outputs/act_lang_pair            # 扫全部 ckpt，出曲线
    $P rl/probe_lang_weight.py --ckpt outputs/act_lang_pair --json     # 机器可读

判读：看 `per_col`（每列 RMS 的比值 = 语言 2 列 vs 其它 6 列），这是唯一可比的口径：
`per_col = (lang_norm/√2) / (other_norm/√6)`。`ratio` 是两块总范数直接相除，含列数差异，别直接比。
参考点（实测）：基座 0.000；v1 混叠数据 1250 步 → encoder per_col≈0.013 / vae_encoder≈0.002。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

TARGET_SUFFIXES = ("robot_state_input_proj.weight",)


def load_ckpts(path: str) -> list[str]:
    """接受 checkpoints 根目录 / 单个 ckpt 目录 / 直接指向 pretrained_model。"""
    if path.endswith("pretrained_model") or os.path.exists(os.path.join(path, "model.safetensors")):
        return [os.path.dirname(path.rstrip("/")) if path.rstrip("/").endswith("pretrained_model") else path]
    dirs = sorted(
        (d for d in glob.glob(os.path.join(path, "checkpoints", "[0-9]*")) if os.path.isdir(d)),
        key=lambda d: int(os.path.basename(d)),
    )
    return dirs


def step_of(ckpt_dir: str) -> int:
    for rel in ("training_state/training_step.json", "pretrained_model/training_step.json"):
        p = os.path.join(ckpt_dir, rel)
        if os.path.exists(p):
            try:
                return int(json.load(open(p)).get("step", -1))
            except Exception:
                pass
    base = os.path.basename(ckpt_dir.rstrip("/"))
    return int(base) if base.isdigit() else -1


def probe(ckpt_dir: str) -> dict:
    from safetensors.numpy import load_file  # 只在 lerobot312 里有

    sf = os.path.join(ckpt_dir, "pretrained_model", "model.safetensors")
    if not os.path.exists(sf):
        sf = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.exists(sf):
        return {"ckpt": ckpt_dir, "error": "no model.safetensors"}
    d = load_file(sf)
    out: dict = {"ckpt": os.path.basename(ckpt_dir.rstrip("/")), "step": step_of(ckpt_dir), "tensors": {}}
    for k, w in d.items():
        if not k.endswith(TARGET_SUFFIXES) or getattr(w, "ndim", 0) != 2 or w.shape[1] < 8:
            continue
        import numpy as np

        lang = w[:, 6:8].astype("float32")
        other = w[:, :6].astype("float32")
        ln = float((lang ** 2).sum() ** 0.5)
        on = float((other ** 2).sum() ** 0.5)
        out["tensors"][k] = {
            "shape": list(w.shape),
            "lang_absmax": round(float(abs(lang).max()), 8),
            "lang_norm": round(ln, 6),
            "other_norm": round(on, 6),
            "ratio": round(ln / on, 6) if on else None,
            # 每列 RMS 比值：语言 2 列 vs 其它 6 列（唯一可比口径）
            "per_col": round((ln / 2 ** 0.5) / (on / 6 ** 0.5), 6) if on else None,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoints 根目录 / 某个 ckpt 目录 / pretrained_model")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ckpts = load_ckpts(a.ckpt)
    if not ckpts:
        print(f"没找到 ckpt: {a.ckpt}", file=sys.stderr)
        return 2
    rows = [probe(c) for c in ckpts]

    if a.json:
        print(json.dumps(rows, ensure_ascii=False, indent=1))
        return 0

    print(f"{'step':>7}  {'tensor':<48} {'lang_absmax':>12} {'lang_norm':>10} {'per_col':>9}")
    for r in rows:
        if r.get("error"):
            print(f"{'?':>7}  {r['ckpt']:<48} {r['error']}")
            continue
        for k, v in r["tensors"].items():
            short = k.replace("model.", "").replace("_robot_state_input_proj.weight", "")
            print(
                f"{r['step']:>7}  {short:<48} {v['lang_absmax']:>12.6g} "
                f"{v['lang_norm']:>10.5f} {v['per_col'] if v['per_col'] is not None else float('nan'):>9.5f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())