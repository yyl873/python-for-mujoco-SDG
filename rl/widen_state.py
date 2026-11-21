#!/usr/bin/env python3
"""把 ACT 预训练 checkpoint 的 observation.state 输入维度加宽（语言条件用）。

背景
----
本仓的 ACT 基座（`act_sim156_pre`）是按 `observation.state = 6 维关节角` 训的，
对应的线性投影 `*_robot_state_input_proj.weight` 是 `(dim_model, 6)`。
语言条件 pilot 把指令 one-hot 拼在 state 末尾（6 -> 8 维），于是：

- `config.json` 里 `input_features["observation.state"].shape` = [6] 会把数据集里的
  8 维 state 挡在外面（`make_policy` 只在 `not cfg.input_features` 时才从数据集推断，
  预训练 config 已经有 6 维了，所以不会自动扩）；
- 权重形状 `(512, 6)` 和 8 维输入对不上，`load_state_dict(strict=False)` 遇到
  **形状不匹配会直接抛**（strict=False 只忽略缺失/多余键，不忽略 shape）。

做法
----
新建一个 pretrained_model 目录：config 改成 8 维，两个 `*_robot_state_input_proj.weight`
右侧**补 0 列**、bias 原样。

★ 补 0 而不是随机：初始化时新维度的输出贡献恒为 0，模型行为和被加宽前的基座
**逐位一致**；能不能用上语言，完全由后续微调的梯度决定。这是"语言是否被使用"
这个实验解释力的前提——否则分不清是语言带来的还是随机扰动带来的。
（前向是 `x @ W.T + b`，输入是 one-hot 的 1.0，所以 dL/dW_new = grad，梯度照常流动。）

用法
----
    .venv-rl/bin/python rl/widen_state.py \
        --src /home/yyl/lerobot/outputs/train/act_sim156_pre/checkpoints/last/pretrained_model \
        --dst /home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model \
        --state-dim 8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def widen_safetensors(src: Path, dst: Path, new_dim: int) -> list[str]:
    import torch
    from safetensors.torch import load_file, save_file

    tensors = load_file(str(src))
    changed = []
    for key in list(tensors):
        if key.endswith("_robot_state_input_proj.weight"):
            w = tensors[key]
            old = w.shape[1]
            if old == new_dim:
                continue
            if old > new_dim:
                raise SystemExit(f"{key} 已经是 {old} 维，不能缩到 {new_dim} 维")
            pad = torch.zeros((w.shape[0], new_dim - old), dtype=w.dtype)
            tensors[key] = torch.cat([w, pad], dim=1)
            changed.append(f"{key} ({w.shape[0]},{old}) -> {tensors[key].shape}")
    if not changed:
        raise SystemExit(f"{src} 里没找到需要加宽的 *_robot_state_input_proj.weight")
    save_file(tensors, str(dst))
    return changed


def patch_config(src: Path, dst: Path, new_dim: int, names: list[str] | None) -> str:
    cfg = json.loads(src.read_text(encoding="utf-8"))
    st = cfg["input_features"]["observation.state"]
    old = list(st["shape"])
    st["shape"] = [new_dim]
    if names:
        st["names"] = names
    dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    return f"input_features.observation.state.shape {old} -> {[new_dim]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ACT 基座 state 维度加宽（补 0 列）")
    ap.add_argument("--src", required=True, help="原 pretrained_model 目录")
    ap.add_argument("--dst", required=True, help="新目录（必须不存在）")
    ap.add_argument("--state-dim", type=int, default=8, help="加宽后的维度")
    ap.add_argument("--names", default=None,
                    help="可选：新 state 列名，逗号分隔，会写进 config 的 names")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    if not (src / "config.json").exists():
        raise SystemExit(f"{src} 不像 pretrained_model 目录（没有 config.json）")
    if dst.exists():
        if not args.overwrite:
            raise SystemExit(f"{dst} 已存在（换路径或 --overwrite）")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    names = [x.strip() for x in args.names.split(",") if x.strip()] if args.names else None
    if names is not None and len(names) != args.state_dim:
        raise SystemExit(f"--names {len(names)} 个，和 --state-dim {args.state_dim} 不一致")

    print("=" * 70)
    print(" widen_state   ACT 基座 state 维度加宽")
    print("=" * 70)
    print(f"  源  : {src}")
    print(f"  目标: {dst}")
    print("  " + patch_config(src / "config.json", dst / "config.json", args.state_dim, names))
    for line in widen_safetensors(src / "model.safetensors", dst / "model.safetensors", args.state_dim):
        print("  " + line)
    for f in ("policy_preprocessor.json", "policy_postprocessor.json",
              "policy_preprocessor_step_3_normalizer_processor.safetensors",
              "policy_postprocessor_step_0_unnormalizer_processor.safetensors"):
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
            print(f"  拷贝: {f}")
    print("  注: normalizer 的 stats 由 lerobot-train 用新数据集的统计量覆盖"
          "（非 --resume 路径），拷来的只是占位")

    # ---- 回读校验 ----
    print("-" * 70)
    cfg = json.loads((dst / "config.json").read_text(encoding="utf-8"))
    from safetensors.torch import load_file
    w = load_file(str(dst / "model.safetensors"))
    ok = True
    dim = cfg["input_features"]["observation.state"]["shape"][0]
    print(f"  回读: state shape = {dim}")
    for k in [k for k in w if k.endswith("_robot_state_input_proj.weight")]:
        last = w[k][:, dim - 1].abs().max().item()
        print(f"        {k} = {tuple(w[k].shape)}  末列 |max| = {last:.1e}（应为 0）")
        ok &= (last == 0.0) and w[k].shape[1] == dim
    print(f"  结论: {'ok' if ok else 'fail'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())