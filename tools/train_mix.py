#!/usr/bin/env python3
"""按"臂"训练 ACT（本地数据集，不联网，参数全部命令行可覆盖）。

为什么要这个脚本而不是裸 lerobot-train
--------------------------------------
数据集不在 HuggingFace Hub 上（repo_id 形如 local/xxx），`lerobot-train` 会在
`get_safe_version` 里去联网要 revision，直接失败。这个脚本沿用 `lerobot/train_local.py`
里已验证过的补丁：`local/` 开头的 repo_id 直接返回本地 revision。

用法（用 lerobot312 解释器，不要用 /usr/bin/python3）
----
  env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
      tools/train_mix.py \
        --root /home/yyl/lerobot_datasets/mix_sim25 \
        --repo-id local/mix_sim25 \
        --output /home/yyl/lerobot/outputs/train/act_mix_sim25 \
        --job-name act_mix_sim25 \
        --steps 20000 --batch-size 16 --num-workers 8 --seed 0

  只想看命令不跑：加 --dry-run。

注意
----
* 输出目录已存在且不加 --resume 会被 lerobot 拒绝（防覆盖），这是好事。
* batch_size / num_workers / steps / seed 必须四条臂完全一致才可比。
* 断点续训加 --resume（lerobot 从 output_dir 里找 last checkpoint）。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="数据集根（含 meta/ data/ videos/）")
    ap.add_argument("--repo-id", required=True, help="数据集标识，务必以 local/ 开头（免联网）")
    ap.add_argument("--output", required=True, help="输出目录（如 lerobot/outputs/train/act_xxx）")
    ap.add_argument("--job-name", default=None)
    ap.add_argument("--policy-repo-id", default=None, help="默认 local/<job_name>")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=None, help="不填沿用 ACT preset（1e-5）")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--eval-split", type=float, default=0.0, help=">0 时按任务留出验证集")
    ap.add_argument("--eval-steps", type=int, default=0)
    ap.add_argument("--video-backend", default=None, help="pyav / torchcodec（默认沿用 lerobot 默认）")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只打印解析出的 argv，不启动训练")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"[fail] 数据集根不存在: {root}", flush=True)
        return 2

    job = args.job_name or Path(args.output).name
    policy_repo = args.policy_repo_id or f"local/{job}"

    argv_list = [
        "lerobot-train",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.root={root}",
        f"--dataset.revision=v3.0",
        "--policy.type=act",
        "--policy.use_vae=false",
        f"--policy.repo_id={policy_repo}",
        f"--output_dir={args.output}",
        f"--job_name={job}",
        f"--policy.device={args.device}",
        f"--steps={args.steps}",
        f"--save_freq={args.save_freq}",
        f"--batch_size={args.batch_size}",
        f"--num_workers={args.num_workers}",
        f"--seed={args.seed}",
        "--wandb.enable=false",
    ]
    if args.lr is not None:
        argv_list.append(f"--policy.optimizer_lr={args.lr}")
    if args.chunk_size is not None:
        argv_list.append(f"--policy.chunk_size={args.chunk_size}")
    if args.n_action_steps is not None:
        argv_list.append(f"--policy.n_action_steps={args.n_action_steps}")
    if args.eval_split > 0:
        argv_list.append(f"--dataset.eval_split={args.eval_split}")
        if args.eval_steps > 0:
            argv_list.append(f"--eval_steps={args.eval_steps}")
    if args.video_backend:
        argv_list.append(f"--dataset.video_backend={args.video_backend}")
    if args.resume:
        argv_list.append("--resume=true")

    if args.dry_run:
        print(" ".join(argv_list))
        return 0

    from lerobot.datasets import lerobot_dataset, utils
    from lerobot.scripts import lerobot_train

    _orig = lerobot_dataset.get_safe_version

    def _patched_get_safe_version(repo_id, revision):
        if str(repo_id).startswith("local/"):
            return str(revision)
        return _orig(repo_id, revision)

    lerobot_dataset.get_safe_version = _patched_get_safe_version
    utils.get_safe_version = _patched_get_safe_version

    sys.argv = argv_list
    print("[train] " + " ".join(argv_list), flush=True)
    lerobot_train.main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
