#!/usr/bin/env python3
"""在 holdout 真机集上离线评测 ACT checkpoint。

指标（单位与数据集 action 相同，SO-101 为 deg）：
  - L1@1   : 每个时间点只取预测动作块的第 1 帧，与当前帧真值比
  - L1@10  : 预测块前 10 帧与未来真值比
  - L1@50 / L1@100 : 更长动作块
  - RMSE 与 gripper 单列误差
所有指标都逐关节输出。

注意：离线 action L1 只用于 A/B/C/D 横向筛选，不能替代真机成功率。

用法：
  env -u PYTHONHOME -u PYTHONPATH \
    /home/yyl/miniforge3/envs/lerobot312/bin/python \
    tools/eval_act_holdout.py \
      --ckpt /home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model \
      --out logs/eval_holdout_armB.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path('/home/yyl/deepseek-harness')
# 默认 HF 缓存在某些环境里是只读的，统一指到工作区
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / '.cache/hf_datasets'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/hf_home'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
(ROOT / '.cache/hf_datasets').mkdir(parents=True, exist_ok=True)
(ROOT / '.cache/hf_home').mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
from torch.utils.data import default_collate

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def load_action_table(root: str):
    """返回 {(episode_index, frame_index): action[6]} 和 {episode_index: n_frames}。"""
    import pyarrow.parquet as pq

    table = {}
    ep_len = {}
    files = sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True))
    if not files:
        raise SystemExit(f'没有找到 data parquet: {root}')
    for f in files:
        t = pq.read_table(f, columns=['episode_index', 'frame_index', 'action'])
        eps = np.asarray(t.column('episode_index').to_pylist(), dtype=np.int64)
        frs = np.asarray(t.column('frame_index').to_pylist(), dtype=np.int64)
        acts = np.asarray(t.column('action').to_pylist(), dtype=np.float32)
        for e, fr, a in zip(eps, frs, acts):
            table[(int(e), int(fr))] = a
            ep_len[int(e)] = max(ep_len.get(int(e), 0), int(fr) + 1)
    return table, ep_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='checkpoint 的 pretrained_model 目录')
    ap.add_argument('--root', default='/home/yyl/lerobot_datasets/holdout_fixed')
    ap.add_argument('--repo-id', default='local/holdout_fixed')
    ap.add_argument('--out', default=None, help='输出 json 路径')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--sample-every', type=int, default=5, help='每 N 帧取一个评测点')
    ap.add_argument('--limit', type=int, default=0, help='最多评测多少个采样点，0=全部')
    ap.add_argument('--horizon', type=int, default=100, help='动作块长度')
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not (ckpt / 'config.json').is_file():
        raise SystemExit(f'不是 pretrained_model 目录: {ckpt}')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA 不可用，自动切到 CPU')
        args.device = 'cpu'

    out = Path(args.out) if args.out else ROOT / 'logs' / f'eval_holdout_{ckpt.parent.parent.name}.json'
    out.parent.mkdir(parents=True, exist_ok=True)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    print(f'[eval] ckpt={ckpt}')
    print(f'[eval] dataset={args.root} device={args.device} sample_every={args.sample_every}')

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision='v3.0')
    action_table, ep_len = load_action_table(args.root)
    print(f'[eval] frames={len(ds)} episodes={ds.meta.total_episodes} action_rows={len(action_table)}')

    policy = ACTPolicy.from_pretrained(str(ckpt)).to(args.device).eval()
    dev_override = {'device_processor': {'device': args.device}}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(ckpt),
        preprocessor_overrides=dev_override,
        postprocessor_overrides=dev_override,
    )

    all_idx = list(range(0, len(ds), max(1, args.sample_every)))
    if args.limit:
        all_idx = all_idx[:args.limit]
    print(f'[eval] sampled points={len(all_idx)} (every {args.sample_every} frames)')

    H = int(args.horizon)
    sum_abs = np.zeros(6, dtype=np.float64)
    sum_sq = np.zeros(6, dtype=np.float64)
    n_steps = 0
    per_h = {h: np.zeros(6, dtype=np.float64) for h in (1, 10, 50, 100) if h <= H}
    per_h_n = {h: 0 for h in per_h}
    per_ep_first = {}
    t0 = time.time()

    for start in range(0, len(all_idx), args.batch_size):
        chunk_idx = all_idx[start:start + args.batch_size]
        samples = [ds[i] for i in chunk_idx]
        gt_chunk = np.zeros((len(samples), H, 6), dtype=np.float32)
        mask = np.zeros((len(samples), H), dtype=np.float32)
        for bi, s in enumerate(samples):
            ep = int(s['episode_index'])
            fr = int(s['frame_index'])
            for k in range(H):
                a = action_table.get((ep, fr + k))
                if a is None:
                    break
                gt_chunk[bi, k] = a
                mask[bi, k] = 1.0

        batch = default_collate(samples)
        for cam in ds.meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float() / 255.0
        batch = pre(batch)

        with torch.no_grad():
            pred_norm = policy.predict_action_chunk(batch)
            pred = post(pred_norm)
        pred = pred.detach().cpu().numpy()[:, :H, :]

        err = np.abs(pred - gt_chunk) * mask[..., None]
        sum_abs += err.sum(axis=(0, 1))
        sum_sq += (((pred - gt_chunk) * mask[..., None]) ** 2).sum(axis=(0, 1))
        n_steps += int(mask.sum())

        for h in per_h:
            mh = mask[:, :h]
            eh = np.abs(pred[:, :h, :] - gt_chunk[:, :h, :]) * mh[..., None]
            per_h[h] += eh.sum(axis=(0, 1))
            per_h_n[h] += int(mh.sum())

        # 按 episode 记录 L1@1，便于看是不是某个视频拖后腿
        e1 = np.abs(pred[:, 0, :] - gt_chunk[:, 0, :])
        for bi, s in enumerate(samples):
            ep = int(s['episode_index'])
            v = per_ep_first.setdefault(ep, [np.zeros(6), 0])
            v[0] += e1[bi]
            v[1] += 1

        if (start // args.batch_size) % 10 == 0:
            print(f'  {min(start + args.batch_size, len(all_idx))}/{len(all_idx)} '
                  f'({time.time() - t0:.0f}s)', flush=True)

    if n_steps == 0:
        raise SystemExit('没有有效评测帧')

    l1 = sum_abs / n_steps
    rmse = np.sqrt(sum_sq / n_steps)
    result = {
        'ckpt': str(ckpt),
        'dataset': args.root,
        'repo_id': args.repo_id,
        'device': args.device,
        'sample_every': args.sample_every,
        'sampled_points': len(all_idx),
        'valid_steps': n_steps,
        'elapsed_s': round(time.time() - t0, 1),
        'joints': JOINTS,
        'l1_chunk': {j: round(float(v), 4) for j, v in zip(JOINTS, l1)},
        'rmse_chunk': {j: round(float(v), 4) for j, v in zip(JOINTS, rmse)},
        'l1_mean': round(float(l1.mean()), 4),
        'gripper_l1': round(float(l1[5]), 4),
        'by_horizon': {
            str(h): {'n': per_h_n[h],
                     'l1_mean': round(float((per_h[h] / max(per_h_n[h], 1)).mean()), 4),
                     'l1': {j: round(float(v), 4) for j, v in zip(
                         JOINTS, per_h[h] / max(per_h_n[h], 1))}}
            for h in per_h
        },
        'per_episode_l1_first': {
            str(ep): {'n': v[1],
                      'l1_mean': round(float((v[0] / max(v[1], 1)).mean()), 4),
                      'l1': {j: round(float(x / max(v[1], 1)), 4)
                             for j, x in zip(JOINTS, v[0])}}
            for ep, v in sorted(per_ep_first.items())
        },
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print()
    print(f'===== {ckpt.parents[2].name} / step {ckpt.parent.name} =====')
    print(f'{"joint":16s} {"L1@1":>9s} {"L1@10":>9s} {"L1@50":>9s} {"L1@100":>9s} {"RMSE":>9s}')
    for i, j in enumerate(JOINTS):
        row = [j]
        for h in (1, 10, 50, 100):
            if h in per_h:
                row.append(per_h[h][i] / max(per_h_n[h], 1))
            else:
                row.append(float('nan'))
        row.append(rmse[i])
        print(f'{row[0]:16s} {row[1]:9.3f} {row[2]:9.3f} {row[3]:9.3f} {row[4]:9.3f} {row[5]:9.3f}')
    print(f'{"MEAN":16s} '
          f'{(per_h[1].mean() / max(per_h_n[1],1)):9.3f} '
          f'{(per_h.get(10, np.zeros(6)).mean() / max(per_h_n.get(10,1),1)):9.3f} '
          f'{(per_h.get(50, np.zeros(6)).mean() / max(per_h_n.get(50,1),1)):9.3f} '
          f'{(per_h.get(100, np.zeros(6)).mean() / max(per_h_n.get(100,1),1)):9.3f} '
          f'{rmse.mean():9.3f}')
    print(f'\n结果: {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
