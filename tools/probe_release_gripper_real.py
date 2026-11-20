#!/usr/bin/env python3
"""在真机 holdout 上找“第二次张开”=释放段，喂给 ACT，检查夹爪输出。"""
from __future__ import annotations
import argparse, glob, os, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate

ROOT = Path('/home/yyl/deepseek-harness')
os.environ.setdefault('HF_DATASETS_CACHE', str(ROOT / '.cache/hf_datasets'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/hf_home'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
(ROOT / '.cache/hf_datasets').mkdir(parents=True, exist_ok=True)
(ROOT / '.cache/hf_home').mkdir(parents=True, exist_ok=True)


def load_actions(root):
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True)):
        t = pq.read_table(f, columns=['episode_index', 'frame_index', 'action'])
        eps = np.asarray(t.column('episode_index').to_pylist(), dtype=np.int64)
        frs = np.asarray(t.column('frame_index').to_pylist(), dtype=np.int64)
        acts = np.asarray(t.column('action').to_pylist(), dtype=np.float32)
        for e in np.unique(eps):
            m = eps == e
            out[int(e)] = acts[m][np.argsort(frs[m])]
    return out


def intervals_above(g, thr=10.0):
    mask = g > thr
    out = []
    i = 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(mask) and mask[j + 1]:
            j += 1
        out.append((i, j))
        i = j + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--root', default='/home/yyl/lerobot_datasets/holdout_fixed')
    ap.add_argument('--repo-id', default='local/holdout_fixed')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='logs/release_gripper_real.csv')
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA unavailable -> cpu')
        args.device = 'cpu'

    acts = load_actions(args.root)
    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision='v3.0')
    policy = ACTPolicy.from_pretrained(args.policy).to(args.device).eval()
    ov = {'device_processor': {'device': args.device}}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.policy,
        preprocessor_overrides=ov, postprocessor_overrides=ov,
    )

    points = []
    for ep in sorted(acts):
        g = acts[ep][:, 5]
        iv = intervals_above(g, 10.0)
        if len(iv) < 2:
            print(f'ep{ep}: only {len(iv)} high intervals, skip')
            continue
        a, b = iv[1]  # second plateau = release
        base = int(ds.meta.episodes[ep]['dataset_from_index'])
        for label, fr in [('release_start', a), ('release_mid', (a + b) // 2),
                          ('release_end', b)]:
            points.append((ep, label, fr, base + fr))
        print(f'ep{ep:2d}: release {a}~{b} gt_max={g[a:b+1].max():.2f}')

    print(f'probe points: {len(points)}')
    rows = []
    for start in range(0, len(points), 8):
        chunk = points[start:start + 8]
        samples = [ds[gidx] for _, _, _, gidx in chunk]
        batch = default_collate(samples)
        for cam in ds.meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float() / 255.0
        batch = pre(batch)
        with torch.inference_mode():
            pred = post(policy.predict_action_chunk(batch))
        pred = pred.detach().cpu().numpy()
        for bi, (ep, label, fr, _) in enumerate(chunk):
            rows.append(dict(episode=ep, phase=label, frame=fr,
                             pred_gripper=float(pred[bi, 0, 5]),
                             gt_gripper=float(samples[bi]['action'][5])))
        print(f'  {min(start + 8, len(points))}/{len(points)}', flush=True)

    import csv
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print('written', out)
    print()
    for label in ('release_start', 'release_mid', 'release_end'):
        vals = [r['pred_gripper'] for r in rows if r['phase'] == label]
        gts = [r['gt_gripper'] for r in rows if r['phase'] == label]
        print(f'{label:14s} n={len(vals):2d}  pred mean={np.mean(vals):6.2f} '
              f'med={np.median(vals):6.2f} min={np.min(vals):6.2f} max={np.max(vals):6.2f} '
              f'| gt mean={np.mean(gts):6.2f}')


if __name__ == '__main__':
    sys.exit(main())
