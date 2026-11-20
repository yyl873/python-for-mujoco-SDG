#!/usr/bin/env python3
"""离线探针：把仿真 release/drop 帧喂给 ACT，检查夹爪输出刻度。

用途：验证 B 的释放失败是否因为夹爪输出被真机数据拉低。
只读，不接真机。
"""
from __future__ import annotations
import argparse, glob, json, os, sys
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


def load_dataset_actions(root: str):
    import pyarrow.parquet as pq
    out = {}
    for f in sorted(glob.glob(os.path.join(root, 'data', '**', '*.parquet'), recursive=True)):
        t = pq.read_table(f, columns=['episode_index', 'frame_index', 'action'])
        eps = np.asarray(t.column('episode_index').to_pylist(), dtype=np.int64)
        frs = np.asarray(t.column('frame_index').to_pylist(), dtype=np.int64)
        acts = np.asarray(t.column('action').to_pylist(), dtype=np.float32)
        for e in np.unique(eps):
            m = eps == e
            order = np.argsort(frs[m])
            out[int(e)] = acts[m][order]
    return out


def source_action(eid: int, jaw_open_deg=85.94, gripper_full=49.08):
    z = np.load(str(ROOT / 'scripts' / f'ep_{eid:06d}.npz'), allow_pickle=True)
    ctrl = np.asarray(z['ctrl'], dtype=np.float64)
    phase = np.array([str(x) for x in z['phase']])
    a = np.degrees(ctrl[:, :6])
    a[:, 5] = gripper_full * (a[:, 5] / jaw_open_deg)
    return a, phase


def map_episodes(ds_actions):
    ids = [int(x) for x in open(ROOT / 'logs/sim156_episodes.txt') if x.strip()]
    src = {}
    for eid in ids:
        a, phase = source_action(eid)
        src[eid] = (a, phase)
    mapping = {}
    for ep, a in ds_actions.items():
        for eid, (s, phase) in src.items():
            if len(a) == len(s) and np.allclose(a[:5], s[:5], atol=1e-4):
                mapping[ep] = (eid, phase)
                break
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--root', default='/home/yyl/lerobot_datasets/sim156')
    ap.add_argument('--repo-id', default='local/sim156')
    ap.add_argument('--episodes', type=int, default=39, help='只测前 N 个 sim episode')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='logs/release_gripper_probe.csv')
    args = ap.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.modeling_act import ACTPolicy

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA unavailable -> cpu')
        args.device = 'cpu'

    ds_actions = load_dataset_actions(args.root)
    mapping = map_episodes(ds_actions)
    print(f'mapped {len(mapping)}/{len(ds_actions)} episodes')

    ds = LeRobotDataset(repo_id=args.repo_id, root=args.root, revision='v3.0')
    policy = ACTPolicy.from_pretrained(args.policy).to(args.device).eval()
    ov = {'device_processor': {'device': args.device}}
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=args.policy,
        preprocessor_overrides=ov, postprocessor_overrides=ov,
    )

    rows = []
    eps = sorted(mapping)[:args.episodes]
    for ep in eps:
        eid, phase = mapping[ep]
        starts = {}
        for name in ('pre_open', 'release', 'drop'):
            idx = np.where(phase == name)[0]
            if len(idx) == 0:
                continue
            if name == 'pre_open':
                starts['pre_open_last'] = int(idx[-1])
            elif name == 'release':
                starts['release_start'] = int(idx[0])
                starts['release_mid'] = int(idx[len(idx) // 2])
                starts['release_end'] = int(idx[-1])
            elif name == 'drop':
                starts['drop_start'] = int(idx[0])
        ep_meta = ds.meta.episodes[ep]
        base = int(ep_meta['dataset_from_index'])
        for label, fr in starts.items():
            rows.append((ep, eid, label, fr, base + fr))

    print(f'probe points: {len(rows)}')
    # batch by 8
    out_rows = []
    for start in range(0, len(rows), 8):
        chunk = rows[start:start + 8]
        samples = [ds[gidx] for _, _, _, _, gidx in chunk]
        batch = default_collate(samples)
        for cam in ds.meta.camera_keys:
            if cam in batch and batch[cam].dtype == torch.uint8:
                batch[cam] = batch[cam].float() / 255.0
        batch = pre(batch)
        with torch.inference_mode():
            pred = post(policy.predict_action_chunk(batch))
        pred = pred.detach().cpu().numpy()
        for bi, (ep, eid, label, fr, gidx) in enumerate(chunk):
            gt = float(samples[bi]['action'][5])
            out_rows.append(dict(episode=ep, source_id=eid, phase=label,
                                 frame=fr, pred_gripper=float(pred[bi, 0, 5]),
                                 gt_gripper=gt))
        print(f'  {min(start + 8, len(rows))}/{len(rows)}', flush=True)

    import csv
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader(); w.writerows(out_rows)
    print('written', out)
    print()
    for label in ('pre_open_last', 'release_start', 'release_mid', 'release_end', 'drop_start'):
        vals = [r['pred_gripper'] for r in out_rows if r['phase'] == label]
        gts = [r['gt_gripper'] for r in out_rows if r['phase'] == label]
        if vals:
            print(f'{label:14s} n={len(vals):2d}  pred mean={np.mean(vals):6.2f} '
                  f'med={np.median(vals):6.2f} min={np.min(vals):6.2f} max={np.max(vals):6.2f} '
                  f'| gt mean={np.mean(gts):6.2f}')


if __name__ == '__main__':
    sys.exit(main())
