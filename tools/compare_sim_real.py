#!/usr/bin/env python3
"""对比 sim 与 real 的 episode 首帧关节分布和 action 分布差（W1）。

用法：
  env -u PYTHONHOME -u PYTHONPATH \
    /home/yyl/miniforge3/envs/lerobot312/bin/python \
    tools/compare_sim_real.py \
      --sim /home/yyl/lerobot_datasets/sim156 \
      --real /home/yyl/lerobot_datasets/real_train_fixed
"""
from __future__ import annotations
import argparse, glob
import numpy as np
import pyarrow.parquet as pq

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def load(root: str):
    """返回：所有 action、所有 state、每个 episode 首帧 state、action - 该 episode 首帧 state。"""
    acts, states, firsts, deltas = [], [], [], []
    for f in sorted(glob.glob(root + '/data/**/*.parquet', recursive=True)):
        t = pq.read_table(f, columns=['episode_index', 'frame_index', 'action', 'observation.state'])
        eps = np.asarray(t.column('episode_index').to_pylist(), dtype=np.int64)
        frs = np.asarray(t.column('frame_index').to_pylist(), dtype=np.int64)
        act = np.asarray(t.column('action').to_pylist(), dtype=np.float32)
        st = np.asarray(t.column('observation.state').to_pylist(), dtype=np.float32)
        order = np.lexsort((frs, eps))
        eps, frs, act, st = eps[order], frs[order], act[order], st[order]
        for e in np.unique(eps):
            m = eps == e
            a, s = act[m], st[m]
            firsts.append(s[0])
            deltas.append(a - s[0])
        acts.append(act)
        states.append(st)
    if not acts:
        raise SystemExit(f'no parquet under {root}')
    return (np.concatenate(acts), np.concatenate(states),
            np.stack(firsts), np.concatenate(deltas))


def w1(a: np.ndarray, b: np.ndarray, n: int = 2001) -> float:
    q = np.linspace(0, 1, n)
    return float(np.trapezoid(np.abs(np.quantile(a, q) - np.quantile(b, q)), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim', required=True)
    ap.add_argument('--real', required=True)
    args = ap.parse_args()

    sa, ss, sf, sd = load(args.sim)
    ra, rs, rf, rd = load(args.real)
    print(f'sim  frames={len(sa):6d} episodes={len(sf):4d}')
    print(f'real frames={len(ra):6d} episodes={len(rf):4d}')

    print('\n[episode 首帧 state]')
    print(f'{"joint":15s}{"sim mean":>10s}{"real mean":>10s}{"sim std":>9s}{"real std":>9s}')
    for i, j in enumerate(JOINTS):
        print(f'{j:15s}{sf[:, i].mean():10.2f}{rf[:, i].mean():10.2f}'
              f'{sf[:, i].std():9.2f}{rf[:, i].std():9.2f}')

    print('\n[action 分布]')
    print(f'{"joint":15s}{"W1":>8s}{"sim p5":>8s}{"sim p95":>9s}{"real p5":>8s}{"real p95":>9s}')
    for i, j in enumerate(JOINTS):
        print(f'{j:15s}{w1(sa[:, i], ra[:, i]):8.2f}'
              f'{np.percentile(sa[:, i], 5):8.2f}{np.percentile(sa[:, i], 95):9.2f}'
              f'{np.percentile(ra[:, i], 5):8.2f}{np.percentile(ra[:, i], 95):9.2f}')

    print('\n[action - 该 episode 首帧 state，去掉起点差异后的 W1]')
    for i, j in enumerate(JOINTS):
        print(f'{j:15s} sim_mean={sd[:, i].mean():7.2f} real_mean={rd[:, i].mean():7.2f} '
              f'W1={w1(sd[:, i], rd[:, i]):7.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
