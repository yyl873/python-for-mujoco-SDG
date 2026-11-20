#!/usr/bin/env python3
"""rl/stack_video.py —— 把两条 episode 录像左右拼成一条对比视频（无 ffmpeg 机器的兜底）

为什么自己拼：这台机器没有 ffmpeg，而 `cv2.VideoWriter` 已经能写 mp4（`rl/eval_ab.py`
用它录像）。所以这里纯 cv2 做水平拼接 + 帧数对齐（短的一边重复末帧）+ 角标文字。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/stack_video.py \
    --left rl/eval/demo_ep0_base.mp4 --right rl/eval/demo_ep0_r.mp4 \
    --left-label "pure ACT (base)" --right-label "ACT + residual RL" \
    --out rl/eval/ab_demo_v2.mp4
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def read_all(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {path}")
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="左右拼接两条录像（帧数按长的一边对齐）")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left-label", default="left")
    ap.add_argument("--right-label", default="right")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)

    L, R = read_all(Path(a.left)), read_all(Path(a.right))
    if not L or not R:
        raise SystemExit("有一条视频是空的")
    n = max(len(L), len(R))
    # 短的一边用末帧补齐，保证左右同步
    L = L + [L[-1]] * (n - len(L))
    R = R + [R[-1]] * (n - len(R))
    if L[0].shape != R[0].shape:
        R = [cv2.resize(f, (L[0].shape[1], L[0].shape[0])) for f in R]

    h, w = L[0].shape[:2]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sink = None
    for cc in ("avc1", "mp4v"):
        sink = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*cc), a.fps, (w * 2, h))
        if sink.isOpened():
            break
    if sink is None or not sink.isOpened():
        raise SystemExit(f"VideoWriter 打不开: {out_path}")

    for i in range(n):
        frame = np.hstack([L[i], R[i]])
        cv2.putText(frame, a.left_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, a.left_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, a.right_label, (w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, a.right_label, (w + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        sink.write(frame)
    sink.release()

    sec = n / a.fps
    size = out_path.stat().st_size / 1e6
    print(f"ok {out_path}  {n} 帧 / {sec:.1f} s / {size:.2f} MB  "
          f"左 {len(Path(a.left).read_bytes()) and a.left}，右 {a.right}")
    if sec < 30:
        print(f"  ⚠️ 时长 {sec:.1f} s < 30 s（spec §9 S4 要求 ≥30 s）")
    return 0


if __name__ == "__main__":
    sys.exit(main())