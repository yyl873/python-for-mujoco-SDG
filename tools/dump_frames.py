#!/usr/bin/env python3
"""从 mp4 抽指定时刻的帧存成 PNG，用于人工核对（夹爪开合、域差、画面异常）。

用法
----
  # 抽第 0 帧与第 3.5 秒的帧（默认存 inspect/frames/）
  env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
    /home/yyl/miniforge3/envs/lerobot312/bin/python tools/dump_frames.py \
      --video /home/yyl/lerobot_datasets/yyl100_dataset_fixed/videos/observation.images.wrist/chunk-000/file-000.mp4 \
      --at 0.0,3.5,8.0 --out inspect/frames/real_ep0_wrist

  # 一次多条视频，按 basename 命名
  ... --video a.mp4 --video b.mp4 --at 1.0

说明
----
* 只解码、不改原始视频；退出码 0 = 全部抽出。
* --at 是秒；不传则取首/中/尾三帧。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def dump(video: str, times: list[float], out_prefix: str) -> list[str]:
    import av
    from PIL import Image

    # 先读信息，未给 times 时取首/中/尾
    with av.open(video) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else None
        n = s.frames
    if not times:
        if dur is None:
            times = [0.0]
        else:
            times = [0.0, dur / 2.0, max(0.0, dur - 1.0 / 30.0)]

    written = []
    with av.open(video) as c:
        stream = c.streams.video[0]
        want = sorted(times)
        idx = 0
        for frame in c.decode(stream):
            ts = float(frame.pts * frame.time_base) if frame.pts is not None else idx / 30.0
            if want and ts + 1e-6 >= want[0]:
                t = want.pop(0)
                arr = frame.to_ndarray(format="rgb24")
                os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
                p = f"{out_prefix}_t{t:.3f}s.png"
                Image.fromarray(arr).save(p)
                written.append(p)
                if not want:
                    break
            idx += 1
    meta = {"video": video, "frames": n, "duration_s": dur, "written": written}
    return written, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--at", default="", help="秒，逗号分隔，如 0,1.5,9.2；留空=首/中/尾")
    ap.add_argument("--out", default="inspect/frames", help="输出前缀（不含扩展名）")
    args = ap.parse_args()

    times = [float(x) for x in args.at.split(",") if x.strip()] if args.at else []
    rc = 0
    for v in args.video:
        name = os.path.splitext(os.path.basename(v))[0]
        prefix = os.path.join(args.out, name)
        try:
            written, meta = dump(v, list(times), prefix)
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {v}: {type(e).__name__}: {e}")
            rc = 1
            continue
        print(f"[ok] {name}  frames={meta['frames']} dur={meta['duration_s']} -> {len(written)} 帧")
        for p in written:
            print(f"     {p}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
