#!/usr/bin/env python3
"""tools/make_assets.py —— 把演示 mp4 压成 README 能直接显示的 GIF

为什么需要：GitHub 的 README 里 `.mp4` 不会自动播放（要 <video> + 附件），而 `.gif` 可以
直接用 `![...](path)` 内嵌。但原视频 1.2~1.8 MB/条，四条就 6 MB，所以这里统一降采样。

用法（需要 .venv-rl：pyav + PIL + cv2）：
    .venv-rl/bin/python tools/make_assets.py
    .venv-rl/bin/python tools/make_assets.py --pairs left_s1 right_s1 --width 480 --fps 8

产物：docs/assets/<name>.gif（默认每条 < 3 MB，超了会告警而不是静默变大）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "rl" / "eval"
OUT = ROOT / "docs" / "assets"


def read_frames(path: Path, every_n: int, width: int) -> list[Image.Image]:
    """解码 → 降采样 → 缩放，返回 PIL 帧。"""
    frames: list[Image.Image] = []
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i % every_n:
                continue
            arr = fr.to_ndarray(format="rgb24")
            h, w = arr.shape[:2]
            nh = int(round(h * width / w))
            small = cv2.resize(arr, (width, nh), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(small))
    return frames


def write_gif(frames: list[Image.Image], out: Path, fps: float) -> int:
    if not frames:
        raise SystemExit("没有解出任何帧")
    out.parent.mkdir(parents=True, exist_ok=True)
    # 统一调色板（用第一帧做 palette）能让 GIF 体积小很多
    pal = frames[0].convert("P", palette=Image.ADAPTIVE, colors=128)
    pframes = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    pframes[0].save(out, save_all=True, append_images=pframes[1:],
                    duration=int(round(1000 / fps)), loop=0, optimize=True)
    return out.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="*", default=["left_s1", "right_s1"],
                    help="对应 rl/eval/demo_<name>.mp4")
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--src-fps", type=float, default=15.0, help="原视频帧率（demo 默认 15）")
    ap.add_argument("--max-mb", type=float, default=3.0)
    a = ap.parse_args()

    every_n = max(1, int(round(a.src_fps / a.fps)))
    bad = 0
    for name in a.pairs:
        src = EVAL / f"demo_{name}.mp4"
        if not src.exists():
            print(f"✗ 缺 {src.relative_to(ROOT)}（先跑 make demo / make demo-right）")
            bad += 1
            continue
        frames = read_frames(src, every_n, a.width)
        dst = OUT / f"demo_{name}.gif"
        size = write_gif(frames, dst, a.fps)
        mb = size / 1048576
        flag = "ok" if mb <= a.max_mb else "warn: 超过 --max-mb"
        print(f"{flag:>18}  {dst.relative_to(ROOT)}  {len(frames)} 帧  {mb:.2f} MB")
    print("\nREADME 引用方式：![demo](docs/assets/demo_left_s1.gif)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
