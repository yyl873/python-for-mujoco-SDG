#!/usr/bin/env python3
"""tools/make_compare_video.py —— 把两条演示视频拼成"同场景左右指令"对比视频

要证明的事：**同一初始状态、只有指令不同 → 箱子落到不同侧**。
所以左指令那条放上面、右指令那条放下面，逐帧对齐（短的那条用末帧补），每条加标题条。

用法（需要 .venv-rl：pyav + cv2 + PIL）：
    .venv-rl/bin/python tools/make_compare_video.py
    .venv-rl/bin/python tools/make_compare_video.py --left left_s1 --right right_s1 --width 960

产物：
    docs/assets/compare_left_right.mp4   （h264，README 里当"完整视频"链接）
    docs/assets/compare_left_right.gif   （降采样，README 里内嵌播放）
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


def read_video(path: Path) -> list[np.ndarray]:
    frames = []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            frames.append(fr.to_ndarray(format="rgb24"))
    return frames


def label_bar(width: int, text: str, color=(120, 255, 120), h: int = 34) -> np.ndarray:
    bar = np.zeros((h, width, 3), np.uint8)
    cv2.putText(bar, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return bar


def compose(left: list[np.ndarray], right: list[np.ndarray],
            left_txt: str, right_txt: str, width: int) -> list[np.ndarray]:
    n = max(len(left), len(right))
    out = []
    for i in range(n):
        a = left[min(i, len(left) - 1)]
        b = right[min(i, len(right) - 1)]
        if a.shape[1] != width:
            scale = width / a.shape[1]
            a = cv2.resize(a, (width, int(round(a.shape[0] * scale))), interpolation=cv2.INTER_AREA)
            b = cv2.resize(b, (width, int(round(b.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        row1 = np.vstack([label_bar(width, left_txt, (120, 255, 120)), a])
        row2 = np.vstack([label_bar(width, right_txt, (120, 200, 255)), b])
        sep = np.full((6, width, 3), 40, np.uint8)
        out.append(np.vstack([row1, sep, row2]))
    return out


def write_mp4(frames: list[np.ndarray], path: Path, fps: int = 15) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    with av.open(str(path), mode="w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height = w, h
        st.pix_fmt = "yuv420p"
        for f in frames:
            assert (f.shape[1], f.shape[0]) == (w, h)
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
    return path.stat().st_size


def write_gif(frames: list[np.ndarray], path: Path, every_n: int, gif_w: int, fps: float,
              max_frames: int = 0, colors: int = 64) -> int:
    imgs = []
    for i, f in enumerate(frames):
        if i % every_n:
            continue
        scale = gif_w / f.shape[1]
        small = cv2.resize(f, (gif_w, int(round(f.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        imgs.append(Image.fromarray(small))
    if max_frames and len(imgs) > max_frames:
        imgs = imgs[:max_frames]
    pal = imgs[0].convert("P", palette=Image.ADAPTIVE, colors=colors)
    pframes = [im.convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    pframes[0].save(path, save_all=True, append_images=pframes[1:],
                    duration=int(round(1000 / fps)), loop=0, optimize=True)
    return path.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="left_s1", help="rl/eval/demo_<left>.mp4")
    ap.add_argument("--right", default="right_s1")
    ap.add_argument("--width", type=int, default=960, help="单条画面宽度（原视频 1280）")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--gif-width", type=int, default=500)
    ap.add_argument("--gif-fps", type=float, default=5.0)
    ap.add_argument("--gif-max-frames", type=int, default=110, help="0=全部（体积会很大）")
    ap.add_argument("--gif-colors", type=int, default=64)
    ap.add_argument("--left-label", default="instr = LEFT   -> lands in box A  (follow)")
    ap.add_argument("--right-label", default="instr = RIGHT  -> lands in box B  (follow)")
    a = ap.parse_args()

    lp, rp = EVAL / f"demo_{a.left}.mp4", EVAL / f"demo_{a.right}.mp4"
    for p in (lp, rp):
        if not p.exists():
            raise SystemExit(f"缺视频 {p.relative_to(ROOT)}（先跑 make demo / make demo-right）")
    left, right = read_video(lp), read_video(rp)
    print(f"左 {len(left)} 帧 / 右 {len(right)} 帧 → 对齐到 {max(len(left), len(right))} 帧")
    frames = compose(left, right, a.left_label, a.right_label, a.width)

    mp4 = OUT / "compare_left_right.mp4"
    s1 = write_mp4(frames, mp4, a.fps)
    gif = OUT / "compare_left_right.gif"
    s2 = write_gif(frames, gif, max(1, int(round(a.fps / a.gif_fps))), a.gif_width, a.gif_fps,
                   a.gif_max_frames, a.gif_colors)
    print(f"✓ {mp4.relative_to(ROOT)}  {s1/1048576:.2f} MB（{a.width}×{frames[0].shape[0]}）")
    print(f"✓ {gif.relative_to(ROOT)}  {s2/1048576:.2f} MB（{a.gif_width} 宽，{a.gif_fps:g} fps）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
