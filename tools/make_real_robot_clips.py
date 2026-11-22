#!/usr/bin/env python3
"""Build the README's real-robot clips from two phone recordings.

The recordings live outside the repository and are plain phone video (one portrait
1992x3840, one 4K), so this tool takes their paths and emits two small assets::

    .venv-rl/bin/python tools/make_real_robot_clips.py \
        --compare   /path/to/comparison.mp4 \
        --finetuned /path/to/fine-tuned.mp4 \
        --compare-label "A  comparison recording" \
        --finetuned-label "B  after real-data fine-tune"

Output: ``docs/assets/real_robot_ab.mp4`` (h264) and ``docs/assets/real_robot_ab.gif``
(what the README embeds).

Three problems this tool solves, in the order they bite:

1. **A 45 s phone video has ~5 s of useful motion.** Pass 1 samples the clip every
   ``--probe-fps`` and records the mean frame-to-frame difference; the window with the
   largest summed motion wins. No hard-coded timestamps to rot.
2. **A portrait recording is 9:16 of mostly floor.** The same pass accumulates a motion
   *map*, so the crop is centred on the region that actually moves (the arm) instead of
   being letterboxed into a 200 px sliver. The crop is widened to ``--aspect`` (4:3).
3. **Phone exports are not always intact.** A truncated HEVC file makes the demuxer raise
   mid-stream, so decoding tolerates damaged packets and reports how much of the window
   survived instead of dying.

Panels are placed side by side at equal height -- the only sane way to combine a portrait
and a landscape clip without one of them shrinking to a postage stamp.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets"

THUMB_WIDTH = 96
LABEL_HEIGHT = 34
SEPARATOR_WIDTH = 8


@dataclass
class Clip:
    """One source recording plus the label drawn above its panel."""

    path: Path
    label: str


@dataclass
class Survey:
    """What pass 1 learned about a recording."""

    path: Path
    fps: float
    n_frames: int
    motion: list[float]          # per *probe*, mean abs diff to previous probe
    activity: np.ndarray         # same-resolution map, summed abs diff
    frame_shape: tuple[int, int]  # (height, width) of the source


# ---------------------------------------------------------------------------
# pass 1: where is the motion?
# ---------------------------------------------------------------------------
def survey(path: Path, probe_fps: float) -> Survey:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"打不开录像 {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / probe_fps)))

    motion: list[float] = []
    activity = None
    previous = None
    shape = (0, 0)
    index = 0
    while True:
        ok = capture.grab()
        if not ok:
            break
        if index % stride == 0:
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                shape = frame.shape[:2]
                scale = THUMB_WIDTH / frame.shape[1]
                thumb = cv2.resize(frame, (THUMB_WIDTH, max(1, int(shape[0] * scale))),
                                   interpolation=cv2.INTER_AREA)
                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
                if previous is None:
                    activity = np.zeros_like(thumb, dtype=np.float32)
                else:
                    diff = cv2.absdiff(thumb, previous)
                    motion.append(float(diff.mean()))
                    activity += diff.astype(np.float32)
                previous = thumb
        index += 1
        if total and index >= total:
            break
    capture.release()
    if not motion or activity is None:
        raise SystemExit(f"{path}: 一只可分析的帧都没有")
    return Survey(path=path, fps=fps, n_frames=index, motion=motion, activity=activity,
                  frame_shape=shape)


def best_window(motion: list[float], window: int) -> int:
    """Index of the ``window``-probe span with the largest summed motion."""
    window = min(window, len(motion))
    running = float(sum(motion[:window]))
    best_value, best_index = running, 0
    for index in range(window, len(motion)):
        running += motion[index] - motion[index - window]
        if running > best_value:
            best_value, best_index = running, index - window + 1
    return best_index


def activity_crop(survey_result: Survey, aspect: float, margin: float = 0.12):
    """Bounding box (x0, y0, x1, y1) in source pixels around the moving region.

    The motion map is thresholded at its 75th percentile of non-zero activity, so a
    lighting flicker in a corner or a subtitle overlay cannot drag the crop away from the
    arm. The box is then widened to ``aspect`` and padded by ``margin``.
    """
    height, width = survey_result.frame_shape
    activity = survey_result.activity
    if activity is None or activity.max() <= 0:
        return (0, 0, width, height)
    thresh = np.percentile(activity[activity > 0], 75) if np.any(activity > 0) else 0
    mask = activity >= max(thresh, 1e-6)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return (0, 0, width, height)

    scale_x = width / activity.shape[1]
    scale_y = height / activity.shape[0]
    y0, y1 = rows[0] * scale_y, (rows[-1] + 1) * scale_y
    x0, x1 = cols[0] * scale_x, (cols[-1] + 1) * scale_x
    centre_x, centre_y = (x0 + x1) / 2, (y0 + y1) / 2
    box_w, box_h = x1 - x0, y1 - y0

    # widen to the requested aspect, then pad
    if box_w / max(box_h, 1e-6) < aspect:
        box_w = box_h * aspect
    else:
        box_h = box_w / aspect
    box_w *= 1 + margin
    box_h *= 1 + margin

    box_w = min(box_w, width)
    box_h = min(box_h, height)
    left = int(round(min(max(centre_x - box_w / 2, 0), width - box_w)))
    top = int(round(min(max(centre_y - box_h / 2, 0), height - box_h)))
    return (left, top, int(round(left + box_w)), int(round(top + box_h)))


# ---------------------------------------------------------------------------
# pass 2: extract the window
# ---------------------------------------------------------------------------
def letterbox(frame: np.ndarray, aspect: float) -> np.ndarray:
    """Centre-crop the frame to ``aspect`` without rescaling (keeps the widest extent)."""
    height, width = frame.shape[:2]
    if width / height > aspect:
        target_w = int(round(height * aspect))
        left = (width - target_w) // 2
        return frame[:, left:left + target_w]
    target_h = int(round(width / aspect))
    top = (height - target_h) // 2
    return frame[top:top + target_h, :]


def read_window(clip: Clip, survey_result: Survey, start_probe: int, probe_count: int,
                args) -> tuple[list[np.ndarray], int, int]:
    """Crop, rescale and decimate the chosen window. Returns frames + (requested, kept)."""
    probe_stride = survey_result.fps / args.probe_fps
    first = int(round(start_probe * probe_stride))
    last = int(round((start_probe + probe_count) * probe_stride))
    stride = max(1, int(round(survey_result.fps / args.fps)))
    crop = activity_crop(survey_result, args.aspect)

    capture = cv2.VideoCapture(str(clip.path))
    if first:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames: list[np.ndarray] = []
    requested = kept = 0
    index = first
    while index < last:
        ok = capture.grab()
        if not ok:
            break
        if (index - first) % stride == 0:
            requested += 1
            ok, frame = capture.retrieve()
            if ok and frame is not None:
                if args.no_crop:
                    region = letterbox(frame, args.aspect)
                else:
                    x0, y0, x1, y1 = crop
                    region = frame[max(0, y0):y1, max(0, x0):x1]
                if region.size:
                    scale = args.height / region.shape[0]
                    frames.append(cv2.resize(
                        region,
                        (max(1, int(round(region.shape[1] * scale))), args.height),
                        interpolation=cv2.INTER_AREA,
                    ))
                    kept += 1
        index += 1
    capture.release()
    return frames, requested, kept


# ---------------------------------------------------------------------------
# composition and writers
# ---------------------------------------------------------------------------
def label_bar(width: int, text: str) -> np.ndarray:
    bar = np.full((LABEL_HEIGHT, width, 3), 28, np.uint8)
    scale = 0.42 if width > 240 else 0.32
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1,
                cv2.LINE_AA)
    return bar


def to_even(frame: np.ndarray) -> np.ndarray:
    """H.264 4:2:0 needs even width and height; pad rather than crop away pixels."""
    height, width = frame.shape[:2]
    pad_h, pad_w = height % 2, width % 2
    if not pad_h and not pad_w:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(16, 16, 16))


def compose(panels: list[tuple[list[np.ndarray], str]]) -> list[np.ndarray]:
    heights = {frames[0].shape[0] for frames, _ in panels if frames}
    if len(heights) != 1:
        raise SystemExit(f"panels must share a height, got {sorted(heights)}")
    count = max(len(frames) for frames, _ in panels)

    stacked_panels = []
    for frames, label in panels:
        width = frames[0].shape[1]
        stacked_panels.append((frames, label_bar(width, label)))
    separator = np.full((LABEL_HEIGHT + heights.pop(), SEPARATOR_WIDTH, 3), 16, np.uint8)

    out = []
    for index in range(count):
        rows = [np.vstack([bar, frames[min(index, len(frames) - 1)]])
                for frames, bar in stacked_panels]
        merged = rows[0]
        for row in rows[1:]:
            merged = np.hstack([merged, separator, row])
        out.append(to_even(merged))
    return out


def write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        for frame in frames:
            assert frame.shape[:2] == (height, width)
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.stat().st_size


def write_gif(frames: list[np.ndarray], path: Path, width: int, fps: float,
              max_frames: int, colors: int = 48) -> int:
    images = []
    for index, frame in enumerate(frames):
        if max_frames and index >= max_frames:
            break
        scale = width / frame.shape[1]
        small = cv2.resize(frame, (width, int(round(frame.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        images.append(Image.fromarray(small))
    palette = images[0].convert("P", palette=Image.ADAPTIVE, colors=colors)
    quantized = [image.convert("RGB").quantize(palette=palette, dither=Image.Dither.NONE)
                 for image in images]
    quantized[0].save(path, save_all=True, append_images=quantized[1:],
                      duration=int(round(1000 / fps)), loop=0, optimize=True)
    return path.stat().st_size


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare", required=True, help="对比录像（任意分辨率/朝向）")
    parser.add_argument("--finetuned", required=True, help="微调后模型的录像")
    parser.add_argument("--compare-label", default="A  comparison recording")
    parser.add_argument("--finetuned-label", default="B  after real-data fine-tune")
    parser.add_argument("--window", type=float, default=9.0, help="每条取多少秒")
    parser.add_argument("--height", type=int, default=380, help="每条画面的高度（像素）")
    parser.add_argument("--aspect", type=float, default=1.34, help="裁剪后的宽高比（默认 4:3）")
    parser.add_argument("--no-crop", action="store_true",
                        help="不裁剪，整帧 letterbox（自动裁剪把机器人切掉时用这个）")
    parser.add_argument("--fps", type=int, default=10, help="抽取帧率")
    parser.add_argument("--probe-fps", type=float, default=4.0, help="找动作段的采样率")
    parser.add_argument("--gif-width", type=int, default=520)
    parser.add_argument("--gif-fps", type=float, default=6.0)
    parser.add_argument("--gif-max-frames", type=int, default=54)
    args = parser.parse_args(argv)

    clips = [Clip(Path(args.compare), args.compare_label),
             Clip(Path(args.finetuned), args.finetuned_label)]
    for clip in clips:
        if not clip.path.exists():
            raise SystemExit(f"找不到录像 {clip.path}")

    probe_count = int(round(max(1.0, args.window) * args.probe_fps))
    panels = []
    for clip in clips:
        info = survey(clip.path, args.probe_fps)
        start = best_window(info.motion, probe_count)
        crop = activity_crop(info, args.aspect)
        crop_text = "off (letterboxed)" if args.no_crop else str(crop)
        print(f"  {clip.path.name}: {info.fps:.1f} fps, {info.n_frames} frames, "
              f"{len(info.motion)} probes -> motion window at probe {start} "
              f"({args.window:g} s), crop {crop_text} "
              f"of {info.frame_shape[1]}x{info.frame_shape[0]}")
        frames, requested, kept = read_window(clip, info, start, probe_count, args)
        if not frames:
            raise SystemExit(f"{clip.path}: 选定窗口里没有解出帧")
        if kept < requested * 0.5:
            print(f"    warn: 只解出 {kept}/{requested} 帧（源文件可能有损坏段）")
        print(f"    kept {kept} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
        panels.append((frames, clip.label))

    frames = compose(panels)
    height, width = frames[0].shape[:2]
    mp4 = OUT / "real_robot_ab.mp4"
    size_mp4 = write_mp4(frames, mp4, args.fps)
    gif = OUT / "real_robot_ab.gif"
    size_gif = write_gif(frames, gif, args.gif_width, args.gif_fps, args.gif_max_frames)
    print(f"✓ {mp4.relative_to(ROOT)}  {size_mp4 / 1048576:.2f} MB  "
          f"({width}x{height}, {len(frames)} frames)")
    print(f"✓ {gif.relative_to(ROOT)}  {size_gif / 1048576:.2f} MB  "
          f"({args.gif_width} wide, {args.gif_fps:g} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())