#!/usr/bin/env python3
"""Generate ``docs/assets/pipeline.svg``: the architecture diagram on the README.

The diagram is generated rather than hand-drawn so its numbers stay attached to the
artifacts they came from::

    python tools/make_pipeline_svg.py            # rewrite docs/assets/pipeline.svg
    python tools/make_pipeline_svg.py --check    # nonzero if the committed file is stale

The weight curve is read from ``rl/eval/lang_weight_curve.json``; the other numbers are
the ones ``docs/PHYSICS_FACTS.md`` and ``docs/REAL_AB_RESULTS_2026-10-28.md`` declare as
citable, and ``tests/test_repo_hygiene.py`` keeps the README honest about them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs" / "assets" / "pipeline.svg"
CURVE_FILE = ROOT / "rl" / "eval" / "lang_weight_curve.json"

WIDTH, HEIGHT = 1440, 1030
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

INK = "#12181f"
MUTED = "#5b6b7c"
FAINT = "#8895a4"
ACCENT = "#0b5cad"
ACCENT_SOFT = "#e8f0fa"
GOOD = "#0f7a4d"
WARN = "#a8480a"
BOX_FILL = "#f8fafc"
BOX_STROKE = "#cdd8e4"

STAGES = [
    ("Scene", "model/scene.xml · SO-ARM101 + two target boxes",
     ["6 randomized dimensions: box pose, grasp jitter, timing, camera, sensor, visuals",
      "data set sdg-v7"]),
    ("Inverse kinematics", "ik/test_ik.py",
     ["24 iterations, end-effector residual 0.92 mm   (gate: below 2 mm)"]),
    ("True-contact grasp + 6 gates", "tools/probe_contact_audit.py → grasp/contact_gates.py",
     ["grip force comes from mj_step, never from writing qpos",
      "penetration 0.02 mm · lift +122.2 mm · descend-segment contact 0/400"]),
    ("Render → batch SDG → lerobot v3",
     "render/test_render.py → collect/collect.py → collect/export_lerobot.py",
     ["15/15 render checks; the renderer never steps physics",
      "92 episodes / 41,922 frames   ·   paired set: 28 episodes / 12,670 frames"]),
    ("Language-conditioned ACT", "rl/run_lang_train.sh",
     ["instruction = 2-dim one-hot on observation.state[6:8]",
      "5,000 steps, 20 checkpoints, resumes after a power cut"]),
    ("Counterfactual evaluation", "rl/run_lang_eval.sh",
     ["19 held-out scenes × 2 seeds × 2 instructions = 76 rollouts",
      "same scene, same seed: only the instruction changes"]),
]

RESULTS = [
    ("p = 0.0094", "grasp 1/12 → 8/12 on real hardware",
     "same steps, same session, same robot: only the training data differs", GOOD),
    ("p = 0.0028", "the two-stage route reaches 9/12",
     "task success p = 0.155 — reported as not significant", MUTED),
    ("40/40 · 15/38", "followed the instruction whenever it landed",
     "Wilson lower bound 0.912; but only 15 of 38 scene pairs were right both ways", GOOD),
    ("52.6%", "the same run under intention-to-treat",
     "40 of 76 rollouts — the denominator that keeps 100% honest", WARN),
]


def esc(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Canvas:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def add(self, chunk: str) -> None:
        self.parts.append(chunk)

    def text(self, x, y, value, size=12, fill=INK, family=SANS, weight="400", anchor="start"):
        self.add(
            f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" fill="{fill}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>'
        )

    def rect(self, x, y, w, h, rx=10, fill=BOX_FILL, stroke=BOX_STROKE, sw=1.2):
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}"/>'
        )

    def line(self, x1, y1, x2, y2, stroke=BOX_STROKE, sw=1.2, marker=False):
        tail = ' marker-end="url(#arrow)"' if marker else ""
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{tail}/>'
        )

    def circle(self, cx, cy, r, fill):
        self.add(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}"/>')


def load_curve() -> list[tuple[int, float]]:
    """The real language-weight curve (``encoder`` column), with the pre-training zeros.

    ``rl/eval/lang_weight_curve.json`` stores ``per_col = (lang_l2/√2)/(other_l2/√6)``
    for the encoder and the VAE encoder; the diagram plots the encoder.
    """
    data = json.loads(CURVE_FILE.read_text(encoding="utf-8"))
    rows = data["rows"] if isinstance(data, dict) and "rows" in data else data
    points = []
    for row in rows:
        step = int(row.get("step") or row.get("checkpoint") or 0)
        weight = float(row.get("encoder") or row.get("per_col") or row.get("weight") or 0.0)
        if step > 0:
            points.append((step, weight))
    points.sort()
    return [(0, 0.0)] + points


def build() -> str:
    c = Canvas()
    c.add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" '
        f'aria-label="SO-ARM101 zero-ROS MuJoCo pipeline architecture">'
    )
    c.add("<defs>")
    c.add(
        f'<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{FAINT}"/></marker>'
    )
    c.add("</defs>")
    c.add(f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>')

    c.text(40, 46, "SO-ARM101 · language-conditioned grasping, zero ROS", size=21, weight="700")
    c.text(40, 70, "Each numbered box is a stage in this repository; each number is regenerated "
                   "from the artifacts by make tables.", size=12, fill=MUTED)
    c.line(40, 84, 1400, 84)

    # ---------------------------------------------------------------- pipeline
    x0, box_w = 40, 664
    top, box_h, gap = 104, 104, 30
    for index, (name, path, lines) in enumerate(STAGES):
        y = top + index * (box_h + gap)
        c.rect(x0, y, box_w, box_h)
        c.circle(x0 + 29, y + 27, 13, ACCENT)
        c.text(x0 + 29, y + 31.5, str(index + 1), size=13, fill="#ffffff", weight="700",
               anchor="middle")
        c.text(x0 + 52, y + 27, name, size=15, weight="650")
        c.text(x0 + 52, y + 49, path, size=11, fill=ACCENT, family=MONO)
        for row, chunk in enumerate(lines[:2]):
            c.text(x0 + 52, y + 71 + row * 17, chunk, size=11, fill=MUTED)
        if index:
            c.line(x0 + box_w / 2, y - gap + 2, x0 + box_w / 2, y - 7, stroke=FAINT, sw=1.6,
                   marker=True)

    aside = top + len(STAGES) * (box_h + gap) - gap + 18
    c.rect(x0, aside, box_w, 88, fill="#fffdf5", stroke="#e5d9b5")
    c.text(x0 + 18, aside + 24, "real_robot/so_arm101_bringup — the only place ROS 2 is allowed",
           size=13, weight="650")
    c.text(x0 + 18, aside + 46, "URDF/xacro + MoveIt2 + ros2_control + an SDGGenerator node.",
           size=11, fill=MUTED)
    c.text(x0 + 18, aside + 64, "The physical A/B (12 trials per model) ran through it. The zero-ROS rule covers the",
           size=11, fill=MUTED)
    c.text(x0 + 18, aside + 80, "simulation and data chain, which never imports rclpy.", size=11, fill=MUTED)

    # ------------------------------------------------------- language channel
    px, panel_w = 740, 660
    c.rect(px, 104, panel_w, 274)
    c.text(px + 20, 132, "The language channel is measurable", size=14, weight="650")
    for row, chunk in enumerate([
        "The instruction is a 2-dim one-hot appended to observation.state (dims 6 and 7).",
        "The ACT checkpoint starts with those two columns zero-padded (6 → 8 dims).",
        "Training moves their weight from 0.00000 to 0.06747 (per-column L2 of the encoder).",
    ]):
        c.text(px + 20, 158 + row * 18, chunk, size=11.5, fill=MUTED)

    curve = load_curve()
    gx, gy, gw, gh = px + 66, 216, panel_w - 136, 100
    vmax = max(value for _, value in curve) * 1.04 or 1.0
    last_step = max(step for step, _ in curve)

    def to_x(step: int) -> float:
        return gx + gw * step / last_step

    def to_y(value: float) -> float:
        return gy + gh - gh * value / vmax

    for frac in (0.0, 0.5, 1.0):
        y = gy + gh - gh * frac
        c.add(f'<line x1="{gx}" y1="{y:.1f}" x2="{gx + gw}" y2="{y:.1f}" stroke="#e6ecf3" stroke-width="1"/>')
        c.text(gx - 10, y + 4, f"{vmax * frac:.2f}", size=10, fill=FAINT, family=MONO, anchor="end")
    points = " ".join(f"{to_x(s):.1f},{to_y(v):.1f}" for s, v in curve)
    c.add(f'<polyline points="{points}" fill="none" stroke="{ACCENT}" stroke-width="2.2" '
          f'stroke-linejoin="round" stroke-linecap="round"/>')
    for step, value in curve:
        c.circle(to_x(step), to_y(value), 2.6, ACCENT)
    c.text(gx, gy + gh + 18, "0", size=10, fill=FAINT, family=MONO)
    c.text(gx + gw, gy + gh + 18, f"{last_step:,} training steps", size=10, fill=FAINT,
           family=MONO, anchor="end")
    final_step, final_value = curve[-1]
    c.text(to_x(final_step) - 4, to_y(final_value) - 9, f"{final_value:.5f}", size=10.5,
           fill=ACCENT, family=MONO, weight="650", anchor="end")
    c.text(px + 20, 372, "Same measurement on the mixed-instruction baseline: 0.01323 → paired data is 1.35× higher.",
           size=11, fill=MUTED)

    # --------------------------------------------------------------- results
    ry = 396
    c.rect(px, ry, panel_w, 264)
    c.text(px + 20, ry + 28, "What the pipeline measured", size=14, weight="650")
    for index, (big, line1, line2, colour) in enumerate(RESULTS):
        y = ry + 58 + index * 48
        c.text(px + 20, y, big, size=15, fill=colour, family=MONO, weight="700")
        c.text(px + 172, y, line1, size=11, fill=MUTED)
        c.text(px + 172, y + 15, line2, size=10.5, fill=FAINT)
        if index < len(RESULTS) - 1:
            c.line(px + 20, y + 24, px + panel_w - 20, y + 24, stroke="#eef2f7", sw=1)

    # ------------------------------------------------------------- reproduce
    cy = ry + 264 + 30
    c.rect(px, cy, panel_w, 118, fill=ACCENT_SOFT, stroke="#c9dcf0")
    c.text(px + 20, cy + 26, "Check it yourself — no GPU, no display, about 10 seconds",
           size=13, weight="650")
    for row, chunk in enumerate([
        "make test        # 102 tests over this repository's own artifacts",
        "make kit-test    # 142 tests over the evaluation kit",
        "make tables      # regenerate docs/RESULTS.md from rl/eval/*.json",
    ]):
        c.text(px + 20, cy + 50 + row * 20, chunk, size=11.5, fill=ACCENT, family=MONO)

    c.text(40, HEIGHT - 22,
           "github.com/yyl873/python-for-mujoco-SDG   ·   MIT   ·   simulation, data, evaluation and "
           "conclusions are mine; ACT/SmolVLA implementations come from lerobot.",
           size=11, fill=FAINT)
    c.add("</svg>")
    return "\n".join(c.parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed SVG differs from the generated one")
    args = parser.parse_args(argv)

    svg = build()
    if args.check:
        if not TARGET.exists():
            print(f"missing: {TARGET}", file=sys.stderr)
            return 1
        if TARGET.read_text(encoding="utf-8") != svg:
            print(f"stale: {TARGET} (run python tools/make_pipeline_svg.py)", file=sys.stderr)
            return 1
        print(f"up to date: {TARGET.relative_to(ROOT)}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(svg, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)} ({len(svg)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())