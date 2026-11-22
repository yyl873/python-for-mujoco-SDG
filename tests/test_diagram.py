"""The README's architecture diagram is generated, so it can go stale — guard it.

A hand-tuned SVG rots silently: someone changes a number in the docs and the picture keeps
lying.  ``tools/make_pipeline_svg.py`` regenerates it from the artifacts and understands
``--check``, which is what this test calls.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

SVG = Path("docs/assets/pipeline.svg")
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"


def _run_check(repo_root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(repo_root / "tools" / "make_pipeline_svg.py"), "--check"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_committed_diagram_is_up_to_date(repo_root):
    result = _run_check(repo_root)
    assert result.returncode == 0, (
        "docs/assets/pipeline.svg 与产物不一致，跑 make svg 重新生成\n"
        f"{result.stdout}{result.stderr}"
    )


def test_diagram_is_well_formed_svg(repo_root):
    root = ElementTree.parse(repo_root / SVG).getroot()
    assert root.tag == f"{SVG_NAMESPACE}svg"
    assert root.get("viewBox"), "缺少 viewBox，GitHub 会按像素裁剪"
    # 无障碍：图片必须能被读屏软件念出来
    assert root.get("aria-label")


def test_diagram_carries_the_headline_numbers(repo_root):
    """图里的数字必须和 README/产物一致，否则图就成了第二个真相来源。"""
    text = (repo_root / SVG).read_text(encoding="utf-8")
    for needle in ("0.02 mm", "+122.2 mm", "p = 0.0094", "p = 0.0028", "0.06747", "52.6%"):
        assert needle in text, f"架构图里缺少 {needle!r}"


def test_no_text_overflows_the_canvas(repo_root):
    """粗估字宽，提前抓住"文字飞出框外"这类只能靠肉眼发现的问题。"""
    root = ElementTree.parse(repo_root / SVG).getroot()
    width, height = float(root.get("width")), float(root.get("height"))
    for element in root.iter(f"{SVG_NAMESPACE}text"):
        x, y = float(element.get("x")), float(element.get("y"))
        size = float(element.get("font-size"))
        is_mono = "mono" in (element.get("font-family") or "")
        em_width = 0.60 if is_mono else 0.52
        estimated = em_width * size * len(element.text or "")
        anchor = element.get("text-anchor", "start")
        left = x if anchor == "start" else (x - estimated / 2 if anchor == "middle" else x - estimated)
        assert left >= 0, f"文字越出左边界: {element.text!r}"
        assert left + estimated <= width, f"文字越出右边界: {element.text!r}"
        assert 0 <= y <= height, f"文字越出上下边界: {element.text!r}"