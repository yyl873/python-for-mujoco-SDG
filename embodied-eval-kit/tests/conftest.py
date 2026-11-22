"""Shared fixtures: put the package on ``sys.path`` and load the real rollout log."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "examples" / "data" / "language_pair_76rows.json"


@pytest.fixture(scope="session")
def real_run() -> dict:
    """A real evaluation log: 19 held-out scenes x 2 seeds x 2 instructions."""
    return json.loads(DATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def real_protocol(real_run):
    from eeval import Protocol

    return Protocol.from_dict(real_run["protocol"])