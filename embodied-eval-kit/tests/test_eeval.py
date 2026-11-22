"""最小回归测试：跑 python -m pytest tests/ 或直接 python tests/test_eeval.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeval import (CleanCfg, Episode, Receipt, aggregate, aggregate_pairs,  # noqa: E402
                   binom_two_sided, exit_code, fisher_exact_2x2, judge_episode,
                   mcnemar_exact, verdict, wilson)

TASKS = ["left", "right"]
EXPECT = {"left": "box_a", "right": "box_b"}


def test_wilson_matches_known_values():
    lo, hi = wilson(40, 40)
    assert abs(lo - 0.9124) < 5e-4, lo      # 40/40 的下界是 91.2%，不是 100%
    assert hi == 1.0
    lo0, hi0 = wilson(0, 76)
    assert lo0 == 0.0 and 0.02 < hi0 < 0.08  # k=0 也不该给出 [0,0]


def test_binom_and_fisher_are_small_for_clear_effects():
    assert binom_two_sided(40, 40) < 1e-9
    assert binom_two_sided(5, 10) > 0.5          # 抛硬币水平不该显著
    assert fisher_exact_2x2(1, 11, 8, 4) < 0.01        # 纯真机 vs 混合数据
    # 本项目真机报告里的 p=0.0028（纯真机 1/12 vs 微调后 9/12）——用本实现复算必须对上
    assert abs(fisher_exact_2x2(1, 11, 9, 3) - 0.0028) < 5e-4
    assert mcnemar_exact(8, 1) < 0.05


def test_clean_judge_catches_penetration_even_when_lift_is_high():
    ep = Episode(lift_mm=[0, 30, 80, 90, 85, 80],
                 tilt_deg=[0, 1, 2, 3, 3, 3],
                 penetration_mm=[0, 0.2, 1.5, 2.0, 1.8, 1.5])
    assert not judge_episode(ep, CleanCfg())["ok"]
    assert any("穿透" in r for r in judge_episode(ep, CleanCfg())["reasons"])


def test_counterfactual_detects_language_use():
    follow = [{"episode_id": i, "seed": 0, "ix": ix,
               "landed_box": EXPECT[TASKS[ix]]} for i in range(10) for ix in (0, 1)]
    ignore = [{"episode_id": i, "seed": 0, "ix": ix, "landed_box": "box_b"}
              for i in range(10) for ix in (0, 1)]
    s_ok = aggregate(follow, TASKS, {}, expect_map=EXPECT)
    s_ok.update(aggregate_pairs(follow, TASKS, EXPECT))
    s_bad = aggregate(ignore, TASKS, {}, expect_map=EXPECT)
    s_bad.update(aggregate_pairs(ignore, TASKS, EXPECT))
    assert s_ok["follow_acc"] == 1.0 and s_ok["pairs_both_follow"] == 10
    assert s_bad["pairs_both_follow"] == 0
    assert verdict(s_ok) == "语言生效" and "未被使用" in verdict(s_bad)


def test_receipt_three_states():
    r = Receipt().add("a", "ok", x=1).add("b", "warn", y=2)
    assert r.worst() == "warn" and exit_code(r) == 1
    r.add("c", "fail", z=3)
    assert r.worst() == "fail" and exit_code(r) == 2
    assert json.loads(r.to_json())["overall"] == "fail"


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"✓ {name}")
            except AssertionError as e:
                fails += 1
                print(f"✗ {name}: {e}")
    print("全部通过" if not fails else f"{fails} 个失败")
    sys.exit(0 if not fails else 1)
