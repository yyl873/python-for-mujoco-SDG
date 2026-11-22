#!/usr/bin/env python3
"""用合成数据演示 eeval 的三件事：区间估计 / 洁净判据 / 反事实聚合。

跑法： python examples/demo_synthetic.py
（纯标准库，不需要 mujoco、不需要数据集）
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eeval import (CleanCfg, Episode, Receipt, aggregate, aggregate_pairs,  # noqa: E402
                   binom_two_sided, exit_code, fisher_exact_2x2, judge_episode,
                   mcnemar_exact, verdict, wilson)

TASKS = ["把药盒放进左边的木箱", "把药盒放进右边的木箱"]
EXPECT = {TASKS[0]: "box_a", TASKS[1]: "box_b"}


def synth_rows(follow_prob: float, land_prob: float, n_scenes: int = 19, seeds: int = 2,
               bias_box: str | None = None):
    """造一份"两条指令 × 场景"的 rollout 记录。

    follow_prob 控制语言是否被遵循；bias_box 非空时"忽略语言"的策略永远落那个箱子
    （模拟本项目 v1 基线：落点 1 箱A / 38 箱B 的偏斜）。
    """
    rng = random.Random(0)
    rows = []
    for scene in range(n_scenes):
        for sd in range(seeds):
            for ix, task in enumerate(TASKS):
                landed = rng.random() < land_prob
                if not landed:
                    box = None
                elif bias_box is not None:
                    box = bias_box
                elif rng.random() < follow_prob:
                    box = EXPECT[task]
                else:  # 反着来
                    box = list(EXPECT.values())[1 - ix]
                rows.append({"episode_id": 2400 + scene, "seed": sd, "ix": ix,
                             "task": task, "landed_box": box})
    return rows


def report(tag: str, rows: list[dict]) -> dict:
    s = aggregate(rows, TASKS, {}, expect_map=EXPECT)
    s.update(aggregate_pairs(rows, TASKS, EXPECT))
    s["binom_p"] = binom_two_sided(s["follow_correct"], s["n_landed_in_some_box"])
    s["verdict"] = verdict(s)
    print(f"\n=== {tag} ===")
    print(f"  rollout {s['n_rollouts']}  落箱 {s['n_landed_in_some_box']}")
    print(f"  follow_acc {s['follow_acc']:.1%}  Wilson {[round(x, 3) for x in s['follow_wilson']]}"
          f"  p={s['binom_p']:.2e}")
    print(f"  ITT {s['follow_acc_itt']:.1%}   配对全对 {s['pairs_both_follow']}/{s['n_pairs']}")
    print(f"  判决：{s['verdict']}")
    return s


def main() -> int:
    print("（以下全部是合成数据，只演示口径怎么算；真实数字见项目 docs/RESULTS.md）")
    good = report("语言被遵循（90% 按指令落箱）", synth_rows(follow_prob=0.90, land_prob=0.55))
    bad = report("语言被忽略（总往右放，模拟 v1 基线）",
                 synth_rows(follow_prob=0.0, land_prob=0.55, bias_box="box_b"))

    print("\n=== 洁净判据（同一个策略，两种口径）===")
    cfg = CleanCfg()
    ep = Episode(lift_mm=[0.0, 10, 60, 70, 80, 75, 60],
                 tilt_deg=[0, 1, 3, 4, 6, 7, 6],          # 倾角超标
                 penetration_mm=[0, 0.1, 0.4, 0.6, 0.9, 0.8, 0.5])  # 穿透超标
    print("  旧口径（只看 max lift）：", "成功" if max(ep.lift_mm) >= cfg.lift_min_mm else "失败")
    r = judge_episode(ep, cfg)
    print("  洁净口径：", "成功" if r["ok"] else "失败", "原因：", r["reasons"])

    print("\n=== 两个模型配对比较（McNemar / Fisher）===")
    print(f"  McNemar b=8 c=1 → p={mcnemar_exact(8, 1):.4f}")
    print(f"  Fisher 纯真机 1/12 vs 微调后 9/12 → p={fisher_exact_2x2(1, 11, 9, 3):.4f}"
          "  ← 与本项目真机报告里的 p=0.0028 一致")

    rc = Receipt()
    rc.add("语言评测", "ok" if good["verdict"] == "语言生效" else "fail",
           follow_acc=f"{good['follow_acc']:.1%}", 配对全对=f"{good['pairs_both_follow']}/{good['n_pairs']}")
    rc.add("洁净判据", "warn" if max(ep.tilt_deg) > cfg.tilt_max_deg else "ok",
           tilt=f"{max(ep.tilt_deg)}deg")
    print("\n=== 三态回执 ===")
    print(rc.to_markdown())
    print(f"\nexit_code = {exit_code(rc)}（0=全绿 / 1=有 warn / 2=有 fail）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
