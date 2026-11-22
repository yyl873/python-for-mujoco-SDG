"""反事实配对评测：同一个场景，换一条指令，落点会不会跟着变？

这是判断"语言到底有没有被用上"的唯一干净方法。单纯看"成功率"没有用——
一个把盒子总往右扔的策略，在指令一半是"右"的情况下也有 50% 成功率。

三个口径必须同时报（本仓库踩过坑）：
  follow_acc   分母=落进箱子的 rollout          → 衡量"落对了没"
  ITT          分母=全部 rollout               → 衡量"整体干成了没"
  配对全对     分母=全部 (场景,seed) 配对       → 最硬：两条指令都要落对
"""
from __future__ import annotations

from collections import defaultdict

from .stats import binom_two_sided, wilson


def _landed_any(box: str | None, boxes: set[str]) -> bool:
    return box in boxes


def aggregate(rows: list[dict], task_order: list[str],
              box_of: dict[int, str], expect_map: dict[str, str] | None = None) -> dict:
    """聚合反事实评测结果。

    rows: 每条形如 {"episode_id", "seed", "ix", "landed_box", "steps", ...}
    task_order: 指令文本列表（下标即 one-hot 位置）
    box_of: 不需要；保留参数以兼容旧调用，实际用 expect_map
    expect_map: {指令文本: 期望落点箱名}；缺省则用 box_of
    """
    boxes = set(expect_map.values()) if expect_map else set(box_of.values())
    expect = expect_map or {task_order[i]: box_of[i] for i in box_of}

    landed = [r for r in rows if _landed_any(r.get("landed_box"), boxes)]
    correct = [r for r in landed if r.get("landed_box") == expect.get(task_order[r["ix"]])]
    k, n = len(correct), len(landed)
    lo, hi = wilson(k, n)
    return {
        "n_rollouts": len(rows),
        "n_landed_in_some_box": n,
        "follow_correct": k,
        "follow_acc": round(k / n, 4) if n else None,
        "follow_acc_itt": round(k / len(rows), 4) if rows else None,
        "follow_wilson": [round(lo, 4), round(hi, 4)],
        "binom_p": round(binom_two_sided(k, n), 6) if n else None,
    }


def aggregate_pairs(rows: list[dict], task_order: list[str],
                    expect_map: dict[str, str]) -> dict:
    """配对层面：同一 (episode, seed) 的两条指令是否都落对。"""
    pairs: dict[tuple, dict] = defaultdict(dict)
    for r in rows:
        pairs[(r["episode_id"], r["seed"])][r["ix"]] = r
    both = any_landed = 0
    for d in pairs.values():
        if len(d) < len(task_order):
            continue
        oks = [d[i].get("landed_box") == expect_map.get(task_order[i]) for i in d]
        if any(d[i].get("landed_box") is not None for i in d):
            any_landed += 1
        if all(oks):
            both += 1
    return {"n_pairs": len(pairs), "pairs_both_follow": both, "pairs_any_landed": any_landed}


def verdict(summary: dict, acc_min: float = 0.80, wilson_lo_min: float = 0.50) -> str:
    """预注册判决（看完数据不许改）。"""
    acc = summary.get("follow_acc") or 0.0
    lo = (summary.get("follow_wilson") or [0.0])[0]
    pairs = summary.get("pairs_both_follow") or 0
    if acc >= acc_min and lo > wilson_lo_min and pairs > 0:
        return "语言生效"
    if acc >= acc_min and lo > wilson_lo_min:
        return "落点随指令变，但配对全对=0（需排查配对数据）"
    return "语言未被使用（≈抛硬币）"
