"""eeval —— 具身策略评测的最小工具包（统计 / 判据 / 反事实聚合 / 三态回执）。

设计目标只有一个：**让"策略到底行不行"变成可复算的数字**，而不是一句"感觉变好了"。
所有函数都不依赖 scipy，纯标准库 + math，方便在任何环境里跑。
"""
from .stats import wilson, binom_two_sided, fisher_exact_2x2, mcnemar_exact
from .judge import CleanCfg, Episode, judge_episode
from .counterfactual import aggregate, aggregate_pairs, verdict
from .receipt import Stage, Receipt, exit_code

__all__ = [
    "wilson", "binom_two_sided", "fisher_exact_2x2", "mcnemar_exact",
    "CleanCfg", "Episode", "judge_episode",
    "aggregate", "aggregate_pairs", "verdict",
    "Stage", "Receipt", "exit_code",
]
__version__ = "0.1.0"
