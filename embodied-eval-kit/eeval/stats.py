"""区间估计与假设检验（纯标准库实现，不依赖 scipy）。

为什么不用朴素"成功率 k/n"：n=40 时 40/40 的"100%"其实下界只有 91%。
所有结论都应该带区间——这是本仓库评测纪律的一部分。
"""
from __future__ import annotations

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 得分区间（适合小样本比例，k=0 或 k=n 也不会给出 [0,0]/[1,1] 这种假精确）。"""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """精确二项检验（双侧）：观测到"至少这么极端"的概率。

    用途：语言到底有没有被用上？如果落点与指令无关，follow_acc 应该 ≈ 抛硬币，
    这里给出的 p 值就是"纯巧合还能这么巧"的概率。
    """
    if n <= 0:
        return 1.0
    obs = _binom_pmf(k, n, p)
    total = 0.0
    for i in range(n + 1):
        pmf = _binom_pmf(i, n, p)
        if pmf <= obs + 1e-12:
            total += pmf
    return min(1.0, total)


def _log_factorial(n: int) -> float:
    return math.lgamma(n + 1)


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Fisher 精确检验（双侧），表为 [[a, b], [c, d]]。

    用途：两个模型各跑 n 次的成功率差异是否显著（本仓库真机 A/B 用的就是它）。
    """
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c

    def p_of(x: int) -> float:
        # 固定边缘时的超几何概率
        num = _log_factorial(r1) + _log_factorial(r2) + _log_factorial(c1) + _log_factorial(n - c1)
        den = (_log_factorial(n) + _log_factorial(x) + _log_factorial(r1 - x)
               + _log_factorial(c1 - x) + _log_factorial(r2 - c1 + x))
        return math.exp(num - den)

    p_obs = p_of(a)
    lo, hi = max(0, min(r1, c1) - min(r1, c1, n)), min(r1, c1)
    total = 0.0
    for x in range(max(0, c1 - r2), min(r1, c1) + 1):
        p = p_of(x)
        if p <= p_obs + 1e-12:
            total += p
    return min(1.0, total)


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验：同一批 (场景, seed) 上比较两个模型，只看"不一致"的对。

    用途：配对 A/B（同一个 episode 跑两个模型）比"各跑一堆取平均"敏感得多——
    因为场景难度被配对消掉了。b/c 是两个方向的不一致计数。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(_binom_pmf(i, n, 0.5) for i in range(k + 1)))
