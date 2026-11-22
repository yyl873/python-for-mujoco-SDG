"""把"成功"拆成可数的几条判据（洁净口径），而不是一句"抬起来就算成功"。

背景（本仓库实测）：只看"最终抬升 ≥50 mm"，会奖励"把盒子压得更深再抬起来"这种作弊策略；
加上"不歪盒 + 不穿透"之后，同一个模型从 32.2% 掉到 13.3%。判据严格度直接决定结论正负。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CleanCfg:
    """洁净判据的阈值。全部用毫米/度，和仿真里的量纲一致。"""
    lift_min_mm: float = 50.0          # 至少抬这么高
    lift_consecutive: int = 5          # 而且要连续保持这么多步（防"瞬间弹起")
    tilt_max_deg: float = 5.0          # 夹稳期间最大倾角
    penetration_max_mm: float = 0.5    # 最大穿模深度
    require_release: bool = False      # 是否要求成功放进箱子（有些任务只要抓取）


@dataclass
class Episode:
    """一条 rollout 的最小轨迹记录（长度相同的三个序列）。"""
    lift_mm: list[float] = field(default_factory=list)       # 每步盒子相对初始的抬升
    tilt_deg: list[float] = field(default_factory=list)      # 每步盒子倾角
    penetration_mm: list[float] = field(default_factory=list)  # 每步穿模深度
    landed_box: str | None = None                            # 最终落进哪个箱子（无则 None）


def judge_episode(ep: Episode, cfg: CleanCfg | None = None) -> dict:
    """返回 {ok, reasons, max_lift_mm, max_tilt_deg, max_penetration_mm, consecutive_ok}。

    `reasons` 会列出**所有**不满足的条件，方便按失败类型计数（失败分布比成功率更有信息量）。
    """
    cfg = cfg or CleanCfg()
    lift = ep.lift_mm or [0.0]
    tilt = ep.tilt_deg or [0.0]
    pen = ep.penetration_mm or [0.0]

    max_lift = max(lift)
    max_tilt = max(tilt)
    max_pen = max(pen)

    # 连续抬升：找最长的一段连续达标
    best = cur = 0
    for z in lift:
        cur = cur + 1 if z >= cfg.lift_min_mm else 0
        best = max(best, cur)
    consecutive_ok = best >= cfg.lift_consecutive

    reasons: list[str] = []
    if max_lift < cfg.lift_min_mm:
        reasons.append(f"lift<{cfg.lift_min_mm}mm(max={max_lift:.2f})")
    if not consecutive_ok:
        reasons.append(f"连续抬升不足{cfg.lift_consecutive}步(最长{best})")
    if max_tilt > cfg.tilt_max_deg:
        reasons.append(f"tilt>{cfg.tilt_max_deg}deg(max={max_tilt:.2f})")
    if max_pen > cfg.penetration_max_mm:
        reasons.append(f"穿透>{cfg.penetration_max_mm}mm(max={max_pen:.2f})")
    if cfg.require_release and ep.landed_box is None:
        reasons.append("未落入目标箱")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "max_lift_mm": round(max_lift, 3),
        "max_tilt_deg": round(max_tilt, 3),
        "max_penetration_mm": round(max_pen, 3),
        "consecutive_ok": consecutive_ok,
    }
