# 文档索引（7 份，其余历史文档都在 git 历史里）

| 文档 | 给谁看 | 内容 |
|---|---|---|
| [RESULTS.md](RESULTS.md) | **先看这个** | 自动生成的结果三表：语言条件评测 / 语言通道权重曲线 / 残差 RL |
| [ROUND_lang_pair_5000.md](ROUND_lang_pair_5000.md) | 想深挖的人 | 语言条件这一轮的完整报告：口径、失败拆解、边界 |
| [REPORT_RESIDUAL_RL.md](REPORT_RESIDUAL_RL.md) | 想深挖的人 | 残差 SAC 的**负结果**与根因（奖励被钻空子） |
| [REAL_AB_RESULTS_2026-10-28.md](REAL_AB_RESULTS_2026-10-28.md) | 想确认真机效果的人 | 真机 12 次/臂 A/B，抓取 1/12 → 9/12（Fisher p=0.0028） |
| [RUNBOOK_SDG.md](RUNBOOK_SDG.md) | 想复现数据的人 | 操作手册：刷数据、导出、1000 条怎么跑 |
| [PHYSICS_FACTS.md](PHYSICS_FACTS.md) | 想改代码/参数的人 | 生效的物理事实、验收数字、不许动的东西 |
| [GRIPPER_RELEASE_DIAGNOSTIC.md](GRIPPER_RELEASE_DIAGNOSTIC.md) | 想看排障过程的人 | "抓到放不掉"的根因链（腕关节约定冲突） |

> 过程类文档（内部计划稿、规格稿、学习笔记等）**不入公开库**，需要时从 git 历史取：
> `git log --diff-filter=D --name-only | grep docs/`。
