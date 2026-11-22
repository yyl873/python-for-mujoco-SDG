# embodied-eval-kit

具身策略评测的最小工具包：**让"策略行不行"变成可复算的数字**，而不是一句"感觉变好了"。

零依赖（纯标准库），三件事：

| 模块 | 解决什么问题 | 关键 API |
|---|---|---|
| `eeval.stats` | 小样本比例不能只报 `k/n`：40/40 的 95% 下界只有 **91.2%** | `wilson(k, n)`、`binom_two_sided(k, n)`、`fisher_exact_2x2(a,b,c,d)`、`mcnemar_exact(b,c)` |
| `eeval.judge` | "抬起来就算成功"会奖励作弊（压得更深）；洁净判据要拆成 4 条 | `CleanCfg`、`Episode`、`judge_episode(ep, cfg)` |
| `eeval.counterfactual` | 判断"语言有没有被用上"：同场景换指令，落点变不变 | `aggregate(rows, task_order, expect_map=…)`、`aggregate_pairs(...)`、`verdict(summary)` |
| `eeval.receipt` | 三态回执 `ok / warn / fail`，**不许把 warn 写成 ok** | `Receipt().add(name, status, **facts)`、`exit_code(receipt)` |

## 为什么需要它（都是踩过的坑）

1. **只报成功率会骗人**：一个"总往右放"的策略，在指令一半是"右"时也有 50% 成功率。
   必须同时报 **follow_acc（分母=落箱）/ ITT（分母=全部）/ 配对全对（分母=全部配对）** 三个口径。
2. **只看抬升会奖励作弊**：本项目残差 RL 在旧口径上是 +5.6pp，按洁净口径（不歪盒 + 不穿透 + 连续抬升）
   却是 26.7% → 13.3%。判据严格度直接决定结论正负。
3. **warn 被写成 ok 等于伪造结论**：流水线自判"成功"、同时打印"约束 ❌ 药盒被撞动 6.85 mm"——正确回执是 warn。

## 快速开始

```bash
python examples/demo_synthetic.py     # 合成数据跑通全部口径（不需要 mujoco/数据集）
python tests/test_eeval.py           # 零依赖测试（也可用 pytest）
```

真实用法（配合自己项目的 rollout json）：

```python
from eeval import aggregate, aggregate_pairs, verdict

TASKS  = ["把药盒放进左边的木箱", "把药盒放进右边的木箱"]
EXPECT = {TASKS[0]: "box_a", TASKS[1]: "box_b"}

s = aggregate(rows, TASKS, {}, expect_map=EXPECT)   # rows: [{episode_id, seed, ix, landed_box}]
s.update(aggregate_pairs(rows, TASKS, EXPECT))
s["verdict"] = verdict(s)      # 预注册判据：follow_acc≥0.80 且 Wilson 下界>0.50 且配对全对>0
```

## 数值校验（不是拍脑袋的公式）

| 输入 | 本实现 | 来源 |
|---|---|---|
| `wilson(40, 40)` | 下界 **0.9124** | 本项目语言条件最终评测 40/40 |
| `fisher_exact_2x2(1,11,9,3)` | **0.0028** | 本项目真机 A/B（纯真机 1/12 vs 微调后 9/12） |
| `binom_two_sided(5,10)` | 1.0（抛硬币水平不显著） | — |

## 设计约束

- **零依赖**：不方便装 scipy 的环境（机器人机载、评测机、CI）优先。
- **小样本友好**：全部用精确检验/区间，不用正态近似。
- **判据与聚合分离**：`judge` 只管"这条算不算成功"，`counterfactual` 只管"语言有没有起作用"，
  互不耦合，方便替换成你任务自己的判据。

## License

MIT
