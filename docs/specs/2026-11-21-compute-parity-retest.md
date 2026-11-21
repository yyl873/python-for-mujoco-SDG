# spec：残差 RL 的"算力对等"复测（10× 步数 + 多进程并行）

- 日期：2026-11-21
- 状态：**待确认**（确认后才写 plan、才动代码）
- 相关：`docs/REPORT_RESIDUAL_RL.md` §17 / §18 / §19 / §20

---

## 1. 问题

现有对残差 RL 的否定结论（§18/§19/§20）有一个没堵上的缺口：
**训练只用了 30 000 步 ≈ 139 条 episode、单环境 `DummyVecEnv`、72 步/秒。**
SAC 在 6 维连续控制上的常见预算是 10 万–100 万步。因此所有负结论都能被一句话质疑：

> "你不是证明了残差 RL 没用，你只是没训够。"

这个 spec 的唯一目的就是**把这句质疑变成可检验的事实**：在**算法、奖励、动作空间、数据池、判据全部不变**
的前提下，只把**算力**提上去，看结论是否翻转。

## 2. 目标 / 非目标

**目标**
1. 在 10× 步数（300 000 步）与多进程并行的条件下，重训 3 个种子的残差 SAC（v1 奖励、±1.43°）。
2. 用**与 §18 完全相同的评测协议**（episodes 100-129 × seeds 3 = 90 次/臂，`judge=target`，
   `jaw_min_deg=16.05`）测这 3 个模型，与"基座+钳爪 78/90"做配对 McNemar。
3. 给出**一句话可引用的判定**：算力提到 10× 后，残差 RL 是否仍然不超过基座。

**非目标（本次明确不做）**
- 不改奖励函数、不改动作空间幅度、不改网络结构、不换算法（否则就不是"算力对等"对照了）。
- 不做 chunk 级残差 / 任务空间残差（那是 §20.4 的另一条线，本次不碰）。
- 不追求 SOTA，不刷 90 次之外的评测集。
- 不动 §18/§19/§20 已有的任何产物文件。

## 3. 现状与证据

| 事实 | 值 | 证据位置 |
|---|---|---|
| 现有训练预算 | 30 000 步 / 种子，**415–427 s**，72 步/秒 | `rl/curves/R1/summary.json` |
| 训练内见过的 episode 数 | **131–143 条**（中位 137） | `rl/curves/R*/s*.jsonl` 的 `episodes` |
| 训练内成功率趋势 | 9 个种子里 8 个**无上升趋势**（前/中/后段 0.00–0.22） | 同上 |
| 并行情况 | `DummyVecEnv([lambda: env])` = **1 个环境，只用 1 核** | `rl/train.py:132` |
| 机器 | 32 核，内存 **15 GB（可用 12 GB）**，RTX 4060 Laptop（**本任务用不上**：`mj_step` 与 ACT 推理都在 CPU） | `free -g`、日志 `No accelerated backend` |
| 评测基线 | 基座+钳爪 **78/90 = 86.7%**，clean 50/90；9 次独立运行完全一致 | §18.2 / §18.4 |
| 现有残差 9 臂（钳爪下） | 无一显著超过基座+钳爪，3 个显著更差 | §18.2 |
| oracle 上界（两种打分） | lift 25/19/9、approach 24/14/4（基座 28/30） | §19.2 / §20.2 |

## 4. 约束

1. **单进程不可行**：300k 步 × 415 s/30k = **~69 min/种子**（串行 3 种子 3.5 h），必须并行。
2. **内存 12 GB 可用**：每个子进程都要加载 ACT 基座 + MuJoCo 场景 + EGL 上下文。
   初值 `n_envs=8`，**必须先 pilot 测内存与吞吐**，超限就退到 4/6。
3. **SIGFPE**：`mj_step` 进程级异常会杀掉进程，并行后概率不变但影响面更大 → 保留 `--resume`，
   外层重试循环（沿用 `rl/run_clamp_all.sh` 的做法）。
4. **不污染结论**：除 `--steps` 与 `--n-envs` 外，命令行参数与 §5 训练 9 个模型时**逐项一致**
   （`--reward v1 --judge strict --residual-deg 1.43 --episodes 0-49`，不加 `--vecnorm-reward`）。
5. **产物隔离**：新 ckpt 写 `rl/ckpt/R1big/s{0,1,2}/`，新曲线写 `rl/curves/R1big/`，
   新评测写 `rl/eval/target_R1big_s*_clamp_90.json`——**不覆盖 R1 的任何文件**。

## 5. 备选方案与取舍

| | 做法 | 代价 | 风险 | 结论 |
|---|---|---|---|---|
| **A** | `SubprocVecEnv` 8 环境 + 300k 步 × 3 种子 + 90 次评测 | **~1.5 h** | 内存不足 → 退到 4 环境（耗时 ×2）；SIGFPE 需重试 | **选它** |
| B | 同 A 但串行单环境 | 3.5–4 h | 无内存风险，但慢一倍且浪费 31 核 | 不选 |
| C | 只跑 1 个种子（省 2/3 时间） | ~50 min | **破预注册判据**（须 3 种子），结论不可引用 | 不选 |
| D | 直接转 VLA，不做本实验 | 0 | 留"没训够"的软肋 | 用户已否 |

**为什么并行不会改变实验公平性**：SAC 的 `total_timesteps` 是所有环境的总步数，
8 个环境只是让同样的 300k 步更快收集完；采样分布来自同一母本池 `0-49`（各进程 seed 不同 → 初始条件不同）。
这是**标准做法**，不是给残差 RL 开小灶。

## 6. 选定方案

A。分三段，每段可独立验收：

1. **Pilot（约 10 min）**：`n_envs ∈ {4, 8}` 各跑 3000 步，量 ①步/秒 ②峰值内存 ③是否 SIGFPE。
   选吞吐/稳定性的最佳点，写进 plan 的实际参数。
2. **训练（约 30–60 min）**：300k 步 × 3 种子，`--resume` + 外层重试；记录 wall time 与 episode 数。
3. **评测（约 40 min）**：3 臂 × 90 次 rollout（`judge=target --jaw-min-deg 16.05`），
   用 `rl/clamp_analysis.py` 与基座+钳爪配对，输出表格 + 一句判定。

**判定规则（先写死，看完数据不许改）**
- **翻转**：任一残差臂 clean 相对基座+钳爪配对 McNemar **p < 0.017**（3 次检验 Bonferroni）
  **且** b > c（净改善）。→ 结论要改写，残差 RL 重新评估。
- **不翻转**：三个种子都达不到上述条件 → 报告写"算力提到 10× 仍无收益"，
  §18/§19/§20 的结论**升级为确证**，残差 RL 永久封存。
- 介于两者之间（如 1 个种子显著）→ 按 §17.6 的预注册判据处理：**仍算不构成确证**，但记录。

## 7. 接口 / 数据 / 文件变化

| 文件 | 变化 | 说明 |
|---|---|---|
| `rl/vec_factory.py` | **新增** | 顶层函数 `make_residual_env(rank, **kw)`，给 `SubprocVecEnv` 用（闭包不能被 pickle） |
| `rl/train.py` | 改 | 新增 `--n-envs`（默认 1 = 现有行为）；`n_envs>1` 时用 `SubprocVecEnv` + `VecMonitor`；`n_envs=1` 路径**代码不变** |
| `rl/run_r1big.sh` | **新增** | pilot + 训练 + 重试 + 评测的一键脚本 |
| `rl/ckpt/R1big/`、`rl/curves/R1big/` | 新增目录 | 新产物，不覆盖 R1 |
| `rl/eval/target_R1big_s*_clamp_90.json` | 新增 | 评测结果 |
| `docs/REPORT_RESIDUAL_RL.md` §21 | 新增 | 结论 |

**兼容性**：`--n-envs 1` 时行为与现在逐字节等价（同一 `DummyVecEnv`）→ 现有 9 个模型与所有历史结论不受影响。

## 8. 失败模式与回滚

| 失败模式 | 症状 | 处理 |
|---|---|---|
| 内存不足 / OOM | 子进程被 kill、系统卡死 | 降 `n_envs` 到 4 或 2；pilot 阶段就量峰值内存，不许盲上 8 |
| EGL 上下文冲突 | 子进程渲染报错或全黑 | 降 `n_envs`；必要时 `LP_NUM_THREADS=1` |
| `SubprocVecEnv` pickle 失败 | `AttributeError: Can't pickle local object` | 用 `rl/vec_factory.py` 顶层函数（已设计） |
| SIGFPE 频繁 | 外层重试超过 10 次 | 记录每次 attempt，最多 10 次；已有 `--resume` 保住进度 |
| 训练不收敛（仍无趋势） | 曲线 `mean_last20` 无上升 | **这是本实验要测的东西之一**，不当作失败，如实记录 |
| 想中途改判据 | 看到数字不理想 | **禁止**。§6 的判定规则先写死 |

**回滚**：所有新产物都在新目录（`R1big` / `*_clamp_90.json` 新文件名），
回滚 = `git checkout -- rl/train.py && rm -rf rl/ckpt/R1big rl/curves/R1big rl/eval/target_R1big_*`。
不动 `rl/ckpt/R1`、`rl/eval/target_R1_*`。

## 9. 验收标准（可执行 + 可观察）

**Pilot 通过**
```bash
bash rl/run_r1big.sh pilot
# 通过：输出里每个 n_envs 都有 "steps/s=?" 与 "peak_rss_gb=?"，且 peak_rss_gb < 10
# 失败：出现 OOM / 子进程异常退出 / peak_rss_gb >= 10（降到 4 再试一次）
```

**训练通过**
```bash
bash rl/run_r1big.sh train
# 通过：3 个 rl/ckpt/R1big/s*/model.zip 存在；3 个 rl/curves/R1big/s*.jsonl 最后一行 episodes >= 1000
#       （30 万步 / 平均 200 步每 episode ≈ 1500 条）
# 失败：任一种子 ckpt 缺失，或 episodes < 600（说明并行没生效）
```

**评测通过**
```bash
bash rl/run_r1big.sh eval && python3 rl/clamp_analysis.py --pat "target_R1big_s*_clamp_90.json"
# 通过：3 个 json 各 180 行（base 90 + base+R 90）；基座行复现 78/90
# 失败：基座行不是 78/90 → 说明评测环境被改动，先查 §18.4 的确定性
```

**结论产出**
在 `docs/REPORT_RESIDUAL_RL.md` §21 写一张表：
`臂 | 训练步数 | 见过 episode | wall | success | clean | 配对 McNemar vs 基座+钳爪 | 判定`
判定列只能取 §6 定义的三档之一。

## 10. 开放问题（不阻塞本实验，留给 VLA 那一步）

1. VLA 的时间预算与目标（面试故事 vs 完整实验）——决定语言条件做到什么深度。
2. 数据集里已经有了 `instruction` 字段（`env_residual.py:273` 读 `meta["instruction"]`），
   但**是否所有 episode 都是同一句**、是否有多任务场景，需要先查（决定 VLA 是"真多任务"还是"单任务+语言前缀"）。
3. 30k 步的 9 个旧模型要不要保留？建议保留（作为"小预算对照"）。