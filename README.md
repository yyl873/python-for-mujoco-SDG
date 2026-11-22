# SO-ARM101 · 纯物理仿真 → 语言条件抓取放置（零 ROS）

一条**零 ROS**的纯 MuJoCo 管线：真物理抓取（不写 `qpos`）→ 批量合成数据 → lerobot 数据集 →
**语言条件 ACT**（"放进左边 / 右边木箱"）→ **反事实配对评测**证明语言真的进了策略。

两块可验证的成果：**① 仿真数据混合显著提升真机抓取成功率**（Fisher p=0.0028）、
**② 配对数据让语言从"完全没用"变成"落箱即跟随 100%"**（配对全对 0/38 → 15/38）。

> 数字全部可由 `make tables` 从 `rl/eval/*.json` 重新生成 → [`docs/RESULTS.md`](docs/RESULTS.md)。

---

## ① 真机：只差"训练数据里有没有仿真"一个变量的对照（每模型 12 次，pilot 规模）

| 训练数据 | 抓取成功率 | 任务成功率 |
|---|---|---|
| 纯真机 139 集 | 1/12 = 8.3% | 1/12 = 8.3% |
| 真机 139 + 仿真 39 | 8/12 = 66.7% | 3/12 = 25.0% |
| 上者 + 真机微调 3000 步 | **9/12 = 75.0%** | 5/12 = 41.7% |

- 抓取提升 **Fisher 精确检验 p=0.0028**（显著），两组 Wilson 95% 区间不重叠。
- 任务成功率（p=0.155）与释放改善（p=0.637）**未达显著**，n=12 属 pilot —— 如实标注。
- 详见 [`docs/REAL_AB_RESULTS_2026-10-28.md`](docs/REAL_AB_RESULTS_2026-10-28.md)、
  根因链 [`docs/GRIPPER_RELEASE_DIAGNOSTIC.md`](docs/GRIPPER_RELEASE_DIAGNOSTIC.md)。

## ② 语言条件：反事实配对评测（留出场景 × 2 seed × 2 指令 = 76 rollout）

| 模型 | 落点 A / B / 未落箱 | follow_acc | 配对全对 | 判定 |
|---|---|---|---|---|
| v1 双指令数据（指令按 episode 区间分配 → 与场景共线） | 1 / 38 / 37 | 56.4%* | **0 / 38** | 语言未被使用 |
| **v2_pair 配对数据（同场景只换指令）** | 25 / 15 / 36 | **100%（40/40）** | **15 / 38** | **语言生效** |

\* v1 的 56.4% 是假象：它只是"总往右放"，凡指令为右的都算对；**配对全对 0/38** 才是真相。
关键变量是数据配对（`collect.py --scene-seed-offset`，同场景两条指令首帧 qpos 差 **0.000e+00**）。

权重层旁证：语言 2 列（state dim 6/7）在基座里**恰好为 0**（6→8 补零），
训练 5000 步后长到 **0.06747**（`per_col` 口径）；同口径下 v2_pair@1250 是 v1 的 **1.35×**。

---

## 它长什么样（演示）

同一场景、同一初始状态，**只改一个 one-hot**，箱子落到不同侧：

```bash
make demo          # 左指令 → rl/eval/demo_left.mp4
make demo-right    # 右指令 → rl/eval/demo_right.mp4
make demo-loop     # 三轮：同场景左右交替，逐轮出片
```

想用**说话**驱动（麦克风 → ASR → 指令 → 实时窗口/视频）：

```bash
P=.venv-rl/bin/python
CKPT=outputs/act_lang_pair/checkpoints/005000/pretrained_model
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl HF_ENDPOINT=https://hf-mirror.com WHISPER_MODEL=tiny \
  $P rl/demo_lang_policy.py --ckpt $CKPT --mic --viewer --loop --realtime \
  --episode 2400 --seed 1 --out rl/eval/demo_mic.mp4
```
语音三后端按序尝试（faster-whisper → vosk → 声纹标定），三者都不可用时**明确报错而不是瞎猜**。

---

## 管线

```
ik/test_ik.py                 逆运动学（24 次迭代，残差 0.9 mm）
      │
grasp/test_grasp.py           真物理抓取探针（法向力 2.6~4 N，药盒重力仅 0.113 N）
      │
tools/probe_contact_audit.py  ★ 六项闸门：下探零接触 / 压盒 ≤0.3 mm / 倾角 ≤5° / 抬升 ≥50 mm / 指缝 / 穿模
      │
render/test_render.py         录轨迹 + 渲染 h264 + 15 项自检（render 阶段不写 ctrl、不调 mj_step）
      │
collect/collect.py            SDG 批量合成（--jobs 并行；--scene-seed-offset 造反事实配对）
      │
collect/export_lerobot.py     → lerobot v3.0 数据集（ACT / SmolVLA 直接训）
      │
rl/run_lang_train.sh          训练壳：断电续跑 / 4 档自动降档 / 显卡闸门（无卡显式失败）
      │
rl/run_lang_eval.sh           反事实评测：同场景两条指令各跑一遍，判"落点是否随指令变"
      │
tools/make_tables.py          结果表自动生成（README 的数字不许手抄）
```

---

## 快速开始（以下都不需要 GPU）

```bash
make help      # 所有目标
make tables    # 从 rl/eval/*.json 重新生成 docs/RESULTS.md（不重跑实验）
make check     # 自检：结果表是否最新、json 字段是否齐、脚本能否编译
make demo      # 出一条演示视频
make audit     # 接触真实性审计（0.5 s/条，纯内存）
make eval      # 重跑 76 条反事实评测（CPU 8~15 分钟）
make train     # 5000 步训练（CPU ≈7.7 h / GPU ≈30 min）—— 不会自动跑
```

环境（本机踩过的坑，不设会直接失败）：

```bash
env -u PYTHONHOME -u PYTHONPATH <python>   # 否则报 No module named 'encodings'
export MUJOCO_GL=egl                       # 渲染只认 egl（glfw/osmesa 都会崩）
```

版本：Python 3.10（工具）/ 3.12（训练，lerobot 0.6.2）· mujoco 3.16.0 · torch 2.7.1 · opencv 4.5.4

---

## 三条硬约束（违反即返工）

1. **零 ROS**：不引入 `rclpy`/`moveit`；`old_ros/` 只作参考，禁止运行。
2. **不伪造抓取**：不写 `data.qpos` 把药盒"焊"在夹爪上；握持只能来自 `mj_step` 的接触力与摩擦。
3. **渲染与控制解耦**：render 阶段不写 `ctrl`、不调 `mj_step`，只对已录轨迹做 `mj_forward` 回放。

---

## 仓库地图

```
model/            MJCF 与 14 个网格（改模型只动这里）
ik/ grasp/ render/ collect/     仿真四步 + 自检
rl/               训练 / 评测 / 演示 / 探针（run_lang_train.sh、eval_lang.py、demo_lang_policy.py…）
tools/            执行器与探针（流水线、接触审计、结果表生成、策略回放）
docs/             方法论与逐轮报告（RESULTS.md 自动生成）
logs/             运行日志与断点档案
```

**不入库**（`.gitignore`，均可由代码重生成）：`outputs/`（ckpt，591 MB/档）、`videos/`、`dataset/`、
`rl/lang_ds/`（1.3 GB 数据集）、`scripts/`（SDG 母本）、`inspect/`。

---

## 我做了什么 / 站在谁的肩膀上

**我的**：真物理 SDG 管线与六项接触闸门 · 反事实配对采集（`--scene-seed-offset`）·
反事实评测与预注册判据（Wilson / 二项检验 / ITT 双口径）· 语言通道权重曲线工具 ·
训练壳硬化（断电续跑、4 档降档、显卡闸门）· 演示与语音链路 · 残差 RL 的负结果分析。

**第三方**：[lerobot](https://github.com/huggingface/lerobot)（数据格式与 ACT/SmolVLA 实现）、
SO-ARM101 本体模型。**策略实现不是我写的**，我写的是数据、评测、工程与结论。

---

## 边界（不许越界引用）

- `follow_acc 100%` 的分母是 **40 条落进箱子**的 rollout；含失败的口径是 **ITT 52.6%**。
- 评测是 **19 个留出场景 × 2 seed**，不是全场景空间；**单 seed 单次评测**，提升幅度不宜当稳定效应。
- 真机 A/B 是 **n=12 的 pilot**：抓取显著（p=0.0028），任务成功率**未达显著**。
- 语言是 **2 维 one-hot（左/右）**，不是自由文本理解；换说法不支持。
- 语言条件策略**只在仿真里评测过**，尚未在真机做闭环评测。

---

## 已知问题 / 下一步

1. **47% 的 rollout 没落进任何箱子**（11 条超时 + 25 条提前脱手；13/38 场景两条都失败）
   → 瓶颈在抓取搬运质量，不是语言通道。
2. **缺现代 BC 对比**：`diffusion` / `vqbet` 在 lerobot 0.6.2 里现成，尚未跑对照。
3. **缺消融**：chunk_size / use_vae / temporal_ensemble_coeff 三个旋钮未做对照（排障字典）。
4. **文本指令版**（one-hot → SmolVLA 文本条件）：数据集已带 `task` 字段，只差改评测脚本。
5. **真机闭环**：语言条件策略尚未在真机跑 20 次评测。

---

## 文档

| 文档 | 给谁看 |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | **自动生成**的结果三表（语言 / 权重曲线 / 残差 RL） |
| [`docs/ROUND_lang_pair_5000.md`](docs/ROUND_lang_pair_5000.md) | 语言条件这一轮的完整报告（口径 / 失败拆解 / 边界） |
| [`docs/REPORT_RESIDUAL_RL.md`](docs/REPORT_RESIDUAL_RL.md) | 残差 SAC 的负结果与根因（奖励被钻空子） |
| [`docs/RUNBOOK_SDG.md`](docs/RUNBOOK_SDG.md) | 操作手册：怎么刷数据、怎么导出、1000 条怎么跑 |
| [`docs/PHYSICS_FACTS.md`](docs/PHYSICS_FACTS.md) | 生效的物理事实与不可动参数 |

## License

MIT（`model/` 内网格素材与第三方库遵循各自许可）。