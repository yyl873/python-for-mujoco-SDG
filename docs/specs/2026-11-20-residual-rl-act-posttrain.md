# spec · ACT 基座上的 RL 后训练（残差 RL PoC，2 天）

- 日期：2026-11-20
- 目标窗口：2 天（D1 = 打通观测口径 + Gym 封装 + pilot；D2 = 训练 + A/B + 交付物）
- 关系：本 spec 只定义"要什么、不要什么、怎么算过"。具体阶段与命令见
  `docs/plans/2026-11-20-residual-rl-act-posttrain.md`（**spec 确认后才写**）。
- 上位文档：`docs/specs/2026-10-26-junior-sim-engineer-interview.md` §G3 已把
  "关卡 6（闭环评估）"列为缺口 —— 本 spec 就是补这一关，同时补上"RL 后训练"这一层能力证明。

---

## 1. 问题

仓库现状：真机遥操数据 → MuJoCo SDG → lerobot v3 导出 → ACT 训练 → **离线 L1** 评测。
缺三件事：

1. **没有闭环评估**。ACT 的"成功率"全部来自真机（12 次/臂，抓取 1/12→9/12，p=0.0028），
   仿真侧只有离线 action L1。`tools/replay_policy.py` 的 `run_closed_loop()` 已写好，
   **但从未跑出过一条闭环指标**（`logs/` 里只有 `rollout_help.log`）。
2. **没有任何 RL 后训练**。仓库零 gymnasium 依赖、零 SB3、零残差层。
3. **没有"策略鲁棒性"证据**。现有 G1-G6 是**脚本轨迹**的接触真实性闸门（`tools/probe_contact_audit.py`），
   不是对"策略"的评测；残差 RL 一旦把物理搞脏，现在没有任何东西会报警。

用户的诉求：往 RL + VLA 方向转，需要一个**能在简历上讲、且每个数字都能被一条命令复现**的项目。

### 已实测的障碍（逐条带出处，全部亲自核对过）

| # | 障碍 | 证据 |
|---|---|---|
| O1 | **没有单一解释器同时有 `mujoco` 和 `torch`/`lerobot`** | `/usr/bin/python3.10`：mujoco 3.16.0 + numpy 1.26.4，无 torch/gymnasium/lerobot。`~/miniforge3/envs/lerobot312`(3.12)：torch 2.7.1+cu126 + lerobot 0.6.2 + gymnasium 1.4.0 + numpy 2.5.4，**无 mujoco** |
| O2 | **SB3 任何解释器都没有** | 三个解释器 `import stable_baselines3` 全部 ModuleNotFoundError |
| O3 | **真机数据训练的 ckpt 与仿真观测存在两处口径差** | ①`yyl100_lance_fixed` 的 `images.env` 通道里装的是腕部画面（`docs/REAL_DATASET_CAMERA_AUDIT.md` §5，血缘相关 +1.0000 / 平均绝对差 0.0000，**未修**）；②`replay_policy.run_closed_loop` 直接 `rig.grab()` 拿原始锐利图，而 SDG 采集过 `test_render.SensorModel`（blur/noise/gain/color jitter） |
| O4 | **本会话沙箱无 GPU** | `/dev/nvidia*` 缺失、`nvidia-smi` 通信失败；渲染退化 llvmpipe 174~197 ms/帧。用户终端有 GPU（1.7 ms/帧双相机） |
| O5 | **磁盘 20 G 可用（86% 已用）** | `df -h /home` → 148G/121G used |
| O6 | **仓库无 `.git`，改动不可回滚** | 仓库根无 `.git`（`.dsh/skills/so-arm101-sim` §1） |

---

## 2. 目标 / 非目标

### 目标（可测量，全部要落到命令）

- **G1 闭环评估闸门**：让"ACT 在 MuJoCo 里闭环"这件事第一次产出可复现指标
  （抬升 mm / 成功布尔 / 接触数），且观测口径对真机 ckpt 正确（O3 两处补偿）。
- **G2 残差 RL 框架**：冻结 ACT 基座 + 6 维残差策略（SAC），标准 Gymnasium 接口，
  训练可跑、可续、可 seed 复现。
- **G3 A/B 证据**：同一批 episode/seed 下 `base` vs `base + residual` 的成功率对照，
  **且**残差策略的 G1-G6 不得退化（两条判据分开报，不合并成一个 ok）。
- **G4 交付物**：README + 训练曲线图 + 成功率柱状图 + ≥30 s 对比视频 + REPORT，
  每个数字后面挂一条可复现命令。
- **G5 叙事对齐**：把"RL 后训练层与策略解耦（`PolicyBase` 接口）"讲清楚，
  明确 ACT 是 IL 策略、**不是 VLA**，VLA 接法是接口已对齐而非本轮验证。

### 非目标（明确不做，防止范围爆炸）

- N1 **不训新 ACT**（不碰 lerobot-train，不改数据集）。基座只用已有 checkpoint。
- N2 **不做权重级 RL 后训练**（档 2：给 ACT 的 CVAE 确定性头加高斯 log-prob 做策略梯度）。
  两天不现实，只写进 REPORT 的"下一步"。
- N3 **不做真机验证**。真机成功率需要上机 20+ 次/臂，本轮时间预算不够，不承诺。
- N4 **不改 MJCF 物理参数**（`model/*.xml` 只读），不动接触段步数 `descend 400 / grasp 300 / lift 400`。
- N5 **不改 `render/test_render.py` / `collect/collect.py` / `tools/replay_policy.py` 的现有行为**，
  只 import 复用；需要的新行为写在新文件里。理由：这三个是已验收的流水线，改了要重跑 100 s 全流水线与 G1-G6。
- N6 **不引入 ROS**（红线 1），不碰 `old_ros/`。

---

## 3. 现状与证据（可直接引用，不许再改口径）

### 3.1 可用基座 checkpoint（读的是 checkpoint 里的 `train_config.json`，不是文档）

| checkpoint | 训练数据 | 条数/帧 | 步数 | 相机标签 |
|---|---|---|---|---|
| `act_yyl151_lance` | `yyl100_lance_fixed`（**纯真机遥操，最早的版本**） | 151 / 67,649 | 20,000 | ⚠️ env/wrist 互换，未修 |
| `act_armA` | `real_train_fixed`（纯真机遥操） | 139 / 62,273 | 22,500 | ✅ 已修 |
| `act_armB` | real + sim 39（混训） | 178 / 80,026 | 22,500 | ✅ |
| `act_sim156_pre` | `sim156`（纯仿真） | 156 / 70,824 | 17,700 | ✅（与渲染同源） |

三个数据集 + 两个真机集**接口契约完全一致**：30 fps、feature 名 `observation.images.env` /
`observation.images.wrist` / `observation.state`、6 维 state（**度**）、
`chunk_size=100 / n_action_steps=100`（3.33 s 开环块）。→ "真机 ckpt 进仿真闭环"的数据前提已满足，
缺的只是 O3 的两处观测侧补偿。

### 3.2 已存在的闭环设施（本 spec 的主要复用对象）

- `tools/replay_policy.py`：`PolicyBase / LerobotPolicy / PlaybackPolicy / HoldPolicy`，
  `run_closed_loop()` 30 Hz、真渲染取图、`_deg_from_ctrl` / `_ctrl_from_deg` 度↔弧度口径、
  返回 `lift_mm(全程最大) / lift_end_mm / hdisp_mm / max_contact / success=(max_lift>=50mm)`。
- `render/test_render.py`：`CameraRig.grab()`（双相机）、`SensorModel`（真机传感器劣化模拟）、
  `contact_audit_from_arrays()` / `evaluate_success()`。
- `model/scene.xml` + `model/so_arm101.xml`：6 个 `position` 执行器（5 关节 + 爪），
  药盒是 `freejoint name="obj_free"`。
- `tools/probe_contact_audit.py`：G1-G6 现成闸门（0.5 s/条，纯内存）。

### 3.3 现有 G1-G6 口径（本 spec 直接沿用，不另立标准）

G1 下探接触 0/400 ｜ G2 压盒 0.39 mm ｜ G3 夹稳倾角 1.06° ｜ G4 抬升 +122.22 mm ｜
G5 指缝 −0.85/−1.07 mm ｜ G6 穿模 0.02 mm。

### 3.4 ★ S0 实测：纯真机基座在仿真闭环里不可用（2026-11-20 实测，命令见 plan §阶段0）

`rl/pilot.py`，15 s/条，30 Hz，判据 `max_lift ≥ 50 mm`：

| 基座 | 观测口径 | 升 | 成功 | TCP↔药盒最近 | 接触峰 |
|---|---|---|---|---|---|
| `act_sim156_pre` | raw | +137.70 mm | ✅ | 24.2 mm | 11 |
| `act_sim156_pre` | sensor | +143.07 mm | ✅ | 23.7 mm | 12 |
| `act_armA` | raw（state=ctrl） | 0 | ❌ | 76.6 mm | 0 |
| `act_armA` | raw（state=qpos） | 0 | ❌ | 76.2 mm | 0 |
| `act_armA` | swap+sensor（state=qpos） | 0 | ❌ | 92.9 mm | 0 |
| `act_yyl151_lance` | raw / swap / sensor / swap+sensor | 0 | ❌×4 | 104~116 mm | 0 |

→ **纯真机基座 0/7，仿真基座在 ep0 上 2/2**。S0 的 4 级 fallback 规则自动落到 **L3 = `act_sim156_pre`**。

补测 3 条 episode（`raw` 与 `sensor` 两种口径）：`raw 1/3`、`sensor 2/3`，合计 **3/6**；
失败模式是"抓到了没抬起来"（ep1 raw，抬升 2.20 mm、接触 12）与"把盒子撞倒"（ep2 两条，倾角 90°）。
→ 基座不是 100%，**给残差 RL 留了提升空间**，也让 A/B 有意义。

域差量化（`tools/compare_sim_real.py`，sim156 ↔ real_train_fixed）：
起点 `wrist_roll` −103.53° vs −86.54°（Δ≈17°）；**动作分布 W1 30~37°**
（shoulder_lift 35.60 / elbow 34.61 / wrist_flex 31.99 / wrist_roll 36.79）。
这就是纯真机基座进不了仿真的原因：它吃到的观测与它训练时的轨迹分布相差一个量级于 15 mm 的夹持窗口。

---

## 4. 约束

1. **零 ROS**；**不伪造抓取**（严禁 qpos 覆写让盒子跟着手走 —— 残差只能通过 `ctrl` + `mj_step` 生效）；
   **渲染与控制解耦**（`test_render.py` 的 render 阶段不许加 `mj_step`，新代码在其外）。
2. 前缀 `env -u PYTHONHOME -u PYTHONPATH`；**别把 Python 输出管道给 head/tail**（SIGFPE）；
   `mj_step` 约 7% 进程级 SIGFPE → 训练/评测入口必须有重试与断点续跑。
3. `MUJOCO_GL=egl` 唯一可用渲染后端。
4. 磁盘 ≤ 20 G：训练落盘只允许 JSONL 曲线 + 小 ckpt（SAC MLP 数 MB），**不落全帧视频**（评测视频按需单条）。
5. 无 git：新文件全部写在**新目录**，动手前把要碰的现存文件复制为 `*.bak`（本 spec 预期为零个）。
6. **闸门与奖励分离**：G1-G6 只做验收判据，**绝不进 reward**（否则 RL 会学着擦边过闸门）。

---

## 5. 备选方案与取舍

| | 档 1 残差 RL（本 spec 选定） | 档 2 权重级 RL 后训练 | 档 3 离线后训练（AWR/加权 BC） |
|---|---|---|---|
| 做法 | 冻结 ACT，外挂 6 维残差（SAC） | 给 ACT 加高斯头，更新部分权重 | 直接在 139/151 条真机数据上加权更新 |
| 两天可交付 | ✅ | ❌ | ✅ |
| 需 MuJoCo | 需 | 需 + 梯度通路 | 不需要 |
| 鲁棒性收益 | 有（天生做纠偏） | 有，更彻底 | 弱（行为约束内） |
| 简历含金量 | 中高（可讲清接缝与取舍） | 最高 | 中低，一眼被看穿 |

**基座决策规则（4 级 fallback，pilot 数据说话，不靠信仰）**：

```
L1. act_yyl151_lance（最早的纯真机版本；需 env/wrist 对调）
L2. act_armA（纯真机、标签干净）
L3. act_sim156_pre（与渲染同源，最可能闭环能跑；但叙事变"仿真基座"）
L4. 脚本 IK 专家（100% 成功、G1-G6 全绿；叙事降级为"残差 RL 叠在上层控制器"）
判据：pilot 3 episode，成功(success=max_lift>=50mm) ≥ 1/3 即采纳该层；否则下移一层。
```

---

## 6. 选定方案

### 6.1 结构

```
        ┌──────────── 30 Hz 控制周期（= 4 个 mj_step @ dt）────────────┐
obs ──▶ │ 基座 ACT（冻结，lerobot 官方推理管线，含 chunk 队列）        │──▶ action_il (6 维, 度)
        │        ↓                                                     │
        │   ctrl = _ctrl_from_deg(action_il) + clip(Δ, ±0.1 rad 单尺度)│──▶ mj_step × N
        │        ↑                                                     │
        │ 残差策略 R（SAC，MlpPolicy，6 维，只吃低维观测）              │
        └──────────────────────────────────────────────────────────────┘
```

- **残差只吃低维观测**（不喂图）：`[q(5,deg), gripper(1), box_pos_rel(3), box_vel_rel(3), contact_n(1), t_norm(1)] = 14 维`。
  理由：沙箱渲染 174 ms/帧、终端 1.7 ms/帧×2 相机；6 维残差的任务是"纠偏"而非"看世界"，
  低维观测把 1~3 万步的训练成本压到可接受，且视觉留在基座里、叙事更干净。
- **动作幅度只缩放一次**：`Box(-0.1, 0.1, shape=(6,))` 即最终幅度，**不再乘 0.1**（原计划的双重缩放 = 残差等于没开）。
- **ACT 的 chunk 队列必须 `policy.reset()`**：跨 episode 不 reset 会接着上一条的队列走（`LerobotPolicy.prepare` 已处理）。
- 观测侧两处补偿（O3）作为**开关**：`--swap-cams`、`--sensor-model`（缺省由 Stage 0 pilot 数据决定，写进 `rl/base_policy.py` 的注释与 REPORT）。

### 6.2 奖励（分段，和 G1-G6 分离）

| 项 | 定义 | 量级 |
|---|---|---|
| 稠密 | `-10 * ‖p_tcp − p_box‖₂`（米） | 0.1 m 处 −1 |
| 抓取 | `grip_contacts ≥ 2` 且 `lift > 5 mm` → `+0.5`/step | — |
| 抬升 | `+2 * tanh(Δ新最大抬升 / 0.05 m)` | ≤ +2 |
| 成功 | `max_lift ≥ 50 mm` 且接触保持 → `+50`，terminated | — |
| 压盒 | G2 口径瞬时压深 > 0.5 mm → `−1`/step | — |
| 撞倒 | 盒倾角 > 45° → `−10`，terminated | — |

终止：成功 / 撞倒；截断：15 s（450 控制步）或药盒水平位移 > 120 mm。

### 6.3 随机化（reset）

药盒 xy ±30 mm、yaw ±15°；药盒质量 ±10%、地面摩擦 ±10%；光照强度 ±10%。
**不抖 roll、不动 descend 步数配比**（skill 硬约束：roll 是指垫对齐轴，步数配比与起点耦合）。

---

## 7. 接口 / 数据 / 文件变化

**新增（全部在新目录，零改动现有文件）**

| 文件 | 职责 |
|---|---|
| `rl/base_policy.py` | 包 `replay_policy.LerobotPolicy`，加 `--swap-cams` / `SensorModel` 两个观测补偿开关，暴露 `act(state_deg, images) -> action_deg` |
| `rl/env_residual.py` | `gymnasium.Env`：`reset/step/_obs/_reward/_terminated`，重试包住 `mj_step`，含 `--selftest` |
| `rl/pilot.py` | 基座 4 级 fallback 的 pilot：N episode 闭环，输出 JSON（含 4 个开关组合） |
| `rl/train.py` | SB3 SAC 训练，多 seed、断点续、JSONL 曲线 |
| `rl/eval_ab.py` | A/B：`base` vs `base+R`，同 episode/seed 集，输出成功率 + G1-G6 |
| `docs/plans/2026-11-20-residual-rl-act-posttrain.md` | 阶段/命令/判据（spec 确认后写） |
| `docs/REPORT_RESIDUAL_RL.md` | 交付物：结论 + 每条数字挂一条命令 |
| `rl/ckpt/`、`rl/curves/`、`rl/eval/` | 产物目录（小文件） |

**解释器方案（O1/O2 的唯一解）**

- ~~P1（首选）~~**已否决**：本会话沙箱把整个文件系统只读挂载（`bwrap --ro-bind / /`），
  只有 `$WORKSPACE` 可写 → `pip install mujoco` 写 `~/miniforge3/envs/lerobot312` 直接
  `OSError: [Errno 30] 只读文件系统`。**且不应该改用户的 lerobot 训练环境**。
- **实际采用 P2**：`/home/yyl/miniforge3/envs/lerobot312/bin/python -m venv --system-site-packages .venv-rl`
  （venv 落在工作区内，`torch 2.7.1+cu126 / lerobot 0.6.2 / gymnasium 1.4.0 / numpy 2.5.4` 全部通过
  继承可见；只往 venv 里装 `mujoco==3.16.0` + `stable-baselines3==2.6.0`）。
- ★ **P2 的坑（已踩并回滚）**：直接 `pip install stable-baselines3` 会把 venv 里的 `torch`
  从 2.7.1 **升到 2.15.0**，与 conda 的 `torchvision 0.22.1+cu126` 打架 →
  `RuntimeError: operator torchvision::nms does not exist`（lerobot 进不去）。
  正确姿势：`pip uninstall -y torch`（退回继承的 2.7.1）+ `pip install --no-deps stable-baselines3==2.6.0`
  + 缺的纯 UI 依赖（matplotlib 及其纯 Python 依赖）也一律 `--no-deps` 逐个装。
  判据：`import torch, torchvision, lerobot, stable_baselines3` 全过，且 `torch.__version__` 仍是 2.7.1。

---

## 8. 失败模式与回滚

| 现象 | 最可能根因 | 立刻看哪 | 回滚 |
|---|---|---|---|
| pilot 3/3 失败、动作几乎不动 | 相机通道互换未补偿 / 未过 SensorModel / 取景不匹配 | 同一 pilot 的 4 组开关对照表 | 下移基座层（L2→L3→L4） |
| 训练 reward 全程平线 | `_compute_reward` 恒 0 / 残差幅度被双重缩放 / episode 永不结束 | `env_residual.py --selftest` 的 step 级打印 | 修 reward，不动基座 |
| 训练中途崩（Floating point exception） | `mj_step` 已知 ~7% SIGFPE | 退出码 + 断点文件 | 自动重试并续跑 |
| 成功率涨了但 G2/G6 变脏 | RL 用奖励 hack 到擦边 | `eval_ab.py` 的 G1-G6 列 | 判 **warn，不是 ok**；收紧压盒惩罚后重训 |
| `pip install` 动了 torch/lerobot | 依赖解析越权 | `pip freeze` diff | 按 before 回滚 → P2 venv |
| 磁盘告警 | 落了全帧视频 | `du -sh rl/` | 删视频，保留 JSONL+曲线 |

---

## 9. 验收标准（命令 + 可观察结果）

| 阶段 | 命令 | 通过判据 | 失败判据 |
|---|---|---|---|
| S0 pilot | `MUJOCO_GL=egl python rl/pilot.py --ckpts lance,armA --episodes 3` | 输出 JSON 含 4 组开关的 `success/lift_mm`；**至少一组 ≥1/3** → 定主臂 | 全部 ≤0/3 → 下移 L3/L4 |
| S1 环境自检 | `python rl/env_residual.py --selftest` | 200 步不崩；obs dim=14、act dim=6 断言通过；`terminated/truncated` 至少各触发 1 次 | 崩溃或维度不符 |
| S2 训练 | `python rl/train.py --seed 0,1,2 --steps 30000` | 3 seed 均值：末 5k reward ≥ 首 5k × 1.2 | 平线 → 回 §8 第 2 行 |
| S3 A/B | `python rl/eval_ab.py --episodes 30 --seeds 3` | ①成功率 Δ(base+R − base) ≥ +10pp **或** 明确记录为 0 并归因；②G1-G6 全部不退化（G2≤0.5 mm、G6≤0.5 mm、G3≤5°） | 任一闸门变脏 → **warn** |
| S4 交付 | `ls docs/REPORT_RESIDUAL_RL.md rl/curves/*.png rl/eval/*.mp4` | 文件存在、视频 ≥30 s、REPORT 每条数字后挂命令 | 缺任一 |

**三态回执纪律**：成功率涨但闸门变脏 = `warn`；成功率不涨但闸门全绿 = `warn`（不是 fail）；
两者都达成 = `ok`；跑不通/判据未达 = `fail`。

**预算**：S0 ≤45 min｜S1 ≤2 h｜S2 ≤3 h（3 seed × 30k 步，GPU 约 25~40 min/seed）｜S3 ≤1.5 h｜S4 ≤1.5 h。
合计机器时间 ≈ 8~10 h，两天窗口有余量；**长跑必须在用户终端（沙箱无 GPU）**。

---

## 10. 开放问题（需要你回答或由 Stage 0 数据裁决）

1. **你的终端 GPU 确认**：`nvidia-smi` 能看到几张卡？（我只在沙箱里看不到，torch 是 cu126 版）
2. **主臂**：由 S0 pilot 数据裁决（L1→L4），不预设。
3. **A/B 的 episode 数**：30 条（每条约 15 s）够不够？若要统计口径，我会给 Wilson 区间而不是裸比例。
4. **交付物语言**：README 中文 / 英文？（面试官可能看这个）
5. **可选档（不占主线）**：要不要在预算有余时用 `lerobot` 里已有的 SmolVLA 权重跑一次**推理**接入同一残差层？
   这能把"VLA"写实，但需要下载权重 + GPU 显存，且不影响 G1-G4。