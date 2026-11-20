# spec：ACT 基座 + 残差 RL（SAC）+ 物理探针的 PoC 框架

- 日期：2026-11-20
- 状态：**待用户确认**（确认后才写 plan 并动手）
- 预算：2 天（我方沙箱无 GPU，长跑与最终计时在用户终端执行）

---

## 1. 问题

现有仓库把"仿真数据 → ACT"这条链跑通了（sdg-v8、act_sim156_pre 等 7 个 checkpoint、真机 A/B 12 次/臂），
但**"策略在仿真里闭环到底行不行"这件事从未被测量过**：

- `tools/replay_policy.py` 有完整的闭环通路（`LerobotPolicy` + `run_closed_loop` + 30 Hz 渲染取图 + 抬升判据），
  但 `logs/` 里只有 `rollout_help.log`（跑过 `--help`）和 `eval_holdout_*.json`（离线 L1），**没有一条闭环指标**。
- 用户拿到的"两天突击计划"假设"封装 Gym + 接入 ACT"是从零开始的工程活，
  并把"跑通 1 万步就有趋势"当作结论。两者都与仓库实况不符（详见 §3）。

本 spec 要解决的问题是：**在 2 天内，用仓库已有的闭环通路，把"残差 RL 相对纯 ACT 是否提升成功率、
且不破坏物理真实性"变成一组可复现的数字。**

## 2. 目标 / 非目标

### 目标（必须达成）
1. G1：一个 `gymnasium.Env` 兼容的**残差环境**，ACT 基座动作 + RL 残差叠加，走真物理 `mj_step`。
2. G2：SAC（stable-baselines3）在该环境上训练成功，3 个 seed 各 ≥30k env step，产出可用 checkpoint。
3. G3：**A/B 对照数字**：纯 ACT vs ACT+残差 RL，在 ID（同分布）与 OOD（域随机化）两套测试集上的成功率 + G1-G6 物理探针。
4. G4：一段"ACT vs ACT+RL 并排"的 mp4 + 一张成功率/奖励曲线图 + README（可直接挂 GitHub / 面试口述）。

### 非目标（明确不做）
- 不重训 / 不改 ACT 基座权重；不训练任何"VLA 大模型"。
- 不追求统计显著（样本量 ≤ 3×20×3，只报 Wilson 区间，不吹 p 值）。
- 不接真机；不做 sim2real 迁移验证（真机成功率已有 `REAL_AB_RESULTS_2026-10-28.md`，本 spec 不碰）。
- 不给 RL 喂图像（见 §6.2 决策 D2）。
- 不动 `render/test_render.py` 的 render 阶段、不动接触段步数、不动任何"不许动的参数"（见 skill `so-arm101-sim` §5）。

## 3. 现状与证据

| 事实 | 证据 |
|---|---|
| 闭环通路已存在，从未量过 | `tools/replay_policy.py` 436 行，`run_closed_loop()` L181；`logs/` 无闭环指标 |
| 解释器分裂，无单一环境能同时做物理+策略 | `/usr/bin/python3.10`：mujoco 3.16.0 / numpy 1.26.4，**无 torch/gym/SB3**；`~/miniforge3/envs/lerobot312`（3.12）：torch 2.7.1+cu126 / lerobot 0.6.2 / gymnasium 1.4.0 / numpy 2.5.4，**无 mujoco**；全部解释器**无 stable-baselines3** |
| ACT checkpoint 可用 | `act_sim156_pre/checkpoints/last/pretrained_model`，`chunk_size=100 / n_action_steps=100 / resnet18 / dim_model=512`，训练到 17700 步 |
| **ACT 一个 chunk 开环执行 3.33 s** | `n_action_steps=100` @ 30 Hz → 残差 RL 有明确的纠偏空间 |
| 观测口径已定 | `observation.state`(6 维，度) + `observation.images.env` + `observation.images.wrist`（`LerobotPolicy.act` L114） |
| 物理模型口径 | 6 个 position 执行器（5 臂关节 + jaw），`sts3215`，`ctrlrange` 弧度制；药盒 `obj_free` freejoint |
| 成功判据已存在且唯一 | `run_closed_loop` 尾部：`success = max_lift >= 50.0 mm`（与 `render/test_render.evaluate_success` 同口径） |
| G1-G6 是物理审计闸门，不是鲁棒性探针 | `render/test_render.contact_audit_from_arrays` L547；实测 G1 0/400、G2 0.39 mm、G3 1.06°、G4 +122.22 mm、G5 −0.85/−1.07 mm、G6 0.02 mm |
| 域随机化口径已有先例 | `collect/collect.py` L527-544：`gz_delta_mm ±1.5`、`camera.env_pos_jitter 0.01~0.03 m`、`color 0.02~0.05`、`ROLL_JITTER_DEG=0` |
| 无 GPU（沙箱）/ 有 GPU（用户终端） | 沙箱 `nvidia-smi` 失败、llvmpipe 174~197 ms/帧；skill 记用户终端 1.7 ms/帧、单条 19.1 s |
| 磁盘余量 | `/home` 148 G，已用 86%，**剩 20 G** |

## 4. 约束

- C1 **红线**：零 ROS；握持必须来自 `mj_step` 接触+摩擦（禁止写 `qpos` 伪造）；`render/test_render.py` 的 render 阶段不写 ctrl、不 `mj_step`。
  （注：RL 闭环环境本身就是控制循环，它写 ctrl 调 `mj_step` 属正常路径；红线 3 只约束 render 阶段的视频回放。）
- C2 **不改现有链路**：`collect/`、`render/`、`export_lerobot.py`、`run_1000.sh`、`run_train_v8.sh` 一行不动。
- C3 **不污染 lerobot 环境**：ACT 训练/导出依赖 `lerobot312`，不得在其中升级/降级任何包。
- C4 本机环境陷阱：`env -u PYTHONHOME -u PYTHONPATH`；**不要**把 Python 输出管道给 `head/tail`（Fatal FPE）；`mj_step` 约 7% 进程级 SIGFPE 需重试。
- C5 时间：2 天。任何单步 pilot 失败不得连累整体（每阶段可独立验收、可回滚）。
- C6 磁盘：训练日志/视频 ≤3 G。

## 5. 备选方案与取舍

| | A **双基座 + pilot 闸门**（选定） | B 纯 ACT 主臂（用户最初贴的计划） | C 纯脚本专家基座 |
|---|---|---|---|
| 基座 | ACT（`act_sim156_pre`）主臂 + 脚本 IK 专家兜底臂 | 只有 ACT | 只用已验证 100% 的脚本专家 |
| 前置 | ≤30 min pilot 量 ACT 闭环成功率 | 无 | 无 |
| 2 天风险 | 中低（必出可交付物） | 高：若 ACT 闭环成功率≈0，两天赔在救 base | 低 |
| 叙事 | 分层解耦 + 退化路径，有取舍 | 最贴"VLA+残差 RL"原话 | VLA 成分下降 |
| 结论 | **采纳** | 作为 A 的失败分支保留 | 作为 A 的兜底臂（同一套 wrapper，只换 `PolicyBase`） |

选定 A 的理由：仓库里"ACT 闭环成功率"是一次都没测过的最大不确定性；先花 30 分钟打掉它，
再决定算力押在哪，比照模板跑 5 万步更重要。A/B/C 三臂共用同一个 wrapper、同一个奖励、同一套探针，切换成本只是换一个策略子类。

## 6. 选定方案

### 6.1 Python 栈（决策 D1）：独立 venv，不动 conda
```
/home/yyl/miniforge3/envs/lerobot312/bin/python -m venv --system-site-packages ~/venvs/rl_sim
~/venvs/rl_sim/bin/pip install "mujoco==3.16.0" "stable-baselines3==2.9.2"
```
理由：`--system-site-packages` 继承 torch/lerobot/gymnasium（不重装 2.5 G CUDA torch），
新增包（mujoco/SB3 及其依赖）落在 venv 内，**conda env 的 site-packages 不被写入**。
回滚：`rm -rf ~/venvs/rl_sim`。

### 6.2 关键决策（默认值，spec 确认时可改）
- **D2 残差只用低维状态，不喂图像**：图像只走 ACT 基座。RL obs 31 维：
  `q(6,deg) + dq(6) + act_target(6,deg) + box_rel_tcp_pos(3,m) + box_linvel(3) + last_residual(6) + t/T(1)`。
  理由：ACT 已在 3.33 s 内开环，残差要做的是"物理纠偏"而非视觉决策；喂图会让 SAC 的样本效率崩掉，2 天跑不出东西。
- **D3 残差幅度**：`action_space = Box(-1,1,(6,))`，实际 `residual = a * 0.05 rad`（≈2.86°/控制周期）。
  **一次缩放，不叠加**（原计划骨架的 `Box(±0.1)` 又乘 `0.1` 是双重缩放，实际 ±0.01 rad，等于没开）。
- **D4 合成方式**：`ctrl_target = clip(ACT_target_rad + residual, ctrlrange)`。控制频率 30 Hz（与 `replay_policy` 同），
  每周期 `mj_step` 到跨过 1/30 s。
- **D5 奖励**（单一公式，写进代码注释，禁止后期调参刷分）：
  ```
  r = 20.0 * (d_prev - d_now)                 # TCP→药盒 距离进展（米→每 mm 0.02）
      + 2.0  if 首次建立握持接触（grip_contacts>0）
      + 100.0 if max_lift >= 50mm → terminated
      - 20.0  if 药盒掉落/翻倒（obj_z < 0.005 或 tilt > 45°）→ terminated
      - 0.02 * ||a||                            # 抑制无谓大幅残差（保平滑度）
  truncated: T_max = 10 s（300 控制周期）
  ```
- **D6 复位与域随机化两套**：
  - **ID 集**：出生位姿取自已验证 spawn 表（`scripts/ep_*.json` 的 spawn），±`gz 1.5 mm`（同 collect 口径）。
  - **OOD 集**：在上述基础上加 mass ×{0.5, 2.0}、friction ×{0.6, 1.6}、相机 `env_pos_jitter 0.01~0.03 m` + `color_jitter 0.02~0.05`（完全复用 collect 口径，不新发明分布）。
  - 训练用 **ID+轻 OOD 混合**；评估分开报 ID / OOD 两栏（否则"提升"无法归因）。
- **D7 探针口径必须先被证明等价于旧口径**：RL rollout 没有 `phase` 标签，`contact_audit_from_arrays` 用不上。
  新写 `rl/probe_rollout.py` 用**相位无关**定义算同样 6 个量，**前置闸门**：在脚本参考轨迹 `scripts/ep_000000.npz` 上
  必须复现旧口径 6 个数字（容差：G1 接触步 0±0、G2 ≤0.5 mm、G3 ≤5°、G4 ≥122 mm、G5 ±1.2 mm、G6 ≤0.1 mm）。
  复现不了 = 探针无效，不许用它评估 RL。

### 6.3 训练规模
- SAC，`MlpPolicy`，`net_arch=[128,128]`，`buffer 200k`，`learning_starts 2000`，3 seeds（0/1/2）。
- 每 seed **30k env step 起，看曲线再决定是否加到 50k**（不承诺 1 万步出趋势——6 维残差 + 成型的基座，
  1 万步只能看出"残差幅度在动"）。
- 预估：用户终端 ≈20~35 min/seed（渲染 1.7 ms/帧、ACT 前向 100 步才一次）。

### 6.4 交付物
```
rl/residual_env.py        ResidualACTEnv（gymnasium.Env） + ACT 基座调用
rl/probe_rollout.py       相位无关 G1-G6 + 平滑度(jerk) + 残差幅度
rl/check_env.py           环境冒烟（shape/NaN/百步）
rl/train_residual.py      SAC 训练入口（--seed --steps --dr --out）
rl/eval_ab.py             A/B 评估（--models base,rl0,rl1,rl2 --suites id,ood --n 20）
rl/make_figs.py           成功率柱状图 + 奖励曲线（matplotlib）
rl/runs/                  ckpt / tensorboard / eval csv
out/ab_id_…mp4            并排视频
RL_README.md              做法、数字、局限（诚实版）
```

## 7. 接口 / 数据 / 文件变化

- **新增目录 `rl/`**（全新，不触碰现有模块）；**新增** `RL_README.md`、`out/`、`rl/runs/`。
- **复用不改**：`render/test_render.Arm / CameraRig / SCENE_XML / FPS / CAM_ENV / CAM_WRIST`；
  `tools/replay_policy._ctrl_from_deg / _deg_from_ctrl / _deg_from_qpos`（**import 复用，不复制**，避免单位口径分叉）。
- **env 契约**：`reset() -> (obs, info)`；`info` 含 `box_pos`、`t`、`act_target_deg`；
  `step(a) -> (obs, r, terminated, truncated, info)`，`info` 含 `lift_mm/max_lift_mm/tcp_box_dist_m/grip_contact_n/success`。
- 数据产物：`rl/runs/<run_id>/{model.zip, eval.json, metrics.csv}`；不写 `scripts/`、不写 `dataset/`。

## 8. 失败模式与回滚

| 风险 | 概率 | 检测 | 处置 |
|---|---|---|---|
| ACT 闭环成功率 = 0（最大风险） | 中高 | Stage 0 pilot，3 条 ep | 切兜底臂（脚本专家），ACT 只作失败对照，诚实写进 README |
| venv 装不上 / mujoco 与 numpy 2 冲突 | 中 | Stage 1 验证命令 | 退回 system python3.10 + pip 装 torch-cpu + SB3（磁盘 +2.5 G） |
| `mj_step` SIGFPE（7%） | 高 | 进程退出码 | 训练循环内 try/重试+断点续训（照 `run_pipeline.py` 做法） |
| ACT chunk 队列跨 episode 残留 | 高 | 第 2 条 ep 起行为突变 | 每 episode 必调 `policy.reset()`（`LerobotPolicy.prepare` 已做，需在 env.reset 里再次确认） |
| 奖励被 hack（推盒下桌/"蹭到就算抬升"） | 中 | G1-G6 + 掉落惩罚 + 看视频 | 惩罚项写死在 D5；成功率判据用 `max_lift≥50mm` 且需 `grip_contacts>0` 同时成立 |
| 训练慢到跑不完 3 seeds | 中 | 首 seed 计时 | 降为 2 seeds ×30k，或降 control_hz 到 20（需重测基座） |
| 磁盘满 | 低 | `df` 门 | 视频只留 4 条 |

回滚：删除 `rl/`、`~/venvs/rl_sim`、`out/`、`RL_README.md` 即完全复原；**本 spec 不修改任何既有文件**。

## 9. 验收标准（可测量）

| # | 命令 | 通过判据 | 失败 |
|---|---|---|---|
| V0 | `env -u PYTHONHOME -u PYTHONPATH ~/venvs/rl_sim/bin/python -c "import mujoco,stable_baselines3,torch,lerobot;print('ok')"` | 打印 `ok` | Traceback |
| V1 | `~/venvs/rl_sim/bin/python tools/replay_policy.py --mode closed-loop --policy /home/yyl/lerobot/outputs/train/act_sim156_pre/checkpoints/last/pretrained_model --episodes 0,1,2 --json` | 打印 `n_success/n`；**n>0 即可继续 ACT 主臂** | n=0 → 走兜底臂分支 |
| V2 | `~/venvs/rl_sim/bin/python rl/check_env.py --steps 100` | `ok`，obs 31 维/action 6 维无 NaN，100 步无异常 | Traceback / NaN |
| V3 | `~/venvs/rl_sim/bin/python rl/probe_rollout.py --episode scripts/ep_000000.npz` | 6 个量与 §3 实测值在 §6.2-D7 容差内 | 超容差 = 探针无效，停止 |
| V4 | `~/venvs/rl_sim/bin/python rl/train_residual.py --seed 0 --steps 30000` | 退出 0，产出 `rl/runs/*/model.zip`，`eval.json` 有 `success_rate` | 无 ckpt / NaN loss |
| V5 | `~/venvs/rl_sim/bin/python rl/eval_ab.py --models base,rl0,rl1,rl2 --suites id,ood --n 20` | 产出 CSV + JSON，每格 n=20，且**报 Wilson 区间** | 缺格 / n≠20 |
| V6 | `~/venvs/rl_sim/bin/python rl/make_figs.py --run-id <id>` | 产出 `rl/runs/<id>/success_ab.png`、`reward_curves.png` | 图缺 |
| V7 | 人工看 1 条 ACT vs 1 条 ACT+RL 的 mp4 | 两者都真实抓取（不是滑过去），G1=0 接触 | 出现"假抓取"即红线违规 |

三态回执：`V0-V2` 全绿 = **ok**；`V1` 成功率 0 但兜底臂成立 = **warn**（写清降级理由）；
`V3` 超容差 / `V7` 假抓取 = **fail**（停止评估）。

## 10. 开放问题（确认 spec 时一并回答）

1. OOD 档位是否就按 §6.2-D6 的 mass/friction/相机三档？要不要加"药盒位置大幅偏移"（可能让 ACT 直接够不到，主要考残差的价值）。
2. 训练是否允许在你终端后台跑（我沙箱无 GPU，只能做 V0/V2/V3 的机械验证与小规模 pilot）？
3. 交付要不要"并排视频 + 简历口述稿"这两样（会额外占 ~1 h，但对面试价值最高）？