# HANDOFF · 残差 RL（ACT 冻结基座 + SAC）· 2026-11-20 12:55

上位文档：`docs/specs/2026-11-20-residual-rl-act-posttrain.md`（已确认）、
`docs/plans/2026-11-20-residual-rl-act-posttrain.md`（S0→S4 五个阶段）。
本文只记**现在的状态、数字、下一刀切哪**，不重复 spec/plan。

---

## 0. 一句话状态（2026-11-20 晚更新：S2/S3 已跑完，见 `docs/REPORT_RESIDUAL_RL.md`）

- **S0-S4 全部有结果了三态**：S0 ok（基座锁 `sim156`）、S1 ok、S2 warn（相对改善 0.130 < 0.20）、
  **S3 warn**、S4 ok（报告 + 3 图 + 36 s 对比视频）。
- **最重要的发现**：残差 RL 在训练时给的动作空间（±5.73° ≈ 5 倍基座步长）下是**负收益**
  （26.4% → 13.2%，seed1/seed2 直接 0/30）；把幅度缩到 ×0.25（±1.43°）后 **3/3 seed 转正**
  （26.7% → 35.6%，Δ +8.9pp，但 G2c 24.82→27.14 mm 超出容差 → warn 不是 ok）。
- 诊断：策略残差均值 3.1–3.7°、p95 达上限 99%、`shoulder_lift` 饱和 32–77% → 残差不是纠偏而是改写
  （`rl/probe_residual.py`）。
- 下一步唯一值得做的：**把动作空间改成 ±1.4° 重训 3 seed**，把"评估期缩放"变成训练内生。

---

## 1. 环境事实（已实测，别再猜）

| 项 | 值 | 证据 |
|---|---|---|
| 执行解释器 | `/home/yyl/deepseek-harness/.venv-rl/bin/python`（py3.12.15） | `cat .venv-rl/pyvenv.cfg` → `include-system-site-packages = true`，基座是 `lerobot312` |
| 前缀 | `env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl` | 缺 `MUJOCO_GL=egl` 渲染起不来 |
| venv 内新增 | `mujoco 3.16.0`、`stable-baselines3 2.9.2` | `.venv-rl` 共 136 MB |
| conda 环境 | **未被污染**：`lerobot312` 里 mujoco / SB3 仍 absent，`site-packages` mtime 停在 10-26 | 符合 spec 约束 C3 / P2 |
| torch | `2.7.1+cu126`（沿用 conda，venv 未重装） | `.venv-rl/bin/pip list` |
| 磁盘 | `/home` 148 G，剩 20 G（87%） | `df -h /home` |

**偏差（有意为之，写清理由）**：spec 写的是 `pip install stable-baselines3==2.9.2`。
照做会让 pip 解析出 `torch>=2.8`，在 venv 里重下 `torch 2.15.0` + `nvidia-cudnn 519 MB`。
实测改用：

```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/pip install --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple stable-baselines3==2.9.2
```

SB3 2.9.2 + torch 2.7.1 + gymnasium 1.4.0 组合已冒烟：`SAC('MlpPolicy', Pendulum-v1).learn(200)` 通过。

---

## 2. S0 关键数字（`rl/eval/pilot.json`）

判据：`success = 全程最大抬升 ≥ 50 mm`，1 episode / 15 s / 30 Hz（450 控制步）。

| 配置 | 抬升 mm | 接触峰 | 盒倾角 | 近盒最近距离 | 墙上时间 | 结果 |
|---|---|---|---|---|---|---|
| lance ep0 raw | 0.0 | 0 | 0.0 | 115.8 mm | 92.6 s | ❌ |
| lance ep0 swap | 0.0 | 0 | 0.0 | 115.9 mm | 92.0 s | ❌ |
| lance ep0 sensor | 0.0 | 0 | 0.0 | 104.1 mm | 104.8 s | ❌ |
| lance ep0 swap+sensor | 0.0 | 0 | 0.0 | 106.0 mm | 105.7 s | ❌ |
| armA ep0 raw（state=ctrl） | 0.0 | 0 | 0.0 | 76.6 mm | 92.1 s | ❌ |
| armA ep0 raw（state=qpos） | 0.0 | 0 | 0.0 | — | 100.4 s | ❌ |
| armA ep0 swap+sensor（state=qpos） | 0.0 | 0 | 0.0 | — | 128.5 s | ❌ |
| **sim156 ep0 raw（state=ctrl）** | **+137.7** | 11 | **0.0** | 24.2 mm | 88.6 s | ✅ |
| **sim156 ep0 sensor（state=ctrl）** | **+143.1** | 12 | **33.9** | 23.5 mm | 105.3 s | ✅（倾角 34° 见下） |

**读法（重要，别只抄"1/1 成功"）**

1. 真机 ckpt 的失败形态是**够不到**：近盒最近距离 76–116 mm，全程 0 接触、抬升 0.0。
   不是"抓了但掉"，是**轨迹走向和仿真里的盒子位置根本不是一回事**。换相机、过 SensorModel、
   换 state 口径（ctrl↔qpos）都救不回来 → 归因方向是"真机数据与仿真场景/取景不同源"，不是观测口径微调。
2. `sim156` 成功靠的是 `observation.state` 取 **ctrl（下发目标）**，不是 qpos；这与
   `rl/env_residual.py::_obs` 的取法一致（`_deg_from_ctrl`），所以 env 侧不用改。
3. `sim156 + sensor` 虽然成功，但**盒倾角 33.9°**，`raw` 是 0.0°。按 G3 口径（≤5°）这已经是脏的，
   选主臂口径时倾向 `raw`（`use_sensor=False`）。**待 3 episode 的复测确认后定稿**。
4. 归因辅助证据：
   - `logs/cmp_sim_real.log`：去掉首帧 state 后 action 的 W1 仍差 12–33 mm（wrist_flex p5/p95
     sim −40…70 vs real +30…78；wrist_roll sim −155…−88 vs real −97…−15）→ 腕部/滚转口径不同源。
   - `logs/cmp_sim_real_lance.log`：`no parquet under /home/yyl/lerobot_datasets/yyl100_lance_fixed`
     → lance 那个 ckpt 对应的数据集路径当前读不到 parquet。

---

## 3. 阶段矩阵

| 阶段 | 判据 | 实测 | 三态 | 证据 |
|---|---|---|---|---|
| S0 pilot | 至少一组 ≥1/3 成功 | sim156 2/2（不同口径各 1 ep），真机 ckpt 0/7 | **ok（带 warn：基座降为仿真数据）** | `rl/eval/pilot.json`、`logs/pilot_s0b.log` |
| S1 env 自检 | 200 步不崩；obs 14 / act 6；terminated、truncated 各 ≥1 | `ok steps=200 obs_dim=14 act_dim=6 reward_sum=-358.22 n_terminated=1 n_truncated=1`，exit 0 | **ok** | `logs/s1_selftest.log` |
| S2 训练 | 3 seed × 30k，末 5k reward ≥ 首 5k × 1.2 | 只做了 256 步入口冒烟（`ratio 1.0`，步数太少不算指数） | **fail（未开始长跑）** | `logs/s2_smoke.log`、`.dsh-tmp/s2_curves/s0.jsonl` |
| S3 A/B | Δ成功率 ≥ +10pp；G2≤0.5mm、G6≤0.5mm、G3≤5° | 无 `rl/eval_ab.py` | **fail（未开始）** | — |
| S4 交付 | REPORT + 2 图 + ≥30 s 视频 | 全缺 | **fail（未开始）** | — |

`rl/env_residual.py --selftest` 的两处修正（已落地）：
1. spec §9 写的是 `--selftest`，原文件不认这个旗标 → 加了一个不改变行为的别名。
2. 200 步 @30 Hz 只有 6.7 s，永远够不到 15 s 超时截断 → selftest 末尾把 `max_seconds` 缩到 2 s
   单独走一遍 `truncated` 分支（复用同一 env 实例，不重载 ACT），现在两项终止都能各触发一次。

---

## 4. 下一刀（按顺序，别跳）

1. **等 3-episode 复测**（另一窗口已在跑）：
   `rl/pilot.py --ckpts sim156 --episodes 0,1,2 --combos raw,sensor --out rl/eval/pilot_sim156.json`
   → 若 `raw` ≥2/3 且倾角干净，**把 `use_sensor` 默认值定为 False** 并写进 `rl/base_policy.py` 注释。
2. **S2 长跑（必须在用户终端，沙箱无 GPU）**：
   ```bash
   env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
     .venv-rl/bin/python rl/train.py --base sim156 --seeds 0,1,2 --steps 30000 --resume
   ```
   崩了（`mj_step` SIGFPE）就原样重跑同一条命令，`--resume` 会接着走。
3. **S3**：写 `rl/eval_ab.py`（`base` vs `base+R`，30 ep × 3 seed，G1-G6 用相位无关口径）→ 出 `rl/eval/ab.json`。
4. **S4**：`docs/REPORT_RESIDUAL_RL.md` + `rl/curves/*.png` + 1 条 ≥30 s 并排视频。

### 还没解决的坑（先说好，到时候别算意外）

- **G1-G6 现在还不能用在 RL rollout 上**：`render/test_render.contact_audit_from_arrays` 依赖
  `phase` 标签，`tools/probe_contact_audit.py` 是脚本轨迹专用。S3 需要先写一个相位无关探针，
  并在 `scripts/ep_000000.npz` 上复现旧口径 6 个数字（容差 G1 0、G2 ≤0.5mm、G3 ≤5°、G4 ≥122mm、
  G5 ±1.2mm、G6 ≤0.1mm），复现不了就不许用它评估 RL。
- 沙箱渲染 llvmpipe 174–197 ms/帧，单条 episode ≈90–105 s；用户终端 1.7 ms/帧，量级差百倍 →
  **长跑与 A/B 只在终端做**。
- 红线段不变：不写 `qpos` 伪造抓取、残差只经 `ctrl` + `mj_step`、`render/test_render.py` 的
  render 阶段不 `mj_step`、G1-G6 不进 reward。

---

## 5. 本次新增/改动的文件

| 文件 | 动作 |
|---|---|
| `.venv-rl/` | 新建（py3.12 + system-site-packages），装 mujoco 3.16.0、SB3 2.9.2 |
| `rl/env_residual.py` | 默认基座 `lance`→`sim156`；`--selftest` 别名；selftest 增加 truncated 通路验证 |
| `rl/train.py` | 默认基座 `lance`→`sim156` |
| `logs/pip_install_sb3_nodeps.log`、`logs/s1_selftest.log`、`logs/s2_smoke.log` | 新增证据 |