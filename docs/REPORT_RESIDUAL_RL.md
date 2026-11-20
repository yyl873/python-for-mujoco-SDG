# REPORT · ACT 冻结基座 + 残差 SAC（2026-11-20）

- 上位文档：`docs/specs/2026-11-20-residual-rl-act-posttrain.md`、`docs/plans/2026-11-20-residual-rl-act-posttrain.md`
- 执行环境：`.venv-rl`（py3.12, mujoco 3.16.0 + torch 2.7.1 + stable-baselines3 2.9.2），**全程 CPU**（沙箱无 GPU）
- 统一前缀：`env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl`，下称 `$PY` = `.venv-rl/bin/python`
- 所有命令都在仓库根 `/home/yyl/deepseek-harness` 执行

---

## 0. 一句话结论 + 三态回执

**残差 RL 在"纠偏量级"的动作空间下有效，在"改写量级"下有害——而第一版 spec 给的动作空间恰恰是后者。**

| 阶段 | 判据 | 实测 | 三态 |
|---|---|---|---|
| S0 基座 pilot | 至少一组 ≥1/3 成功 | 真机 ckpt `lance` 0/4、`armA` 0/3；`sim156` raw 1/1、sensor 2/3 | **ok**（基座降为仿真数据 ckpt） |
| S1 Gym 环境 | 200 步不崩；obs 14 / act 6；terminated、truncated 各 ≥1 | `ok steps=200 obs_dim=14 act_dim=6 n_terminated=1 n_truncated=1` | **ok** |
| S2 训练 | 3 seed × 30k，相对改善均值 ≥0.20 | 0.178 / 0.085 / 0.126 → **均值 0.130** | **warn** |
| S3 A/B（训练幅度 ±5.73°） | Δ成功率 ≥+10pp 且闸门不退化 | 26.4% → 13.2%，**Δ = −13.2pp** | **fail** |
| S3 A/B（幅度 ×0.25 = ±1.43°） | 同上 | **26.7% → 35.6%，Δ = +8.9pp**（3/3 seed 为正）；但 G2c 24.82→27.14 mm 超出 +0.5 mm 容差 | **warn** |
| S4 交付物 | REPORT + 2 图 + ≥30 s 视频 | 本文件 + `rl/curves/success_ab_scale*.png` + `reward_curves.png` + `rl/eval/ab_demo_v2.mp4` | **ok** |

---

## 1. 复现入口（每个数字挂一条命令）

```bash
# S0：基座 pilot（9 组）
$PY rl/pilot.py --ckpts lance,armA --episodes 0 --duration 15 --out rl/eval/pilot.json
$PY rl/pilot.py --ckpts sim156 --episodes 0,1,2 --combos raw,sensor --out rl/eval/pilot_sim156.json

# S1：环境自检
$PY rl/env_residual.py --selftest --base sim156 --steps 200

# S2：训练 3 seed × 30k + 用存量曲线重算判据
$PY rl/train.py --base sim156 --seeds 0,1,2 --steps 30000 --resume
$PY rl/s2_verdict.py

# S3：A/B（训练幅度）
$PY rl/eval_ab.py --base sim156 --policy rl/ckpt/s0/model.zip --episodes 0-29 --resume --out rl/eval/ab_s0_30.json
$PY rl/eval_ab.py --base sim156 --policy rl/ckpt/s1/model.zip --episodes 0-29 --resume --out rl/eval/ab_s1_30.json
$PY rl/eval_ab.py --base sim156 --policy rl/ckpt/s2/model.zip --episodes 0-29 --resume --out rl/eval/ab_s2_30.json

# S3 消融：残差幅度 ×0.25（不改权重，只缩幅度）
for s in 0 1 2; do $PY rl/eval_ab.py --base sim156 --policy rl/ckpt/s$s/model.zip \
    --episodes 0-29 --res-scale 0.25 --resume --out rl/eval/ab_s${s}_scale0.25.json; done

# 诊断 + 图 + 视频
$PY rl/probe_residual.py --policy rl/ckpt/s1/model.zip --steps 150
MPLCONFIGDIR=$PWD/.mplcache $PY rl/make_figs.py --ab 'rl/eval/ab_s*_scale0.25.json' --curves 'rl/curves/s*.jsonl'
# 视频：分别录基座与残差（--video-tag 选臂），再左右拼接
$PY rl/eval_ab.py --base sim156 --episodes 0 --video rl/eval/demo_base.mp4 --video-tag base
$PY rl/eval_ab.py --base sim156 --policy rl/ckpt/s0/model.zip --episodes 0 --res-scale 0.25 \
    --video rl/eval/demo_r.mp4
$PY rl/stack_video.py --left rl/eval/demo_base.mp4 --right rl/eval/demo_r.mp4 \
    --left-label "pure ACT (base)" --right-label "ACT + residual RL" --out rl/eval/ab_demo_v2.mp4
```

---

## 2. S0：基座选谁（pilot，`rl/eval/pilot.json`）

判据 `success = 全程最大抬升 ≥ 50 mm`，1–3 episode / 15 s / 30 Hz。

| 配置 | 抬升 | 接触峰 | 近盒最近距离 | 结果 |
|---|---|---|---|---|
| `lance`（真机数据）raw / swap / sensor / swap+sensor | 0.0 | 0 | 104–116 mm | ❌ 0/4 |
| `armA`（真机数据）raw（state=ctrl） | 0.0 | 0 | 76.6 mm | ❌ |
| `armA` raw / swap+sensor（state=qpos） | 0.0 | 0 | — | ❌ 0/2 |
| **`sim156`（仿真数据）raw** | **+137.7 mm** | 11 | 24.2 mm | ✅ |
| **`sim156` sensor** | **+143.1 mm** | 12 | 23.5 mm | ✅ |
| `sim156` 3 episode 复测 | raw **1/3**、sensor **2/3** | — | — | ✅ 过闸门 |

**读数**：真机 ckpt 的失败形态是**够不到**（0 接触、最近 76–116 mm），换相机通道、过 SensorModel、
换 state 口径（ctrl↔qpos）都救不回来 → 真机数据与仿真场景不同源，不是观测口径微调能解决的。
**基座锁定 `sim156`**，代价是叙事从"真机数据后训练"降级为"仿真闭环上验证残差 RL"。

---

## 3. S1：Gymnasium 环境（`$PY rl/env_residual.py --selftest`）

`ok steps=200 obs_dim=14 act_dim=6 reward_sum=-358.22 n_terminated=1 n_truncated=1`，exit 0。
`rl/env_residual.py` 修了两处：① spec §9 写的 `--selftest` 旗标原来不被 argparse 接受；
② 200 步 @30 Hz = 6.7 s 永远够不到 15 s 超时，selftest 末尾把 `max_seconds` 缩到 2 s 单独验证 `truncated` 通路。

---

## 4. S2：训练（3 seed × 30k 步，`rl/curves/`）

| seed | episode 数 | 首四分位奖励 | 末四分位奖励 | Δ | 相对改善 | 墙钟 |
|---|---|---|---|---|---|---|
| 0 | 171 | −299.47 | −246.10 | **+53.37** | 0.178 | 566 s |
| 1 | 172 | −289.27 | −264.61 | **+24.66** | 0.085 | 579 s |
| 2 | 171 | −285.47 | −249.36 | **+36.10** | 0.126 | 565 s |

- 三个 seed 的奖励**都在变好**（负值向 0 靠），但相对改善均值 **0.130 < 0.20** → S2 判 **warn**：30k 步不够收敛。
- ⚠️ **旧 `rl/curves/summary.json` 里的 `ratio` 字段是错的**：它算 `tail/head`，在负奖励上
  `−246 / −299 = 0.82 < 1.2` 会被读成"训练失败"。正确判据见 `rl/s2_verdict.py` + `rl/curves/summary_v2.json`。
- 图：`rl/curves/reward_curves.png`（3 条曲线 + 首/末四分位均值）。

---

## 5. S3：A/B（每格 30 episode，ep0–29，`--seeds 1`；base 均为同一个冻结 `sim156` 基座）

### 5.1 用训练时的动作空间（±5.73°，≈ 5× 基座步长）

| seed | base | base+R | Δ |
|---|---|---|---|
| 0 | 8/31 = 25.8% | 12/31 = 38.7% | **+12.9pp** |
| 1 | 8/30 = 26.7% | 0/30 = 0.0% | **−26.7pp** |
| 2 | 8/30 = 26.7% | 0/30 = 0.0% | **−26.7pp** |
| **合计** | **24/91 = 26.4%** [18.4, 36.3] | **12/91 = 13.2%** [7.7, 21.6] | **−13.2pp** |

闸门（闭环近似口径）：base G2c 24.82 / G6c 24.82 / G3c 100.7° → base+R 21.09 / 21.09 / 95.3°（形式上"不退化"，但成功率崩了）。

### 5.2 残差幅度 ×0.25（±1.43°，≈ 1.2× 基座步长，权重不变、只缩幅度）

| seed | base | base+R | Δ |
|---|---|---|---|
| 0 | 8/30 = 26.7% | 12/30 = 40.0% | **+13.3pp** |
| 1 | 8/30 = 26.7% | 11/30 = 36.7% | **+10.0pp** |
| 2 | 8/30 = 26.7% | 9/30 = 30.0% | **+3.3pp** |
| **合计** | **24/90 = 26.7%** [18.6, 36.6] | **32/90 = 35.6%** [26.4, 45.9] | **+8.9pp** |

补充：seed1 在 ×0.10 也是 8/30 → 11/30（+10.0pp），说明**幅度只要收到纠偏量级，结果就稳定为正**。

闸门：base+R 的 G2c/G6c 最大到 **27.14 mm**（base 24.82）→ 超出 `+0.5 mm` 的"不退化"容差，
按文件头判据记 **warn**，不是 ok。绝对数两边都很脏（脚本专家口径是 0.39 mm），所以 **G2c/G6c/G3c 不可冒充 spec 的 G2/G6/G3**。

**三态**：成功率达标（+8.9pp）+ 闸门略退化 → **warn**；不是 fail（fail 指跑不通），也不是 ok（ok 要求两者都达成）。

---

## 6. 为什么会这样：残差幅度诊断（`rl/probe_residual.py`）

| seed | \|残差\| 均值 | p95 | 饱和比例（\|a\|>0.9×上限） | shoulder_lift 饱和 |
|---|---|---|---|---|
| 0 | 3.65° | 5.52° | 24.4% | **77%** |
| 1 | 3.13° | 5.49° | 16.3% | 57% |
| 2 | 3.54° | 5.54° | 19.7% | 32% |
| 对照：基座自身逐步动作幅度 | **1.17°** | — | — | — |

**结论**：策略输出的残差是基座自身步长的 **2.7–3.1 倍**，p95 已达上限的 99%，`shoulder_lift` 长期贴在上限。
此时"残差"已经不是纠偏，而是**改写基座动作**——所以同一套奖励训出来的三个策略表现剧烈分化
（+12.9 / −26.7 / −26.7pp）。把幅度缩到 25% 后三个 seed 全部转正，印证了这个解释。

**这是一条比"+10pp"更有用的工程结论**：残差 RL 的动作空间必须按"基座步长"定标（约 1×），
而不是按"关节全幅"定标（±0.1 rad 看着小，实际是 5×）。

---

## 7. 诚实局限（别在简历/面试里越界）

1. **样本量不足**：每格 n=30（合计 90），Wilson 区间 26.4–45.9 vs 18.6–36.6 **大幅重叠**，**不能声称显著**。
   只能说"方向为正、3/3 seed 为正、幅度缩小后稳定为正"。
2. **×0.25 是评估期缩放，不是重训**：它证明的是"幅度约束很重要"，不是"已经有一个好策略"。
   真要把这个结果变成策略性质，必须把动作空间改成 ±1.4° 重训。
3. **S2 没收敛**：相对改善 0.130 < 0.20（spec 判据），30k 步对 14 维 obs + 6 维动作偏少。
4. **崩溃过 5 次**：`mj_step` 已知的进程级 SIGFPE（浮点异常）。`eval_ab.py` 已加 `--resume`；
   seed0 的 A/B 因此是**分块跑**的，其中 ep13 一条在两次尝试里都崩，最终以 31 条（多跑一条 ep0）补齐。
5. **旧文件不可引用**：`rl/eval/ab_s0_20_LEGACY.json`（原名 `ab_s0.json`，20 条，7/20 → 9/20，"+10pp"）是**旧版 eval 代码**产出，
   在当前代码 + 当前样本下不复现。本 REPORT 只认 `ab_s*_30.json` / `ab_s*_scale0.25.json`。
6. **闸门口径**：闭环没有相位标签，只能给 G2c/G3c/G6c 近似；脚本轨迹的 G1–G6（`tools/probe_contact_audit.py`）
   不能直接套到学习型策略上（连纯 ACT 自己都是 G2c 21.7 mm / G3c 105°）。
7. **无 GPU**：全部 CPU，1 seed 30k 步 ≈ 9.5 min，90 条 A/B ≈ 13 min/格。

---

## 8. 下一步（按性价比排序）

1. **把动作空间改成 ±1.43° 重训 3 seed**（≈30 min/seed）：把"评估期缩放"变成"训练内生"，这是唯一能写进简历的下一步。
2. **奖励对齐成功率**：现在 `−10·‖tcp−box‖` 主导，抬升项只有 `2·tanh(Δ/0.05)`；把成功项提前/加大，或加"接触后奖励"。
3. **训练 30k → 60k**，看 `improve` 是否过 0.20；同时把 A/B 每格从 30 条提到 100 条，才有资格谈显著。
4. **残差门控**：只在 `max_lift < 5 mm`（接近/抓取阶段）放开残差，抬升段锁死——物理上更合理。

## 9. 交付物地图

| 产物 | 路径 |
|---|---|
| 本文档 | `docs/REPORT_RESIDUAL_RL.md` |
| 成功率图（训练幅度） | `rl/curves/success_ab_scale1.png` |
| 成功率图（×0.25） | `rl/curves/success_ab_scale025.png` |
| 奖励曲线 | `rl/curves/reward_curves.png` |
| 对比视频（36 s，4 段：ep0/ep9 基座失败残差成功、ep1/ep6 两边都成功） | `rl/eval/ab_demo_v2.mp4`（左=纯 ACT，右=ACT+残差） |
| A/B 原始结果 | `rl/eval/ab_s{0,1,2}_30.json`、`rl/eval/ab_s{0,1,2}_scale0.25.json` |
| 残差诊断 | `rl/eval/residual_probe*.json` |
| 代码 | `rl/{base_policy,env_residual,pilot,train,eval_ab,s2_verdict,probe_residual,make_figs,stack_video}.py` |