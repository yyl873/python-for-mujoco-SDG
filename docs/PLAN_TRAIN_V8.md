# 训练清单 · sdg-v8（A / B / C / D / E）

> 本文件只讲"拿哪批数据、跑多少步、怎么判"。**物理与采样的事实**见 `docs/PHYSICS_FACTS.md`，
> 混训比例的论证见 `docs/PLAN_MIX_SIM_REAL.md`，本文件是它的**执行版 + 两处口径修正**。
> 生成于 sdg-v8（反向生成）落地之后，所有数字均为本机实测或由实测帧数直接算出。

---

## §0 三道闸门（不过就别花 GPU）

| # | 闸门 | 当前状态 | 通过标准 |
|---|---|---|---|
| **G0** | 仿真母本齐全 | ⚠️ 目前只有 6 条 smoke | `scripts/index.json` 的 `n_episodes=1000`、`n_accepted=1000`、版本全 `sdg-v8` |
| **G1** | **夹爪量纲对齐** | ❌ **未定案（唯一硬阻塞）** | 见 §1 |
| **G2** | 数据口径（防重复 / 防泄漏） | ✅ 已查清 | 见 §2 |

跑 G0：`bash run_1000.sh`（已补退出码检查 / `--jobs 8` / 每批核对条数）。

---

## §1 夹爪量纲：为什么它是硬阻塞

实测范围（`meta/stats.json` + 母本 npz）：

| 来源 | gripper 取值 | 0 点含义 |
|---|---|---|
| 真机 `real_train_fixed` | 0.62 ~ **49.08** | 0 = 闭合 |
| 真机 `real_train_corners` | 0.62 ~ **42.27** | 0 = 闭合 |
| 真机 `holdout_*` | 0.62 ~ 40.11 / 34.02 | 0 = 闭合 |
| 仿真 jaw（sdg-v8） | 0 ~ 127.88°（实际只用到 0 ~ **85.94**） | 0 = 闭合 |

`collect/export_lerobot.py` 的 `--gripper-space` 已有三档，但
`episode_arrays(..., real_full: float = 100.0)` **把真机满量程写死成 100**，与实测的 42~49 不符。
直接用它导出，夹爪列会落在 100 尺上，与真机的 49 尺**差约 2 倍** → 混训后策略会把"开合指令"
学到错误尺度，**B / C / E 的结果全部作废**。

**二选一（推荐 a，已实现）**

- **a** 已给 `export_lerobot.py` 加 `--gripper-full`（默认 100 保持兼容），导出仿真时传实测值：
  `fixed` 族用 **49.08**、`corners` 族用 **42.27**，映射 `sim 0（闭） → 0`、`sim 85.94（开） → gripper_full`。
  **方向用 `--gripper-space real-closed`**（真机 0 = 闭合），不是 `real-open` —— 实测证据：
  真机在"搬运/抬起"段 gripper ≈ 7~10（小端），在"接近/张开"段才到 28~49（大端），
  所以小端 = 闭合。旧 plan 里写的 `real-open` 会**把夹爪列反向**。
- **b** 在真机侧把夹爪列乘 `85.94 / gripper_full`（≈1.751 / 2.034）映射到仿真角度尺。

判定方法（两端都要量，别只看最大值）：
1. 闭合端：取抬升段（`lift` 阶段、药盒已被夹住）gripper 值的**众数** → 应 ≈0；
2. 张开端：取 `approach/descend` 段的最大值 → 就是 `gripper_full`；
3. 两端都成立才允许混训，写进 `meta/sim_provenance.json`。

---

## §2 数据口径（已逐项对账）

| 数据集（`~/lerobot_datasets/…`） | 条数 | 帧数 | fps | gripper max | 角色 |
|---|---|---|---|---|---|
| `real_train_fixed` | 139 | 62,273 | 30 | 49.08 | **训练（主）** |
| `real_train_corners` | 22 | 8,588 | 30 | 42.27 | 训练（扩展） |
| `holdout_fixed` | 12 | 5,376 | 30 | 40.11 | **测试（主）** |
| `holdout_corners` | 8 | 3,282 | 30 | 34.02 | 测试（扩展） |
| `yyl100_dataset_fixed` | 151 | 67,649 | 30 | 49.08 | ⚠️ = fixed 训练+测试，**禁止再与上面两个同用** |
| `yyl100_corners` | 30 | 11,870 | 30 | 42.27 | ⚠️ = corners 训练+测试，**禁止再与上面两个同用** |
| `yyl100_lance_*` | 151 / 30 | 同上 | 30 | 同上 | ⚠️ 同上（Lance 格式副本，内容同 `yyl100_*`） |

**关系已证**：`139+12 = 151`、`62,273+5,376 = 67,649`；`22+8 = 30`、`8,588+3,282 = 11,870`
→ `yyl100_*` 就是"训练+测试"打包在一起。两种写法等价（合并后再 `exclude_episodes` 排除测试条），
但**绝不能把 `yyl100_*` 与 `real_train_*` / `holdout_*` 同时喂进去**（重复计数 = 泄漏）。

### 两套口径（选一套，别混）

| 口径 | 训练池 | 测试池 | 说明 |
|---|---|---|---|
| **P1（推荐·主实验）** | `real_train_fixed` **139 条 / 62,273 帧** | `holdout_fixed` 12 / 5,376 | 与 sdg-v8 的取景**严格同分布**（v8 只复刻了 `_fixed` 的"垫子切底"视角） |
| **P2（扩展·对齐既有 plan）** | fixed + corners **161 条 / 70,861 帧** | holdout 固定+corners **20 条 / 8,658 帧** | 样本多 14%，但 corners 的取景不同（垫子完整）、腕角习惯不同（抓取 roll 中位 −87.5° vs −61.4°），仿真臂会天生少匹配 12% |

**为什么推荐 P1**：sdg-v8 的盒朝向窗口是照着 `_fixed` 的视角与腕角分布设计的。
在 P2 下，"仿真 vs 真机"的差异里会混进"fixed vs corners"的批次差异，结论不好归因。
corners 单独跑一组即可（它本身也是真实数据，不该丢）。

---

## §3 五个臂

### 表 A：按你的表（P2 口径，161 条）——数字完全自洽，可直接用

| 臂 | 数据 | 训练帧 | 步数（bs8，等 epoch ≈2.25） | 单臂用时 | 作用 |
|---|---|---|---|---|---|
| **A** | 161 真机 | 70,861 | **20,000** | 1.8 h | 基线（对照现有 `act_yyl151_lance`） |
| **B** | 161 + sim 39（20% 帧） | 88,606 | **25,000** | 2.2 h | 主推 |
| **C** | 161 + sim 156（=1:1） | 141,841 | **40,000** | 3.6 h | 上限探针 |
| **D** | 156 纯仿真 | 70,980 | **20,000** | 1.8 h | 下限参照（预期真机成功率≈0） |
| **E** | sim 预训 → 161 微调 | 455k → 70.9k | ⚠️ 见 §7 | — | 可选 |

### 表 B：按 P1 口径（139 条，与 v8 严格同分布）

| 臂 | 训练帧 | 步数 | 用时 |
|---|---|---|---|
| A′ | 62,273 | **17,500** | 1.6 h |
| B′（sim 20% → 34 条） | 77,841 | **22,000** | 1.9 h |
| C′（sim 1:1 → 139 条） | 124,546 | **35,000** | 3.1 h |
| D | 70,980 | **20,000** | 1.8 h |

> 步数口径：`steps ≈ 0.2817 × 训练帧数`（= bs8 下约 2.25 epoch）。四臂等 epoch，
> 保证"加仿真"不改变每样本训练量。想与旧模型严格对齐也可全部固定 20k 步（旧口径）。

---

## §4 建数据集（官方 merge，本机已跑通 2+2 最小验证）

```bash
cd /home/yyl/lerobot
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
export HF_HUB_OFFLINE=1
DS=/home/yyl/lerobot_datasets
export -n PYTHONHOME PYTHONPATH

# 1) 导出仿真（★方向是 real-closed：真机 0=闭合；满量程按目标真机族传实测值）
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 /home/yyl/deepseek-harness/collect/export_lerobot.py \
  --dst $DS/sim156 --repo-id local/sim156 --episodes "$EP156" \
  --units deg --gripper-space real-closed --gripper-full 49.08 --overwrite

# 2) sim39 = 从 sim156 删掉 39..155（保持 B ⊂ C，便于对照）
$PY -m lerobot.scripts.lerobot_edit_dataset \
  --operation.type=delete_episodes --operation.episode_indices="[$(seq -s, 39 155)]" \
  --repo_id=local/sim156 --root=$DS/sim156 \
  --new_repo_id=local/sim39 --new_root=$DS/sim39

# 3) arm A（P1 口径：只用 fixed）
$PY -m lerobot.scripts.lerobot_edit_dataset --operation.type=merge \
  --new_repo_id=local/armA --new_root=$DS/armA \
  --operation.repo_ids="['local/fixed']" --operation.roots="['$DS/real_train_fixed']"

# 4) arm B / C：在 A 后面接 sim 子集（repo_ids 与 roots 一一对应）
$PY -m lerobot.scripts.lerobot_edit_dataset --operation.type=merge \
  --new_repo_id=local/armB --new_root=$DS/armB \
  --operation.repo_ids="['local/fixed','local/sim39']" \
  --operation.roots="['$DS/real_train_fixed','$DS/sim39']"

# 5) arm D
$PY -m lerobot.scripts.lerobot_edit_dataset --operation.type=merge \
  --new_repo_id=local/armD --new_root=$DS/armD \
  --operation.repo_ids="['local/sim156']" --operation.roots="['$DS/sim156']"
```

`EP156` 的取法沿用 `PLAN_MIX_SIM_REAL.md` §1.3（按 |θ| 等距抽，别取前 156 条）；
**sdg-v8 下请改成按 `posture.box_yaw_deg` 等距抽**，理由同 §1.3：要覆盖最散的盒朝向。

⚠️ merge 会**复制视频**：`sim` 一条约 6 MB，四臂合计约 8~10 GB。训完删数据集、留 ckpt。

---

## §5 训练

```bash
cd /home/yyl/lerobot
export HF_HUB_OFFLINE=1
/home/yyl/miniforge3/envs/lerobot312/bin/python train_mix.py \
  A /home/yyl/lerobot_datasets/armA ""        # 最后一参 = exclude 列表（P2 口径下填 20 条测试 id）
```
（`train_mix.py` 已在本机存在，负责给 `get_safe_version` 打离线补丁。）

参数：`--policy.type=act`、`--batch_size=8`、`--num_workers=6`、`--save_freq=5000`、
`--policy.device=cuda`、`--wandb.enable=false`。bs8 实测 **0.32 s/步**。
数据慢就先 `--operation.type=reencode_videos`（av1 软解是瓶颈，h264 解码快 3~5×）。

---

## §6 评测（离线指标不能当结论）

1. **离线**：只在测试池（P1: 12 条 `holdout_fixed`；P2: 20 条）上算 action L1 / 神谕成功率代理 —— 只用来筛，不写进结论。
2. **真机**：每臂 ≥**20 次**同一套摆放脚本的实抓，记录成功率 / 失败模式。
3. 顺序：先 A vs B（这两个定生死），C/D 只在 B 有信号后才值得跑。

---

## §7 已知风险（按严重度）

| # | 风险 | 数字依据 | 处置 |
|---|---|---|---|
| 1 | **E 的预训练等于没预训** | 你的表 28.4k 步 / 455k 帧 / bs8 = **0.5 epoch** | 改 `sim156 × 5 epoch = 44k 步`（3.9 h）；仍嫌贵就砍掉 E |
| 2 | **B 的剂量小** | 1:4 只在 161 条上 +20% 帧；真机自身方差可能淹没 | 先跑；无信号就把 B 提到 1:2（sim 80 条）再试 |
| 3 | **D 的真实成功率预期≈0** | 纯仿真 → 真机 zero-shot，视觉域差（锐度差 7~24×）、轨迹差 25.8° | 只当下限基线，不要当候选交付 |
| 4 | 仿真与真机不是同一条轨迹分布 | `DATA_HEALTH_181.md`：跨域轨迹差 25.8° | 仿真是**正则项**，比例甜点 1:4~1:3，不要上 50% 以上 |
| 5 | A 可能欠拟合 | 20k 步 ≈ 2.25 epoch，ACT 常规预算更大 | A 看 loss 曲线；没平台就 +50% 重跑 A，再重跑 B |

---

## §8 时间与磁盘预算（单卡、bs8、6 workers）

| 阶段 | 时长 |
|---|---|
| 生成 1000 条母本（有 GPU，`run_1000.sh`，jobs 8） | 1~1.5 h |
| 导出 sim156（16.6 s/条） | ≈0.75 h |
| 建 sim39 + 三个 merge | ≈0.5 h |
| 训练 A/B/C/D（表 A 步数） | ≈9.4 h |
| 训练 A′/B′/C′/D（表 B 步数，P1 口径） | ≈8.4 h |
| E（改 44k 预训后） | +5.5 h |
| **合计（A~D，P1 口径）** | **≈11 h**（可夜里跑） |

磁盘：母本 1.2 GB + 四个合并数据集 8~10 GB + ckpt（每 5k 步 ≈0.4 GB）→ **留 ≥15 GB**。

---

## §9 一句话

**先跑 G0 与 G1（夹爪量纲），再用 P1 口径跑 A′ → B′，C′/D 看情况，E 只有在 C′ 有信号且预训练加到 44k 步时才做。**