# HANDOFF · SO-ARM101 Sim2Real sdg-v8 / A·B 实验现状

> 日期：2026-10-26  
> 用途：换窗口继续。先读这一份，再读 `docs/PLAN_TRAIN_V8.md` 和 `docs/PHYSICS_FACTS.md`。  
> 当前结论：**A/B 已训练完；离线 A/B 没有显著差异；下一优先是真机 A/B 评测，不是直接加量跑 E。**

---

## 1. 一句话结论

- `armA`、`armB` 已训练完，checkpoint 可正常加载。
- 离线 `holdout_fixed` 上，B 只在**短时 L1 和夹爪**上略好；**长动作块和 RMSE 更差**，12 条测试集上差异不显著。
- 仿真数据的主要问题**不是位置数量少**，而是：
  1. 每条 sim 的**起点关节完全固定**；
  2. 起点与真机均值差 7～17°；
  3. 去掉起点差异后，lift / elbow / wrist_flex 的 action 分布仍有 **25～33° 的 W1 距离**。
- 因此：**不要直接加大数据量跑 E。** 先上真机 A/B；若 B 有信号，再做“起点 + 轨迹分布”对齐的 pilot，最后才考虑 E。

---

## 2. 数据与模型现状

### 2.1 数据集

| 数据集 | episodes | frames | 说明 |
|---|---:|---:|---|
| `real_train_fixed` | 139 | 62,273 | P1 口径，A 的训练池 |
| `holdout_fixed` | 12 | 5,376 | 测试池，**禁止训练** |
| `sim156` | 156 | 70,824 | 从 957 条成功母本中按 `box_yaw_deg` 等距抽样 |
| `sim39` | 39 | 17,753 | sim156 的前 39 条 |
| `armA` | 139 | 62,273 | 纯真机基线 |
| `armB` | 178 | 80,026 | 139 真机 + 39 sim（20% 帧） |
| `armC` | 295 | 133,097 | 139 真机 + 156 sim（=1:1），**未训练** |
| `armD` | 156 | 70,824 | 纯仿真，**未训练** |

路径：`/home/yyl/lerobot_datasets/`

### 2.2 模型

| 臂 | 最终 checkpoint | 状态 |
|---|---|---|
| armA | `/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500` | ✅ 17,500 步完成 |
| armB | `/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500` | ✅ 22,500 步完成（中途崩过，已 resume） |
| armC | — | 未训练 |
| armD | — | 未训练 |

验证：`ACTPolicy.from_pretrained` 加载 A/B 均返回 34.2M 参数，`model.safetensors` 178 个 tensor 可读。

---

## 3. A/B 离线评测结果

命令：

```bash
cd /home/yyl/deepseek-harness
nohup bash tools/run_eval_ab_cpu.sh > logs/eval_holdout_AB.log 2>&1 &
```

产物：

```text
logs/eval_holdout_armA.json
logs/eval_holdout_armB.json
logs/eval_holdout_AB.log
```

口径：`holdout_fixed` 每 10 帧取一个点，共 538 个采样点；指标单位为 deg。

| 指标 | armA | armB | B−A | 相对 |
|---|---:|---:|---:|---:|
| L1@1 | 2.490 | 2.408 | −0.082 | **−3.3%** |
| L1@10 | 2.618 | 2.627 | +0.009 | +0.3% |
| L1@50 | 3.524 | 3.679 | +0.155 | +4.4% |
| L1@100 | 4.227 | 4.688 | +0.461 | **+10.9%** |
| RMSE | 8.302 | 9.198 | +0.896 | +10.8% |

夹爪单列：

| 夹爪 L1 | armA | armB | 变化 |
|---|---:|---:|---:|
| L1@1 | 2.133 | 1.435 | **−32.7%** |
| L1@10 | 2.117 | 1.692 | −20.1% |
| L1@50 | 2.547 | 2.450 | −3.8% |
| L1@100 | 2.911 | 2.950 | +1.3% |

逐 episode：

```text
B 更好的 episode: 7/12
paired diff (B-A) = -0.081
bootstrap 95% CI = [-0.295, +0.106]
sign-test p = 0.387
```

**解读：**

- B 在短时动作和夹爪决策上有局部改善；
- B 的长时动作块明显漂移；
- 12 条测试集上**不显著**，不能据此宣布 B 优于 A；
- 离线 L1 不能替代真机成功率。

---

## 4. 数据诊断：为什么怀疑仿真分布

### 4.1 对象位置覆盖

sim156：

```text
x 范围 81.3 mm
y 范围 77.9 mm
离垫心半径 2.5 ~ 55.8 mm
5mm 网格占用 121 格
最近邻距离 中位 3.2 mm
盒朝向 0 ~ 179.7°
```

有 156 条 / 121 个 5mm 网格，不算“只有几个位置”，但空间范围偏窄。**位置数量不是当前最硬的证据。**

### 4.2 起点关节分布（关键）

| 关节 | 真机首帧 mean | 真机 std | 仿真首帧 mean | 仿真 std |
|---|---:|---:|---:|---:|
| shoulder_pan | −8.3 | 3.5 | 0.0 | **0.0** |
| shoulder_lift | −103.8 | 0.9 | −100.0 | **0.0** |
| elbow_flex | 95.5 | 0.2 | 88.3 | **0.0** |
| wrist_flex | 77.3 | 1.3 | 70.7 | **0.0** |
| wrist_roll | −86.5 | 6.2 | −103.5 | **0.0** |
| gripper | 1.1 | 1.2 | 0.0 | **0.0** |

**每条仿真 episode 都从同一个固定旧姿态起步，真机起点有变化。**

### 4.3 action 分布 W1

全部 action：

| 关节 | W1 |
|---|---:|
| shoulder_pan | 11.15 |
| shoulder_lift | 35.60 |
| elbow_flex | 34.61 |
| wrist_flex | 31.99 |
| wrist_roll | 36.79 |
| gripper | 10.66 |

把“action 减去该 episode 首帧 state”后，排除起点偏差：

| 关节 | 去掉起点后的 W1 |
|---|---:|
| shoulder_pan | 8.87 |
| shoulder_lift | **32.80** |
| elbow_flex | **28.24** |
| wrist_flex | **25.64** |
| wrist_roll | 19.80 |
| gripper | 11.78 |

结论：**不只是起点不同，轨迹形状本身也不同。** 这正好解释 B 的“夹爪短时变好、长时变差”。

复现命令：

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
  tools/compare_sim_real.py \
  --sim /home/yyl/lerobot_datasets/sim156 \
  --real /home/yyl/lerobot_datasets/real_train_fixed
```

---

## 5. 当前不建议做的事

### 5.1 不要直接加大仿真数量跑 E

原因：

1. 现有 sim 分布已经偏，更多同分布数据 = 放大偏差；
2. E 会把偏差先固化到预训练权重，再用 139 条真机微调，不一定纠得回来；
3. 156 条 sim 做“预训练”太小；按 plan 至少 5 epoch / 44k 步才算公平；
4. 真机 A/B 还没做，连 B 是否有信号都未确定。

### 5.2 不要直接跑 C（1:1）

当前趋势是 sim 比例越高，长时动作块越差。C 很可能进一步放大长时漂移，应该等数据对齐后再跑。

### 5.3 旧 sim 数据不能靠后处理修正

不能简单给 action 加偏移量。起点、IK、接触、夹爪闭合都是按旧起点重新仿真出来的，**只能重新生成**。

---

## 6. 环境/稳定性问题（重要）

这台机器在长任务中反复出现：

```text
kernel BUG at fs/inode.c:649!
clear_inode -> shmem_evict_inode
Comm: deepseek-harnes
```

以及：

- `python segfault`
- `ValueError: bad marshal data`（pyc 被写坏）
- `TypeError: 'ReLU' object is not iterable`
- `RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED`
- `nvidia-smi` 偶发连不上驱动，`torch.cuda.is_available()` 变 False

**已做的规避：**

1. 长任务用 `systemd-run --user` 或 `nohup`，不要挂在前台/桌面进程下。
2. `run_train_v8.sh` 已加：
   ```bash
   export PYTHONDONTWRITEBYTECODE=1
   export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$HOME/.cache/pycache_lerobot}"
   ```
3. 训练 `train_mix.py` 已加：
   - `--policy.push_to_hub=false`（离线不再报 hub 错误）
   - `--resume-config`（resume 自动找 `train_config.json`）
4. 训练/评测前确认：
   ```bash
   nvidia-smi
   env -u PYTHONHOME -u PYTHONPATH \
   /home/yyl/miniforge3/envs/lerobot312/bin/python -c \
   "import torch; print(torch.cuda.is_available())"
   ```

如果 `torch.cuda.is_available()==False`，先重启或重载 nvidia 模块，**不要在 cuda False 状态续训**。

---

## 7. 已修改/新增的脚本

| 文件 | 作用 |
|---|---|
| `run_train_v8.sh` | export 改用 `$PY`；merge 加 `--operation.concatenate_videos=false`；加 pyc 环境 |
| `tools/train_mix.py` | 加 `--policy.push_to_hub=false`、`--resume-config` |
| `tools/export_sim156_chunked.sh` | 分块导出 sim156，可断点续跑 |
| `tools/run_armB_resume_loop.sh` | B 崩了自动从最新 checkpoint 续训 |
| `tools/eval_act_holdout.py` | holdout 离线 action L1/RMSE 评测 |
| `tools/run_eval_ab_cpu.sh` | A/B 一键 CPU 评测 |
| `tools/compare_sim_real.py` | sim/real 首帧分布 + action W1 对比 |

备份：

```text
run_train_v8.sh.bak.*
run_train_v8.sh.bak.merge.*
```

---

## 8. 推荐下一步（按优先级）

### 第 1 步：真机 A/B，各 ≥20 次（最高优先）

这是唯一能定生死的实验。记录：

- 成功率
- 失败模式：没抓到 / 抓掉 / 放偏 / 撞盒 / 夹爪提前闭合
- 动作是否抖动
- 夹爪开合是否正常

如果 B 真机没有优势：不要加量跑 E，先回到数据分布问题。  
如果 B 有局部优势（夹爪好、长时差）：正好对应本文件的诊断，进入第 2 步。

### 第 2 步：数据分布 pilot（不要全量重刷）

目标：

```text
sim 首帧状态均值 ≈ real 首帧均值
sim 首帧 std ≈ real 首帧 std
action − first_state 的 W1 降到 < 10~15°
```

最小改动：

1. 在 `collect/collect.py` 新增 `START_POSE="real"`，起点取真机首帧均值：
   ```text
   [-8.3, -103.8, 95.5, 77.3, -86.5]
   ```
2. 每条 episode 在真机 std 附近采样：
   ```text
   pan ±3.5°, lift ±0.9°, elbow ±0.2°, wflex ±1.3°, wroll ±6.2°
   ```
3. **先只生成 100～200 条**，导出 39～80 条；不要直接 1000 条。
4. 用 `tools/compare_sim_real.py` 检查 W1；W1 没降下来就停，不要训练。

注意：`collect/collect.py` 目前硬编码写 `scripts/` 和 `videos/`，重刷前必须：
- 备份旧目录，或
- 给 `collect.py` 加 `--out-root` / 环境变量输出根。

同时把 `DATASET_VERSION` 升到 `sdg-v9`，避免新旧混用。

### 第 3 步：B2 小实验

```text
armA：139 真机（不变）
B2：139 真机 + 39～80 条新 sim
```

先离线对比，再真机 ≥20 次。  
只有 B2 有信号，才值得铺量。

### 第 4 步：E（最后）

前提：

```text
sim 分布已对齐
sim 预训练 ≥ 5 epoch（约 44k 步）
保留预训练 checkpoint
```

否则 E 只是“把偏差学得更牢”，不是有效的 Sim2Real 证明。

---

## 9. 现有数据如何处置

| 数据 | 处置 |
|---|---|
| `real_train_fixed` / `holdout_fixed` | 保留，继续用 |
| `armA` | 保留，纯真机基线 |
| A/B 最终 checkpoint | 保留，作为旧分布对照 |
| 1000 条母本、`sim156`、`sim39` | **保留但隔离**；新实验不建议混入 |
| `armB` / `armC` / `armD` 数据集 | 保留作历史对照；不要当新 B2/E 的输入 |
| 旧 `scripts/`、`videos/` | 重刷前备份，不要覆盖 |

---

## 10. 需要下一个窗口确认的问题

1. 真机 A/B 是否已经跑过？结果如何？
2. 是否接受“先做 100～200 条 pilot，而不是直接 1000 条”？
3. 是否允许修改 `collect.py` 增加 `--out-root` 和逐 episode 起点采样？
4. 时间线：真机评测、pilot、B2、E 各能分配多少天？
5. 如果时间不够，是否只保留 A/B 真机结论，放弃 E？

---

## 11. 关键文件索引

```text
docs/PLAN_TRAIN_V8.md                 # 原始训练计划
docs/PHYSICS_FACTS.md                 # 物理事实
docs/HANDOFF_SIM2REAL_V8.md           # 本文件
logs/eval_holdout_armA.json           # A 离线结果
logs/eval_holdout_armB.json           # B 离线结果
logs/train_v8_AB.log                  # A/B 训练日志（含崩溃/续训）
logs/train_v8_B_resume.log            # B 续训日志
tools/eval_act_holdout.py             # 离线评测
tools/run_eval_ab_cpu.sh              # A/B 一键评测
tools/compare_sim_real.py             # sim/real 分布对比
tools/export_sim156_chunked.sh        # 分块导出
tools/run_armB_resume_loop.sh         # 自动续训
```

---

## 12. 最终建议

**现在最该做的是真机 A/B 20 次，而不是加数据跑 E。**  
如果 B 真机有信号，再按“起点 + 轨迹分布”做 100～200 条 pilot；W1 降下来后训练 B2；E 放在最后。
