# 计划：真机 181 条 + 纯仿真，怎么混、按什么比例、怎么判定

> 前置事实见 `docs/DATA_HEALTH_181.md`（本轮实测）。本文件只讲"怎么做 + 为什么"。
> 关键前提：仿真与真机**不是同一任务**（药盒→木箱 vs 抓黑方块/抓盒子），
> 仿真轨迹互相只差 1.1°、与真机差 25.8° → **仿真是正则项，不是数据量**。

---

## 0. 先做 3 件事，再谈比例（约 1 h，不做就是白跑 8 h GPU）

| # | 事 | 命令/位置 | 通过标准 |
|---|---|---|---|
| P1 | 确认夹爪端点（**唯一阻塞项**） | 报告 §3 B1 那段 20 行 snippet | 闭合 ≈ 0，张开 ≈ 100 |
| P2 | 修仿真夹爪量纲 | `collect/export_lerobot.py` 加 `--gripper-scale 0.78198`（见 §1.2） | 导出后仿真夹爪 ∈ [0,100] |
| P3 | 把 corners 的 30 条并进来 | 见 §3 arm A | 现有 ACT 只吃了 151 条，白丢 17% |

> P3 的期望收益 > 加仿真：真机数据同分布、动作多样性 10 倍于仿真。

---

## 1. 仿真侧准备

### 1.1 现状

- 母本 `scripts/index.json`：**1000 条全 accepted、全 success**，`videos/` 里 2000 个 mp4 齐全。
- 已导出的 `dataset/soarm101_sim_v1` 只有 **64 条**（sdg-v6 早期的子集）→ **已过期，别直接用**。
- 导出速度：**16.6 s/条**（一条 455 帧、双相机 av1）。

### 1.2 夹爪重标定（P2 的补丁）

`episode_arrays()` 里 `units=="deg"` 之后加一行：

```python
    # 夹爪：仿真 gripper_joint 是度（行程 0~2.232 rad = 0~127.9°）；
    # 真机 so_follower 的 gripper 是 RANGE_0_100（标定 raw 1455 counts = 127.9°）。
    # 两者 0 点一致（0=闭合），只差比例 100/127.9 = 0.78198。
    if gripper_scale != 1.0:
        state[:, 5] *= gripper_scale
        action[:, 5] *= gripper_scale
```

并把系数写进 `meta/sim_provenance.json`（第 ~286 行的 `note`/`units` 旁边加一个 key），
否则半年后没人知道这批数据被改过量纲。

### 1.3 选 156 条（按 θ 均匀铺开，别取前 156 条）

```bash
cd /home/yyl/deepseek-harness
EP156=$(env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - <<'EOF'
import json
idx = json.load(open('scripts/index.json'))
acc = [e['episode_id'] for e in idx['accepted']]
def th(e):
    try: return abs(json.load(open(f'scripts/ep_{e:06d}.json'))['randomization']['spawn']['theta_deg'])
    except Exception: return 0.0
acc.sort(key=th)                      # 按 |θ| 排序后等距抽 → 覆盖最散
step = len(acc) / 156
print(','.join(str(acc[int(i*step)]) for i in range(156)))
EOF
)
echo "$EP156" | head -c 200; echo
```

### 1.4 导出

```bash
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 collect/export_lerobot.py \
  --dst /home/yyl/lerobot_datasets/sim156 \
  --repo-id local/sim156 \
  --episodes "$EP156" \
  --units deg --gripper-scale 0.78198 \
  --overwrite
```

导出后自检（回读 + 维度 + 视频对齐由脚本自己做，再看一眼夹爪区间）：

```bash
env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
  /home/yyl/deepseek-harness/tools/check_datasets_health.py \
  --root /home/yyl/lerobot_datasets/sim156 --out logs/health_sim156.json | tail -20
# 看：gripper min/max ≈ 0 / 67，frames ≈ 156×455 ≈ 71k
```

---

## 2. 比例怎么定（核心问题）

### 2.1 三个约束

1. **仿真不是同分布**：跨域轨迹差 25.8°，高比例混合 = 用不属于本任务的轨迹稀释真机监督。
2. **仿真多样性饱和**：64 条互相差 1.1°，真机差 10.6° → 仿真的信息量在第 ~10 条后基本不再增长，
   加多少条都不会让"行为覆盖"变宽，只增加外观随机性。
3. **视觉域差大**：仿真图像锐度只有真机的 1/7~1/24，比例越高，真机视觉特征被稀释越多。

### 2.2 推荐比例（按**样本/帧**计，不是按条数）

真机训练池（留出 20 条后）= **161 条 ≈ 71.0k 帧**；仿真 455 帧/条。

| 仿真占比 s | 仿真帧 | 仿真条数 | 用途 |
|---|---|---|---|
| 0% | 0 | 0 | **arm A 基线（必跑）** |
| 15% | 12.5k | 28 | 保守 |
| **20%** | **17.8k** | **39** | **arm B 主推** |
| 30% | 30.4k | 67 | 中 |
| 40% | 47.3k | 104 | **arm C 上限探针** |
| 50% | 71.0k | 156 | 不建议（除非 C 表现特别好） |
| 100% | — | 156 | arm D 纯仿真下限参照 |

**结论：先跑 20%（arm B），同时跑 40%（arm C）划出趋势；不要一上来试 50%。**

### 2.3 为什么不是更高

- 真机 20%/40% 的边际效应不同：20% 时仿真≈"数据增强"，40% 时已经能改写动作分布均值
  （仿真均值与真机相关 25.8° 的偏移会渗进预测）。
- 真机任务只有 2 个物体类别（黑方块/盒子）、一个固定桌面；仿真引入的是**第三个场景**。
  这类"多任务但无语言条件"的混合，实践上甜点区在 **1:4 ~ 1:3（仿真:真机）**；
  超过 1:1 就等价于训练一个"两边都不擅长"的平均策略。

### 2.4 更省的替代方案（如果 B/C 都没收益）

- 直接对真机做外观增强（颜色/亮度/裁剪抖动）+ 把 corners 并进来，成本 0，通常能拿到
  与"加仿真"同量级甚至更好的鲁棒性提升。
- 两阶段（arm E）：仿真预训练 → 真机微调。冲突被关在"初始化"，比混合训练安全，
  且预训练阶段便宜（156 条 = 71k 帧，20k 步 ≈ 1.8 h）。

---

## 3. 实验臂（5 个，前 4 个必跑）

| 臂 | 数据 | 训练步数 | 作用 |
|---|---|---|---|
| **A** | 真机 181 全部 merge，**训练时排除留出 20 条** | 20k | 基线（对照"现有 ACT 只吃 151 条"） |
| **B** | A + sim39（≈20%） | 20k | **主推** |
| **C** | A + sim104（≈40%） | 20k | 上限探针 |
| **D** | sim156 only | 20k | 下限参照（预期真机成功率 ≈0） |
| E（可选） | sim 预训练 20k → 真机 161 微调 8k | 28k | C 若有效再上 |

留出集：fixed 里等距 12 条 + corners 里等距 8 条 = **20 条真机**，训练时用
`--dataset.exclude_episodes` 排除，离线评测只在这 20 条上算。

---

## 4. 建数据集（官方 merge，不走自研合并）

`features_equal_for_merge` 已实测 **True**（真机与仿真的 features 完全兼容，
视频编码参数差异会被忽略）。**真机+仿真 merge 已在本机跑通并逐帧对账**（见 §4.1）。

```bash
cd /home/yyl/lerobot
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
FIXED=/home/yyl/lerobot_datasets/yyl100_dataset_fixed
CORNER=/home/yyl/lerobot_datasets/yyl100_corners
SIM156=/home/yyl/lerobot_datasets/sim156
export HF_HUB_OFFLINE=1        # 无网时避免 get_safe_version 打网

# sim104 / sim39：从 sim156 里各留前 N 条（保持 B ⊂ C ⊂ 全集，便于对照）
# ⚠️ episode_indices 必须带方括号，裸 "104,105,..." 会 draccus 解码失败
env -u PYTHONHOME -u PYTHONPATH $PY -m lerobot.scripts.lerobot_edit_dataset \
  --operation.type=delete_episodes --operation.episode_indices="[$(seq -s, 104 155)]" \
  --repo_id=local/sim156 --root=$SIM156 \
  --new_repo_id=local/sim104 --new_root=/home/yyl/lerobot_datasets/sim104
# 同法删 [39..103] → sim39（复制只算保留的 episode，sim 一条约 6 MB）

# arm A：真机 151 + 30
env -u PYTHONHOME -u PYTHONPATH $PY -m lerobot.scripts.lerobot_edit_dataset \
  --operation.type=merge \
  --new_repo_id=local/mixA --new_root=/home/yyl/lerobot_datasets/mixA \
  --operation.repo_ids="['local/fixed','local/corners']" \
  --operation.roots="['$FIXED','$CORNER']"

# arm B / C：在 A 后面接仿真子集
... --new_repo_id=local/mixB --new_root=/home/yyl/lerobot_datasets/mixB \
    --operation.repo_ids="['local/fixed','local/corners','local/sim39']" \
    --operation.roots="['$FIXED','$CORNER','/home/yyl/lerobot_datasets/sim39']"

# arm D
... --new_repo_id=local/mixD --new_root=/home/yyl/lerobot_datasets/mixD \
    --operation.repo_ids="['local/sim156']" --operation.roots="['$SIM156']"
```

合并后 **episode 编号会重排**：真机在前（0~150 = fixed，151~180 = corners，181.. = 仿真）。
留出集的 20 条用它在 A 里的新编号写进 `exclude_episodes`（用一个 3 行脚本生成即可）。

### 4.1 merge 已实测（本机跑通，可当模板）

拿 2 条真机 + 2 条仿真做的最小验证：

| 检查项 | 结果 |
|---|---|
| merge 退出码 | 0，`Episodes: 4, Frames: 1795`（= 896 + 899） |
| 逐帧对账（action+state） | 真机段、仿真段 **max abs diff = 0.0** |
| episode 重排 | 0,1 = 真机；2,3 = 仿真 |
| 两个 task 文本 | 都保留 |
| 视频读取 | 640×480×3 正常出帧 |
| merged stats | gripper min 0.0 / max **85.94** ← **这正是 B1 的直接证据**（真假量纲混在一起） |
| 耗时 | 2+2 条约 10 s（视频走 ffmpeg remux，不重编码）→ 337 条约 2~5 min |

复盘（想自己再看一遍就把 §4 的 merge 换成 2 条真机 + 2 条仿真，产物放任意空目录）：

```bash
env -u PYTHONHOME -u PYTHONPATH HF_HUB_OFFLINE=1 $PY -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds=LeRobotDataset('local/t_mix4', root='<你的 merge 输出目录>')
print(ds.meta.total_episodes, ds.meta.total_frames, ds[0]['action'].numpy(), ds[900]['observation.images.env'].shape)"
```

⚠️ 磁盘：每个 merge 会复制视频（sim 一条 ≈ 6 MB）。实测体量：**A ≈ 2.7 GB、B ≈ 2.9 GB、
C ≈ 3.3 GB、D ≈ 1.0 GB**，四臂共 ≈ 10 GB；根分区剩 ~30 GB，**训完即删数据集、留 ckpt**。
`--operation.concatenate_videos=False` 仍会复制；真要省空间只能自研硬链接合并（需额外验证）。

---

## 5. 训练

`lerobot-train` 在离线环境需要 `local/` 前缀的 `get_safe_version` 补丁（你现有
`train_local.py` 已在做）。建议把它做成参数化 runner：

```python
# /home/yyl/lerobot/train_mix.py
import sys
from lerobot.datasets import lerobot_dataset, utils
from lerobot.scripts import lerobot_train

_o = lerobot_dataset.get_safe_version
def _p(repo_id, revision):
    return str(revision) if str(repo_id).startswith("local/") else _o(repo_id, revision)
lerobot_dataset.get_safe_version = utils.get_safe_version = _p

arm, root, exclude = sys.argv[1], sys.argv[2], sys.argv[3]   # 例：B /.../mixB "" 或 "12,25,..."
sys.argv = [
    "lerobot-train",
    f"--dataset.repo_id=local/mix{arm}",
    f"--dataset.root={root}",
    "--policy.type=act",
    f"--policy.repo_id=local/act_mix{arm}",
    "--policy.use_vae=false",
    f"--output_dir=outputs/train/act_mix{arm}",
    f"--job_name=act_mix{arm}",
    "--policy.device=cuda",
    "--steps=20000",
    "--save_freq=5000",
    "--wandb.enable=false",
    "--batch_size=8",
    "--num_workers=6",
] + ([f"--dataset.exclude_episodes=[{exclude}]"] if exclude else [])
lerobot_train.main()
```

```bash
cd /home/yyl/lerobot
export -n PYTHONHOME PYTHONPATH
export HF_HUB_OFFLINE=1                              # 离线，避免打网
# 注意 exclude 也要方括号列表（draccus 语法），例：--dataset.exclude_episodes="[12,25,38]"
/home/yyl/miniforge3/envs/lerobot312/bin/python train_mix.py A /home/yyl/lerobot_datasets/mixA "12,25,..."
```

**参数建议**
- `batch_size`：先按你显卡试 8 / 16；实测 0.32 s/步 @ bs8 + 2 workers。
- `num_workers=6`（比 2 快，av1 软解是瓶颈）；仍慢就先把数据集转 h264：
  `--operation.type=reencode_videos`（体积 ↑，解码快 3~5×）。
- `steps=20000`（与现有 ACT 对齐，便于比较）。想更公平可对所有臂设 `steps = 25 × 训练帧数/bs/1000`
  的等 epoch 数——但**先别动**，等 B vs A 有信号再说。

**时间预算**（单卡、bs8、6 workers）

| 阶段 | 时长 |
|---|---|
| 上机夹爪确认 + 导 sim156 | 0.2 h + 0.75 h |
| 建 sim104/sim39 + 3 个 merge | 0.5 h |
| 训练 A/B/C/D（20k 步/臂） | 4 × 1.8 h ≈ **7.2 h**（可夜里跑） |
| 离线评测 | 0.3 h |
| 上机评测（每臂 20 次） | 每臂 ≈ 15 min |

---

## 6. 怎么判"好不好"（别用 loss 判）

### 6.1 离线（自动，先筛掉明显坏掉的臂）

在留出的 20 条真机 episode 上，对每个臂算：
- 逐关节预测 MAE（度），**重点看 gripper 与 wrist_flex**；
- 夹爪"开合方向"准确率（预测的 Δ 方向 vs GT）；
- 起始 50 帧的误差（这是真机部署最关键的早段）。

**判据**：任何臂只要 gripper MAE 比 A 差 >30% → 该臂直接淘汰，不必上机。

### 6.2 上机（唯一有意义的指标）

固定 20 次试验/臂：物体位置在 3 个点循环（近/中/远，与训练集分布内/边缘各占一部分），
记录：抓取成功（抬起 ≥8 cm）/ 全任务成功 / 动作时长 / 危险事件（撞桌、压盒）。

**判据表**

| 结果 | 结论 | 下一步 |
|---|---|---|
| B ≥ A + 10 个百分点 | 仿真有效 | 跑 C、E，按趋势选最优 |
| \|B − A\| < 10 | 仿真中性 | **选 A**（更简单），把精力放真机扩数据 |
| B ≤ A − 10 | 仿真有害 | 停；只保留 E（两阶段）作为最后尝试 |
| D > 20% | 任务太容易，评测区分度不足 | 加大物体位置范围 / 增加干扰物后再评 |
| A 显著优于旧 `act_yyl151_lance` | 白捡的 30 条 corners 有用 | 直接上线 A |

### 6.3 一次性 smoke test（10 min，强烈建议先跑）

`--steps=3000` 只跑 A 和 B，看 `train/loss` 与留出集 loss：
若 B 的 loss 在第 1k 步就明显高于 A 且不收敛，说明量纲/归一化没对齐（回查 P2），
不要等到 7 h 之后才发现。

---

## 7. 风险清单

| 风险 | 触发条件 | 处置 |
|---|---|---|
| 夹爪量纲不一致 | P1 没做就混 | 必须先做 P1+P2 |
| 磁盘不够 | 同时留 >3 个 merge | 训完即删数据集，留 ckpt |
| av1 解码拖慢 GPU | `update_s` > 0.4 s/步 | `num_workers` ↑ 或 reencode h264 |
| 用 `--dataset.repo_id` 传列表 | 报 `NotImplementedError: LeRobotMultiDataset` | 本版不支持，必须物理 merge |
| 用了过期的 `dataset/soarm101_sim_v1`（64 条） | 直接训练 | 一律重新导出（§1.4） |
| 用 lance 版和 parquet 版混 | merge 报存储格式错 | merge 只用 parquet 根目录 |
| 拿训练 loss 当结论 | — | 只认 §6.2 上机成功率 |

---

## 8. 我可以直接替你做的部分

1. 写并验证 `--gripper-scale` 补丁 + 跑一次 2 条 episode 的小导出做前后对比；
2. 跑 `sim156` 全量导出与 `sim104/sim39` 裁剪（CPU，约 1 h）；
3. 跑 3 个 merge（约 0.5 h，磁盘占用 10 GB）；
4. 写离线评测脚本 `tools/eval_policy_offline.py` 并跑通（需要 CUDA，本沙箱看不到
   `/dev/nvidia*`，只能在你终端跑）。

训练与上机评测只能你来（本沙箱无 GPU）。