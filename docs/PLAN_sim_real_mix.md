# 计划：真机 181 条 + 纯仿真 + 混合训练（ACT）

> 写于 2026-10-25。所有数字都是本机实测，不是估计；复现命令附在每节。
> 结论先行：**181 条能直接训，但现在还不能和仿真混** —— 夹爪列语义没定案之前混训会把
> 夹爪通道学成双模。先定案（10 秒物理检查），再按 §5 的比例网格开训。

---

## 1. 体检结果（已跑完）

### 1.1 盘点

| 数据集 | 条 | 帧 | 每相机视频 | 任务文本 | 盘 | 备注 |
|---|---|---|---|---|---|---|
| `yyl100_dataset_fixed`（真机） | 151 | 67 649 | 16 文件 av1 | `Grab the black cube` | 2.3 G | 每条约 448 帧 |
| `yyl100_corners`（真机） | 30 | 11 870 | 1 文件 av1 | `Grab the box` | 395 M | 308~578 帧 |
| `soarm101_sim_v1`（仿真·已导出） | 64 | 29 176 | env 2 / wrist 1 | `Pick up the medicine box and place it into the wooden box.` | 391 M | sdg-v6 |
| 仿真母本 `scripts/index.json` | 1000 | ≈455/条 | 已渲 1000×2 | 同上 | 1.9 G + 166 M npz | 想补多少条都能补 |
| `yyl100_lance_fixed` / `yyl100_lance_corners` | 151 / 30 | — | lance | — | 2.3 G / 395 M | 内容 = 上面两个 parquet 版，**不是新数据** |

**真机总量 181 条 / 79 519 帧**；当前已导出的仿真 64 条 / 29 176 帧。

### 1.2 一致性矩阵

| 项 | 结论 | 证据 |
|---|---|---|
| codebase / fps / robot_type | ✅ 全同 | v3.0 / 30 / `so_follower` |
| state/action 维度与关节名顺序 | ✅ 全同 | 6 维，`shoulder_pan…gripper.pos` |
| 相机 key、分辨率、codec | ✅ 全同 | `observation.images.env/wrist`，640×480，av1 |
| 时间戳 | ✅ 全 1/30 精确、每 episode 单调、无丢帧 | `dt med=0.03333 min=max` |
| NaN / Inf | ✅ 0 | action+state 全扫 |
| 视频完整性 | ✅ 181 + 64 条全解码，帧数与 meta **0 不符** | 两个相机都查了 |
| 每步跳变（teleport） | ✅ p99 ≤ 4.2°，>20° 仅 1 步 | 无明显硬拉 |
| **夹爪列语义** | ⚠️ **未定案（阻塞混训）** | 见 §2.1 |
| 关节覆盖 | ⚠️ 仿真窄 | 见下表 |
| 任务/物体 | ⚠️ 不同任务 | 仿真=药盒→木盒(含 place)，真机=抓黑方块/盒子 |
| 视觉域差 | ⚠️ 很大 | 见 §2.3 |
| 成功/失败标签 | ❌ 真机没有 | 仿真有 G1–G6 门限，真机没有 |

关节跨度（度，整集统计）：

| 关节 | 真机 fixed | 真机 corners | 仿真 |
|---|---|---|---|
| pan | 76.0 | 72.8 | 60.0 |
| lift | 158.2 | 161.2 | 110.4 |
| elbow | 148.4 | 150.3 | **53.9** |
| flex | 86.9 | 63.8 | 77.0 |
| roll | 127.4 | 123.3 | **13.7** |
| gripper | 49.7 | 42.1 | 85.9 |

### 1.3 复现命令（你要的"检测命令"）

```bash
cd /home/yyl/deepseek-harness
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python     # 必须用这个解释器（有 lerobot/pyav/pyarrow）

# 一趟查结构+数值（快，~10 s/数据集）
env -u PYTHONHOME -u PYTHONPATH $PY tools/check_datasets_health.py \
  --root /home/yyl/lerobot_datasets/yyl100_corners --mode data --out logs/health_corners.json

# 一趟查视频（慢，~90 s/151 条；pyarrow 与 pyav 同进程在 151 条那批上会 SIGBUS，所以拆两趟）
env -u PYTHONHOME -u PYTHONPATH $PY tools/check_datasets_health.py \
  --root /home/yyl/lerobot_datasets/yyl100_dataset_fixed --mode video --out logs/health_fixed_video.json
```

已落盘：`logs/health_*_data.json`、`logs/health_fixed_video.json`、`logs/health_soarm101_sim_v1.json`。
退出码 0=ok / 1=warn / 2=fail。

---

## 2. 体检发现的 6 个问题与处置

### 2.1 夹爪列语义没定案（**唯一阻塞项**）

事实：

| | 起点值 | 终点值 | 峰值 | 仿真语义（已知） |
|---|---|---|---|---|
| 真机 fixed | 0.43（95/151 条恰为 0.0） | 0.37 | 49.68 | — |
| 真机 corners | 3.40 | 3.18 | 42.58 | — |
| 仿真 | 0.00 | 0.00 | 85.94 | **0 = 全闭**（jaw=0，爪甲合拢）；85.94° = 全张（1.5 rad） |

真机数据里 `0` 是硬停止位（95 条起点恰为 0.0，末端 115 条回到 0.0），所以 `0` 必是机械行程的一端；
但**哪一端**从数值上定不了：0=张开 或 0=闭合，两条解释都能自洽（详见 §9 的推理）。
这一位决定"混训前要不要把仿真夹爪翻过来"。

**10 秒定案**：打开 `inspect/frames/` 下 4 组图（已抽好）：

```
inspect/frames/real_ep0_env/file-000_t0.000s.png      ← 真机 home 帧（env 侧视）
inspect/frames/real_ep0_env/file-000_t4.000s.png      ← 真机 4 s 处（夹爪转折点）
inspect/frames/sim_ep0_env/file-000_t0.000s.png       ← 仿真 home 帧（jaw=0）
inspect/frames/sim_ep0_env/file-000_t4.600s.png       ← 仿真 descend 段（jaw=85.9°，全张）
```

看真机 home 帧：**两指是贴在一起（闭合）还是分开（张开）？**

- 闭合 → 真机 `0=闭合`，与仿真同向 → 用 `--gripper-space real-closed`（只是 0..100 vs 0..86 的尺度差）
- 张开 → 真机 `0=张开`，与仿真反向 → 用 `--gripper-space real-open`（翻转 + 尺度）

`collect/export_lerobot.py` 已加好这个开关（`--gripper-space {sim,real-open,real-closed}`，实测导 2 条通过回读校验）：

```bash
env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
  collect/export_lerobot.py --dst /home/yyl/lerobot_datasets/sim44_realunits \
    --episodes <见 §5.2> --gripper-space real-open --overwrite
```

> 顺带一个物理事实：真机峰值 49.68、grasp 平台 25~32；按本仓库实测的夹爪几何
> （指面间距 ≈ 1.31 mm/deg，jaw 0° ≈ 合拢）折算，真机抓的物体厚约 22~29 mm（若 0=闭合）
> 或 43~74 mm（若 0=张开）。黑方块是 25 mm 量级 → 也偏向"0=闭合"，但仍以看图为准。

### 2.2 仿真的"有效多样性 ≈ 1 条轨迹"

1000 条母本全部来自同一套 15 段脚本（init→approach→descend→grasp→…→home），只随机化了
θ ±10°、相机 ±2°、光照、噪声、材质增益、节奏系数。表现在数据上就是：roll 只跨 13.7°、
elbow 只跨 53.9°，而真机是 123~127° / 148~150°。

**含义：再导出 500 条仿真 ≈ 加外观抖动，不是加样本量。** 所以仿真占比要按"外观增强"来给，
不是按"20 条真机 ≈ 20 条仿真"来给。

### 2.3 视觉域差 7~17 倍（Laplacian 方差，越高越清晰）

| | env | wrist |
|---|---|---|
| 真机 fixed | 421 | 1381 |
| 真机 corners | 991 | 460 |
| 仿真 | **57.5** | **35.8** |

亮度均值差不多（137~145），差在**高频细节**：仿真图明显更"糊/更假"。视觉主干（resnet18）
会在两个世界之间分叉，仿真占比越高，真机信号越被稀释 —— 这是把上限压在 1:1 以内的第二个理由。

### 2.4 真机 fixed 的 151 条全是 448/449 帧（`distinct_len=2`）

固定窗口录制/重采样。副作用是 demo 可能被截断（ep0 就是"抓了一下又松手"的半截动作），
也是它的 gripper 曲线看起来反直觉的原因。corners 那 30 条是变长的（308~578，`distinct_len=28`），
更像正常示教。**处置：两个真机数据集一起训，别只用 fixed。**

### 2.5 `yyl100_dataset_fixed` 的全局 `index` 不连续（0..68096，67649 行）

说明这份数据集被编辑过（删过 episode）。按位置读没问题（`lerobot` 用 episode meta 定位），
但**不要**拿它做行号去重/切分，也不要假设 `index == 行号`。

### 2.6 训练数据加载是瓶颈

实测：batch 8 / `num_workers=2` / 20 000 步 = **108 min**（5 000 步 ≈ 27 min，来自
`outputs/train/act_yyl151_lance/checkpoints/*` 的 mtime 间隔）。av1 软解 + 2 个 worker
大概率是瓶颈。**处置**：`--num_workers 8 --batch-size 16`；空间够的话把 av1 转 h264 再加一档。

---

## 3. 混合比例：给多少仿真

按**帧**算（真机 79 519 帧；仿真平均 455 帧/条）：

| 臂 | sim:real | 仿真帧占比 | 需要仿真条数 | 需要帧数 |
|---|---|---|---|---|
| S0 纯真机（基线） | 0 | 0 % | 0 | 79 519 |
| S1 | 1:8 | 11 % | 22 | 9 940 |
| **S2 ★ 主臂** | **1:4** | **20 %** | **44** | 19 880 |
| S3 | 1:1 | 50 % | 87 | 39 760（上限探测） |
| S4 纯仿真（对照） | ∞ | 100 % | 64 | 29 176 |
| S5 两阶段 | 预训+微调 | — | 44 | — |

**为什么 1:4 是主臂**

1. 仿真的有效多样性 ≈ 1 条轨迹（§2.2）→ 加到 20 % 已经把"外观/光照/相机抖动"喂满了，再加是重复。
2. 视觉域差 7~17 倍（§2.3）→ 仿真超过 1:1 时，模型会把一半容量花在"渲染图长什么样"上。
3. 任务与物体不同（仿真=药盒→木盒，真机=黑方块/盒子）→ 仿真能教的只有"关节量纲 + 夹爪命令语义 +
   相机系下的视觉伺服鲁棒性"，教不了任务；真机才是任务信号。
4. **上限探测**：如果 S3(1:1) 打不过 S2(1:4)，说明仿真增益已经饱和，以后就别再往上加，省下的算力拿去重做仿真场景
   （把场景换成真机的黑方块 + 真机相机位，这条路的上限比调比例高得多）。

**必须按 real-pass 对齐，不然结论是假的**

纯真机 79 519 帧、batch 16 → 每 real-pass ≈ 4 970 步。混合改变的是「每 epoch 的组成」，不是总步数：

| 臂 | 训练步数 | 真实 real-pass | 说明 |
|---|---|---|---|
| S0 | 20 000 | 4.0 | 基线 |
| S1 | 22 500 | 4.0 | 与 S0 等价曝光 |
| S2 | 25 000 | 4.0 | 与 S0 等价曝光 |
| S3 | 40 000 | 4.0 | 贵；也可只跑 25 000 并注明 real-pass 只有 2.5 |

**只按 step 对齐会让"仿真多的臂"看起来更差（真机没看够），或让"仿真多的臂"看起来更好（正则更强）——
两个方向都会骗你。** 预算紧就把 S3 降到 25 000 步并在报告里标 real-pass。

---

## 4. 工具（已写好并冒烟）

| 工具 | 用途 | 状态 |
|---|---|---|
| `tools/check_datasets_health.py` | 数据集体检（结构/数值/视频/夹爪/重复/域差） | ✅ 已跑通 3 个数据集 |
| `collect/export_lerobot.py --gripper-space` | 仿真导出时把夹爪列换到真机量纲 | ✅ 导 2 条回读通过 |
| `tools/train_mix.py` | 按臂训练 ACT（本地数据集免联网、参数全可覆盖） | ✅ `--dry-run` 通过 |
| `tools/dump_frames.py` | 抽帧存 PNG（夹爪定案、域差目视） | ✅ 已抽 8 张 |
| `lerobot-edit-dataset --operation.type=merge` | 官方合并数据集（重算 stats） | 未跑，见 §5.3 |

---

## 5. 执行步骤

### 5.0 前置（一次性，10 min）

1. 看 §2.1 的 4 张图，定下 `--gripper-space` 用 `real-open` 还是 `real-closed`，写进本节注释。
2. 确认磁盘：`df -h /home/yyl` —— 每个混合数据集 ≈ 2.7 G（真机）+ 仿真子集，留 15 G 以上。
3. 确认 GPU 可见：`nvidia-smi -L`（在你自己终端跑；AI 会话里看不到设备节点是正常的）。

### 5.1 划分训练/留出集（一次性）

从 181 条里留 36 条永远不训（fixed 留 28、corners 留 8），用于离屏评测与可比性：

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - <<'PY'
import json
holdout = {"fixed": sorted(range(0, 151, 5))[:28], "corners": [0, 4, 9, 13, 17, 21, 25, 29]}
print("holdout: fixed", holdout["fixed"])
print("holdout: corners", holdout["corners"])
json.dump(holdout, open("logs/holdout.json", "w"))
PY
```

> 之后每条臂都必须排除这些 episode：`lerobot-edit-dataset --operation.type delete_episodes`
> 生成 `real_train_fixed` / `real_train_corners` 两份训练用副本（一次性，之后所有臂共用）。

### 5.2 导出仿真子集（22 / 44 / 87 / 175 条）

按步长在 1000 条母本里撒点（保证 θ/光照箱覆盖，不要取前缀）：

```bash
cd /home/yyl/deepseek-harness
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
pick () {  # $1 = 要几条
  env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 -c "
import json,sys
n=int(sys.argv[1]); acc=json.load(open('scripts/index.json'))['accepted']
ids=[e['episode_id'] for e in acc]; step=max(1,len(ids)//n)
print(','.join(map(str,ids[::step][:n])))" "$1"
}
for N in 22 44 87 175; do
  env -u PYTHONHOME -u PYTHONPATH $PY collect/export_lerobot.py \
    --dst /home/yyl/lerobot_datasets/sim${N} --repo-id local/sim${N} \
    --episodes "$(pick $N)" --gripper-space real-closed --overwrite
done
```

耗时实测 **17 s/条**（含 av1 重编码）：22 条 ≈ 6 min，44 ≈ 13 min，87 ≈ 25 min，175 ≈ 50 min。

### 5.3 建 5 个混合数据集（官方 merge，重算 stats）

```bash
export HF_HUB_OFFLINE=1
EDIT=/home/yyl/miniforge3/envs/lerobot312/bin/lerobot-edit-dataset
D=/home/yyl/lerobot_datasets

merge () {  # $1=输出名  $2=sim 条数（0=纯真机）
  roots="['$D/real_train_fixed','$D/real_train_corners'"
  ids="['local/real_train_fixed','local/real_train_corners'"
  [ "$2" != "0" ] && roots="$roots,'$D/sim$2'" && ids="$ids,'local/sim$2'"
  env -u PYTHONHOME -u PYTHONPATH $EDIT --new_repo_id "local/$1" --new_root "$D/$1" \
    --operation.type merge --operation.repo_ids "$ids']" --operation.roots "$roots']"
}

merge mix_S0_sim0   0      # 纯真机基线
merge mix_S1_sim22  22     # 1:8
merge mix_S2_sim44  44     # 1:4  ★
merge mix_S3_sim87  87     # 1:1
```

> `--operation.concatenate_videos false` 可省一次拼接，但视频仍会**复制**（不硬链），不要指望省盘。
> 纯仿真臂 S4 直接用 `$D/sim64`（已存在）或 `sim44_realunits`，不用 merge。

### 5.4 训练（每条臂一条命令，参数除步数外完全一致）

```bash
cd /home/yyl/deepseek-harness
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python
T="$PY tools/train_mix.py --batch-size 16 --num-workers 8 --save-freq 5000 --seed 0"

env -u PYTHONHOME -u PYTHONPATH $T --root $D/mix_S0_sim0  --repo-id local/mix_S0_sim0  \
  --output /home/yyl/lerobot/outputs/train/act_S0_realonly --job-name act_S0_realonly --steps 20000
env -u PYTHONHOME -u PYTHONPATH $T --root $D/mix_S1_sim22 --repo-id local/mix_S1_sim22 \
  --output /home/yyl/lerobot/outputs/train/act_S1_sim8  --job-name act_S1_sim8  --steps 22500
env -u PYTHONHOME -u PYTHONPATH $T --root $D/mix_S2_sim44 --repo-id local/mix_S2_sim44 \
  --output /home/yyl/lerobot/outputs/train/act_S2_sim4  --job-name act_S2_sim4  --steps 25000
env -u PYTHONHOME -u PYTHONPATH $T --root $D/sim64 --repo-id local/sim64 \
  --output /home/yyl/lerobot/outputs/train/act_S4_simonly --job-name act_S4_simonly --steps 20000
```

预算（按实测 108 min/20k@batch8 折算 batch16 约 60~80 min/20k）：**S0+S1+S2+S4 ≈ 4~5.5 h**，
加 S3 约 +1.5 h。两阶段臂 S5：先用 `sim64` 跑 8 000 步，再用 `--resume` 换到真机数据继续 15 000 步
（注意 `--resume` 会沿用旧 normalizer，换数据集时要显式对比 stats，或者干脆不续，改为初始化=仿真权重 + 真机 stats）。

### 5.5 评测

**离线（零成本，先筛）** —— 留出 36 条真机 episode，用 `lerobot-rollout`/`lerobot-eval` 或自写脚本对比：

1. 每个关节的 chunk 前 10 步 action L2（重点看 `wrist_roll` / `gripper`）
2. 夹爪通道方向一致率（预测的"闭合时刻"与真值的符号是否一致）—— 这是检验 §2.1 是否搞错的最灵敏指标
3. 夹爪闭合时刻（onset）误差，单位帧

**在线（唯一能拍板的指标）**：真机每臂 20 次，5 个摆放位置 × 4 次，成功 = 抓起并放进目标，
另外单独记 1st-grasp 成功率。

**决策规则**：混合臂在线成功率 ≥ 纯真机基线 **+2/20**，且没有任何一个位置比基线更差 → 采纳混合；
否则退回纯真机，把算力投到"按真机任务重做仿真场景"（见 §7）。

---

## 6. 不建议做的事

- ❌ 把仿真导到 1000 条再 1:1 混 —— 有效多样性只有 1 条轨迹，多导只是重复（§2.2）。
- ❌ 夹爪没定案就混训 —— 夹爪通道会学成双模，表现是"能过去但抓不住"或者"贴着物体不动"。
- ❌ 只跑纯仿真臂就宣布"仿真能迁移" —— 真机任务不同，纯仿真臂的作用只是量化域差下限。
- ❌ 用 step 对齐来比较不同比例的臂（§3 的 real-pass 说明）。
- ❌ 在 1:1 那臂上用 20 000 步就跟 20 000 步的纯真机比 —— 它的真机曝光只有一半。
- ❌ 只看 train loss 判断混合好坏 —— 仿真是更"光滑"的数据，loss 一定更低，但真机成功率不一定。

---

## 7. 更值得做的一步（如果 §5 结果不理想）

现在的仿真场景（药盒 + 木盒 + place 段落）和真机任务（黑方块 / 盒子）不是同一个任务，
所以混合的上限被任务差异锁死。**把 `collect/` 的场景随机化换成真机场景 —— 同一个黑方块、
同一组相机位姿（用真机视频反推 env/wrist 相机外参）、同样"抓起→放下"的段落**，再重跑 SDG。
这一改的收益比在 0.1~1.0 之间扫比例高一个数量级，而且 `collect.py` 的随机化框架、接触审计、
导出链路都不用动。

---

## 8. 待你回答的一个问题

> 录制开始时（home 姿态）夹爪是**完全张开的**还是**完全闭合的**？
> 看一眼 `inspect/frames/real_ep0_env/file-000_t0.000s.png` 即可。

这一位定下来，§5.2 的 `--gripper-space` 就能定死，后面全是机械执行。

---

## 9. 附：夹爪方向的推理记录（为什么数值定不了案）

- 支持 **0=张开**：95/151 条起点恰为 0.0、115 条末点回到 0.0，是机械停止位；人类示教时习惯
  让夹爪张着待机；ep0 的 `f0-80 grip=0 → f96-144 爬到 25.2 并停住 → f160 回 0` 读作
  "张开接近→合上抓→松开"。
- 支持 **0=闭合**：仿真的 home 就是"爪甲收起"（jaw=0=闭合），而仿真 home 与真机 home 的
  其余 5 个关节只差 5~15°（说明仿真 home 是照真机抄的）；按夹爪几何 1.31 mm/deg 折算，
  0=闭合时真机抓的物体 ≈ 22~29 mm（黑方块合理），0=张开时要 43~74 mm（偏大）。
- 两条都自洽 → 必须看图或看机器人，不能靠数据推。
