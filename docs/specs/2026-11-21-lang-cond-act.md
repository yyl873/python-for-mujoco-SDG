# spec：语言条件 ACT 管线（阶段 1：修视频编码器；阶段 2：用现有数据跑通"行为指令"条件化）

- 日期：2026-11-21
- 状态：**待确认**（确认后才写 plan、才动代码）
- 上游：`docs/specs/2026-11-21-compute-parity-retest.md`（残差 RL 算力对等复测，**未执行**，先排在本 spec 之后）
- 相关事实来源：本 spec §3 全部为本次实测，不是估算

---

## 1. 问题

项目目前**没有语言输入**：ACT 的观测只有 `observation.state(6)` + 两路图像，不是 VLA。
但"直接加个文本编码器"在当前条件下**做不了也不成立**，有三个硬事实：

1. **数据里语言是常数**：`scripts/` 1000 条 episode 的 `instruction` 全部是同一句话
   （`Pick up the medicine box and place it into the wooden box.`），`task` 全部是 `pick_place_v1`。
   → 现在接文本编码器 = 给网络喂常量输入，学不到任何条件依赖。
2. **本机出不了视频**：`cv2 4.13.0` 的 FFMPEG 构建**没有 libx264**，`PATH` 里没有 `ffmpeg`，
   而 `collect/collect.py` 硬编码 `avc1` → 任何数据重采在第一步就崩
   （实测 `RuntimeError: VideoWriter 打不开 ... codec=avc1`）。
3. **本机没有可用的文本模型**：HF hub 缓存为空、本机无 `tokenizer.json`、**网络不可用**
   → CLIP / SentenceTransformer 这条"标准 VLA 路线"现在无法执行。

所以本 spec 只做两件能做的事：**把编码器修好（解锁未来所有数据生成）**，
**用零重采的方式把"语言条件"管线跑通并建立可验证的评测协议**。

## 2. 目标 / 非目标

**目标**
1. **G1 修编码器**：让 `collect/collect.py` 能在本机产出 h264 视频。
   验收 = 采 1 条 episode，`videos/ep_XXXXXX_env.mp4` 与 `_wrist.mp4` 都存在、各 436 帧、pyav 能解码。
2. **G2 造出"多指令"数据（零重采）**：不改任何图像，只在 LeRobot 数据集副本里
   **按 episode 重贴 `tasks`**——用已在数据里的 `randomization.timing_scale` 分成
   slow / normal / fast 三档，对应三句指令。
3. **G3 语言条件策略**：把指令向量拼进 `observation.state`（6 → 6+N），
   用 lerobot ACT 微调（**不改 lerobot 源码**）。
4. **G4 反事实评测协议**：同一初始条件（同 episode_id / seed）分别给 slow 与 fast 指令，
   比较 **episode 时长**与**平均关节速度**，判断语言是否真的被使用。

**非目标（本次明确不做）**
- 空间 grounding（双目标箱 / 左-右指令）→ 另起 spec（A′），需要改 MJCF + 重采 ~500 条（实测 ~3.5 h 渲染）
- 引入 CLIP 或任何外部文本模型（无网络、无本地权重，见 §3）
- 重采任何数据（本 spec 零重采）
- 改动 `scripts/`、现有 `sim156` 基座权重、现有 LeRobot 数据集原件
- 残差 RL / 算力对等复测（那是另一个 spec）

## 3. 现状与证据（本次全部实测）

| 事实 | 实测值 | 出处 |
|---|---|---|
| 单条 episode 物理 + 专家 | **0.71 s** | `collect.py --n 1 --no-render`（/usr/bin/time） |
| 每 episode 帧数 | **436 grab = 872 帧**（env + wrist，640×480） | `ep_000000.json` 的 `n_frames` / `video_frames` |
| 单次 grab（2 相机） | **199–299 ms** | `test_render.CameraRig.grab` 基准 |
| 渲染单条 | ~87–130 s | 上式 × 436 |
| 渲染并行扩展 | 1/4/8/12 进程 → 3.3/10.3/14.7/17.4 grab/s（**12 进程饱和 ~5×**） | 12 进程并发基准 |
| 1000 条全量重采 | **~7 h**（12 进程） | 上式外推 |
| 指令取值 | 1000/1000 同一句 | `scripts/ep_*.json` 统计 |
| `timing_scale` | **0.90–1.099**，182 个不同值，连续 | 前 400 条统计 |
| 视频编码可用性 | `cv2` 无 libx264；**`pyav 15.1.0` 可编 libx264/h264（实测成功）**；conda ffmpeg 有 libx264+libsvtav1 | 本机实测 |
| 网络 | **不可用**；无本地文本模型 | `urllib` 直连超时、`find` 无 tokenizer.json |
| ACT 是否支持任意 state 维度 | **支持**：`modeling_act.py:415/467` 用 `input_features[OBS_STATE].shape` 建投影层 | lerobot 源码 |
| LeRobot 是否支持每 episode 不同 task | **支持**：`meta/episodes/*.parquet` 有 per-episode `tasks` 列；`tasks.parquet` 是 task→index 表 | 数据集元数据 |
| 专家脚本如何定位目标箱 | **按名字查** `target_box`（`grasp/test_grasp.py:184`） | 源码（利好 A′） |

## 4. 约束

1. **离线**：语言编码必须确定性、不依赖网络/权重。
2. **零重采**：本 spec 不渲染任何新 episode（渲染是唯一贵的东西，~100 s/条）。
3. **不动 lerobot 源码**：ACT 的输入维度来自 `input_features`，所以把指令向量 **concat 进
   `observation.state`** 是唯一"零侵入"路径。副作用：CVAE 的 state 编码器也会吃到指令
   （这是**期望**行为——正是条件化生效的地方）。
4. **不动原件**：新数据集写到 `rl/lang_ds/`；`scripts/`、`rl/dagger_ds/` 只读。
5. **保留现有结论**：不改 `sim156`、不改 `sim156_plus_dagger3`、不改 `rl/eval/target_*`。

## 5. 备选方案与取舍

### 5.1 编码器修复

| | 做法 | 取舍 |
|---|---|---|
| **A** | 把 `collect/collect.py` 的 `_VideoWriter` 换成 **pyav libx264**（进程内） | **选它**。无外部进程、无 PATH 依赖；实测可用；接口与现有 `write()/release()` 最接近 |
| B | 子进程调 conda 的 `ffmpeg`（有 libx264） | 可行，但引入进程管理与临时帧目录，且 `mix_dataset.py` 已证明这条路易踩格式坑 |
| C | 改回 `mp4v` | **拒绝**：`test_render.py:1357` 明确写了训练侧需要 h264，改回会破坏 lerobot 读取 |

### 5.2 语言编码方式（无 CLIP 可用）

| | 做法 | 取舍 |
|---|---|---|
| **A** | **手工语义特征** `[is_slow, is_normal, is_fast]`（3 维 one-hot）+ 预留 `[is_left, is_right]` 两位 | **选它**。离线、可复现、**同一段代码以后能直接扩到空间指令** |
| B | 只用 2 维 `[slow, fast]` | 更省，但 normal 档没有锚点，难以判断"是否只是回到默认" |
| C | 指令字符串哈希成随机向量 | 拒绝：没有语义结构，迁移到 A′ 时要从头再来 |
| D | CLIP / SentenceTransformer | **现在不可行**（无网络、无本地权重） |

> 诚实标注：**one-hot 不是"语言理解"**。它是一个受控的指令标识，用来做**反事实可检验**的条件化。
> 真正的语言泛化（同义句、未见组合）必须等有文本编码器之后，那是另一个 spec。

### 5.3 三档指令怎么来

| | 做法 | 取舍 |
|---|---|---|
| **A** | 用已有 `randomization.timing_scale` 分档：≤0.95 = slow、0.95–1.05 = normal、≥1.05 = fast，重贴 `tasks` | **选它**。零重采。风险见 R1 |
| B | 强制极端速度（0.5×/1.4×）重采 ~300 条（~2.1 h 渲染） | 效果更明显，但破"零重采"，且本 spec 的目标是跑通管线 |
| C | 不按速度，按 `spawn.y` 分档（近/远）贴不同指令 | 指令与几何耦合，语义上不像"指令"；放弃 |

## 6. 选定方案

**5.1-A + 5.2-A + 5.3-A**，两阶段：

- **阶段 1（编码器）**：`collect/collect.py` 的 `_VideoWriter` 内部换 pyav；保持对外接口不变。
- **阶段 2（管线）**：
  1. `rl/relabel_tasks.py`：从 `sim156_plus_dagger3` 生成副本 `rl/lang_ds/v1`，
     按 `timing_scale` 把每条 episode 的 `tasks` 改成三句之一，`tasks.parquet` 扩到 3 行；
     **同时把 `observation.state` 从 6 维扩到 9 维**（末尾拼指令 one-hot），并重算 `stats.json`。
  2. 用 lerobot ACT 微调（`--policy.pretrained_path` 指向 sim156 基座；lr 1e-5；≤3000 步；
     因为只是加条件输入，不需要 10k 步）。
  3. `rl/eval_lang.py`：固定 episode/seed，**切换指令**跑两条（slow 指令 vs fast 指令），
     记录成功率、episode 时长、平均 |qvel|、关节目标速度。

## 7. 接口 / 数据 / 文件变化

| 文件 | 变化 |
|---|---|
| `collect/collect.py` | `_VideoWriter` 换 pyav（对外 `write/release` 不变）；**不改任何默认参数** |
| `rl/relabel_tasks.py` | **新增**：生成 `rl/lang_ds/v1`（dataset 副本 + 3 条 task + state 扩维 + stats） |
| `rl/lang_ds/v1/` | **新增**（不改原件） |
| `rl/eval_lang.py` | **新增**：反事实指令评测 |
| `rl/eval/lang_*.json` | **新增** |
| `/home/yyl/lerobot/outputs/train/act_lang_v1/` | **新增** ckpt（不动 act_dagger_ft_1000） |

## 8. 失败模式与回滚

| 风险 | 症状 | 处理 |
|---|---|---|
| **R1 timing_scale 效果太小**（我的主要担心） | 三档行为差异只有 ±10%，策略学不出差异，反事实测不出 | **这本身就是结论**：按 §9 判据判"语言未被使用"，然后升级到 A′（极端速度或空间指令）。不硬凑 |
| R2 state 扩维后 stats 不匹配 | 归一化错、loss 不降 | `relabel_tasks.py` 必须重算 stats（新维度用 one-hot 的均值/方差），并在训练前打印 state 统计核对 |
| R3 指令与 episode 完全共线（指令→速度一一对应） | 模型可能只靠图像看出速度而不看指令 | 反事实测试**必须换指令不换场景**，这正是本 spec 的核心评测；仅此一条能排除混杂 |
| R4 pyav 写出的 mp4 lerobot 读不了 | 数据集校验失败 | 阶段 1 的验收包含"用 lerobot 读回来并解码一帧" |
| R5 微调把基座带坏 | 单任务成功率从 86.7% 掉下来 | 阶段 2 附带**回归检查**：用同一 ckpt 跑 30 次原始指令评测，clean 不应显著低于基座（>15pp 掉幅即停） |

**回滚**：`git checkout -- collect/collect.py`；`rm -rf rl/lang_ds rl/eval/lang_* /home/yyl/lerobot/outputs/train/act_lang_v1`。原件一律不动。

## 9. 验收标准（可执行 + 可观察）

**阶段 1：编码器**
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python collect/collect.py \
    --n 1 --start 9000 --instruction "test only"
# 通过：scripts/ep_009000.{npz,json} + videos/ep_009000_env.mp4 + _wrist.mp4 都存在
#      且 pyav 能打开并解码，帧数 = 436
# 失败：仍报 VideoWriter/编码器错误，或帧数不符 → 停
# 清理：rm -f scripts/ep_009000.* videos/ep_009000_* scripts/index/ep_009000.json && 重建 index
```

**阶段 2a：数据集副本**
```bash
python3 rl/relabel_tasks.py --out rl/lang_ds/v1
# 通过：tasks.parquet 有 3 行；episodes parquet 的 tasks 列出现 3 种取值；
#      observation.state 维度 6 → 9；stats.json 里 state 的 mean/std 长度 = 9；
#      三档 episode 数量都 ≥ 100
```

**阶段 2b：训练（≤3000 步）**
```bash
lerobot-train --policy.type=act --dataset.repo_id=...（本地路径） \
  --policy.pretrained_path=<sim156 基座> --batch_size=8 --steps=3000 ...
# 通过：ckpt 落盘；loss 与 DAgger 微调同量级（不爆 NaN）
```

**阶段 2c：反事实评测（本 spec 的核心）**
```bash
python3 rl/eval_lang.py --ckpt <新 ckpt> --episodes 100-119 --seeds 2
# 通过（语言被使用）：同一 (episode, seed) 下，
#   给 slow 指令的平均 episode 时长 ≥ 给 fast 指令的 1.05 倍，且方向在 ≥80% 的配对上成立
# 不通过（语言未被使用）：差异 < 5% 或方向随机
#   → 这就是负结果，写进报告，升级到 A′（空间指令），不改判据
```

## 10. 开放问题（不阻塞本 spec）

1. 阶段 2b 的 3000 步够不够？加条件输入后可能需要更多；先按 3000 步跑，看 loss 与反事实差异再定。
2. 三档 episode 数量是否均衡（`timing_scale` 是连续均匀分布 → 中档会自然更多）；
   若不均衡，是否需要对中档下采样（会影响数据量，需在 plan 里给数字）。
3. 一句 `instruction` 在 `scripts/` 层面是全局的（`--instruction` 是整个 run 的参数）；
   本 spec 不重采，所以只在 **LeRobot 副本**里重贴标签——这会与 `scripts/` 不一致，
   必须在 `rl/lang_ds/v1/README` 里写清来源与转换脚本，避免以后误认。
4. 以后有网络 + 文本权重时，5.2-A 的三维 one-hot 换成 CLIP 嵌入（`observation.state` 6+384），
   其余管线不变——这是本 spec 选 5.2-A 的主要理由。