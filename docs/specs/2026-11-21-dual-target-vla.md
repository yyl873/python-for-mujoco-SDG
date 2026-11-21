# spec：双目标箱空间 grounding（A′）—— 让 ACT 学会"按指令把药盒放进哪个箱子"

- 日期：2026-11-21
- 状态：**待确认**（确认后才写 plan、才动代码）
- 取代：`2026-11-21-lang-cond-act.md` 的阶段 2（B：`timing_scale` 行为指令，天花板仅 ±10%，
  用户已选择放弃）；**该文档的阶段 1（pyav 视频编码器修复）原样继承为本 spec 阶段 1**
- 未执行：`2026-11-21-compute-parity-retest.md`（残差 RL 算力对等复测）

---

## 1. 问题

项目没有语言输入，而现有数据里语言是**常数**（1000/1000 都是同一句 instruction），
所以"加文本编码器"在当前数据上等于喂常量，不构成 VLA。

要做出**可检验**的语言条件，必须让同一个场景里存在**两个合理的放置目标**，
并且**只有指令能区分它们**。本 spec 就做这件事：场景里放两个目标箱，指令说左还是右。

**为什么不用"速度指令"（上一个 spec 的 B）**：`timing_scale` 只有 0.90–1.099（±10%），
行为差异太小，很可能既学不出也测不出，天花板太低。空间目标则是一个**二元可判**的指标。

## 2. 目标 / 非目标

**目标**
1. **G1 修视频编码器**：`collect/collect.py` 在本机能产出 h264（pyav libx264），
   验收 = 1 条 episode 出 2 个 mp4、各 436 帧、pyav 可解码、lerobot 可读。
2. **G2 场景加第二个目标箱**：`target_box_b`，候选位置 `[0.29, +0.132, 0.025]`（镜像），
   **IK 实测可达（末端误差 0.4 mm）**。
3. **G3 目标参数化**：专家脚本 / 环境 / 判据都从"按名字查 `target_box`"升级为"按参数选目标箱"，
   **默认值仍是 `target_box`**（旧链路行为不变）。
4. **G4 采两个任务的数据**：每任务 **250 条**，共 **500 条**，同一新场景、指令分别为
   "…into the **left/right** wooden box"。
5. **G5 语言条件 ACT**：指令向量拼进 `observation.state`（6 → 8，两箱 one-hot），
   用 `sim156` 基座微调，**不改 lerobot 源码**。
6. **G6 反事实评测（核心）**：同一 `episode_id/seed`、同一场景，**只换指令**，
   主指标 = **指令跟随准确率**（放进的箱子是否等于指令指定的箱子），
   **chance = 50%**。

**非目标**
- 速度/风格条件化（已放弃）
- CLIP / 外部文本模型（无网络、无本地权重）
- 三箱以上、或"抓哪个物体"类指令（先只做放置目标这一维）
- 真机、残差 RL、算力对等复测
- 改动 `scripts/`、现有 `sim156` 基座、现有 LeRobot 数据集原件

## 3. 现状与证据（本次实测，非估算）

| 事实 | 值 | 出处 |
|---|---|---|
| 现有目标箱 | `target_box` @ `[0.29, −0.132, 0.025]` | `model/scene.xml:119` |
| 第二个箱候选位置可达性 | 镜像 `[0.29,+0.132,0.06]` → **IK 误差 0.4 mm**；`[0.24,+0.132]` 0.8 mm；`[0.34,+0.132]` 0.4 mm；`[0.36,−0.132]` 0.5 mm | `ik/test_ik.solve_ik_to_xyz` 实测 |
| 专家如何定位目标 | **按名字**：`mujoco.mj_name2id(..., "target_box")` | `grasp/test_grasp.py:184` |
| 环境/判据如何定位目标 | `self.arm.tbox` → `_tbox_xy` | `rl/env_residual.py` |
| 单条物理+专家 | **0.71 s** | `collect.py --no-render` 实测 |
| 每 episode 帧数 | **436 grab = 872 帧**（640×480 × 2 相机） | `ep_000000.json` |
| 单次 grab | 199–299 ms | `test_render.CameraRig.grab` 基准 |
| 渲染并行扩展 | 12 进程饱和 **~5.2×**（17.4 grab/s） | 12 进程并发基准 |
| **500 条采集预算** | **~3.5 h**（12 进程） | 500×436 / 17.4 |
| 视频编码 | `cv2` 无 libx264；**`pyav 15.1.0` 可编 libx264（实测）**；conda ffmpeg 有 libx264 | 本机实测 |
| 网络 / 文本模型 | 均不可用（无网络、HF 缓存空、无 tokenizer.json） | 实测 |
| LeRobot 每 episode 独立 task | **支持**：`meta/episodes/*.parquet` 有 per-episode `tasks` 列 | 数据集元数据 |
| ACT 接受任意 state 维度 | **支持**：`modeling_act.py:415/467` 按 `input_features[OBS_STATE].shape` 建投影层 | lerobot 源码 |
| 旧数据（1000 条）的场景 | **只有 1 个木箱** | `video_frames`/MJCF |

## 4. 约束

1. **数据不能混用旧场景**：加了第二个箱子之后，旧 1000 条的图像里没有它。
   混用会让"同一指令对应两种视觉场景"，所以**训练集必须是新场景重采的 500 条**。
   旧 1000 条只作为**回归对照**（单箱场景下基座 86.7% 的历史数字），不参与训练。
2. **离线**：指令编码必须确定性（两箱 one-hot 2 维），不依赖网络/权重。
3. **不改 lerobot**：指令向量 concat 进 `observation.state`（6→8）；CVAE 也会吃到它（期望行为）。
4. **默认参数不变**：专家/环境/判据的默认目标仍是 `target_box` → 旧链路（`eval_ab`、`view_policy`、
   残差训练）行为不受影响。
5. **不动原件**：新数据集写 `rl/lang_ds/dual/`；`scripts/` 只读（新数据用 `--start 2000+` 或独立目录）。
6. **样本量**：每任务 250 条（ACT 在 ~50 条就能学单任务；DAgger 那轮 480 条即够微调）。

## 5. 备选方案与取舍

### 5.1 第二个箱的位置

| 候选 | IK 误差 | 取舍 |
|---|---|---|
| **镜像 `[0.29, +0.132, 0.025]`** | 0.4 mm | **选它**：视觉上左右分明，最像"空间指令"；两箱关于 y=0 对称，chance 基线干净 |
| 同侧 `[0.36, −0.132, 0.025]` | 0.5 mm | 前后/远近区分，语言上是 "front/back"，不如左右直观 |
| `[0.24, +0.132]` | 0.8 mm | 与现有箱投影重叠风险更高，放弃 |
| `[0.34, +0.132]` | 0.4 mm | 也可用；先按镜像做，pilot 不行再换 |

> ⚠️ IK 可达 ≠ 专家能完成任务。**pilot 必须实测专家成功率**（见 §9 阶段 2），
> 不通过就换位置，这是本 spec 最大的技术风险。

### 5.2 数据规模

| | 做法 | 渲染成本 | 取舍 |
|---|---|---|---|
| **A** | 新场景重采 **500 条**（每任务 250） | **~3.5 h** | **选它**。样本量与 DAgger 那轮同量级，够微调 |
| B | 重采 1000 条（每任务 500） | ~7 h | 更稳但翻倍；收益不确定 |
| C | 旧 1000 条当任务 A + 新采 500 条当任务 B | ~3.5 h | **拒绝**：旧数据图像里没有第二个箱子，观测不一致 |
| D | 只采 200 条（每任务 100） | ~1.4 h | 可作为 pilot 规模；正式训练可能欠拟合 |

### 5.3 语言编码

| | 做法 | 取舍 |
|---|---|---|
| **A** | 两箱 **one-hot 2 维** 拼进 `observation.state`（6→8） | **选它**。离线、可复现、零侵入 |
| B | 手工语义特征 `[is_left, is_right]` | 与 A 等价（one-hot 就是它），不必分开 |
| C | 指令字符串哈希 | 拒绝：无语义结构，换任务要重来 |
| D | CLIP | 不可行（无网络/无权重）；**有网络后把 2 维换成 384 维嵌入，其余管线不变** |

> 诚实标注：one-hot 是**受控的指令标识**，不是"语言理解"。它能回答的问题是
> **"策略是否真的按指令走"**（可检验、可反事实），不能声称"理解了同义句/未见组合"。

## 6. 选定方案

**5.1-A（镜像）+ 5.2-A（500 条）+ 5.3-A（one-hot 2 维）**，分六个阶段（见 plan）。

## 7. 接口 / 数据 / 文件变化

| 文件 | 变化 | 兼容性 |
|---|---|---|
| `model/scene.xml` | **新增** `target_box_b` body（镜像位置，同几何同材质） | 旧 episode 不含它 → 只影响新采集 |
| `collect/collect.py` | ①`_VideoWriter` 换 pyav ②新增 `--target-box`（默认 `target_box`） | 默认行为不变 |
| `grasp/test_grasp.py` | 目标箱 id 由名字参数化（默认 `target_box`） | 默认行为不变 |
| `rl/env_residual.py` | 新增 `target_body` 参数；`_tbox_xy`/判据跟着走 | 默认 `target_box` |
| `rl/eval_ab.py` | 新增 `--target-box`、`--instruction`（转发给 env） | 默认不变 |
| `rl/relabel_tasks.py` | **新增**：生成 `rl/lang_ds/dual`（3 或 2 条 task + state 6→8 + 重算 stats） | 新目录 |
| `rl/eval_lang.py` | **新增**：反事实指令评测（同场景换指令） | 新文件 |
| `rl/lang_ds/dual/`、`rl/eval/lang_*.json` | **新增产物** | 不覆盖原件 |
| `/home/yyl/lerobot/outputs/train/act_lang_dual/` | **新增** ckpt | 不动 `act_dagger_ft_1000` |

## 8. 失败模式与回滚

| 风险 | 症状 | 处理 |
|---|---|---|
| **R1 专家放不进第二个箱** | pilot 采集 `accepted=false` / `reject_reasons` 非空 | 换位置（`[0.34,+0.132]` / `[0.36,−0.132]`）或调整箱高；**连换 3 个位置都不行 → 停止，回到 spec** |
| **R2 两箱互相干扰** | 搬运途中手臂撞到另一个箱 → `off_table`/接触违规上升 | 加大两箱间距（镜像 y 改 ±0.15）；pilot 必测 |
| R3 state 扩维后 stats 不匹配 | loss 不降或 NaN | `relabel_tasks.py` 显式重算 stats，训练前打印核对 |
| R4 pyav 写的 mp4 lerobot 读不了 | 数据集校验失败 | 阶段 1 验收包含"lerobot 读回并解码一帧" |
| R5 指令与视觉共线（模型靠图像而非指令） | 换指令后行为不变 | **这正是核心评测**：同场景换指令；若跟随准确率 ≈50% → 如实判"未使用语言" |
| R6 微调把单任务能力带坏 | 新 ckpt 在两个箱都要求"任一"时也不如基座 | 附带回归：跑 30 次原始单箱场景，clean 不得比基座低 >15pp |
| R7 采集被 SIGFPE 打断 | 进程崩 | 沿用 `--start` 续跑 + 外层重试（脚本里写死） |

**回滚**：
```bash
git checkout -- model/scene.xml collect/collect.py grasp/test_grasp.py rl/env_residual.py rl/eval_ab.py
rm -rf rl/lang_ds/dual rl/eval/lang_* /home/yyl/lerobot/outputs/train/act_lang_dual
rm -f scripts/ep_0020*.{npz,json} videos/ep_0020*_*.mp4     # 新采集（编号段专属）
```

## 9. 验收标准（可执行 + 可观察）

**阶段 1：编码器**
```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python collect/collect.py \
    --n 1 --start 9000 --no-keep-failed
# 通过：videos/ep_009000_env.mp4 与 _wrist.mp4 都在；pyav 打开后 frames == 436
# 失败：仍报编码器错误 / 帧数不符 → 停
```

**阶段 2：第二箱 pilot（**决定 A′ 是否可行**）**
```bash
# 每目标各采 5 条
... collect.py --n 5 --start 2010 --target-box target_box_b --instruction "...right..."
# 通过：5 条全部 accepted=true，且 place 阶段无 reject；两箱间距下无异常接触
# 失败：见 R1/R2 → 换位置重试，最多 3 个位置
```

**阶段 3：正式采集 500 条（~3.5 h，12 进程）**
```bash
# 250 条 target_box（left）+ 250 条 target_box_b（right），编号段 2000-2499
# 通过：scripts/ep_002*.json 共 500 个，accepted 率 ≥ 90%，两档各 250
# 失败：accepted < 90% → 先查 reject_reasons，不许直接拿去训
```

**阶段 4：数据集副本**
```bash
python3 rl/relabel_tasks.py --out rl/lang_ds/dual
# 通过：tasks.parquet 2 行；episodes 的 tasks 出现 2 种取值（各 250）；
#      observation.state 维度 6→8；stats.json state 长度 8
```

**阶段 5：微调（≤5000 步）**
```bash
lerobot-train --policy.type=act --policy.pretrained_path=<sim156> --steps 5000 --batch_size 8
# 通过：ckpt 落盘、loss 与 DAgger 微调同量级；不爆 NaN
```

**阶段 6：反事实评测（本 spec 核心，判据预注册）**
```bash
python3 rl/eval_lang.py --ckpt <新 ckpt> --episodes 100-119 --seeds 2
# 主指标：指令跟随准确率 = 放进"指令指定箱"的次数 / 总次数
# 通过（语言生效）：配对跟随准确率 ≥ 80%，且 Wilson 95% 下界 > 50%（chance）
# 次级（不崩能力）：正确指令下的 succ ≥ 单箱基座 86.7% − 10pp = 76.7%
# 不通过：跟随准确率 ≈ 50%（≈ 抛硬币）→ 判"语言未被使用"，负结果写进报告，禁止改判据
```

## 10. 开放问题

1. 两箱的**视觉区分度**够不够？材质/颜色是否要不同（`material` 已可参数化）——
   若两箱长得一样，模型可能只能靠位置区分，而不是靠"指令 + 空间关系"。pilot 后决定是否给第二个箱换材质。
2. 500 条（每任务 250）是否足够？若 loss 收敛慢或跟随准确率低，再加 250 条（+1.75 h）。
3. 指令文本措辞：`"...into the left/right wooden box"` 还是
   `"...into the wooden box on the left/right"`——对 one-hot 编码无差别，但要写进数据卡。
4. 评测集的 held-out 场景（episodes 100-129）在新场景下需要重新确认专家也能成功
   （新场景多了一个箱子，`scripts/` 里 100-129 的轨迹是单箱场景录的，**不能直接回放**）
   → 评测必须用**新采集的 held-out 编号段**（计划里另留 2400-2449 中的一部分作 held-out）。

---

## 11. 执行记录（2026-11-21 晚，实测）

| 阶段 | 状态 | 实测数字 |
|---|---|---|
| 1 编码器（pyav/libx264） | ok | 采集 100 条，出 184 个 mp4（92 条正样本 × 2 相机 + 负样本） |
| 2 第二箱 pilot | ok | 实际落地位置 **(0.206, +0.245, 0.025)**（不是 §5.1 的镜像位，见下） |
| 3 采集 | ok | 编号段 2000–2099 采 100 条，**验收通过 92 条**（`.lang_ids.txt`），另 8 条 `reject_reasons` 非空 |
| 4 数据集副本 | ok | `rl/lang_ds/v1`：**92 条 / 41922 帧 / 604 MB**；`observation.state` **8 维**（6 关节 + `instr_left`/`instr_right`）；`tasks.parquet` **2 行**；episodes 的 tasks 分布 **45 left / 47 right**；`meta/sim_provenance.json` 的 `task_order` 与 `tasks.parquet` 同序；回读 `len=41922 episodes=92 state(8,) action(6,) img=ok stats=ok`；**图像写入失败 0 条** |
| 5 加宽基座 | ok | `act_sim156_lang8/pretrained_model`：config 的 state=8（补 0 列），ACT 可直接加载（52 M 参数，无 shape 冲突） |
| 5 微调 5000 步 | **段错误（rc=139）** | 用户 GPU 终端：**跑到 839/5000 步（4:59，2.87 step/s）段错误、核心已转储**，此前 loss 正常（l1 0.109 / kld 0.002），退出时 20 个泄漏信号量 → 死在 dataloader worker 侧；当时 `save_freq=1000` → **一步 ckpt 都没落**。沙箱 CPU 实测 6.5–8 s/步（不可作为方案） |
| 5b h264 备用数据集 | ok | `rl/lang_ds/v1_h264`（`--vcodec h264`）：92 条 / 41922 帧 / **317 MB**，每相机**单文件** 41922 帧（AV1 那版是 20409+21513 / 40529+1393 的怪切分），回读 `img=ok stats=ok`、图像写入失败 0、导出 4 分 43 秒 |
| 6 反事实评测 | 框架已跑通，**正式结果待训练** | 框架首跑修掉 3 处 bug（见 §11.1）；控制组 4 条 rollout / 14 s；76 条约 4.5 min（CPU）。判据预注册，不许看完数据再改 |

**训练段错误的头号嫌疑（编码栈，不是模型）**：`rl/lang_ds/v1` 的 mp4 是 **AV1(libdav1d)**
（把 `file-000.mp4` 的 codec 打出来确认过），而 4 worker × prefetch 4 并发解码；本机 AV1 栈有前科：
`collect/export_lerobot.py:139` 记着 svt-av1 **编码** 92 条时同样 **rc=139**，而历史上训练成功的
`dagger_v1` 是 **h264**。已建好 h264 版数据集作为对照，并按此重写训练脚本：
`SAVE_FREQ` 默认 1000→**250**、`ATTEMPTS=3` 自动 `--resume` 续跑、`WORKERS`/`BACKEND` 可切。
先跑 h264 版（单变量：只换编码），崩溃点 839 步已被 1200+ 步覆盖。

**held-out 段已补采（本 spec §10.4 的硬要求）**：原默认 `--episodes 2060-2079` 里
大部分 episode **已经进了训练集**（`--lang-onehot` 导出了全部 92 条），拿它当 held-out
等于在训练场景上做反事实，跟准确率会被顶高。补采 **2400–2419**（左 10 + 右 10）：
**19/20 验收通过**（2410 落 `no_grasp_candidate(3次)/no_lift(-17.6mm)/not_in_box`，
与训练段 8 条被拒同一失败类），编号写在 `.lang_holdout_ids.txt`，
母本/index/视频齐全（20 npz、38 mp4、20 个 index）。评测用这 19 条初始条件。

**卡死根因（本次修掉）**：导出不是算错，是**进程随会话被 SIGTERM 带走**——第一次跑到
`[57/92]`、`Error writing image` **0 条**、无 traceback 就断。修法：`setsid nohup ... < /dev/null &`
起独立会话 + 用日志进度行轮询（`pgrep -f` 会探到自己的 cmdline）。复跑：**20:35:59 → 20:41:03，
5 分 4 秒，92/92，rc=0**。

**偏差记录（别再当成镜像位）**：`model/scene.xml` 里 `target_box_b` 的真实 `pos` 是
`(0.206, 0.245, 0.025)`，不是 §5.1 选的镜像 `(0.290, +0.132)`。采集（`--target-box target_box_b`）
与评测（`rl/eval_lang.py` 的 `BOX_XY`）都按真实值走，数据与判据自洽；但 §5.1 的"两箱几何镜像
→ chance 基线干净"论证**不再严格成立**，报告里要注明 chance=50% 只是名义基线。

**下一步命令（用户终端，有 GPU）**：
```bash
cd /home/yyl/deepseek-harness
# 微调（先跑 h264 版：单变量只换编码；≈30 min @ 2.87 step/s，输出 ~/lerobot/outputs/train/act_lang_dual）
DS_ROOT=$PWD/rl/lang_ds/v1_h264 STEPS=5000 bash rl/run_lang_train.sh
# 若仍 139（说明与编码无关）→ 换回 AV1 + 主进程解码：
# DS_ROOT=$PWD/rl/lang_ds/v1 WORKERS=0 STEPS=5000 bash rl/run_lang_train.sh
# 再不行还有一档：BACKEND=pyav（换解码后端，不改数据）
# 反事实评测：19 条训练集外场景 × 2 seed × 2 指令 = 76 条 rollout（重试壳，崩了自动续）
bash rl/run_lang_eval.sh
```

### 11.1 评测链路的首次实跑与三处真 bug（2026-11-21 夜，沙箱 CPU）

训练之前先把「训练完之后才用得上」的评测链路跑通，避免训练 30 分钟后再发现评测跑不起来。
实跑暴露的问题（都已修 / 已记）：

| # | 问题 | 实测症状 | 处置 | 状态 |
|---|---|---|---|---|
| 1 | 加宽基座的 config 带 `names` | `rl/eval_lang.py` 一启动就 `DecodingError: 'input_features.observation.state': The fields names are not valid for PolicyFeature` | `act_sim156_lang8/pretrained_model/config.json` 在 `input_features.observation.state` 里写了 8 个名字，而 lerobot 0.6.2 的 `PolicyFeature` 只有 `(type, shape)`。训练侧不受影响（`--policy.pretrained_path` 只吃权重，config 由 CLI+数据集重建）→ **能训不能用**。修法：`tools/replay_policy.py` 新增 `load_policy_config()`，把不认的键剥掉后用临时副本解码，权重/归一化仍读原目录，不改进 ckpt 文件 | 修好 |
| 2 | 加宽基座**不能直接推理** | 8 维状态进管线 → `RuntimeError: The size of tensor a (8) must match the size of tensor b (6)` | 同一 ckpt 的 `policy_preprocessor.json` 里 normalizer 的 `observation.state` 仍是 `shape:[6]`（safetensors 里 mean/std 也只有 6 维），加宽只改了策略输入层与 config。它是**权重初始化**，评测必须指向微调后的 ckpt | 已定性 |
| 3 | 「崩溃自动续跑」原本是坏的 | `--resume=true` 单独用 → `ValueError: A config_path is expected when resuming a run`（rc=1），3 次重试全部秒退 | lerobot 0.6.2 的 `_resolve_resume_checkpoint()` 要求同时给 `--config_path` 指向上次的 `train_config.json`。已改为 `--resume=true --config_path="$OUT/checkpoints/last/pretrained_model/train_config.json"`；本机验证：跑 6 步落 `000006` → 续跑改成 12 步 → 新增 `000009`/`000012`，**没有从头重训** | 修好并验过 |
| 4 | `mj_step` 偶发 SIGFPE 会打断整轮评测 | 第一次评测死在第 2 条 rollout（`rc=136` 浮点异常、核心转储）；**同一条命令再跑则 4 条全过** → 是偶发，不是确定性 | 新增 `rl/run_lang_eval.sh`：每完成一条 rollout 就落盘 + `--resume` 跳已完成 + 连续两轮没有新行就停（那才是确定性触发，需换场景） | 修好 |

评测框架本身已跑通（控制组：6 维基座 `act_sim156_pre` + `--no-instr`）：
**4 条 rollout / 14 s ≈ 3.5 s 一条**（CPU，`max_seconds=15` 上限内提前判停）→
76 条（19 场景 × 2 seed × 2 指令）**约 4.5 分钟**，CPU 足够，不必抢 GPU。

训练侧速率复核（CPU，batch 8、4 worker、h264 数据集）：**6.0–6.5 s/step**（5000 步 ≈ 8.5 h），
与上轮记录的 6.5–8 s/step 一致 → CPU 仍然不是方案；GPU 2.87 step/s 下 5000 步 ≈ 30 min。

### 11.2 第二轮实跑（2026-11-21 21:52，用户 GPU 终端 + 沙箱 CPU）

**（a）段错误与视频编码无关**：换成 h264 数据集（`rl/lang_ds/v1_h264`）后 GPU 上**仍然 rc=139**，
这次因为 `SAVE_FREQ=250` + 续跑参数已修好，三次 attempt 分别推进到 ~1000 / 1250 / **1473 步**，
ckpt 落了 `001000`、`001250`。→ 结论：**不再指望它不崩**，改成「让它崩了能续 + 自动降档」：

| 档 | 参数 | 触发条件 |
|---|---|---|
| 0 | 配置原样（4 worker） | 起始档 |
| 1 | `--num_workers=0`（解码退回主进程，没有 worker 进程） | 同档连续崩 3 次 |
| 2 | `num_workers=0` + `--dataset.video_backend=pyav` | 再连续崩 3 次 |

`ATTEMPTS` 默认 60：即使每 300 步崩一次，也能靠续跑磨到 5000 步（每次崩最多丢 250 步 ≈ 1.5 min）。

**（b）评测被 `mj_step` SIGFPE 打断，根因已定位**：`python -X faulthandler` 直接指到
`grasp/test_grasp.py:256 in step → reset → __init__ → env_residual.py:135`，
即 **`mj_step` 在 env 构造期的 `reset(settle=1200)` 里**偶发浮点异常（rc=136）；
同一条命令第二次跑就过 → 偶发、与场景无关，但**一次崩溃会把整轮 76 条评测带走**。
处置：`rl/eval_lang.py` 改成**每条 rollout 一个子进程**（`_spawn_rollout`，`--inproc` 可关），
子进程崩只丢那一条并自动重试（`--crash-retries` 默认 3），仍不通才记 `crashed=true`
（不进 follow 分子分母、但进 ITT 分母）。整轮 76 条实跑：**467 s（7.8 min），rc=0**，
其中 2 条是三次重试后仍崩（`ep2417 s1 B`、`ep2418 s1 B`）。

**（c）1250 步中间 ckpt 的反事实评测（框架已验证，结论待 5000 步）**：

| 项 | 数值 |
|---|---|
| 落点分布 | 左箱 27 / 右箱 22 / 都没进 27（共 76） |
| follow_acc（分母=落进任一箱） | **27/49 = 55.1%**，Wilson [41.3%, 68.2%]，对 chance=50% 的二项检验 p=0.568 |
| A(左指令)落左箱 / B(右指令)落右箱 | 15/37 · 12/37（都低于 50%，说明还有个**位置偏好**而非指令跟随） |
| 配对全对 | **0/27** |
| 判定 | 语言未被使用（≈抛硬币） |

→ 1250 步只有 5000 步的 1/4，「语言没生效」符合欠训练预期，**不能当结论**；
结果留档 `rl/eval/lang_counterfactual_ckpt1250.json`，正式结果等 5000 步 ckpt 用同一条命令重跑。

### 11.3 第三轮：崩溃换了一种类型（CUDA 内部断言，不是 SIGSEGV）

| 项 | 实测 |
|---|---|
| 症状 | 续跑到 **step 55/3750**，进度条卡在同一步约 **20 s**（速度 2.93 → 1.19 step/s），随后 `RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED at "c10/cuda/impl/CUDAGuardImpl.h:34"`（rc=1，有 traceback，在 `accelerator.backward(loss)` 里） |
| 与 139 的区别 | 前几次是**无 traceback 的段错误**（worker 侧、20 个泄漏信号量）；这次是 autograd backward 的 C++ 断言，且**先卡 20 s 再报错** |
| 已排除 | 续跑配置本身正常：`train_config.json` 里 `policy.device=cuda`、`batch_size=8`、`num_workers=4`、`seed=1000`、`steps=5000`；数据侧在 CPU 上连取 400 batch（h264 / AV1 各一轮）都不崩 |
| 剩下两种可能 | ① **驱动/供电面**：GPU 卡死后回到驱动层（`dmesg` 找 Xid）；② **torch 轮子面**：`2.7.1+cu126` 对本机 **sm_120（RTX 5060 Laptop）**覆盖不全，某个 backward 算子落到错误设备 |

**新增 `rl/probe_gpu_health.py`（判读工具，1~2 分钟，只碰显卡、不碰数据集和 dataloader）**：
A 段纯 matmul + backward；B 段用真 ckpt 跑 ACT 前向 + backward（batch 8，与训练同 shape）。

| 结果 | 结论 | 下一步 |
|---|---|---|
| A 段就炸 | 显卡/驱动面 | `dmesg \| tail -30 \| grep -iE 'xid\|reset'`、`nvidia-smi -q \| grep -iE 'xid\|throttle'` 的尾部贴回来 |
| A 过、B 炸 | torch/lerobot 算子面 | `CUDA_LAUNCH_BLOCKING=1` 复跑定位算子；仍崩就换 cu128 及以上的 torch 轮子（对 sm_120 支持更全） |
| A、B 都过 | 计算面没问题 | 嫌疑落在 dataloader worker / 视频解码 → 用 `WORKERS=0` 跑训练 |

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
    rl/probe_gpu_health.py --steps 200
```
### 11.4 第三轮加固与 CPU 兜底（22:06，另一窗口记录；与 §11.3 同一现象）

续跑（从 `001250` 起、剩余 3750 步）跑到 **55 步**时报：
`RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED at "c10/cuda/impl/CUDAGuardImpl.h":34`，
且进度条在同一行卡了约 20 s 才报错（2.93 → 1.19 step/s）。这是 **CUDA 侧**错误
（`CUDAGuardImpl`），不是数据/编码/训练逻辑问题；同一进程此前已多次 SIGSEGV，
反复崩会留下**粘滞的 CUDA context**，只有重启或 `nvidia-smi --gpu-reset` 才清。

已排除的：`checkpoints/last/pretrained_model/train_config.json` 里 `policy.device=cuda`、
`num_workers=4`、`steps=5000` 都正常（不是 resume 把设备/步数带坏）。

脚本对应加固（`rl/run_lang_train.sh`）：
- 档位阶梯扩到 4 档：档0 原样 → 档1 `workers=0` → 档2 `+pyav` → 档3 `+CUDA_LAUNCH_BLOCKING=1`；
- **任何 rc≠0 都重试**（原来只重试 139/136，这条 CUDA assert 会直接放弃）；
- 每次 attempt 的完整日志留 `rl/logs/attempt_N_HHMMSS.log`；
- 识别到 CUDA 类错误自动打印驱动面证据（`nvidia-smi --query-gpu`、`Xid`/`Pending`/`Retired`、
  `journalctl -k` 里的 nvrm/nvidia 行），省得再手翻；
- `已训 >= STEPS` 时直接退出，不空跑。

**并行兜底**：把 GPU 侧的 `001250` ckpt 复制到 `outputs/act_lang_dual_cpu/`（591 MB），
用 CPU 续跑剩余 3750 步（6.0–6.5 s/step ≈ 6.5 h，`SAVE_FREQ=500`），不依赖 GPU 是否稳定。
GPU 侧一旦跑通，直接用那边的 ckpt 评即可，两边互不影响（输出目录不同）。
