# 合成数据操作手册（没接触过这个仓库也能照着做）

目标：生成一批"机械臂抓药盒放进木盒"的仿真数据，并导出成 ACT 能直接训练的数据集。
全程不需要 ROS，不需要懂 MuJoCo。

---

## 0. 你最终会得到什么

| 产物 | 位置 | 一条多大 | 用途 |
|---|---|---|---|
| 轨迹 + 标签 | `scripts/ep_XXXXXX.npz` / `.json` | ≈140 KB | 动作、关节角、接触力、相机外参、随机化记录、验收结果 |
| 双相机视频 | `videos/ep_XXXXXX_env.mp4` / `_wrist.mp4` | ≈1.6 MB | 640×480 / 30fps / h264 |
| 训练数据集 | `dataset/<名字>/` | ≈6 MB | lerobot v3.0 格式，ACT 直接训练 |
| 体检/统计 | `scripts/index.json`、`splits/`、`DATA_CARD.md` | 几 KB | 清单、划分、覆盖度 |

---

## 1. 准备（3 分钟，只做一次）

```bash
cd /home/yyl/deepseek-harness
nvidia-smi            # 能列出显卡 = 渲染会走 GPU（快 100 倍）。列不出来也能跑，只是慢
```

两条**必须记住**的规则：

1. **跑仿真的命令前面要加 `env -u PYTHONHOME -u PYTHONPATH`，并用 `/usr/bin/python3`。**
   不加会报 `ModuleNotFoundError: No module named 'encodings'`（本机环境被一个 AppImage 污染了）。
2. **导出数据集 / 训练要用另一个 python：`~/miniforge3/envs/lerobot312/bin/python`。**
   那个环境才有 lerobot + pyarrow。两个环境不能混用。

渲染后端固定 `MUJOCO_GL=egl`（已默认，显式写上更保险）。

---

## 2. 总览：五步，一条线

```
① 采集 ──> ② 体检 ──> ③ 切分/统计 ──> ④（想看图才做）──> ⑤ 导出 lerobot ──> ⑥ 训练
   物理+视频        帧数对不对        训练清单/数据卡        拼图/预览           ACT 能读的数据集
```

每步都有明确判据。**前一步没过，不要往下走。**

---

## 3. 第 ① 步：采集

### 3.1 先只刷物理（可选，5 分钟看清成败率）

```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 100 --no-render --seed 3
```

- **不做**：不出视频。只写 `scripts/`。
- **耗时**：100 条约 30 秒（1000 条约 5 分钟）。
- **看什么**：最后一行 `接受率`。正常是 100%；掉到 90% 以下说明抓取出问题了，先别继续。
- **什么时候用**：想快速确认参数没被改坏；或者先看多样性，再决定要不要出视频。

### 3.2 出视频（训练真正要的）

```bash
# 物理已经刷过：只补渲染
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --render-only --episodes 0,1,2,3 --jobs 8

# 一步到位（物理 + 视频）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 20 --jobs 8 --seed 3
```

- **耗时**（有 GPU）：单条 ≈19 秒；8 进程并行时**每条 23~36 秒/进程**（并发互相抢 GPU/CPU），
  **1000 条约 60~75 分钟**；无 GPU 时约慢 5 倍。
- **看什么**：每条一行 `✅ ep_XXXXXX  抬升 +1xx mm ...`；末尾 `数据集: N 条（收 N / 丢 0）`。
- **注意**：`--jobs 8` 会自动开 8 个进程、跑完自动合并清单，**不需要**你再跑 `--rebuild-index`。

### 3.3 采集参数——每个后缀什么时候用

| 后缀 | 作用 | 什么时候加 |
|---|---|---|
| `--n N` | 生成 N 条 | 必填 |
| `--start K` | 从编号 K 开始 | 续跑（先跑 100 条，之后加 100 条 → `--start 100 --n 100`） |
| `--jobs K` | K 进程并行 | **总是加**，本机填 8；填 1 就是单进程（慢 8 倍） |
| `--seed S` | 固定随机种子 | 想复现同一批数据就填死（推荐 3）；不填则按时间随机 |
| `--no-render` | 只跑物理、不出视频 | 先看成败率，或只想快速刷轨迹 |
| `--render-only` | 不跑物理，把已有轨迹重渲成视频 | 当初用了 `--no-render`，现在要视频；或视频坏了要重出 |
| `--episodes 0,3,7` | 配合 `--render-only` 指定编号 | 补渲特定几条 |
| `--quick` | 出 320×240 | 只想快速看流程。**别拿去训练** |
| `--keep-failed` / `--no-keep-failed` | 负样本（没抓起来的）落不落盘 | 默认落（负样本是稀有资源）；只想留成功的加 `--no-keep-failed` |
| `--render-failed` | 负样本也出视频 | 要研究失败模式时加（每条多一份渲染时间） |
| `--burn-counter` | 视频左上角烧帧号 | **只给排查用。训练数据绝对不能开**（等于在像素里埋捷径） |
| `--reset` | 清空清单重刷 | 想从头开始（不会删已有的 npz/视频文件） |
| `--shard I/K` | 手动分片 | 一般不用；`--jobs` 内部就是用它 |

### 3.4 合成 1000 条的完整剧本

**先说结论**：一次跑完 ≈40 分钟采集 + ≈4.6 小时导出，占 ≈8 GB。建议**先跑 100 条、训练通了再上 1000**。

### 3.4.1 先记住怎么"脱手跑"和"看进度"（踩过坑）

```bash
# ① 脱手启动：setsid 让它脱离当前终端，终端关了/Ctrl+C 都不影响
cd /home/yyl/deepseek-harness
setsid nohup env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 1000 --jobs 8 --seed 3 > logs/collect_1000.log 2>&1 &
disown
echo "已启动，PID=$!"        # 记下这个数

# ② 看进度：只"看一眼"就返回，绝不要用 tail -f
tail -n 5 logs/collect_1000.log
grep -c "✅ ep_" logs/collect_1000.log     # 已完成条数

# ③ 它还在跑吗
pgrep -af "collect.py --n 1000" | head -3

# ④ 想停（已完成的那几条不会坏；最多丢正在跑的 1~8 条）
pkill -f "collect.py --n 1000"

# ⑤ 崩了/停了怎么续：先重建清单，找出最小的缺失编号 M，从 M 接着跑
/usr/bin/python3 collect/collect.py --rebuild-index
/usr/bin/python3 - <<'EOF'                   # 打印缺号（可能是中间断的，不是只断尾巴）
import pathlib
have = {int(p.stem.split('_')[1]) for p in pathlib.Path('scripts/index').glob('ep_*.json')}
print("已完成", len(have), "最小缺号", min(set(range(0, 1000)) - have))
EOF
# 假设打印出 最小缺号 64，就从 64 跑到 999（已完成的重跑一遍也无害：同 seed 结果逐位相同）
setsid nohup env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --start 64 --n 936 --jobs 8 --seed 3 \
    > logs/collect_resume.log 2>&1 &
```

**为什么不能用 `tail -f`**：它按设计就是"跟着文件一直等"，永远不返回，看起来像卡死。
看进度一律 `tail -n 20`（看一眼）或 `watch -n 20 'grep -c "✅ ep_" logs/collect_1000.log'`。

**为什么会"跑到一半没了"**：如果命令是在前台跑的，`Ctrl+C` 会同时杀掉采集本身；
`nohup` 只挡"终端关闭"（SIGHUP），挡不住 `Ctrl+C`（SIGINT）。所以要用 `setsid nohup ... & disown`。

方案 A：一次跑完（推荐，简单）

```bash
# 用上面 3.4.1 的 ① 启动，② 看进度；≈60~75 分钟
tail -3 logs/collect_1000.log     # 期望：数据集: 1000 条（收 1000 / 丢 0，接受率 100%）
```

方案 B：分 5 批（每批 200 条，中断了好恢复，也能边看边跑）

```bash
for s in 0 200 400 600 800; do
  env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
      collect/collect.py --start $s --n 200 --jobs 8 --seed 3 \
      > logs/collect_${s}.log 2>&1
  tail -2 logs/collect_${s}.log     # 每批结束看一眼
done
```

**1000 条的预算表**（有 GPU；括号内是没有 GPU 时）

| 步骤 | 耗时 | 产物体积 |
|---|---|---|
| 采集（视频） | ≈60~75 min（无 GPU 时 ≈8~9 h） | ≈1.6 GB |
| 体检 + 切分 + 统计 | ≈2 min | 几 MB |
| 导出 lerobot | **≈4.6 h（单进程，没有并行开关）** | **≈6 GB** |
| 合计 | ≈5.5~6 h | ≈8 GB（磁盘需 ≥10 GB 空闲） |

省导出时间的两个办法（不改代码，只是参数）：

- `--drop-phases init,home,tail`：剪掉开头的待机段和结尾的归位段，帧数 −22%，导出和存储同比减少；
- 分批导出到不同目录（`--episodes 0,1,…` 加 `--dst dataset/part1`），各进程可并行跑；
  但**训练时要能读多个数据集目录**，这点取决于你的 lerobot 版本是否支持，先小规模试一次。

---

## 4. 第 ② 步：体检（必做，别跳）

```bash
# a) 视频/轨迹是否对齐（帧数、fps、尺寸、能不能解码）
/usr/bin/python3 tools/inspect_episode.py --all --check-only

# b) 数据契约（相机外参、标签、清单自洽、字段形状/dtype）
/usr/bin/python3 collect/check_dataset.py
```

- **判据**：`体检: N 条，问题 0 条`（退出码 0）且 `check_dataset` 打印 `总判定: ✅ 全部通过`。
- **1000 条耗时**：a 十几秒，b 约 1~2 分钟。
- **有问题**：打开 `scripts/report.json` 看每条的 `problems`；视频坏的那条删掉 mp4 后
  `collect.py --render-only --episodes N` 重出。

---

## 5. 第 ③ 步：切分 + 覆盖度（出训练清单和数据卡）

```bash
/usr/bin/python3 collect/split.py            # train/val/test 划分
/usr/bin/python3 collect/stats.py --card     # 覆盖度统计 + DATA_CARD.md
```

- **产物**：`scripts/splits/{train,val,test}.json` + `manifest.json`；`scripts/stats.json|png`、`scripts/DATA_CARD.md`。
- **什么时候用**：每次采集完都做一次（几秒钟），`DATA_CARD.md` 是给训练侧看的"这批数据长什么样"。

---

## 6. 第 ④ 步：想用肉眼看的时候才出图

```bash
/usr/bin/python3 tools/inspect_episode.py --episodes 0,7,13   # 只出这 3 条
/usr/bin/python3 tools/inspect_episode.py --failed --limit 5  # 先看没抓起来的
/usr/bin/python3 tools/inspect_episode.py --every 10          # 每 10 条抽 1 条
/usr/bin/python3 tools/inspect_episode.py --demo              # 看 demo/ 那条单条演示
```

- **产物**：`inspect/ep_XXXXXX_strip.jpg`（6 帧拼图，≈0.4 MB）+ `ep_XXXXXX_preview.png`（夹稳那一帧）。可以随时删。
- **参数**：`--render`（这条没视频就先补渲）、`--check-only`（只体检不出图）、`--tile-w`（拼图单格宽度）。
- **为什么单独一步**：采集阶段故意不出任何拼图/预览/avi，避免"每条都花时间转格式"。

---

## 7. 第 ⑤ 步：导出成 ACT 训练数据集

```bash
# 先干跑体检（不写盘，只看会导出几条、共多少帧）
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --check

# 正式导出（写完会自动回读校验；目标目录已存在要加 --overwrite）
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --dst dataset/soarm101_sim_v1 --overwrite
```

- **耗时/体积**：≈16.3 秒/条，≈6.1 MB/条（实测 64 条 = 1045 秒 / 390.6 MB）。
- **注意**：AV1 编码器（libsvtav1）偶发段错误（实测 66 条时崩过一次，重跑即成功）。
  若崩了：目录会留下写坏的 parquet，删掉重建即可（`--overwrite` 会自动删）；建议**分批导出**（每次 ≤ 40 条）降低重跑成本。
- **判据**：末尾打印 `回读 : len=… episodes=… frames=… img=ok stats=ok`，没有 ❌。
- **导出参数**：

| 后缀 | 作用 | 什么时候用 |
|---|---|---|
| `--src/--videos` | 母本目录 | 换机器后指向新路径 |
| `--dst` | 输出目录 | 必填 |
| `--units deg\|rad` | 关节角单位 | 默认 `deg`（与真机数据集一致）；做纯仿真研究才用 `rad` |
| `--task "..."` | 自然语言任务 | 想换任务文本时 |
| `--episodes 0,3,7` | 只导这几条 | 试水、分批导出 |
| `--limit N` | 最多导 N 条 | 同上 |
| `--include-failed` | 连负样本一起导 | 默认不导（ACT 不建议混负样本） |
| `--drop-phases init,home,tail` | 剪掉指定阶段 | 想去掉长长的待机/归位段 |
| `--overwrite` | 覆盖已有目录 | 重导时加 |
| `--check` | 只体检不写盘 | 正式导出前先看 |
| `--no-verify` | 跳过回读校验 | 不推荐 |

导出结果的字段（与真机 SO-101 数据集同名同单位）：

| 特征 | 内容 |
|---|---|
| `observation.state` | 6 维当前关节角（pan/lift/elbow/flex/roll/gripper），**度** |
| `action` | 6 维下发的关节目标角，**度** |
| `observation.images.env` / `.wrist` | 640×480 视频（av1 / yuv420p / crf30） |
| `robot_type` | `so_follower` |

---

## 8. 第 ⑥ 步：训练（可选）

```bash
source ~/miniforge3/bin/activate && conda activate lerobot312

lerobot-train \
  --dataset.repo_id=local/soarm101_sim_v1 \
  --dataset.root=/home/yyl/deepseek-harness/dataset/soarm101_sim_v1 \
  --policy.type=act --policy.repo_id=local/act_soarm101 \
  --policy.device=cuda \
  --output_dir=outputs/act_soarm101 --batch_size=8 --steps=100000
```

- 显存 8 GB：`--batch_size=8` 稳；32 会 OOM。
- 想先确认"数据读得出来"：`--steps=3` 跑三步就停（loss 会打印出来）。
- **换机器**：把 `scripts/`、`videos/`、`dataset/` 三层拷过去即可；要重新导出时只改
  `--src/--videos/--dst` 三个路径，代码里没有硬编码路径。
- **想看策略在仿真里抓不抓得起来**：
  `python3 tools/replay_policy.py --mode closed-loop --policy <checkpoint 目录> --episodes 0,1,2`
  （开环对照用 `--mode open-loop`；判据与训练标签同口径：峰值抬升 ≥ 50 mm）。

---

## 9. 出错怎么办

| 现象 | 原因 | 处置 |
|---|---|---|
| `No module named 'encodings'` | 少了 `env -u PYTHONHOME -u PYTHONPATH` | 命令前面补上 |
| `No module named 'lerobot'/'pyarrow'` | 用错解释器 | 导出/训练用 `~/miniforge3/envs/lerobot312/bin/python` |
| 进程忽然消失、没报错 | `mj_step` 偶发 SIGFPE（约 7%） | 原命令重跑；已完成的不会丢 |
| `Fatal Python error: Floating point exception` | 把 python 输出管道给了 `head`/`tail` | 改成 `> logs/x.log 2>&1` 再看文件 |
| 采集日志里 `❌ ... no_lift` | 抓取没成（参数被改过） | 先跑 `tools/probe_contact_audit.py`（0.9 秒看六项闸门） |
| `frames[env] 452 != 453` | 视频帧数与轨迹不一致 | 删该条 mp4，`--render-only --episodes N` 重出 |
| 导出报"没有 video_env" | 当初用了 `--no-render` | `--render-only --episodes …` 补渲 |
| `VideoWriter 打不开` | 磁盘满 | `df -h`；1000 条要 ≈10 GB 空闲 |
| 导出报 huggingface 缓存只读 | 受限环境 | 脚本已默认把 `HF_HOME` 指到仓库内 `.cache/` |
| 命令"卡死"不动 | 大概率是 `tail -f`（永不返回）；也可能真在跑 | `tail -n 20 log` 看一眼；`pgrep -af collect.py` 确认进程 |
| 采集跑一半没了、`index.json` 里条数不对 | 前台跑被 `Ctrl+C` 杀掉；`--jobs` 模式只在结束时合并清单 | `collect.py --rebuild-index` 重建清单，再从 `--start N` 续跑 |

---

## 10. 绝对不要做的事

- ❌ 改 `data.qpos` 伪造握持，或在采集里覆写药盒位姿（出生落地那一次除外）。
- ❌ 改接触段步数（`descend 400 / grasp 300 / lift 400`）：放慢下探会把药盒推倒，抓取全废。
- ❌ 把 `--burn-counter` 出的视频拿去训练。
- ❌ 拿 `--quick`（320×240）的数据去训练。
- ❌ 把不同 `dataset_version` 的数据混在一起训练（清单里版本不同就分开）。
- ❌ 改指垫位置却不重算：指垫与"起止姿态 + 接近路径 + 步数配比"是耦合的。

改参数前先读 [`PHYSICS_FACTS.md`](PHYSICS_FACTS.md)。