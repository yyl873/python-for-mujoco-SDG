# 从仿真到 ACT 训练：一次采集、一步导出、零改动训练

本文只讲**数据怎么用**，不讲物理怎么调（那是 `docs/HANDOFF.md` 的事）。

一句话：
> `collect/` 采集只出【母本】（npz + json + mp4），`collect/export_lerobot.py` 一步把它变成
> **lerobot v3.0 数据集**，`lerobot-train --policy.type=act` 直接开训。换机器只改 3 个路径参数。

---

## 1. 两段产物，各管一件事

```
/home/yyl/deepseek-harness
├── scripts/                    母本·轨迹与标签（每条 ≈ 122 KB）
│   ├── ep_000000.npz           time/qpos/qvel/ctrl/phase/tcp_pos/obj_pos/接触力/逐帧外参
│   ├── ep_000000.json          标签 + 全部随机化参数 + 抓取参数 + 验收指标
│   ├── index.json              清单（accepted/rejected，权威来源是 index/ 片段）
│   └── splits/                 train/val/test 划分
├── videos/                     母本·视频（每条两路，h264 mp4）
│   ├── ep_000000_env.mp4       外部相机 640×480 @30fps
│   └── ep_000000_wrist.mp4     腕部相机 640×480 @30fps
├── inspect/                    ★按需排查产物（拼图/预览；不进训练集，随时可删）
└── dataset/<名字>/             ★导出结果 = lerobot v3.0 数据集（训练直接读这个）
    ├── meta/info.json          fps / features / robot_type / 路径模板
    ├── meta/stats.json         各特征 min/max/mean/std/q01…q99
    ├── meta/tasks.parquet      任务文本 → task_index
    ├── meta/episodes/chunk-000/file-000.parquet   每条的长度/起止索引/视频时间戳
    ├── data/chunk-000/file-000.parquet            state/action/timestamp/index…
    └── videos/observation.images.{env,wrist}/chunk-000/file-000.mp4
```

**为什么分两段**：写 lerobot 数据集要 pyarrow + lerobot（只有 conda `lerobot312` 有）；
跑 MuJoCo 要系统 python3.10。分开后两边互不污染，而且导出只搬数据、不跑物理，几分钟就能重跑。

---

## 2. 三步命令（照抄即可）

```bash
cd /home/yyl/deepseek-harness

# ① 采集（20 条；渲染吃 CPU，--jobs 并行；软件 GL 实测 8 进程 ≈ 3x）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 20 --jobs 8 --seed 3

# ② 体检母本（帧数/维度/视频对齐，不解码，秒级）
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --check

# ③ 导出成 lerobot 数据集（会自动回读校验一遍）
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --dst dataset/soarm101_sim_v1 --overwrite
```

采集端的三种"少花时间"姿势：

| 目的 | 命令 |
|---|---|
| 只想快点刷物理（4 s/条，无视频） | `collect.py --n 100 --no-render` |
| 事后只想看某几条的视频 | `collect.py --render-only --episodes 4,9,17 --jobs 4` |
| 只想看拼图/预览（不解码全片，秒级） | `tools/inspect_episode.py --every 10` |

---

## 3. 训练

```bash
source ~/miniforge3/bin/activate && conda activate lerobot312

lerobot-train \
  --dataset.repo_id=local/soarm101_sim_v1 \
  --dataset.root=/home/yyl/deepseek-harness/dataset/soarm101_sim_v1 \
  --policy.type=act \
  --output_dir=outputs/act_soarm101_sim_v1 \
  --batch_size=8 --steps=100000
```

数据集自检（不训练也能确认"读得出来"）：

```bash
lerobot-dataset-viz --repo-id local/soarm101_sim_v1 \
    --root /home/yyl/deepseek-harness/dataset/soarm101_sim_v1 --episode-index 0
lerobot-info --dataset.repo_id local/soarm101_sim_v1
```

---

## 4. 字段与单位（与真机数据集逐字段同构）

| 特征 | 本数据集 | 来源 | 单位 |
|---|---|---|---|
| `observation.state` | `(6,)` float32 | npz `qpos[:, :6]`（当前关节角） | **度** |
| `action` | `(6,)` float32 | npz `ctrl[:, :6]`（当帧下发的关节目标角） | **度** |
| `observation.images.env` | video 480×640×3 | `videos/ep_XXXXXX_env.mp4` 原帧 | - |
| `observation.images.wrist` | video 480×640×3 | `videos/ep_XXXXXX_wrist.mp4` 原帧 | - |
| `timestamp` / `frame_index` | 自动 | fps=30 | s / 帧号 |
| `task` / `task_index` | 自动 | `--task`（默认取母本 instruction） | - |

`names` 顺序与真机 SO-101 一致：
`shoulder_pan.pos / shoulder_lift.pos / elbow_flex.pos / wrist_flex.pos / wrist_roll.pos / gripper.pos`，
`robot_type = so_follower`。

★实测对齐证据（同机已有真机数据集 `~/lerobot_datasets/yyl100_corners`）：
`codebase_version=v3.0`、`fps=30`、同一套 6 个关节名、`robot_type=so_follower`、
state/action 都在**度**量纲（pan −34.9~36.9 / lift −103.7~57.8 / …）。
所以本数据集与真机数据**可以混合训练、也可以共用同一套归一化**，不必二次改代码。

`--units rad` 可切成弧度（仿真里 MuJoCo 原生就是弧度）；默认 `deg` 就是为了"迁移不改代码"。

---

## 5. 换机器怎么迁（零代码改动）

1. 整目录拷走（只这两层就够训练，`dataset/` 自带视频）：

   ```
   scripts/  videos/  dataset/soarm101_sim_v1/
   ```

   ```bash
   rsync -av --progress /home/yyl/deepseek-harness/{scripts,videos,dataset} user@train:/data/sim/
   ```

2. 新机器上想**重新导出**（比如换 `--units`、换 `--task`、去阶段）时，只改参数：

   ```bash
   python collect/export_lerobot.py \
       --src /data/sim/scripts --videos /data/sim/videos \
       --dst /data/sim/dataset/soarm101_sim_v1 --units deg --overwrite
   ```

   全部路径都是参数；代码里没有绝对路径。`meta/sim_provenance.json` 会记下母本
   `dataset_version`、`index.json` 的 sha、导出的 episode 列表与单位，事后能对账。

3. 训练侧不需要任何改动：`LeRobotDataset` 只认 `meta/info.json`。

---

## 6. 排查视频：按需，不是每条都出

采集**不再**生成任何观看用格式（没有 avi、没有预览 png、没有拼图、不烧帧号）。
要看得自己点：

```bash
# 抽 3 条看拼图（env 一行 + wrist 一行，6 帧，jpg ≈ 0.4 MB/条）
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/inspect_episode.py --episodes 0,7,13

# 先看失败的（负样本）
... tools/inspect_episode.py --failed --limit 5

# 只体检不出图：100 条全扫，只读容器头，秒级
... tools/inspect_episode.py --all --check-only

# demo 目录（render/test_render.py 的产物）
... tools/inspect_episode.py --demo
```

排查产物落 `inspect/`：`ep_XXXXXX_strip.jpg`（抽帧拼图）、`ep_XXXXXX_preview.png`（夹稳那帧大图）。
体检项：视频帧数 == 轨迹帧数、fps≈30、尺寸 640×480、可解码。**对不上就报 ⚠️ 并 exit 1。**

---

## 7. 常见坑（都实测过）

| 现象 | 原因 | 处理 |
|---|---|---|
| 导出报"没有 video_env/video_wrist" | 采集时用了 `--no-render` | `collect.py --render-only --episodes …` 补渲 |
| 训练 loss 不动 / 夹爪不闭合 | 单位搞错（deg 当 rad 用） | 导出后核对 `meta/stats.json` 的 action min/max 是否在 ±180 内 |
| 想只要"抓取段"不要长长的待机段 | 母本整条录（含 init/home/tail） | 导出加 `--drop-phases init,home,tail`（先 `--check` 看帧数变化） |
| 回读校验报 huggingface 缓存只读 | 沙箱里 `~/.cache` 不可写 | 导出脚本已默认把 `HF_HOME` 指到仓库内 `.cache/`；自己设了 `HF_HOME` 则尊重你的 |
| 负样本没进数据集 | 默认只导 `success=true` | 需要就加 `--include-failed`（ACT 不建议混负样本，除非做对比） |
| 20 条渲了半小时 | 跑在看不到 `/dev/nvidia*` 的环境（AI 沙箱/容器/无 X 的 rootless 环境）→ MuJoCo 退化为 llvmpipe 软件 GL（640×480 双相机 ≈ 200 ms/帧） | 在普通终端跑即走 GPU（单条 19 s、8 进程 8 条 33 s）；确认手段：`nvidia-smi` 能列出卡 |

---

## 8. 当前这批实测数字（可直接引用）

| 项 | 数值 |
|---|---|
| 母本版本 | `sdg-v6`（物理同 v5，视频容器 AVI→MP4） |
| 每条 | 446~453 帧 / 14.9~15.1 s @30fps，两路视频合计 ≈ 1.2~1.9 MB |
| 采集耗时 | **有 GPU：单条 19.1 s（含启动）、8 进程 8 条 33.1 s，100 条约 4~5 min**；无 GPU（沙箱内）75~90 s/条 |
| 导出耗时 | 332 s / 20 条（≈16.6 s/条：解码 h264 + av1 crf30 重编码两路） |
| 导出体积 | **121 MB / 20 条 / 9102 帧**（≈13 KB/帧，视频占绝大部分） |
| 训练自检 | ACT 52M 参数、CPU 1.25 s/step：`lerobot-train … --policy.type=act --steps=3` 跑通，loss 90.2→82.7→61.7 |
