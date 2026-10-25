# 第八轮：视频产物解耦 + ACT 直训数据集（sdg-v6）

日期：2024-11 · 影响面：`collect/`（产物）、`render/`（产物）、`tools/`（新增排查工具）、新增训练交付链路

## 0. 用户原话与本轮判据

> 「之后的合成数据时不要每次都把视频转换格式方便我查看，把他们分开，生成完数据后假如 100 条，
> 我想排查视频是否有问题我自己去运行代码生成视频查看而不是每条都生成一次。我准备用 act 去训练，
> 在之后的数据希望可以直接训练，我迁移时方便不要做二次修改。」

拆成三条硬判据：

| # | 判据 | 本轮做法 | 验收 |
|---|---|---|---|
| J1 | 采集不再"每次都转格式" | 容器统一 **MP4/h264**；`make_video.py` 删除；采集路径不再出 avi/预览/拼图/帧号 | 采集产物只有 `npz + json + 2×mp4` |
| J2 | 排查视频按需、可指定条目、不每条都出 | 新增 `tools/inspect_episode.py`（选择器 + 拼图/预览/体检；缺片可补渲） | `--episodes 0,7,13` / `--failed` / `--every N` / `--all --check-only` 四路都验过 |
| J3 | 数据**直接**可训 + 迁移零改动 | 新增 `collect/export_lerobot.py` → **lerobot v3.0** 数据集；全部路径走参数 | `lerobot-train --policy.type=act` **3 步跑通**（loss 90.2 → 61.7） |

## 1. 关键数字（实测，本机）

| 项 | 数字 |
|---|---|
| 视频体积（455 帧 640×480，单路） | MJPG AVI **13.1 MB** → MP4/h264 **0.68~0.71 MB**（≈1/19） |
| 两条视频/条 | 13.3 MB（20 条 266 MB）→ **1.2~1.9 MB**（20 条 27 MB） |
| 渲染成本 | **有 GPU：`rig.grab` 1.7 ms/帧，单条 19.1 s（含启动）、8 进程 8 条 33.1 s**；无 GPU（看不到 `/dev/nvidia*` 时）退化为 174~197 ms/帧、75~90 s/条 |
| 并行采集 `--jobs` | 有 GPU：4 进程 20.1 s / 8 进程 33.1 s；无 GPU：4 进程 148.7 s、8 进程 16 条 8 min |
| 导出到 lerobot | **332 s / 20 条**（≈16.6 s/条，解码 + av1 crf30 重编码两路） |
| 导出体积 | **121.0 MB / 20 条 / 9102 帧**（≈13 KB/帧） |
| 训练自检 | ACT 52M 参数，CPU 1.25 s/step，`data_s≈0.05-0.11`（视频解码正常），3 步 loss 90.2→82.7→61.7 |
| 母本体检 | `inspect_episode --all --check-only` **20/20 无问题**；`check_dataset` **20/20 通过** |

## 2. 改了什么（文件级）

**产物与容器**
- `collect/collect.py`：`AviSink` → **`Mp4Sink`**（`avc1`）；路径 `ep_XXXXXX_env|wrist.mp4`；
  `paths` 键 `env_avi/wrist_avi` → **`video_env/video_wrist`**；`DATASET_VERSION=sdg-v6`；
  新增 **`--jobs K`**（自动摊成 K 个 `--shard` 子进程 + 结束后聚合清单，每进程限 `LP_NUM_THREADS`）；
  新增 **`--render-only --episodes …`**（不跑物理，从 npz 纯运动学重渲）。
- `render/test_render.py`：`VideoSink` → `avc1`；拼图/预览单帧改为 **`--inspect` 才出**；
  `episode_0.npz`（图片数组）改为 **`--export-npz` 才出**（默认不再写 800 MB）。
- `render/make_video.py`：**删除**（npz→AVI 的唯一用途已消失）。
- `tools/run_pipeline.py`：`video` 阶段退役（ik/grasp/render 三段）；顺手修了它自带
  `--selftest` 的一个老 bug（假日志里缺 `[schema] ✅`，断言本来就红）。
- `collect/dataset.py` / `check_dataset.py` / `split.py`：读侧支持新键，**保留对 `env_avi/wrist_avi` 的回退**
  （sdg-v5 及更早的数据仍可读）。

**新增工具**
- `tools/inspect_episode.py`：选择器（`--episodes/--failed/--every/--all/--limit`）+ 出图
  （`ep_XXXXXX_strip.jpg` 6 帧拼图、`ep_XXXXXX_preview.png` 夹稳帧）+ 体检（帧数/fps/尺寸/可解码，
  不过就 ⚠️ + exit 1）+ `--render` 缺片补渲（子进程调 `collect.py --render-only`，纯运动学）。
- `collect/export_lerobot.py`：母本 → lerobot v3.0；`--units deg|rad`（默认 deg）、`--task`、
  `--episodes/--limit/--include-failed/--drop-phases`、`--check`（只体检）、`--overwrite`；
  写完自动回读校验（`len / episodes / frames / 图像形状 / stats`），并落
  `meta/sim_provenance.json`（母本版本、index sha、单位、episode 列表）。
- `docs/MIGRATION_ACT.md`：采集 → 导出 → 训练 → 迁移 → 排查 的一条龙说明。

## 3. 关键决策（含被否决的方案）

1. **容器选 MP4/h264，不选 mp4v、不留 AVI**：h264 是 lerobot/pyav/ffmpeg/浏览器四边都吃的格式；
   AVI 只为了"双击播放"，而双击播放 MP4 也行。实测 h264 比 MJPG 小 19 倍。
2. **视频必须照出，不能默认 `--no-render`**：训练要像素。省时间靠 `--jobs`，不靠不渲染。
3. **排查产物一律不进默认路径**：连 demo（`render/test_render.py`）也一样，`--inspect` 才出——
   否则"我只想看一条"就会变成"每条都生成 4 MB 拼图"。
4. **单位用 deg 而不是 rad**：本机已有的真机数据集（`~/lerobot_datasets/yyl100_*`）就是
   `shoulder_pan.pos … gripper.pos` + 度，`robot_type=so_follower`。用 deg 才能与其共用归一化、混训。
   `--units rad` 保留给纯仿真研究。
5. **导出用 lerobot 自己的 `LeRobotDataset.create()`**，不手写 parquet：
   编码器参数（`libsvtav1 / yuv420p / g=2 / crf=30 / preset=12`）与真机数据集逐字段一致，
   统计量也由 lerobot 算，避免"我拼的 parquet 训练时才炸"。
6. **负样本默认不进训练集**（`--include-failed` 才进）：ACT 不做隐式负样本学习，
   混进去只会教它"偶尔别抓"。
7. **`--jobs` 走子进程 + `--shard`，不用 multiprocessing/线程**：EGL 上下文与 MuJoCo 模型
   在 fork 之后再建上下文容易踩坑；`--shard` 这条路径本来就有，产物天然互不覆盖。

## 4. 未收口 / 风险

- ~~本机没有可用 GPU~~ **已更正**：本机是 RTX 4060 Laptop（驱动 595.99.02，正常）。
  之前记录成“软件渲染/100 条 50 min”是**AI 的执行沙箱屏蔽了 `/dev/nvidia*`** 导致的假象，
  用户自己在终端跑是 GPU 速度（100 条约 4~5 min）。
- **夹爪量纲差异**：本仿真 `gripper` 关节 0~86°（张开 2.232 rad≈128°），
  真机数据集里 gripper 只有 1.2~42.3°，两者物理量程不同。混训/迁移时需确认部署侧的夹爪映射
  （本轮只保证"字段名/单位/维数一致"，不保证"量程一致"）。
- **负样本样例 `demo_neg/` 是上一代的**（sdg-v5 物理、mp4 已是 h264）：通路本身没变，
  要更新就重跑 `render/test_render.py --gz 0.080 --out demo_neg --quick`（≈85 s）。
- **`--jobs` 与 `--max-reject` 组合**：各 worker 独立计数，总量会放大 K 倍（未做全局配额）。
- **`export_lerobot.py` 依赖 lerobot 0.6.2 的 `LeRobotDataset.create` 签名**（yuv420p / crf / preset 走默认）；
  升级 lerobot 后先跑 `--check` + 导 2 条试水。

## 5. 复现（照抄）

```bash
cd /home/yyl/deepseek-harness

# 采集（并行；视觉随机化与物理不变）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 20 --jobs 8 --seed 3

# 母本体检（秒级，不解码）
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/inspect_episode.py --all --check-only

# 按需看图（只出你点的那几条）
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/inspect_episode.py --episodes 0,7,13

# 导出 lerobot 数据集 + 回读校验
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --dst dataset/soarm101_sim_v1 --overwrite

# 训练
source ~/miniforge3/bin/activate && conda activate lerobot312
lerobot-train --dataset.repo_id=local/soarm101_sim_v1 \
  --dataset.root=/home/yyl/deepseek-harness/dataset/soarm101_sim_v1 \
  --policy.type=act --policy.repo_id=local/act_soarm101 \
  --output_dir=outputs/act_soarm101_sim_v1 --batch_size=8 --steps=100000
```
