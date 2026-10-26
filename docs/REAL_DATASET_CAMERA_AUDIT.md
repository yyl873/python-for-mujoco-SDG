# 真机数据集相机标签排查报告（2026-10-26）

## 1. 结论

真机数据集里 `observation.images.env` 与 `observation.images.wrist` 两路相机的**内容与文件夹名互换过**，
涉及三个 `_fixed` 数据集；三个 `_corners` 数据集标签正确。**已修正**，修完后逐文件复测 12 个流全部一致。

| 数据集 | 修复前 `...images.env` 实拍内容 | 修复前 `...images.wrist` 实拍内容 | 处理 |
|---|---|---|---|
| real_train_fixed | 腕部相机（随臂移动） | **环境相机**（固定） | 已互换 |
| yyl100_dataset_fixed | 腕部相机 | **环境相机** | 已互换 |
| holdout_fixed | 腕部相机 | **环境相机** | 已互换 |
| real_train_corners | **环境相机** | 腕部相机 | 无需改动 |
| holdout_corners | **环境相机** | 腕部相机 | 无需改动 |
| yyl100_corners | **环境相机** | 腕部相机 | 无需改动 |
| yyl100_lance_fixed | Lance 格式无 mp4；元数据血缘证明 env 路=腕部画面 | 同左 | **同样互换，未修**，见 §5 |
| yyl100_lance_corners | Lance 格式无 mp4；元数据血缘证明 env 路=环境相机 | 同左 | 标签正确，见 §5 |

修复后的实测值（见表末「复核」章节的命令）：

| 数据集/流 | static% | pad_cv | 判定 |
|---|---|---|---|
| holdout_fixed/env | 50.5 | 0.25 | 正确（固定） |
| holdout_fixed/wrist | 1.3 | 0.80 | 正确（随臂） |
| real_train_fixed/env | 37.7 | 0.26 | 正确（固定） |
| real_train_fixed/wrist | 1.4 | 0.77 | 正确（随臂） |
| yyl100_dataset_fixed/env | 38.9 | 0.24 | 正确（固定） |
| yyl100_dataset_fixed/wrist | 1.5 | 0.77 | 正确（随臂） |
| holdout_corners/env → wrist | 43.0 / 2.3 | 0.47 / 0.60 | 正确 |
| real_train_corners/env → wrist | 33.9 / 2.3 | 0.37 / 0.95 | 正确 |
| yyl100_corners/env → wrist | 37.0 / 1.9 | 0.34 / 0.78 | 正确 |

## 2. 判定方法（为什么不是靠文件夹名）

两条独立指标，逐 mp4 文件测量（`tools/scan_real_cams.py`）：

| 指标 | 定义 | 固定环境相机 | 腕部相机 | 实测间隔 |
|---|---|---|---|---|
| **A. static%**（主判据） | 均匀采样 16 帧，逐像素灰度时序标准差 < 8 的像素占全图百分比 | 33.9% ~ 50.5% | 1.3% ~ 2.3% | 差 15 倍以上 |
| **B. pad_cv**（辅助） | 绿色拾取垫连通域面积的时间变异系数 std/mean | ≤ 0.60 | 0.43 ~ 0.95 | 有重叠，只做边界确认 |

A 的物理含义：相机固定在桌面上时，桌子/背景/墙必然有大范围像素全程不动；相机装在夹爪上时整幅画面随臂移动，
几乎没有静止像素。B 只是佐证（固定相机下垫子在画面里大小恒定；腕部相机接近→离开会让它撑满或离开画面）。

阈值取 `static% ≥ 12 → 固定`、`static% ≤ 6 → 随臂`、中间带（6~12）才用 B 裁决 —— 现实数据里**从未出现过中间带**。

## 3. 根因与旁证

`_corners` 与 `_fixed` 是同一套录制脚本、同一台机器人，只有相机标签对调，最可能的原因是一次重新插拔后
**USB 设备枚举顺序变了**，而录制脚本按设备序号的先后给两路流命名（哪个先枚举到就叫 `env`），于是名字跟着设备跑。

旁证（6/6 命中）：`meta/info.json` 里 `features` 的键顺序正好与互换状态一致 ——

- 三个互换的数据集：`observation.images.wrist` 排在 `observation.images.env` **前面**
- 三个正确的数据集：`observation.images.env` 排在前面

也就是说，写键的顺序与写错标签是同一个原因造成的，因此这个键顺序可以当作低成本预警信号
（`tools/scan_real_cams.py` 会把该顺序打进 `summary.txt`）。

## 4. 这次改了什么（不只是改文件夹名）

只改目录名会让统计量和画面内容错配，所以 `tools/fix_cam_labels.py` 一次改了四处：

1. `videos/observation.images.env/` ↔ `videos/observation.images.wrist/` 目录互换（同盘 rename，O(1)，不复制 7.9 GB）
2. `meta/info.json` 里 `features` 两项的先后顺序互换
3. `meta/stats.json` 里两个键的整块数值互换（像素 min/max/mean/std/q01… 属于画面内容，跟着内容走）
4. `meta/episodes/**/*.parquet` 里 `stats/observation.images.{env,wrist}/*` 共 20 列/文件互换
   （holdout_fixed 1 个文件、real_train_fixed 1 个、yyl100_dataset_fixed 16 个，合计改动 360 列）；
   `videos/observation.images.{env,wrist}/{chunk_index,file_index,from_timestamp,to_timestamp}`
   两路实测完全相同，故不动。

备份与回滚：每个数据集下写 `.cam_fix_backup/<时间戳>/`（info.json、stats.json、episodes parquet 原件）
和 `.cam_label_fix.json` 清单。回滚命令：

```bash
PYTHONPATH=.cache/pylibs env -u PYTHONHOME /usr/bin/python3 tools/fix_cam_labels.py --revert   # pyarrow 由 tools/get_pyarrow_libs.py 准备
```

校验修复正确性的硬证据：`stats.json` 里新 `env` 的 mean = 修复前 `wrist` 的 mean
（`0.508751/0.534441/0.532593`），新 `wrist` 的 mean = 修复前 `env` 的 mean
（`0.581700/0.601304/0.483861`），逐位相等。

## 5. 未覆盖 / 存疑

- **yyl100_lance_fixed / yyl100_lance_corners 仍然没有修**：视频以 Lance 格式存在 `videos.lance/`
  （各 2.3 GB，schema 为 `video_key, chunk_index, file_index, file_size, moov_*, kf_*, video_bytes(blob)`），
  每个 (相机, chunk, file) 一行。已确认的事实（`meta/episodes` 里逐 episode 的逐路像素统计，151/30 个 episode 全对齐）：

  | 数据集 | 相关性 lance.env~mp4.env | lance.env~mp4.wrist | 平均绝对差 | 判定 |
  |---|---|---|---|---|
  | yyl100_lance_fixed ↔ yyl100_dataset_fixed | −0.17 | **+1.0000** | **0.0000** | env 路装的是腕部画面 → **同样互换，未修** |
  | yyl100_lance_corners ↔ yyl100_corners | **+1.0000** | −0.58 | **0.0000** | 标签正确 |

  注意这是**元数据血缘**证据（lance 版的统计量与修复前 mp4 版逐位相同，说明它就是从那份跨标签的数据转出来的），
  不是像素级验证 —— 这个 lance 版本（13.0.0）没有 `ds.blobs()` 读取接口，`video_bytes` 只能拿到
  `{kind, position, size, blob_id, blob_uri}` 结构，取不出字节来解码。
  正确做法是**先修 mp4 版（已完成）再重跑这两份的转换脚本**，而不是手工改 Lance。
- **conda 环境的正确启动方式**（之前记的"环境一启动就崩"是我调用方式不对）：
  ```bash
  env -u PYTHONHOME -u PYTHONPATH bash -lc '
    source ~/miniforge3/bin/activate && conda activate lerobot312 &&
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH && python -c "import lance,pyarrow,av"'
  ```
  三个坑缺一不可：DSH 会话里 `PYTHONHOME` 被 AppImage 污染（必须先 `env -u`）、必须走 `conda activate`
  （直接调 `envs/lerobot312/bin/python` 会崩）、必须把 `$CONDA_PREFIX/lib` 放到 `LD_LIBRARY_PATH` 前面
  （否则 `/lib/x86_64-linux-gnu/libstdc++.so.6` 太旧，报 `CXXABI_1.3.15 not found`）。
  实测可用版本：lance 13.0.0 / pyarrow 26.0.0 / av 15.1.0。
- 真机**环境相机的取景分两个批次**，详见 §8。

## 6. 复核方式

```bash
# 1) 全量复测（94 个 mp4，逐文件）——所有行都应显示「正确」
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/scan_real_cams.py --frames 16 --workers 8

# 2) 只看计划（dry-run），确认没有新的需要互换的数据集
PYTHONPATH=.cache/pylibs env -u PYTHONHOME /usr/bin/python3 tools/fix_cam_labels.py

# 3) 重新导出 20 张环境相机帧（脚本按内容自动选路，不再依赖文件夹名）
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/export_real_env_frames.py
```

## 7. 产物

| 文件 | 内容 |
|---|---|
| `tools/scan_real_cams.py` | 逐文件判相机（static% + pad_cv），输出 CSV / summary / 拼图 |
| `tools/fix_cam_labels.py` | 四处同步互换 + 备份 + 自检 + 回滚 |
| `tools/export_real_env_frames.py` | 导出 20 张环境相机帧 + 3 张仿真参考 + 拼图 + 绿区统计 |
| `inspect/cam_scan/cam_scan.csv` | 94 个 mp4 的逐文件指标与判定（已被 .gitignore 忽略） |
| `inspect/cam_scan/summary.txt` | 按数据集/流聚合的结论 + 异常与线索 |
| `inspect/cam_scan/montage_all_files.png` | 94 个文件各一帧缩略图，绿=固定相机、蓝=随臂相机 |
| `inspect/cam_scan/montage_env_vs_wrist_pairs.png` | 每个数据集一行：[env]×2 时间点 ｜ [wrist]×2 时间点 |
| `tools/scan_env_framing.py` | 量环境相机取景：垫子中心/直径/贴边/逐帧跳变 |
| `inspect/env_framing/` | 47 个环境相机文件的取景明细 + 汇总 + 取景拼图 |
| `inspect/env_cam_20/` | 20 张真机环境相机帧（640×480）+ 3 张仿真参考 + `montage_env_vs_sim.png` + `index.txt` + `pad_stats.txt` |
## 8. 环境相机取景实测：是「全部这个视角」还是「只有一部分」

测法：`tools/scan_env_framing.py` —— 每个环境相机 mp4 均匀采样 24 帧，用最大绿色连通域当拾取垫，
记录中心 `(cx, cy)`（0~1 归一化）、占画面面积 `area%`、贴边情况 `touches`（`R`=贴右、`B`=贴下、`inside`=完整）。

| 批次 | 文件 | 采样帧 | 垫子中心中位数 | 直径(px) | 完整帧占比 | 贴边帧占比 | 贴哪边 |
|---|---|---|---|---|---|---|---|
| real_train_fixed | 16 | 384 | (0.817, 0.837) | 212~268 | **0%** | **100%** | 右 + 下 |
| yyl100_dataset_fixed | 16 | 384 | (0.817, 0.833) | 212~268 | **0%** | **100%** | 右 + 下 |
| holdout_fixed | 12 | 288 | (0.822, 0.833) | 203~268 | **0%** | **100%** | 右 + 下 |
| real_train_corners | 1（22 ep） | 24 | (0.859, 0.639) | 234 | 75% | 25% | 右 |
| yyl100_corners | 1（30 ep） | 24 | (0.852, 0.634) | 240 | 75% | 25% | 右 |
| holdout_corners | 1（8 ep） | 24 | (0.860, 0.645) | 217 | 83% | 17% | 右 |

结论：

1. **不是随机"部分"，而是按批次分两个机位**。三个 `_fixed` 批次共 44 个视频、1056 个采样帧，
   **每一个文件、每一帧**垫子都同时贴住右边和下边（只有极少数帧只贴下边）—— 相机被固定成这样，不是偶发。
2. 三个 `_corners` 批次是**另一种机位**：垫子在 (0.85, 0.64)，直径小约 13%，多数帧完整，约 1/5 帧蹭到右边框。
   两批之间垫子中心纵向差 **0.20 个画面高 ≈ 96 px**，直径差约 13% —— 相当于相机被抬高/前移过。
3. **批次内部没有中途移动**：逐帧时间线（每个文件 40 个采样点）里 `cy` 只在 ±0.03~0.06 内抖动，
   不存在台阶式跳变；`cx` 的极差 0.05~0.24 来自机械臂遮挡垫子造成的检测抖动，不是相机移动。
4. 文件之间 `diam` 在 212~268 之间波动、而 `cx/cy` 几乎不动 —— 遮挡严重（臂压在垫子上）的文件测出的
   可见绿区更小，属于检测侧效应，不代表机位变了。

对照本仓库仿真：`videos/*_env.mp4` 里垫子中心 (0.71, 0.60)、直径 260 px，**更像 `_corners` 那批**
（cy 0.60 vs 0.64、直径 260 vs 234），完全不像 `_fixed` 那批（cy 0.83，垫子被右下角切掉）。

产物：`inspect/env_framing/env_framing.csv`（47 个文件逐文件）、
`inspect/env_framing/summary_env_framing.txt`、`inspect/env_framing/montage_env_framing.png`
（每个数据集一行，早/中/晚三帧，十字标出垫子中心，贴边帧标题变橙色）。

复核：

```bash
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/scan_env_framing.py --frames 24 --workers 8
```

## 9. 更正 §8 的两点：mp4 是打包容器 / 垫子到底被切多少

### 9.1 「47 个 mp4」不是 47 条数据：lerobot v3.0 把多条 episode 打包进一个 mp4

`info.json` 里 `video_files_size_in_mb: 200`，录制脚本把多段 episode 顺序拼进同一个 mp4，
每条 episode 的时间段记在 `meta/episodes/**/*.parquet` 的
`videos/<相机>/{file_index, from_timestamp, to_timestamp}` 里。所以按文件数出来的数会远小于实录条数：

| 数据集 | 实录条数 | 总帧数 | env 路 mp4 文件 | 每个 mp4 装几条 |
|---|---|---|---|---|
| real_train_fixed | 139 | 62273 | 16 | 8.7 |
| yyl100_dataset_fixed | 151 | 67649 | 16 | 9.4 |
| holdout_fixed | 12 | 5376 | 12 | 1.0 |
| real_train_corners | 22 | 8588 | 1 | 22.0 |
| holdout_corners | 8 | 3282 | 1 | 8.0 |
| yyl100_corners | 30 | 11870 | 1 | 30.0 |
| **合计** | **362** | | **47** | |

按 episode 逐条测（`tools/scan_env_framing.py --unit episode`，读上表的时间段逐条采样 8 帧）：

| 数据集 | 条数 | 每帧都贴边 | 部分贴边 | 从不贴边 |
|---|---|---|---|---|
| real_train_fixed | 139 | **139** | 0 | 0 |
| yyl100_dataset_fixed | 151 | **151** | 0 | 0 |
| holdout_fixed | 12 | **12** | 0 | 0 |
| real_train_corners | 22 | 4 | 3 | 15 |
| holdout_corners | 8 | 1 | 1 | 6 |
| yyl100_corners | 30 | 5 | 5 | 20 |
| **合计** | **362** | **312** | **9** | **41** |

结论不变但更精确：**`_fixed` 三批 302 条实录全部（302/302）垫子被裁**；`_corners` 三批才是另一种机位。

### 9.2 单帧判「贴边」会被机械臂遮挡污染，改用「绿色掩码时序并集」

机械臂压在垫子上时绿色连通域会缩小、可能离开边框，于是同一机位会被判成「完整」。
把 16 帧的绿色掩码做并集（遮挡在不同帧落在不同位置，并集能还原垫子真实投影范围）后：

| 数据集 | 文件 | 垫子可见 x 范围 | 可见 y 范围 | 可见 宽×高 (px) | 面积% | 贴边 |
|---|---|---|---|---|---|---|
| real_train_corners | 1 | 358 ~ 639 | 162 ~ 421 | 282 × 260 | 15.5 | 右 |
| holdout_corners | 1 | 384 ~ 639 | 160 ~ 417 | 256 × 258 | 16.1 | 右 |
| yyl100_corners | 1 | 352 ~ 639 | 160 ~ 419 | 288 × 260 | 17.0 | 右 |
| real_train_fixed | 16 | 370~420 ~ 639 | 284~291 ~ 479 | 220~270 × **189~196** | 12.1~15.1 | 右 + **下** |
| holdout_fixed | 12 | 371~435 ~ 639 | 284~291 ~ 479 | 205~269 × **189~196** | 11.5~14.2 | 右 + **下** |
| yyl100_dataset_fixed | 16 | 370~420 ~ 639 | 284~291 ~ 479 | 220~270 × **189~196** | 12.1~14.9 | 右 + **下** |

读法（垫子真实直径按可见高度 ~258~270 px 估）：

- **`_corners` 三批：垂直方向完整**（可见高度 256~260 ≈ 直径），只有**右缘正好压在右边框**上
  —— 属于像素级的「蹭边」，不是被切掉一大块。
- **`_fixed` 三批：可见高度只有 189~196 px**，比直径少 **65~80 px（约 25~30%）**
  —— 垫子**下缘被画面切掉**，同时右缘也压在边框上。

所以 §8 里「垫子被右下角裁掉」这个说法对 `_fixed` 成立（下边确实被切），对 `_corners` 应改成
「右缘正好贴边框、上下完整」。

产物：`inspect/env_framing/env_framing_episodes.csv`、`summary_env_framing_episodes.txt`（按条）、
`inspect/env_framing/env_union.csv`、`summary_env_union.txt`（按时序并集，抗遮挡）。

复核：

```bash
# 按实录逐条（需要 pyarrow）
PYTHONPATH=$PWD/.cache/pylibs env -u PYTHONHOME /usr/bin/python3 tools/scan_env_framing.py --unit episode --frames 8 --workers 8
# 抗遮挡的绿色并集
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/scan_env_framing.py --unit union --frames 16 --workers 8
```

## 10. 用仿真复刻这个视角（已落地）

用户决定：不改真机相机，让仿真 `env_camera` 复刻真机 `_fixed` 批次的视角。

### 10.1 解算：只动 pos，xyaxes / fovy 不变

目标由 §9 的绿色并集实测给出（640×480）：垫子宽 **270 px**、左缘 **370**、右缘 **639**（压在右边框）、
上缘 **286**（284~291），下缘超出画面 → 可见高 **194 px**（真机 189~196）。

`tools/solve_env_camera.py` 用针孔模型（fovy=30 → f_px=895.69）迭代：
像素宽度反解深度（`w ∝ 1/depth`），再由目标像素反投影出相机位置，每次都用与真机**同一个绿色检测器**复测。
一次迭代即收敛：

| | pos | 实测垫子 bbox | 宽×高 | 可见高 |
|---|---|---|---|---|
| 改前 | `0.65 0 0.55` | (322,178,581,398) | 260×221 | 未切，整块在画面内 |
| **改后** | **`0.5814 -0.0418 0.5919`** | **(370,286,639,479)** | **270×194** | **194 px（下缘被切）** |
| 真机 `_fixed` 目标 | — | (370,286,639,479) | 270×~258 | 189~196 px |

即：相机沿视线方向前移约 2.4 cm、向 −y 侧移约 4.2 cm（让垫子落到右下角并被底边切掉）。

### 10.2 顺带修掉一个会让改动"静默失效"的坑

`collect/collect.py::apply_camera_jitter` 原本把基准位姿写死成 `np.array([0.65, 0.0, 0.55])`，
只改 XML 的话每条 episode 都会被它拽回旧值、而且**画面看起来完全正常**，属于最难查的那类静默错。
已改成从模型现读：

```python
base_pos  = np.array(model.cam_pos[cid], dtype=float).copy()   # = XML 里的 env_camera
base_fovy = float(model.cam_fovy[cid])
```

（`render_episode` / `extrinsics_only` 都是先 `_pristine_model()` 还原再调用，所以此刻就是名义值。）

### 10.3 抖动之后还剩多少"贴着边框"

`tools/probe_env_cam_jitter.py` 用 12 个 seed 跑 `apply_camera_jitter` + 渲染 + 同一个绿色检测器：

| 指标 | 仿真（抖动后 12 帧） | 真机 `_fixed`（302 条） |
|---|---|---|
| 贴下边框 | **100%**（12/12） | **100%** |
| 完全在画面内 | **0%** | 0% |
| 贴右边框 | 50%（6/12） | 100% |
| 垫子左缘 x0 | 336~423 | 370~420 |
| 垫子上缘 y0 | 260~303 | 284~291 |
| 垫子宽 | 217~284 | 205~270 |

读法：**"下缘被切"这个主要特征在抖动下 100% 保住**，`0%` 整块进画面也保住了；
差异在于仿真的抖动（±1~3 cm / fovy±3° / yaw±2° / pitch±2°）比真机批次内的实际位移大，
所以「贴右边框」只有一半、上缘散布 ±20 px。这是**刻意保留的域随机化**，不是 bug ——
想更贴真机就把 `camera.env_pos_jitter_m` 压小，想更鲁棒就保持。

### 10.4 版本与既有产物

- `DATASET_VERSION` 由 `sdg-v6` → **`sdg-v7`**（`collect/collect.py`），`docs/PHYSICS_FACTS.md` §1/§7/§8 同步。
- `scripts/` 里的 1000 条母本仍是 **sdg-v6（旧相机）**，`ep_000000.json` 里 `dataset_version` 可查；
  `videos/`、`dataset/`、`inspect/env_cam_20/sim__*.png` 也全是旧相机出的。
  **要得到新视角的数据必须重跑采集**（`collect/collect.py --reset …`），否则仿真侧仍是旧取景。
- 演示管线自检（`render/test_render.py record` + `render`）在新相机下 15 项全部 ✅，
  其中「物块像素数 首/中/末 = 4371/5378/5636（需 >200）」确认药盒仍在画面里。

### 10.5 复核命令

```bash
# 解算（只打印，不改文件）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 tools/solve_env_camera.py
# 抖动后取景分布
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 tools/probe_env_cam_jitter.py --n 12
# 演示管线自检
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 render/test_render.py record --out <tmp>
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 render/test_render.py render --out <tmp> --npz <tmp>/episode.npz
```
