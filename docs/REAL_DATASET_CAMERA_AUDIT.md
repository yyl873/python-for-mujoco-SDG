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
