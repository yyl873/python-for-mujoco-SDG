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
| yyl100_lance_fixed | 全为 Lance 格式无 mp4，无法解码验证 | 同左 | 仅从元数据推断「疑似同样互换」，见 §5 |
| yyl100_lance_corners | 同上 | 同上 | 仅从元数据推断「标签正确」，见 §5 |

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

- **yyl100_lance_fixed、yyl100_lance_corners**：视频以 Lance 格式存于 `videos.lance/`（各 2.3 GB），
  没有 mp4 文件，本机没有可用的 `lance` 模块（`~/miniforge3/envs/lerobot` 与 `lerobot312` 两个 conda 环境
  目前一启动就崩，系统 Python 无 lance/pyarrow），因此**无法解码验证**。
  只能按 §3 的元数据规则推断：`yyl100_lance_fixed` 的 features 首相机是 `wrist` → 疑似同样互换；
  `yyl100_lance_corners` 首相机是 `env` → 标签正确。
  正确做法是**先修 mp4 版再重跑这两份的转换脚本**，而不是手工改 Lance。
- 本机 conda 环境（`~/miniforge3/envs/lerobot`、`lerobot312`）启动即崩，与本次修改无关，但会让依赖它们的
  导出脚本（`collect/export_lerobot.py`）暂时跑不起来。
- 真机**环境相机的取景本身有问题**：20 张环境相机帧里 20 张的绿色拾取垫都被画面右边和下边裁掉
  （包围盒恒为 `x1=639 / y1=479`）。这和"相机视角好像变了"一致，属于**物理相机朝向**问题，
  与标签互换无关，是否要调真机相机或反过来把 sim 的 `env_camera` 对齐到这个歪视角，需另行决定。

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
| `inspect/env_cam_20/` | 20 张真机环境相机帧（640×480）+ 3 张仿真参考 + `montage_env_vs_sim.png` + `index.txt` + `pad_stats.txt` |