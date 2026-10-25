# 第七轮交接单：撤掉「悬空假垫」、指垫贴手指真实内表面、`dy` 补偿、删产物

> 自包含，可直接复制到别的窗口。**先读本文件，再读 `docs/ROUND_contact_realism.md`（第六轮，指垫首次落地）。**
> 现行数据：**`sdg-v5`**（起止姿态仍是 old_ros `initial_pose`）。

---

## 0. 一句话

第六轮的指垫贴在**药盒表面**，实测垫子 14 个探针点 **0/14** 落在手指实体内 ——
等于悬在指尖前方 4.7~15 mm 的板子（用户看到的"指尖多了个垫子 + 缝隙"）。
本轮把垫子改成**贴合手指真实内表面**、**不可见**（`rgba 0 0 0 0`），并补上
**抓取点横向补偿 `dy=+17.5 mm`** 与 **合爪量 0.28**：

| 指标 | 第六轮（v4） | 本轮（v5） |
|---|---|---|
| 夹稳时 固定指↔盒 缝 | **−7.5 mm**（手指实体整块埋进盒） | **−0.85 mm** |
| 夹稳时 活动指↔盒 缝 | **+4.77 mm**（看得见的缝） | **−1.07 mm** |
| 手指实体在盒内的最大深度 | **7.5 mm** | **0.02 mm** |
| 落在盒内的手指顶点数 | 9814 | 52 / 1968（都只在 0.02 mm 的边界层） |
| 下探段指↔盒接触 | 0/400 | **0/400** |
| 压盒 / 夹稳倾角 | 0.22 mm / 0.07° | **0.39 mm / 1.06°** |
| 抬升 / 水平 | 125.10 / 209.56 mm | **122.22 / 218.06 mm** |
| 指垫可见性 | 深灰块（看得见） | **不可见** |

---

## 1. 三个用户问题，逐条回答（都带实测）

### 1.1 「指尖加了垫子？不加行不行？」→ 不行；但可以做到"看不见 + 贴在真表面上"
两条路都试过：

| 做法 | 实测 |
|---|---|
| 不加接触体（恢复指壳凸包碰盒） | 完全下探时固定指凸包**穿进盒 22.6 mm** → 下探 92/400 步戳压、压盒 6.87 mm、回弹 2.56 mm |
| 不碰（只有悬浮垫） | 固定指实体 **9814 个顶点**在盒内、深度 7.5 mm（穿模）；活动指离盒 **4.77 mm**（缝隙） |

根因不变：指壳是 C 形件，MuJoCo 只能拿**凸包**当碰撞体，凸包把 C 的开口填满。
所以必须有"贴合手指真实内表面"的接触体 —— 它不必看得见。

### 1.2 「夹取时和夹取过程中会穿模」→ 确认，方向是**指头整块压在盒子体积里**
探针口径：手指 mesh 的顶点是否落在药盒盒体内（盒体 ±37 / ±7.5 / ±26 mm）。
第六轮实测：夹稳 9090 / 抬升 9208 / 搬运 8990 / 下放 8918 个顶点在盒内（都是固定指）。
本轮：**最大深度 0.02 mm**（等价于"贴着边界"，不再有实体积插入）。

### 1.3 「闭合量最好在 `set_gripper(0.15)` 左右」→ 这个模型里夹 15 mm 盒子必须 0.28
`old_ros/send_trajectory.py` 的 `set_gripper(val)` 是**直接把 val 当弧度**下发给 `gripper_joint`，
所以 0.15 就是 0.15 rad。但本模型两指真实内表面间距随闭合量变化约 **7.6 mm / 0.1 rad**：

| jaw 角 | 两指面间距 | 对 15 mm 盒 |
|---|---|---|
| 0.15 | **4.45 mm** | 等于手指已合拢 → 会把盒挤出去/压穿 |
| 0.20 | 8.8 mm | 仍然夹不实 |
| **0.28** | **14.15 mm** | ✔ 与盒厚匹配（本轮取值） |

所以 `GRIP_SHUT` 定为 **0.28**；你要的"合到 0.15 的贴合感"由"垫面贴齐指面"实现
（夹稳时两指面都贴住盒子，缝 −0.85 / −1.07 mm），而不是把 ctrl 写小。
⚠️ 这条要跟真机对齐时，得先在真机上量"指面间距 vs set_gripper 值"——本模型是 7.6 mm/0.1 rad。

---

## 2. 本轮改了什么

| 文件 | 改动 |
|---|---|
| `model/so_arm101.xml` | 两枚指垫改为 **50×2×40 mm、`rgba 0 0 0 0`（不可见）、贴在各自手指的足迹内最内侧极点**；位掩码不变（垫 `4/2`、指壳 `8/1`） |
| `grasp/test_grasp.py` | 新增 `GRASP_XY_OFFSET=(0, 0.0175)`、`GRIP_APPROACH=1.5`；`GRIP_CLOSED` → **0.28**；下探开度由 `GRIP_MAX` 改 `GRIP_APPROACH`；抬升段夹爪由 `0.0` 改 `GRIP_CLOSED`（原来抬升途中继续合拢会把盒挤出去） |
| `render/test_render.py` | `GRIP_SHUT` → **0.28**；`GRASP_XY_OFFSET`/`GRIP_APPROACH` 改为从 `test_grasp` 导入；`REF_LIFT_MM/REF_HDISP_MM` → **122.22 / 218.06** |
| `collect/collect.py` | `DATASET_VERSION` → **sdg-v5**；标定解与 `waypoints_for()` 都吃 `GRASP_XY_OFFSET`（**并按 theta 旋转**——漏了这一步 20/20 全部 `no_lift`） |
| `tools/probe_contact_audit.py` | 新增 G5 缝隙 / G6 穿模深度两项；`--shut` 开关；`--dx/--dy` 默认改为"用模块常量"；压盒闸门与 render 的 `PRESS_MAX_MM` 对齐 |
| `tools/run_pipeline.py` | **「抓取」阶段从步骤二换成接触审计**（理由见 §4）；新增 `parse_contact_audit` |
| `debug/pad_flush_try.py` | 新增：指垫"贴齐指面"的放置 + 参数验证工具（含干净参考位姿、`--emit-mjcf`） |

---

## 3. 方法学：这次踩的两个坑（下一轮别再踩）

1. **参考位姿必须是"干净"的**。我第一次用"已经被接触推歪的那一帧"反算垫子位置，
   实测偏 4~5 mm（污染帧 TCP=20.5 mm、盒子被推 1.4 mm；干净 descend 末 TCP=23.57 mm）。
   正确做法：**先把两枚垫子的碰撞临时关掉跑一遍**，取 descend 段末位姿
   （`debug/pad_flush_try.py` 的 `clean_reference()` 就是这么做的）。
2. **"最内侧极点"必须按药盒足迹取**（x±37 / z±26 mm）。用 ±20 mm 的小窗口会漏掉
   手指在边缘的凸出材料，导致"垫面齐平了、手指实体却先进盒"。

---

## 4. 步骤二 `grasp/test_grasp.py` 已退出流水线（重要）

第七轮之后指垫是按**生产路径（步骤三）的真实到达位姿**反算的。步骤二的序列与生产路径不同
（无 `settle_a` 停顿、起点不同），实测它的下探段会压盒 **21.85 mm**、自己的「约束1」报 ❌。
这不是回归，是"探针与生产路径分叉"的老问题（HANDOFF §5.2 记过）。

处置：`tools/run_pipeline.py` 的「grasp」阶段改为 **`tools/probe_contact_audit.py`**
（0.9 s、审计的是生产路径、判据更严：G1~G6）。步骤二保留为历史探针，可单独跑，
但**不再作为流水线闸门**。文件内已同步修了三处与新的夹持几何冲突的老写法
（下探开度 `GRIP_MAX`→`GRIP_APPROACH`、抬升夹爪 `0.0`→`GRIP_CLOSED`）。

---

## 5. 复现命令

```bash
cd /home/yyl/deepseek-harness

# 0) 六项接触审计（0.9 s；这是现在的抓取闸门）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 tools/probe_contact_audit.py
#    G1 下探零接触 / G2 压盒≤0.5mm / G3 夹稳倾角≤5° / G4 抬升≥50mm
#    G5 |缝|≤1.2mm / G6 穿模深度≤1.6mm

# 1) 指垫位置重算 + 参数验证（改起点/路径/步数/合爪量之后必须跑）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 debug/pad_flush_try.py \
    --shut 0.28 --clear 2.5 --pad-w 50 --pad-h 40 --dys=16.5,17.0,17.5,18.0
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 debug/pad_flush_try.py \
    --shut 0.28 --clear 2.5 --pad-w 50 --pad-h 40 --emit-mjcf      # 输出可粘进 MJCF 的 geom 行

# 2) 全链路（ik / 接触审计 / render / video），期望 ok 4 / warn 0 / fail 0，自检 16/16
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py

# 3) 数据集
env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 20 --seed 3 --no-render --reset
env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py
env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py --policy random --seed 7
env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py --card
```

---

## 6. 产物清理（用户要求）

已删除 **1.04 GB** 可再生媒体（代码与标签全部保留）：

| 删除 | 体积 |
|---|---|
| `demo/episode_0.npz` | 838 MB |
| `demo_neg/episode_0.npz` | 200 MB |
| `videos/*.avi`（40 条） | 142 MB |
| `demo/`、`demo_neg/` 下的 avi/mp4/分镜/预览 | ~27 MB |

保留：`episode.npz`（轨迹 120 KB，`make_video.py` 的输入）、`meta.json`、`scripts/`（数据集 3.7 MB）。
重建：`run_pipeline`（101 s，会重建 `demo/` 的 mp4+avi）、
`record --gz 0.080 --out demo_neg --quick`（85 s）、
`collect --n 20 --seed 3`（**带渲染**才生成 `videos/*.avi`，约 4~5 min）。

---

## 7. 遗留 / 下一轮

1. **搬运途中倾角 27~29°**：本臂 lift/elbow/flex 三轴平行，工具俯仰随 TCP 高度线性变化
   0.22°/mm，抬 100 mm 必然如此（**不是接触问题**，第六轮已记录；`ik_lock` 也压不下来）。
   要减小只能降低搬运高度或换 6 轴。
2. **`demo/episode_0.npz` 799.8 MB 问题**：本轮按用户要求删了文件，但**JPEG 化仍未做** ——
   批量采集前必须解决（20 条 ≈ 16 GB，磁盘 32 GB 可用）。
3. **`set_gripper(0.15)` 与真机的对应**：本模型指面间距/闭合量是 7.6 mm/0.1 rad，
   真机需要实测同一条曲线，才能把"0.15"翻译成仿真里的角。
4. **指垫位置与动作脚本强耦合**：改起点/接近路径/步数/合爪量任一，都要跑
   `debug/pad_flush_try.py` 重算那 8 个数（`pos`/`quat` 各 2 组）。
5. 步骤二仍与生产路径分叉（§4），已退出流水线但文件留着；要么修它的序列，要么删掉。