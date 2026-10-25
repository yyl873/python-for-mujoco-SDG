# 本轮交接：录制起止姿态改成 old_ros 的 initial_pose（sdg-v3）

> 2024-11 第五轮。上一轮是 `docs/ROUND_golden_start.md`（起点 = 遥操 golden pose，sdg-v2）。
> 本轮把**录制开始与结束的姿态**换成真机 ROS 录制脚本的 `initial_pose`，并把数据升到 `sdg-v3`。
> 代码真身：`render/test_render.py`、`grasp/test_grasp.py`、`collect/collect.py`、
> 新增 `tools/probe_grasp_robustness.py`。真机参考：`old_ros/send_trajectory.py`（**只读，禁止运行**）。
>
> 本机环境：所有 python 调用必须 `env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3`
> （否则报 `No module named 'encodings'`）；渲染加 `MUJOCO_GL=egl`。

---

## 1. 目标与结论

用户要求：**开始录制和结束录制都用 old_ros 录制脚本的姿态**，并参考它的动作流程
（`old_ros/send_trajectory.py:369` 开录前、`:552` 录完后，用的是同一个

```
initial_pose = [0.0, -1.74535, 1.54036, 1.23277, -1.80689, 0.0]   # 弧度 + 夹爪
             = pan 0° / lift −100° / elbow 88.26° / wrist_flex 70.63° / wrist_roll −103.53°
```

结论：**已落地**（`DATASET_VERSION = "sdg-v3"`，`START_POSE = "oldros"`）。第 0 帧实测
`[-0.0, −100.00, 88.26, 70.66, −103.53]°`、爪甲 0.0；总步数仍 **7500 步 = 15.0 s / 455 帧**。
**意外收获：抓取不再临界** —— 20 条验收从 v2 的 18/20 变成 **20/20**（候选次数 1~7，v2 是 1~13）。

代价/注意：
- 换起点**必须**同时把 `(init_lift, approach_retract, approach_above)` 从 `(500,800,700)`
  改成 `(700,800,500)`，否则抓取全灭（§2.2）。
- old_ros 的 `safe_z = 0.20` **不能照搬**，`RETRACT_Z` 必须保持 0.24（§3.2）。
- 起手爪甲仍按上一轮用户要求保持收起，不照搬 old_ros 的 `set_gripper(0.8)`（§3.3）。

---

## 2. 可行性分析（动手前先算清楚，全部实测）

### 2.1 姿态本身：合法、安全、可达

| 关节 | initial_pose | 模型限位 | 判定 |
|---|---|---|---|
| shoulder_pan | 0.00° | ±110.00° | ok |
| shoulder_lift | −100.00° | ±100.00° | **超限 2.07e−05 rad**（≈0.001°，标定值舍入） |
| elbow_flex | 88.26° | −96.83° ~ 88.24° | **超限 3.60e−04 rad**（≈0.02°） |
| wrist_flex | 70.63° | ±91.67° | ok |
| wrist_roll | −103.53° | ±180.00° | ok |

两处超限在 `Arm.apply_start_pose("oldros")` 里 `np.clip` 掉，落定后实测 `qpos` 就是限位上的值。

正运动学（`mj_forward`，只读，无 mj_step）：

| 姿态 | TCP (mm) | 离药盒 (mm) | jaw→药盒 (mm) | 臂部最低点 z (mm) | 接触 |
|---|---|---|---|---|---|
| **old_ros initial_pose** | (168.1, 7.6, 21.6) | **151.4** | 203.0 | 62.4 | 0 |
| golden（v2 起点） | (231.9, 149.7, 38.4) | 87.5 | 104.1 | — | 0 |
| home（q=0 塌陷位） | (391.4, −0.3, 226.3) | 238.3 | 238.5 | — | 0 |

也就是说 old_ros 起点比 golden **离药盒更远、更安全**：v2 那轮"手指贴着药盒、开口就有推倒风险"
的隐患在这里天然不存在。

### 2.2 只换起点不重扫步数 = 抓取全灭

单次运行、其它一概不动（步数配比仍是 golden 那套 500/800/700）：

| timing_scale | 0.90 | 0.95 | 1.00 | 1.05 | 1.10 |
|---|---|---|---|---|---|
| 抬升 (mm) | 0.00 | 0.00 | **0.00** | 0.00 | 0.00 |

失败签名与 v2 记录的"临界翻车"完全一致：合爪后药盒被横向甩射 37 mm、z 27.77 → 9.27 mm。
逐段追踪定位到问题不在路径而在**合爪起点状态**：

```
段          obj_z (mm)      tcp_z (mm)     结论
init        27.8 → 27.8     22  → 22       与 golden 不同（golden 是 37→36）
approach    27.8 → 27.8     22  → 162      到高位，未碰盒
descend     27.8 → 21.5     162 → 39       与 golden **逐段同值**（21.5 / 39）
grasp       20.9 → 9.3      35  → 33       ★ 翻车在这里（min 2.1，盒被压进垫子再甩出）
```

### 2.3 重扫三段自由空间步数后：救回来，而且更稳

固定 `init_lift + approach_retract + approach_above = 2200`（其余段不动，总步数仍 7500），
共扫 41 组，**5 组成功**：

| 组合 (init_lift, retract, above) | 抬升 (mm) | 水平 (mm) |
|---|---|---|
| 600, 800, 600 | 103.55 | 221.0 |
| **700, 800, 500**（选定） | 103.45 | 221.0 |
| 700, 1000, 300 | 103.23 | 221.0 |
| 800, 700, 500 | 103.60 | 221.0 |
| 800, 900, 300 | 103.22 | 221.0 |

对这几组再做 `timing_scale 0.90~1.10` 十档复核（每组真跑 10 次）：

| 起点 / 组合 | 10 档通过 | 抬升范围 (mm) |
|---|---|---|
| **oldros / (700,800,500)** | **10/10** | 103.41 ~ 103.88 |
| oldros / (800,700,500) | **10/10** | 103.41 ~ 103.88 |
| oldros / (700,1000,300) | **10/10** | 103.04 ~ 103.66 |
| oldros / (800,900,300) | **10/10** | 103.05 ~ 103.58 |
| oldros / (600,800,600) | 4/10 | 0 或 103.5 |
| golden / (500,800,700)（v2 现状） | **2/5**（0.95 / 1.05 / 1.10 全灭） | 103.25 ~ 103.31 |

=> **old_ros 起点 + (700,800,500) 反而消除了 v2 的临界性**，这也是本轮接受率回到 20/20 的原因。

### 2.4 A/B 陷阱：起点与步数配比是**耦合**的，不能单独 A/B

实测（都用生产代码路径）：

| 起点 | 步数配比 | 结果 |
|---|---|---|
| oldros | (700,800,500) | 十档节奏全过 ✅ |
| oldros | (500,800,700) | 5 档全灭 ❌ |
| golden | (500,800,700) | 2/5（0.95/1.05/1.10 灭） |
| golden | (700,800,500) | 单跑 1.00 档：抬升 **0.00 mm** ❌ |

所以 `render/test_render.py record --start-pose golden`（会沿用 v3 的 700/800/500）**必然失败**，
那是预期行为，不是回归。要 A/B golden 必须同时换回 (500,800,700)，最省事的办法是用探针：
`tools/probe_grasp_robustness.py --start-pose golden --combos "500,800,700"`。

⚠️ 顺手纠正一个旧论断：`make_step_plan` 的老注释写"接触段步数不变，所以抓得住不受影响" ——
**错**。`settle_a` 本身就在可缩放集合里，而更关键的是"进入 descend 时的残留速度/PD 滞后"
由前面几段自由段的节奏决定。旧文档给的失败档位清单（v2 写的 0.90/0.92/0.95/0.98/1.03/1.06）
本轮也复现不出来（实测 0.90 与 1.00 过、0.95/1.05/1.10 灭）——**这张临界集本身不稳定，
不要把它当铁律，要现测**。现测工具就是 `tools/probe_grasp_robustness.py`（§6）。

---

## 3. 实现（四个文件 + 一个新工具，**没改 MJCF**）

| 文件 | 改动 |
|---|---|
| `grasp/test_grasp.py` | 新增 `OLDROS_JOINTS` / `OLDROS_Q_DEG` / `OLDROS_Q_RAW`（度→弧度，按 `ARM_JOINTS` 顺序）；`Arm.apply_start_pose()` 新增 `pose="oldros"` 分支（写 `qpos` + `ctrl`，两处超限 `np.clip`）；`reset(pose=)` 文档同步。 |
| `render/test_render.py` | `N_INIT_LIFT 500→700`、`N_ABOVE 700→500`（合计不变，`TOTAL_STEPS` 断言仍 7500）；`POSE_WITH_INIT_LIFT = ("golden","oldros")` 取代只认 golden 的 `is_golden`；`START_POSE_DEFAULT="oldros"`；`record` 子命令加 `--start-pose {oldros,golden,home}`；`record_meta` 新增 `start_joints_deg`（实际首帧关节，度）+ `dataset_version="sdg-v3-demo"`，`golden_joints_deg` 仅在 golden 下保留（旧数据兼容）。 |
| `collect/collect.py` | `DATASET_VERSION="sdg-v3"`、`START_POSE="oldros"`；`_start_q()` 把姿态名映射到关节目标（数值只在 test_grasp 定义一次）；每条 meta 增 `start_joints_deg`；聚合 `index.json` 增顶层 `start_pose` / `start_joints_deg`。 |
| `tools/probe_grasp_robustness.py` | **新增**。换起点/换配比/换节奏后 30 秒内回答"还抓得住吗"，纯内存跑物理、不落盘。 |

### 动作时序（发给下游必须一致；总步数 7500 = 15.0 s 未变）

```
init_hold  200 (0.4s)  在 initial_pose 静止，爪甲收起（0.0）
init_lift  700 (1.4s)  关节空间竖直抬出到 z=0.24（≈0.156 m/s）
approach_retract 800   高位撤回点 → 途中张爪到 1.5
approach_above   500   药盒正上方 z=0.16
descend 400 / grasp 300 / settle_a 400 / lift 400   ← 接触段，锁定不许动
settle_b 300 / transport 600 / place 300 / pre_open 100 / release 300 / drop 300
retract 450 / home 1300 / tail 150
合计 7500 步 = 15.0 s / 455 帧 @30fps
```

`init_lift` 那 700 步的帧仍标 `phase="approach"`（`init` 只表示"静止的起点"），
数据契约与 v2 逐字段相同 —— `verify` 的 11 项、`dataset.py` 的 6 条契约都不用改。

### 3.1 与 old_ros 动作流程的对照

| old_ros | 本片 | 是否一致 |
|---|---|---|
| 开录前 `send_joint_waypoints(q_now, initial_pose, 3.0s)` | `settle` 前 `apply_start_pose("oldros")` 写 qpos+ctrl | ✅ 起点姿态相同 |
| `start_recording()` 后先静止 | `init_hold` 200 步静止 | ✅（0.4 s vs 它的 sleep 1.0 s） |
| 动作1 原地垂直抬升到 `safe_z=0.20` | `init_lift` 700 步抬到 `RETRACT_Z=0.24` | ⚠️ 高度不同（见 §3.2） |
| 动作2 高空对齐（抓取姿态反推高位解） | `approach_retract` + `approach_above` | ✅ 结构相同 |
| 动作3 精确下探（含 −4/+10 mm 局部补偿） | `descend` 400 + `gz=0.012` 标定补偿 | ✅ 等效 |
| 动作4 `set_gripper(0.15)` 合爪 | `grasp` 300 步合到 `GRIP_SHUT=0.20` | ⚠️ 数值语义不同（见 §3.3） |
| 动作5 垂直拔起 → 动作6 飞木盒上方 → 6.5 下放 | `lift` → `settle_b` → `transport` → `place` | ✅ |
| 动作7 开爪 + 45 帧自由落体 | `release` 300 + `drop` 300 | ✅（真物理，未写 qpos） |
| 动作7.5 原地垂直撤离 +0.10 m | `retract` 450 → `RETRACT_Z` | ✅ |
| 动作8 回 `end_pose`（= initial_pose），静止 1.0 s | `home` 1300 + `tail` 150 = 2.9 s | ⚠️ 时长不同（见 §3.4） |

### 3.2 有意偏离①：抬升高度 0.24 而不是 old_ros 的 0.20

沿用（700,800,500）与十个节奏档，只改 `RETRACT_Z`：0.18 / 0.20 / 0.22 / 0.24 / 0.26 / 0.28 / 0.30
中 **只有 0.24 抓得住**，0.20 与 0.22 在 0.95 / 1.00 / 1.05 三档全灭（抬升 0）。
`→ RETRACT_Z 保持 0.24，old_ros 的 safe_z 是给真机标定的，本模型不适用。`

### 3.3 有意偏离②：起手爪甲保持收起，不照搬 `set_gripper(0.8)`

old_ros 在开录后立刻把夹爪张到 0.8（ROS 侧 joint 值，单位 rad，与本模型 ctrl 语义不同源：
本模型 0=收起、2.232=全开、闭合用 0.20）。上一轮用户明确要求"起手和收尾爪甲是收起的预设姿态，
张开只出现在抓取前后"，所以这里保持 `GRIP_PRESET=0.0`，张开仍在 approach 段做。
实测首帧爪甲 0.000、init 段漂移 ≤0.0004 rad（≈0.02°，PD 微渗）。

### 3.4 有意偏离③：结束段 2.9 s vs old_ros 的 4.0 s

old_ros 是"3.0 s 回位 + 1.0 s 静止"；本片受"总时长 7500 步不变"约束，`home 1300 + tail 150 = 2.9 s`。
要严格对齐 4.0 s 就得从别处挪 1.1 s，会破坏刚验证过的步数配比 —— 需要的话另起一轮重扫，别顺手改。

---

## 4. 验收（全部重跑过，三态结论）

| 条目 | 关键量 | 状态 |
|---|---|---|
| `render/test_render.py all` | 自检 **15/15 全绿**、`总判定 ✅`、rc=0 | **ok** |
| 第 0 帧姿态 | `[-0.00, −100.00, 88.26, 70.66, −103.53]°`、爪甲 0.000、init 段药盒 Δz 0.000 mm | **ok** |
| 末帧回位 | 各关节偏差 0.8 / 1.8 / 1.0 / 2.0 / 0.4°（阈值 5.7°）、末帧夹爪 0.000 | **ok** |
| 轨迹一致性 | 抬升 **+103.45 mm**（基准 104.13 ±5）、水平 **221.01 mm**（基准 222.0 ±5） | **ok** |
| 时长/帧数 | 7500 步、455 帧、**14.998 s**、30 fps | **ok** |
| `tools/probe_grasp_robustness.py` | oldros + (700,800,500) → **10/10**；golden + (500,800,700) → 2/5（对照） | **ok** |
| `collect/collect.py --n 20 --seed 3 --no-render` | **接受率 20/20**，抬升 101.5~104.5 mm，候选次数 1~7，耗时 6.7 s | **ok** |
| `collect/check_dataset.py` | **20/20 通过**、清单自洽 | **ok** |
| `collect/split.py --policy random --seed 7` | **16 / 2 / 2**，`index.json.splits` 已回填 | **ok** |
| `collect/stats.py --card` | sdg-v3、接受率 100%、数据卡已重写 | **ok** |
| `tools/run_pipeline.py` | 4 段 **ok 3 / warn 1 / fail 0**，**100.7 s**；render 自检 15/15、success=true（warn 来自步骤二"约束1 ❌"，与起点无关） | **ok** |
| `demo_neg/`（负样本通路） | 新起点下重做：`--gz 0.080 --quick` → `success=false`、`lift_too_low`、verify 15 行全 ⚠️/✅ 无 ❌ | **ok** |

对比 v2：接受率 18/20 → **20/20**；候选次数 1~13 → 1~7；抬升 102.1~104.7 → 101.5~104.5 mm。

产物（sdg-v3）：
- `demo/env_camera.mp4`、`demo/wrist_camera.mp4`、`demo/sim_{env,wrist}_640x480.avi`、
  `demo/contact_sheet.png`、`demo/meta.json`、`demo/episode{,_0}.npz`
- `scripts/ep_000000..19.{npz,json}`、`scripts/index.json`、`scripts/index/`（20 片段 + `_policy.json`）
- `scripts/splits/{train,val,test}.json`、`scripts/DATA_CARD.md`、`scripts/stats.{json,png}`、`scripts/report.json`
- `videos/ep_000000_{env,wrist}.avi`（抬升 103.31mm / 水平 187.6mm / 7482 步 / 14.96 s）、
  `videos/ep_000001_{env,wrist}.avi`（101.69 / 201.7 / 7352 步 / 14.70 s）
- `demo_neg/`（负样本对照件，200 MB，含 README 说明怎么重做）
- 上一批 sdg-v2 的索引快照留在 `scripts/index.sdg-v2{,.json}`（只读备份，别删）

---

## 5. 现在的"抓取鲁棒性"地图（下一轮必须记住）

1. **换起点 / 改自由段步数 / 改 roll-gz 之前，先跑
   `tools/probe_grasp_robustness.py`**（30 秒）。它跑的就是生产代码路径
   （`run_episode` + `judge`），纯内存、不落盘。
2. 当前配置：oldros + (700,800,500) + roll −90° + gz 0.012 → 十档节奏全过。
   同起点还有三组备胎：(800,700,500)、(700,1000,300)、(800,900,300)，同样 10/10。
   若哪天 20 条验收掉到 18 以下，按这个顺序换组合复测，**不要并行叠加改法**
   （v2 轮试过的"合爪期间保持 ctrl / 下探后停 0.4 s / 下探前停 0.4 s"三种改法全部让抓取全灭）。
3. 仍然成立的根因级限制：指垫是刚性曲面网格（线接触），"必须 μ=1.2 才抓得住"、
   "爪甲闭合有缝、下探戳盒回弹"。真修法是按【张开位姿】设计 TPU 平垫并重扫 gz/roll，
   属于独立一轮工作（v2 实测"按闭合位姿贴平垫"会让压盒 6.07→24.19 mm、抬升归零）。

---

## 6. 复现命令

```bash
cd /home/yyl/deepseek-harness
P="env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3"

# 0) 先确认这条路还通（30 s，纯内存）
MUJOCO_GL=egl $P tools/probe_grasp_robustness.py
MUJOCO_GL=egl $P tools/probe_grasp_robustness.py --start-pose golden \
    --combos "500,800,700" --scales "0.90,0.95,1.00,1.05,1.10"   # A/B 对照

# 1) 单条 demo + 15 项自检（约 92 s）
MUJOCO_GL=egl $P render/test_render.py all
MUJOCO_GL=egl $P render/test_render.py record --start-pose golden   # A/B 用

# 2) 20 条数据（不渲染，约 7 s）
#    注意：重刷前先 rm -rf scripts/index scripts/index.json（--reset 会接着刷默认 5 条、不退出）
$P collect/collect.py --start 0 --n 20 --seed 3 --no-render

# 3) 前两条补 640x480 渲染（约 95 s/条）
$P collect/collect.py --start 0 --n 2 --seed 3

# 4) 体检 / 切分 / 数据卡 / 四段流水线
$P collect/check_dataset.py
$P collect/split.py --policy random --seed 7
$P collect/stats.py --card
$P tools/run_pipeline.py
```

坑（沿用上一轮，仍然成立）：
- 进程级 `SIGFPE`（signal 8）随机出现（**批量跑时实测比 7% 高**，新建模型/多进程更易触发），
  出现在 `mj_step`。**重试即可**，批量脚本要用"从下一条未完成 id 续跑"的循环
  （fragment 机制保证不重不漏）。本轮 20 条一次过、两条渲染也一次过。
- `collect.py --reset` **单独跑不会退出**，会接着生成默认 5 条；清索引直接
  `rm -rf scripts/index scripts/index.json`。
- 探针/批量都用 `> /tmp/x.log 2>&1` 落盘再读，**别把 python 输出管道给 head/tail**（稳定触发 SIGFPE）。

---

## 7. 未收口（下一轮候选）

1. **`demo/episode_0.npz` 仍是 raw uint8 图像数组（799.8 MB/条）** —— HANDOFF §6.2-5 的
   "改 JPEG 字节流"仍未做，这是批量采集前的硬阻塞（要同时改 `make_video.py` /
   `dataset.py` / `validate_images` / `collect --keep-frames`）。
2. **步骤二 `grasp/test_grasp.py` 起点仍是 home**（`REF_LIFT_MM=104.13`、约束1/2 口径都基于它）。
   本轮按用户裁决只做 §5.2(b)：文档口径改成引用步骤三，**物理没动**，`run_pipeline` 的 warn 依旧。
3. 结束段时长 2.9 s vs old_ros 4.0 s（§3.4）—— 要保持"总时长 7500 步不变"就是这样，需另起一轮。
4. 数据卡在 n=20 下仍报部分"空箱 / 偏薄箱"（θ、yaw、vignette 等），属样本量问题不是缺陷。
5. 起始姿态是否需要与真机逐帧对齐（例如真机 `initial_pose` 落定后实测值 vs 本模型 clip 后的值
   差 0.02°）：目前按"模型限位内可取到的最近值"处理，若要严格一致，需要真机给出落定后的关节读数。

> ⚠️ **本文件记录的是历史（sdg-v3）。现行是 `sdg-v4`（2024-11 第六轮）：
> 指尖碰撞体改成两片平面指垫 + 碰撞位掩码后，下探段对药盒零接触、压盒 0.22 mm、
> 夹稳倾角 0.07°、抬升 +125.10 mm、水平 209.56 mm，20/20 接受。
> 现行结论一律看 `docs/ROUND_contact_realism.md`；本文件里的抓取数字（抬升 ~103 mm 那一套）
> 已被取代，不要当现行实测引用。**

