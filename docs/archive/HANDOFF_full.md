# SO-ARM101 纯物理仿真 — 工作区交接文档

> 生成时间: 2025-10-24
> 工作区: `/home/yyl/deepseek-harness`
> 用途: 换到 VS Code DeepSeek 继续时，把本文件整个贴过去即可接上进度

---

## 0. 一句话现状

**步骤一（IK）✅ · 步骤二（物理抓取）⚠️ · 步骤三（渲染采集）✅ · 步骤四（SDG 批量）✅ ·
数据落地（success 标签 / 负样本 / schema 校验）✅**

药盒能被纯物理抓取、抬升约 103 mm、搬运约 221 mm 到木盒里，松爪掉落后机械臂归位。全程零 `qpos` 覆盖。
渲染管线已产出 455 帧 × 2 相机 640×480 的双相机数据，含逐帧标注、`success` 顶层标签、
npz 契约校验与 MP4/AVI 视频；抓取失败的 episode 也照样导出（`demo_neg/` 是一条实测负样本）。

> ★★★★ **现行数据 `sdg-v6`（2024-11 第八轮，训练交付版）**：视频容器 AVI(MJPG) → **MP4(h264)**
> （455 帧 640×480：13 MB → 0.7 MB），采集路径**不再出任何观看用产物**（无 avi 转码、无预览 png、
> 无拼图、不烧帧号），要看得用 `tools/inspect_episode.py` 按需生成；新增
> `collect/export_lerobot.py` —— 一步把母本导成 **lerobot v3.0 数据集**（`observation.state`/`action`
> 6 维关节角，单位 deg，`robot_type=so_follower`，与真机数据集逐字段同构），ACT 直接训练。
> 新增 `--jobs K` 并行采集（软件 GL 下实测 4 进程 4 条 148.7 s ≈ 37 s/条）。
> 物理与轨迹与 v5 **逐位相同**（只换容器），老清单留档 `scripts/index.sdg-v5.json`。
> 交接单：`docs/MIGRATION_ACT.md`（训练/迁移）+ 本文。
>
> ★★★ **上一版 `sdg-v5`（2024-11 第七轮，指垫贴真表面）**：两枚指垫改成
> **50×2×40 mm、不可见（`rgba 0 0 0 0`）、贴合手指真实内表面**，并补上抓取点横向补偿
> `dy=+17.5 mm` 与合爪量 `0.28`。实测：下探 **0/400**、压盒 **0.39 mm**、夹稳倾角 **1.06°**、
> 抬升 **+122.22 mm**、水平 **218.06 mm**、夹稳时两指缝 **−0.85 / −1.07 mm**（贴合）、
> 手指实体在盒内最大深度 **0.02 mm**（第六轮是 7.5 mm = 穿模）。20 条验收 **20/20**，
> `run_pipeline` **ok 4 / warn 0**，render 自检 **16/16`。详见 `docs/ROUND_pad_flush.md`。
>
> ★★ 上一版 **`sdg-v4`（2024-11 第六轮，接触真实性）**：指尖碰撞体改成"两片平面指垫 +
> 碰撞位掩码"，下探段对药盒 **零接触**、压盒 **0.22 mm**（旧 6.87）、夹稳倾角 **0.07°**（旧 10.95）、
> 抬升 **+125.10 mm**、水平 **209.56 mm**，20 条验收 **20/20**，`run_pipeline` **ok 4 / warn 0**。
> 用户反馈的"下落抓取必然戳到药盒、产生回弹"就是本轮修掉的；详见 `docs/ROUND_contact_realism.md`。
>
> ★ 上一版 **`sdg-v3`（2024-11 第五轮）**：录制起止姿态 = old_ros 录制脚本的
> `initial_pose`（pan 0°/lift −100°/elbow 88.26°/wrist_flex 70.63°/wrist_roll −103.53°），
> 20 条验收 **20/20**。上一轮的 golden 起点（sdg-v2，18/20）已被取代，
> 两份交接单：`docs/ROUND_oldros_start.md`（现行）、`docs/ROUND_golden_start.md`（历史）。

> 文件已按功能重组到子目录（`model/` `ik/` `grasp/` `render/` `collect/` 等），
> 每个目录都有 `README.txt`，总览见仓库根 `README.md`。

---

## 1. 运行环境（必须照抄，否则跑不起来）

### 1.1 Python 解释器的坑

本机 `python3` 的 `PYTHONHOME` 被一个 AppImage 挂载污染，直接跑会崩：

```
Fatal Python error: init_fs_encoding: failed to get the Python codec
ModuleNotFoundError: No module named 'encodings'
```

**所有命令前面必须加 `env -u PYTHONHOME -u PYTHONPATH`**：

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH python3 test_grasp.py
```

### 1.2 已确认的版本

| 组件 | 版本 |
|---|---|
| Python | 3.10.12 (`/usr/bin/python3`) |
| mujoco | 3.16.0 |
| numpy | 1.21.5 |

装法（如需重装）：`env -u PYTHONHOME -u PYTHONPATH pip install mujoco numpy`

### 1.3 打开可视化器

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH python3 -m mujoco.viewer --mjcf=mjcf/scene.xml
```

左侧 Control 面板最后一个滑块 = `gripper_joint`：

- `0.00` → 闭合（留微小间隙）
- `1.70` → 张开最大
- 关节限位 `±2.3 rad`（`wrist_roll`），超出会被 clip

---

## 2. 已完成的工作

### 步骤一 `ik/test_ik.py` ✅

纯 MuJoCo DLS（阻尼最小二乘）逆运动学，无 ROS。

- 用 `mj_jacSite` 取雅可比，`dq = Jᵀ(JJᵀ + λ²I)⁻¹·err`，`λ=0.05`
- 目标 `[0.20, 0.10, 0.05]` → **24 次迭代收敛，误差 0.000923 m**
- 含中文数学原理注释（雅可比 / 阻尼 / 迭代收敛）

```bash
env -u PYTHONHOME -u PYTHONPATH python3 ik/test_ik.py
env -u PYTHONHOME -u PYTHONPATH python3 ik/test_ik.py 0.25 -0.05 0.10  # 自定义目标
```

### 步骤二 `grasp/test_grasp.py` ✅

纯物理抓取完整闭环（`test_physics.py` 是早期的探索版，保留作参考）。

```bash
env -u PYTHONHOME -u PYTHONPATH python3 grasp/test_grasp.py                 # 完整流程
env -u PYTHONHOME -u PYTHONPATH python3 grasp/test_grasp.py --no-transport  # 只抓不搬
env -u PYTHONHOME -u PYTHONPATH python3 grasp/test_grasp.py --roll -96 --gz 0.014
```

**实测结果（默认配置 `roll=-90°, gz=0.012`，2024-11 复跑）：**

| 指标 | 数值 |
|---|---|
| 下探时药盒被压低 | **6.3 mm**（约束1 ❌，属已知缺陷：固定指望料件先于指垫触盒） |
| 合爪接触点 | 6~8 个（jaw 4 + grip 2~4） |
| 合爪法向力 | 2.6 ~ 4.0 N（μ=1.2 时） |
| 抬升高度 | **+102.66 mm** |
| 抬升后接触 | 8 个（jaw 4 + grip 4） |
| 水平搬运 | **206.0 mm** → 到达木盒 |
| 与木盒碰撞 | **0 / 600 步**（约束2 ✅） |

力学裕度：药盒重力 `0.1128 N`，摩擦系数 `μ=10`，理论最低需要法向力 `0.00564 N`，实测 `34.70 N` → **6151 倍裕度**。

### 步骤三 `render/test_render.py` ✅

渲染采集管线，record / render 两段用数据契约解耦（render 阶段不写 `ctrl`、不调 `mj_step`，
只对已录制的轨迹做 `mj_forward` 正运动学回放）。

```bash
env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py            # 正式 640x480，约 34s
env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py --quick    # 320x240，约 17s
env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py render --inspect   # 额外出拼图/预览
env -u PYTHONHOME -u PYTHONPATH python3 tools/inspect_episode.py --demo         # 事后按需看图
```

**产出（全部在 `demo/`）：**

| 文件 | 内容 |
|---|---|
| `env_camera.mp4` / `wrist_camera.mp4` | **MP4/h264** / 30fps / 640x480 / ~455 帧（训练侧直接吃） |
| `demo/episode_0.npz` | 图片数组（仅 `--export-npz` 才出，约 810 MB；默认不出） |
| `episode.npz` | 物理状态轨迹 ~120 KB（重渲染不必重跑物理） |
| `meta.json` | **顶层 `success`** + 逐帧标签 + 相机内外参（fx=fy=895.69, cx=320, cy=240）+ `schema` 摘要 |
| `contact_sheet.png` | 8 格关键帧拼图（仅 `--inspect`） |
| `preview_env.png` / `preview_wrist.png` | 抓取瞬间单帧（仅 `--inspect`） |

**实测（sdg-v4，2024-11 第六轮）：** 15.0 s / 455 帧 × 2 相机 640×480，schema + verify 全绿
（打印 **16 行**：视频与物块可见性各按两相机分行，另加"接触真实性"一行）。
抬升 **+125.10 mm**、水平 **209.56 mm**、压盒 **0.22 mm**、夹稳倾角 **0.07°**；
尾段是"下放进木盒 → 松爪掉落 → 提出 → 归位"。
（sdg-v3 时代这里是"与步骤二逐位相同：103.47 / 221.01"—— 那两个数字已作废。）

#### 步骤三的数据工程（2024-11 新增：success 标签 / 负样本 / 契约校验）

只加"导出与校验"的辅助代码，**物理控制与渲染核心一行未动**。

| 项 | 实现 | 判据/约定 |
|---|---|---|
| success 标签 | `evaluate_success(rec_meta, ep)`（第 2 段） | 抬升 = z(lift 末) − z(grasp 末)；`>= 50 mm` 且搬运末帧没塌回峰值一半 → true |
| 失败原因 | 写进 `meta.json.success_info.reasons` | `lift_too_low`（没抬起来）/ `slipped`（抬起来又掉）/ `no_metrics`（旧 npz） |
| 负样本导出 | `cmd_render` 不因 success=false 中止 | 视频 + npz + meta 全写；verify 的轨迹一致性/落在木盒/腕部像素 3 项降级为 ⚠️ |
| 契约校验 | `_TRAJ_SCHEMA` + `validate_trajectory` / `validate_images` / `validate_npz_schema`（第 10 段） | 轨迹 16 键 + meta_json、图像 env/wrist/fps；形状与 dtype 精确比对，不符即 `SystemExit` |
| 退出码 | record=抓取判定、render=导出健康、all=任一非 0 则非 0 | 负样本 record 返回 1 但 npz 已落盘 |

- 阈值 50 mm 的依据：药盒半高 26 mm、木盒口 z = 50 mm；正样本实测 **125.10 mm**，余量 75 mm。
- schema 校验**第一次跑就抓到真缺陷**：全片零接触时 `sum([])` 是 Python `int 0`，
  `fn_jaw/fn_grip/fn_sum` 整列会被存成 `int64` —— 同一份契约的 dtype 竟然取决于数据长什么样。
  已在 `EpisodeRecorder.push` 用 `float()` 强制（负样本 100% 会踩到）。
- 造负样本：`python3 render/test_render.py --gz 0.080 --out demo_neg --quick`（抓取高度抬到 80 mm → 夹空），
  实测 `success=false, reason=lift_too_low`，视频/npz/meta 全部正常落盘。

#### 步骤三的视觉随机化（2024-11，第二轮修正）

`SensorModel` 五层，顺序按物理链路（光学 → 光电 → 电学 → ISP）：

| # | 层 | 参数 | 默认 | 说明 |
|---|---|---|---|---|
| 1 | 运动虚影 | 长度 = `k`×像素速度，方向随机 | k=0.6，上限 4 px，静止=0 | 直线核（`cv2.filter2D`），非高斯 |
| 2 | 暗角 | 角落亮度乘 (1−v)、径向幂 | v=0.30, pow=2 | 小镜头渐晕 |
| 3 | 曝光 | 逐帧整帧增益 | ±1% | 真机 AGC 逐帧微动，故意保留 |
| 4 | 噪声 | 高斯 σ | 3 DN | 读出噪声 |
| 5 | 色彩抖动 | 亮度/对比度/饱和度 **慢漂移** | ±3%，τ=2 s | 不再是逐帧重掷 |

**第一轮的两个错误（用户反馈驱动修正）**

1. **色彩抖动逐帧重掷 → 视频在"闪"**
   旧实现每帧独立 `uniform(±8%)`，等于把真机 AGC 换成随机数发生器。
   实测 demo 视频静止段（init，机械臂不动）帧间平均亮度差 **mean 6.25 / max 21.04 DN**。
   现在：每条视频抽一次基准，帧间用一阶低通慢漂移（τ=2 s），幅度 ±3%。
   独立复测新视频：init 段 **mean 0.99 / p95 2.37 / max 2.82 DN**（≈不可见）；
   分解后残差几乎全部来自"故意保留的曝光 ±1%"（关掉它 → 0.03 DN）。
   verify 新增一行哨兵：**init 帧间 |Δ| p95 < 2.5 DN**。

2. **运动虚影与运动无关 → 静止也在糊，像相机在抖**
   旧实现每帧独立抽 1~3 px、75% 的帧都有。用户确认真机（100 万像素腕带 / 200 万像素环境相机）
   只在物体或镜头相对运动时才出虚影。现在由调用方传入该帧 TCP 在**图像平面上的像素速度**，
   长度 = `clip(k·speed_px, 0, 4px)`，<0.2 px 视为静止直接不糊。
   实测：speed 0→0.00 px（边缘能量与无模糊一致）、1→0.60、5→3.00、20→4.00 px（封顶）；
   两台相机各自算（腕部像素速度更大，天然更糊）。

- 随机量全部来自 `np.random.default_rng(seed)`（慢漂移同一 RNG 驱动）→ 同 seed 逐像素复现
  （实测同 seed 完全一致，换 seed 最大像素差 23）。
- `meta.json.sensor` 记"配置 + drawn（均值/最小/最大）+ drift(alpha,τ)"；
  `collect.py` 每条 cfg 随机化并记录同一组（k 0.45~0.80、上限 3.0~4.5 px、暗角 0.22~0.38、
  慢漂移 ±2%~±5%、τ 1.2~3.0 s、曝光 ±0.5%~2%）。
- 关掉：`--no-motion-blur --vignette 0 --color-jitter 0`（连同 `--blur-sigma 0 --noise-sigma 0 --gain-jitter 0` 即干净原图）。

#### 步骤三的动作脚本（2024-11 改版）

| # | 阶段 | 步数 | 时长 | 内容 |
|---|---|---|---|---|
| 1 | `init` | 200 | 0.40 s | **真机 golden pose 静止**（pan −36°/lift +4°/elbow +14°/wrist_flex +55°/wrist_roll −104°，爪甲收起） |
| 2 | `approach` | 500 | 1.00 s | **笛卡尔竖直抬出**到 z=0.24（手指就在药盒旁边，必须先抬离台面；爪甲保持收起） |
| 3-4 | `approach` | 800+700 | 1.60+1.40 s | 抬出点 → 药盒上方撤回位 z=0.24 → 正上方 z=0.16；**边移动边张爪到 1.5** |
| 4 | `descend` | 400 | 0.80 s | 竖直下探（★接触段，锁定） |
| 5 | `grasp` | 300 | 0.60 s | 合爪 GRIP_SHUT=0.20（★接触段，锁定） |
| 6 | `settle_a` | 400 | 0.80 s | 夹稳确认 |
| 7 | `lift` | 400 | 0.80 s | 竖直抬升（★接触段，锁定） |
| 8 | `settle_b` | 300 | 0.60 s | 抬升后停 |
| 9 | `transport` | 600 | 1.20 s | 水平搬到木盒上方 z=0.16 |
| 10 | `place` | 300 | 0.60 s | 保持夹持下放到盒口 z=0.115（新增） |
| 11 | `pre_open` | 100 | 0.20 s | 松手前停稳（新增） |
| 12 | `release` | 300 | 0.60 s | 开爪，药盒自由下落（新增） |
| 13 | `drop` | 300 | 0.60 s | 落进木盒并静止（新增） |
| 14 | `retract` | 450 | 0.90 s | 竖直提出木盒到 z=0.24，**边退边收爪**（新增） |
| 15 | `home` | 1300 | 2.60 s | 回初始姿态 = 首帧姿态 + 爪甲收起（新增） |
| 16 | `tail` | 150 | 0.30 s | 收尾静止（新增） |

**夹爪开合脚本（2024-11）**：预设 `GRIP_PRESET=0.0`（收起，与 keyframe 一致）→ approach 边移动
边张开到 `GRIP_APPROACH=1.5`（≈86°，够跨过 74mm 盒宽）→ 合爪 `GRIP_SHUT=0.2` → 释放张开到
`GRIP_RELEASE=1.2` → retract/home 边退边收回到 0.0。**起手和收尾都不是张开姿态，也不再每次顶到
最大 2.232**（扫描过 0.6~2.232 都能抓起来，取 1.5/1.2 留余量；`--grip-open/--grip-release` 可覆盖）。

合计 7500 步 = 15.00 s（真机平均 448 帧 ≈ 14.93 s）。旧版结尾 `N_HOLD=2000`（4.0 s 纯静止，
占 27%）已删除；静止占比降到 24.7%，扣掉 `init` 只剩 14%，最长连续静止段 4.0 s → 0.63 s。
接触段步数一律不动（放慢下探会把药盒推倒：z 0.0278 → 0.0041）。

**渲染性能（软件光栅，无可用 GPU）：** 640×480 约 70–90 ms/帧/相机，双相机整片 34 s。

---

### 步骤四 `collect/collect.py` ✅（SDG 批量生成）

一条命令随机刷数据：场景旋转 + 径向/偏航抖动 + 节奏抖动 + 相机/噪声抖动，
每条都真跑物理、自动验收，过了才渲染，产物按类分家落到 `scripts/`（动作脚本）与 `videos/`（视频）。

```bash
env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 20 --no-render  # 只刷脚本，秒级/条
env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 5 --quick       # 320x240，约 78s/条
env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --start 20 --n 5    # 续刷，编号不撞
```

**随机化带宽是实测出来的**（细节见 `collect/README.txt`）：

| 维度 | 取值 | 依据 |
|---|---|---|
| 场景旋转 θ | **±10°**（原 30°~75° 已废弃，会把药盒甩出绿垫） | 见下方"绿垫约束"：绕 pan 轴是严格同构（残差 0.00 mm），但旋转半径 271.6 mm，转多了就离开垫子 |
| 径向抖动 | +0~6 mm | 负向 -2 mm 就掉 |
| 药盒偏航 | ±1.5° | 再大指垫错开窄端面（-1° 就掉） |
| 光照 | 方位 0~360°、俯仰 5~25°、强度 0.35~0.70 | 原来固定 `(0,0,-1)/0.5`，是最大的视觉 sim2real 缺口之一 |
| 材质 | 药盒/木盒/桌面 rgba ×0.90~1.10 | 只改明暗不改色相 |
| 相机朝向 | yaw/pitch ±2°（另加位置 ±3 cm、fovy ±3°） | 避免模型学到"永远同一个灭点" |
| 节奏 | 自由段/停顿段 ×0.90~1.10 | 接触段一步不动 |
| 环境相机 | 位置 ±3 cm、fovy ±3° | 腕部相机随臂运动，不能动 |
| 传感器 | 模糊 0.6~1.0、噪声 2~4 DN、曝光抖动 **0.5~2%**（逐帧）+ 运动虚影/暗角/慢漂色彩抖动 | 与 demo 同一套 SensorModel |
| 摩擦 μ | **不随机**（固定 1.2） | TPU 对纸盒的实测标定值，改了等于换机器人 |

**绿垫硬约束（2024-11 用户要求：药盒不许离开绿色拾取垫）**：

```
垫：中心 (0.31, 0.090)、半径 96.0 mm、顶面 z=0.002   （全部运行时从 model/scene.xml 读）
药盒竖立 footprint 半对角 = hypot(37, 7.5) = 37.8 mm
留边 2 mm  ->  盒心离垫心最多 96 - 37.8 - 2 = 56.2 mm
而"绕 pan 轴的同构旋转"会让盒心沿半径 271.6 mm 的圆弧走：
    由 56.2 mm 的弦长反解  ->  |θ| ≤ 11°；实测 10° 时 footprint 最远点 92.9 mm（垫内），
    11° 时 96.0 mm（贴边） -> 带宽取 ±10°
```

每条 episode 都**真落定一次**，用实际 `qpos` 算 footprint 四角：
`max|角点 - 垫心| ≤ 94.0 mm` 且 `p0.z ≥ 27.5 mm`（贴在垫面而不是落到低 2 mm 的桌面）；
不满足就重采样出生位姿（≤12 次），仍不满足则记 `rejected(spawn_off_pad)` 不渲染。
实测 `--n 20 --seed 91 --no-render`：**20/20 全部在垫上**，footprint 47.2~80.3 mm，
落定偏航 -10.7°~+8.4°，抬升 ≥102.1 mm，重采样 0 次。

**两条关键修正（都是踩出来的）**：
1. 场景旋转必须绕 pan 关节轴（底座 (0.0388, 0)），不是世界原点 —— 否则 TCP 与药盒差 39 mm。
2. 药盒可能落在拾取垫（顶面 z=0.002）或桌面（z=0），2 mm 高差就足以让指垫错过窄端面；
   所以每条先落定、量出 `p0.z`，再按 `p0.z - 0.02777` 补偿抓取高度。

**实测：** `--n 20 --seed 91 --no-render`（垫内约束版）→ 收 20 / 丢 0（接受率 100%），
**7.4 s** 单条合计（v1.1 起含"无渲染也写相机外参"，比 v1 的 5.5 s 多 +0.1 s/条）；
含 SIGFPE 重试的墙钟 7~9 s。单条抬升 ≥102 mm、水平 180~250 mm、450 帧、14.1~15.7 s。

---

### 步骤五 数据工程四件套 ✅（2024-11，sdg-v1.1；现数据为 sdg-v2）

> 版本沿革：sdg-v1.1（本节四件套）→ **sdg-v2**（起始姿态换成真机 golden pose，
> 见步骤三第 3 条）。四件套的代码没有变，只是 `dataset_version` 跟着数据走。

生成器只保证"跑通了才落盘"，不保证"落下去的东西能被训练直接用"。这一轮补的是
**数据生产之后**的四件事：切分、读取、体检、覆盖度。新增文件都在 `collect/`：

| 文件 | 作用 | 一句话理由 |
|---|---|---|
| `dataset.py` | 读数据的唯一入口 | 6 条数据契约（帧对齐/外参方向/内参/有损/标签/版本）写进代码，不靠口口相传 |
| `split.py` | train/val/test 划分 | 划分必须是**数据产物**；等训练时临时切会出现"同一份数据两套指标" |
| `check_dataset.py` | 批量体检 | 刷 1000 条混进几条 dtype 错/帧数错，训练时才发现，返工成本极高 |
| `stats.py` | 覆盖度 + 数据卡 | 没有它 `--n 1000` 是盲刷：accept_rate 好看 ≠ 每一维都覆盖到了 |

**修掉的两个数据安全坑**（都可复现，不是猜测）：

1. **续跑会冲掉 `index.json`**：老版 `main()` 用 `accepted, rejected = [], []` 起步、
   每次整份覆盖清单 —— `--n 20` 后再 `--start 20 --n 5`，前 20 条清单就没了
   （npz/视频还在，但没有清单 = 孤儿文件）。现在每条先写 `scripts/index/ep_XXXXXX.json`
   （原子写：`.tmp` + `os.replace`），`index.json` 由片段聚合生成。
   **实测**：`--n 5` 后 `--start 5 --n 3` → 清单 8 条齐全（老版只剩 3 条）。
2. **采样 RNG 没绑 episode**：老版 `default_rng(base_seed)` 顺序抽，`--start 20` 抽到的是
   eid 0 的随机量 —— "一次跑 25 条"和"跑 20 + 续 5"是**两套不同样本**。
   改成 `SeedSequence([base_seed, eid])` 后两种跑法逐位相同（实测：单独重跑 ep_107 与
   续跑那次的 keys 完全一致）。因为采样结果变了，版本号升到 **sdg-v1.1 / schema_version=2**。

**崩溃语义**：`kill -9` 打断后核对 —— 片段 12 条、npz 12 条、无 `.tmp` 残留、无孤儿
（`--rebuild-index` 能从片段完整恢复）。

**体检口径里的两个"实测才知道"**：
* `time` 是**时间栅格采样**（0.002 s 步长 / 30 fps = 16.67 步），相邻帧间隔在
  0.032 / 0.034 s 之间交替，**不是** `dt == timestep`；平均 gap 也**小于** 1/fps
  （阶段末 force 补帧是"额外多插的帧"，不占时间格）。正确的判据是"帧数 ≈ 时间格数
  (1.00~1.20) 且没有 gap > 2.5/fps"。
* 每条会有 **1~2 帧 dt==0 的重复行**（force 补帧与上一条常规帧落在同一个 sim step，
  同 qpos、同 phase）。来自 `EpisodeRecorder.push` 的强制落帧设计，不影响状态正确性，
  体检容忍 ≤2%，超过说明录制逻辑被改坏。**没有为此改 `test_render.py`**。

**交叉校验（守两条外参路径同源）**：无渲染路径用 `collect._cam_math`（纯运动学，
约 0.1 s/条，复用同一 `MjModel` 但每次先把 `cam_quat`/`mat_rgba` 还原成原始值，
否则乘性随机化会逐条累积漂移）。与渲染路径 `CameraRig` 逐位比对：**外差 0.0e+00**，
内参一致；`check_dataset.py --cross-extrinsics` 可随时复验。

**实测数字（本机）**：

| 项 | 结果 |
|---|---|
| 20 条 `--no-render` | 5.9~7.4 s（+0.1 s/条 = 外参保证；随动作候选重试次数波动），墙钟含重试 7~9 s |
| 负样本落盘 | npz + json + 双相机外参齐全、`success=false`、无 AVI，体检通过 |
| 切分可复现 | `split.py --seed 0` 连跑两次 sha 相同；θ-holdout 下 test 全 `|θ|>5°`、train 全 ≤5° |
| 覆盖度 | 13 条样例：`sim_time ≈ 6.30 + 8.68×timing_scale`（r²=1.000）→ 节奏随机确实生效 |
| 清单自洽 | 条目数 == 片段数，无孤儿/缺文件 |

---

## 3. 对模型做的全部修改（4 处，务必保留）

### 3.1 `mjcf/scene.xml` — 删除药盒与机械臂的碰撞排除

原文件有 5 条 `<exclude>` 把夹爪和药盒之间的碰撞关掉了（视觉欺骗）。**已全部删除**，整个 `<contact>` 块现在是空的。

### 3.2 `mjcf/scene.xml` — 软化药盒接触（第 74-81 行）

```xml
<!-- 原: solref="0.02 1" -->
<geom name="object_physics" type="box" size="0.0365 0.0075 0.026"
      friction="10.0 0.1 0.1"
      solref="0.04 1" solimp="0.9 0.95 0.001" condim="4" rgba="0 0 0 0"/>
```

原因：药盒仅 11.5 g，默认 `solref="0.02"` 接触刚度太高，夹爪一碰产生 **20.89 N** 冲量（重力的 185 倍），物块直接弹飞。

效果：峰值接触力从 20.89 N 降到 8.39 N，法向夹持力从 0.005 N 提到 0.42 N（80 倍）。

### 3.3 `mjcf/scene.xml` — 挪动目标盒（第 101 行）

```xml
<!-- 原: <body name="target_box" pos="0.290 -0.132 0.025"> -->
<body name="target_box" pos="0.440 0.090 0.025">
```

**为什么必须挪**：夹爪沿 `y` 夹住药盒的 15 mm 厚度，若搬运方向也是 `y`，药盒会被顺着夹口挤出去。实测：

| 搬运方向 | 结果 |
|---|---|
| 沿 `y`（原位置） | ❌ 第 334 步掉落 |
| **沿 `x`（新位置）** | ✅ **全程不掉，8 个接触点保持** |
| 沿 `-y` 慢 3 倍 | ❌ 第 489 步掉落（与速度无关） |
| 抓起后滚转腕部再沿 `y` | ❌ 第 252 步掉落 |

### 3.4 `mjcf/so_arm101.xml` — 电机壳退出碰撞

`upper_arm_link` 和 `lower_arm_link` 下的 `sts3215_03a_v1` 电机壳**只保留视觉、退出碰撞**（删除对应的 `<geom class="collision">`）。

原因：电机壳从连杆侧面凸出约 2 cm，下探时会先于指尖撞桌面，把手臂顶偏（造成 TCP 跟踪误差 12.5 mm）。

替换成：
```xml
<geom
  class="visual"
  pos="-0.11257 -0.0155 0.0187"
  quat="4.56308e-16 -0.707107 0.707107 -1.37383e-16"
  mesh="sts3215_03a_v1"
  material="sts3215"
/>
```

碰撞 geom 总数：`23 → 21`。

### ⚠️ 3.5 一个已回退的改动（不要重犯）

`so_arm101.xml` 第 113 行 `gripper_joint` 的 `axis` **必须保持 `"0 0 1"`**。

曾误以为夹爪开合轴不对，改成 `"0.98384 0.17903 0"`，结果夹爪变成水平旋转、无法闭合。**已回退**。

**教训**：用 mesh 顶点包围盒去反推夹持几何极易出错（铰链附近的顶点恒相邻，会掩盖真实间隙）。凡是涉及夹爪的判断，**直接用 `mujoco.viewer` 肉眼看**，比任何数值推断都可靠。

---

## 4. 关键物理事实（全部实测，可直接引用）

| 事实 | 数值 |
|---|---|
| 时间步长 | `0.002 s`（默认，未显式设置） |
| 药盒质量 | `0.0115 kg`，重力 `0.1128 N` |
| 药盒尺寸 | `0.073 × 0.015 × 0.052 m`（x 长 / y 厚 / z 高） |
| 药盒静置位置 | `[0.295, 0.090, 0.02777]` |
| 药盒顶面 | `z = 0.05377` |
| 药盒摩擦 | `μ = 10.0`（法向）/ `0.1 / 0.1`（切向） |
| 夹爪传动 | `kp=150, kv=10, forcerange=±3.35 N` |
| `gripper_joint` 语义 | `1.70` = 张开最大，`0.00` = 闭合 |
| `wrist_roll` 限位 | `±2.3 rad = ±131.8°` |
| 重力下 PD 稳态误差 | `elbow_flex` 约 `0.075 rad`（4.3°），导致 TCP 偏差 10~16 mm |
| 加 `kp` 能否消除该误差 | ❌ 不能，`kp=600` 会冲过头 -29.5 mm 扎进桌子 |

### 抓取为什么要 `wrist_roll = 105°~120°`

夹口张开方向在世界系里不天然对准药盒 15 mm 厚度向（y），需要腕部滚转掰正。扫描结果：

- `75°` → 只有 `jaw` 单侧接触（`grip` 0 个），抬升 112~116 mm 但边缘易滑
- `82~120°` → `jaw 4 + grip 2` 双侧接触，稳定
- `120°` → 最优（已验证配置）

### 路径为什么要"高位撤回"

从 home 姿态直接插值到目标会让手臂**扫倒药盒**（实测药盒 z 从 0.0278 掉到 0.0093，位移 3.4 cm）。

原因：home 关键帧的 `ctrl` 默认为 0，落定后手臂塌到 `q≈0`，TCP 在 `(0.39, 0, 0.22)`。

**正确路径**：`home → 高位撤回(z=0.24) → 药盒正上方(z=0.16) → 竖直下探`

---

## 5. 已知限制（不是 bug，是几何决定）

1. **搬运必须垂直于夹持轴**。药盒只有 15 mm 厚可夹，夹持轴就是厚度方向，沿该方向移动必然挤出。这是唯一无法绕过的物理约束。
2. **下探末段会有约 11 步 `gripper_link` 接触**。药盒被压下 0.75 mm，属正常抓取动作，不影响稳定性。
3. **`wrist_roll` 只有 ±131.8°**（2024-10 已按真机标定改成 ±180°；旧记录留此说明扫描时的"多组结果相同"现象）。
4. **PD 稳态误差客观存在**。IK 是纯运动学解（无重力），物理执行有约 0.075 rad 的肘部偏差。若要精确到位，需在 IK 目标里预加补偿，而不是加大 `kp`。

---

## 5.0 2024-11 的两处修正

1. **木盒长宽朝向**：用户实测"142 是 y 向、115 是 x 向"，旧模型写反了。
   `mjcf/scene.xml` 里 `target_box` 已对调（x 半宽 57.5 / y 半宽 71，壁 4mm），
   内腔 x±53.5 / y±67mm，`diaginertia` 的 Ix/Iy 一起对调；verify 的落点判据同步改成
   `|Δx|≤50mm、|Δy|≤62mm`。（当时留的备份 `_bak_before_boxswap.xml` 已随精简删除。）
2. **夹爪不再"起手/收尾都张开、且每次开最大"**：见上表的夹爪脚本 ——
   `init`/`tail` 都是收起的预设姿态 `GRIP_PRESET=0.0`（与真机待机画面一致，
   golden 起点下爪甲闭合也正好指的是同一个值），张开只出现在 approach→descend 前后
   （`GRIP_APPROACH=1.5`）和 release（`GRIP_RELEASE=1.2`），张合动作全部塞进已有的移动段，
   不额外增加静止帧。verify 第 9 项现在校验"末帧爪甲 ctrl == 预设 0.0"。

---

## 5.1 「为什么夹取不是从正上方垂直伸入」——实测结论（2024-11）

`debug/probe_topdown.py` 把这个问题量化了，四条实测事实：

1. **工具本来就是斜的**：现有抓取位 `gripper_link` 的手指纵轴（−z_link）=
   `(0.663, 0.213, −0.718)`，与"正下方"夹角 **44.1°**，从 +x 侧斜伸进去。
   纯竖直的是 **TCP 轨迹**（`q_gsp = ik([x, y, gz])` 的 x、y 全程等于药盒中心）。
2. **原因在设计，不在几何**：`Arm.ik()` 只有 **3 个位置约束**（阻尼最小二乘），
   `wrist_roll` 由 seed 冻结后被 `np.clip` 夹住 —— 管线里从来没有对工具**姿态**求过解，
   倾角就是"从种子出发顺手落到哪儿算哪儿"。
3. **竖直在运动学上可达**：同一拾取点改用「位置 IK + 零空间姿态优化」后，
   手指倾角可压到 z=90mm 处 11.6°、z=30mm 处 0.4°。
   ⚠️ 第一版探针用"位置+姿态加权最小二乘"直接解，得到"0/60 收敛"的**假结论**
   （两个量纲不同的残差互相拉扯；wt=0 时 22/25 收敛到 0.00mm）—— 教训：姿态优化要走零空间，
   不要和位置混在一个最小二乘里。
4. **真正卡住的是手指结构**：钉住 `roll`（让张开方向横跨 15 mm 厚度）后逐 10 mm 下探，
   用 `mj_geomDistance` 量"指尖结构件 ↔ 药盒"的真实间隙：
   TCP z=70mm 时 10.0 mm，**z=60mm 时 0.0 mm**（药盒顶面 54mm），再往下是穿透
   −10 ~ −22 mm。也就是说竖直通道被指尖/固定指那块结构件占满，指垫降不到能夹住
   52mm 高盒体中部的高度。而且这对手指是**铰链指**（不是平行指垫），
   合爪方向与手指纵轴不垂直 —— roll 扫遍 360° 也只能让合爪方向与 y（15mm 厚度向）
   对齐到 **0.706**（1.0 才算真正横跨厚度）。

**因此：** 现阶段保留"斜进入 + 夹两个 15×52 窄端面"（这也正是真机上唯一能用的手法）。
要做真正的竖直俯抓，前提是给两指加**真实 TPU 指垫 geom**（平面指垫、两指闭合方向严格相对、
结构件退出碰撞），属于模型改动 → 单独 `dataset_version` + 单独验证，不能顺手改。

---

## 5.2 步骤二 / 步骤三 两条执行路径的数字不一致（**sdg-v4 起：两条路径都 ✅，问题消失**）

**现状（sdg-v4，2024-11 第六轮）：这个矛盾已经消失** —— 两条路径的约束1 都是 ✅。

| 项 | step2 `test_grasp.py` | step3 自身报告 | 说明 |
|---|---|---|---|
| 抬升 | +120.75 mm | +125.10 mm（sdg-v4 实测） | 两条路径阶段序列仍不同（step2 无 `settle_a`），差值 4 mm 属口径差 |
| 水平位移 | 203.6 mm | 209.56 mm（sdg-v4 实测） | 同上 |
| 约束1（下探不碰倒） | ✅ | ✅（下探段指↔盒接触 **0/400 步**） | 修前 step2 是 ❌ Δz −6.85 mm |
| 末端握持 | Fn 0.298 N，8 个接触 | 夹持 Fn 0.21~0.25 N | 都是"两指双侧" |

`tools/run_pipeline.py` 现在的三态回执：**`ik ok / grasp ok / render ok`（ok 3 / warn 0 / fail 0）**
（`video` 段第八轮已退役：mp4 就是最终格式，不再 AVI 转码）。
★ 第七轮起「grasp」阶段换成 `tools/probe_contact_audit.py`（六项审计，测的是生产路径）；
步骤二 `grasp/test_grasp.py` 保留为历史探针，但它与生产路径已分叉（下探段压盒 21.85 mm），
不再当流水线闸门 —— 详见 `docs/ROUND_pad_flush.md` §4。

历史（保留，说明当时为什么选 b 而不是改 step2）：

- 裁决（2024-11 第五轮，用户确认）：**步骤二 `test_grasp.py` 是历史探针，物理不动**；
  对外引用一律用步骤三。理由是两条路径的阶段序列早已不同步（步骤二没有 `settle_a` 停顿）。
- 旧数字：step2 抬升 +102.66 / 水平 206.0 / 约束1 ❌ Δz −6.85 mm；step3 +103.45 / 221.01；
  README 旧记录 +104.13 / 222.0（重标定前的历史值，**永远不要引用**）。
- 旧取证：下探完成时 `TCP z=0.0353`（目标 0.012，停在上方 23.3 mm）；`Arm.move()` 是关节空间
  线性插值（非笛卡尔直线）+ PD 稳态误差 → 末端没到点、侧向扫到药盒顶面 → 药盒被压下 6.85 mm。
  当时试过把下探开度改成 1.5：Δz 仍 −6.85 mm（排除"夹爪开太大"）。
- ★ 第六轮查出真正的根因：不是"步数不同步"，而是**指尖碰撞体是 C 形件的凸包**，
  完全下探时固定指的凸包**穿进药盒 22.6 mm**，两指之间在碰撞里是实心的 —— 所以怎么调参都压不平。
  换成平面指垫 + 位掩码后，两条路径的下探都不再碰盒。详见 `docs/ROUND_contact_realism.md`。

---

## 6. 后续计划（步骤三已完成）

### 6.1 已完成（原任务书要求，全部满足）

- [x] 加载模型，设定相机 `env_camera` / `wrist_camera`
- [x] OpenCV 高斯模糊 + 随机噪声模拟真机传感器（还加了曝光抖动）
- [x] 输出 MP4（mp4v）+ AVI（MJPG）+ 可选 PNG 序列
- [x] 逐帧标注 `meta.json`（位姿 / 接触力 / 相机内外参）—— 原任务书没要求，SDG 必需

### 6.2 下一步建议

~~1. **多 episode 域随机化**~~ —— 已由 `collect/collect.py` 完成（垫内约束版，接受率 100%）。

已完成的（原缺口）：负样本可导出（`test_render.py`：`success=false` 照样落盘）、
npz 契约校验（第 10 段 schema）。

还差（按影响排序，都没开始）：

1. **闭环控制 / 失败检测**（最大缺口）
   现在整条链路是开环盲走：没有接触反馈、没有抓空检测、没有失败重试；
   图像只输出不回灌。真实 SDG 至少要有"夹爪闭合后力/位移判断是否夹空"。
2. **执行器真实性**
   目前只有裸 position servo（`na=0`）：没有速度/加速度限幅、没有编码器量化
   （4096 counts ≈ 0.088°）、有控制延迟和 30 Hz 动作台阶要模拟；
   `so_arm101.xml` 里的 `backlash` 类（±0.5°）**定义了但没有任何关节引用**。
   ★ 2024-11 第二轮已**量化后决定暂不做** 30 Hz 动作台阶：把现有 500 Hz 真轨迹按
   零阶保持重采样，TCP 偏差 mean 1.14 mm（30 Hz）/ 3.53 mm（10 Hz），
   合爪末帧 TCP 高度只差 +0.06 mm（30 Hz）—— 远小于抓取窗口（±3 mm），
   而且后端没有真机控制律可模拟。真要做，应先真机实测"下发→到位"延迟再合成。
3. **接触 / 材料 —— ★ 已被第六/七轮解决（详见 `docs/ROUND_contact_realism.md`、`docs/ROUND_pad_flush.md`）**
   **现行实现（第七轮）**：两指各一枚 **50×2×40 mm、不可见、贴合手指真实内表面**的指垫
   （`pad_gripper_link` / `pad_jaw_link`），配 `dy=+17.5 mm` 与合爪 0.28。
   指壳 mesh 碰撞体经 `contype/conaffinity` 位掩码**不再与药盒碰撞**（只与通用件碰）。
   实测（`tools/probe_contact_audit.py`）：下探段指↔盒接触 **0/400 步**、压盒 **0.22 mm**、
   回弹 0.22 mm、夹稳倾角 **0.07°**、抬升 **+125.10 mm**、接触法向 z 分量均值 **0.000**（纯侧夹）。
   下面是**历史**（修前），保留是为了记住"为什么不能靠调参"：

   **（历史）结论：简单贴平垫不可行**
   指垫是刚性曲面网格（线接触），TPU 柔性没建模 —— 这是"必须 μ=1.2 才抓得住"的根源，
   也是用户反馈"爪甲闭合时有缝隙 + 下探戳到药盒回弹"的根因。实测（2024-11 第二轮）：
     * 接触法线（药盒系）≈ (0, ±0.75, 0.66)：两指夹的是 15 mm 薄边，接触点 z 44~48 mm
       而药盒顶 z 47.5 mm —— **两指压在盒顶边缘**，不是夹住窄端面，所以下面必然有缝。
     * 合爪末帧 **Fn_jaw = 0**（只有固定指单侧接触），jaw 要到 settle_a 末才碰。
     * descend 末帧接触是 `object↔gripper_link/geom26`（固定指结构件），Fn 3.41 N 压在盒顶，
       药盒被压进桌面/绿垫（底面 −4.5 mm）→ 抬升前先"压盒 6.3 mm"、回弹 1.7 mm。
     * gz 扫描：0.004~0.040 里只有 **0.012~0.016 能抓起来**，压盒/回弹压不到 0。
       **试过把默认提到 0.014（压盒 6.30→6.07、回弹 1.72→1.57、Fn_jaw 1.30→1.83），
       但被步骤二否决**：`test_grasp.py` 主流程没有 settle_a 停顿，合爪后立刻抬升，
       0.014 时 jaw 侧还没吃上力 → 末端握持接触 8→**0** 个、约束2 ❌（药盒 z 掉到 0.0405）。
       所以 gz 保持 0.012。**参数不能只对"带停顿的步骤三"好。**
     * **起始姿态改成真机 golden pose（2024-11 第四轮，已落地 → sdg-v2）**：
       目标姿态 pan −36° / lift +4° / elbow +14° / wrist_flex +55° / wrist_roll −104°，
       夹爪闭合（= 用户真机遥操录制的第 0 帧）。落地方式与踩坑记录：

         1) **写姿态的方式**：`Arm.apply_start_pose("golden")` 同时写 `qpos` 与 `ctrl`
            （只写 qpos 不写 ctrl，落定期手臂会朝 ctrl=0 塌下去）。实测两种写法落定后
            药盒 Δz 都是 −0.23mm、0 接触 —— 也就是说"塌下去"并不会碰到药盒，
            **上一轮"关节第一步把药盒扫倒（z 27.8→9.3）"的判断是错的**，真正翻车的是后面的合爪。
         2) **路径**：golden 下手指贴着药盒侧面（jaw 最近 34mm、gripper 49mm，与盒同高）。
            实测从 golden 直接关节插值到撤回位**也可以**（药盒 Δz=0.00mm、0 接触），
            但 `init_lift`（笛卡尔竖直抬出 200mm、爪甲保持收起）保留：它顺手给了 PD 收敛时间，
            并让"张爪"发生在离药盒 200mm 之外（张爪只在 approach 段做）。
         3) **步数不变**：init 800 拆成 `init_hold 200 + init_lift 500`，
            再把 approach 调成 800+700（原 800+600），合计仍 **7500 步 = 15.0 s / 450 帧**。
            扫 (init, retract, above) 共 15 组（每组恒定 2200 步）后选的 (700, 800, 700)：
            这一组 golden 抬升 +103.31mm，且 700/850/650、800/800/600 两组也行；
            其余 12 组有 golden 或 home 一侧全灭 —— 步数与成败是**非单调**的临界关系。
         4) **已知的临界性（重要）**：固定节奏下 golden 起点对 `timing_scale` 很敏感 ——
            s=0.90/0.92/0.95/0.98/1.03/1.06 单次运行会失败（抬升 0），
            s=1.00/1.10 成功；而 home 起点 0.90~1.10 全成功。
            失败签名：合爪开始后 0.2 s 内药盒被横向甩出 29mm 并躺平（z 27.77→9.27）。
            试过三种"消除临界"的改法，**全部让抓取全灭**（16 个节奏档全 0）：
              · 合爪期间保持 ctrl=q_gsp（不再冻结当前 qpos）
              · 下探后停 0.4s 收敛再合爪
              · 下探前在 q_abv 停 0.4s 收敛
            => 现在这一点"抓得住"依赖下探结束时 ~5° 的 PD 滞后，**不要去消除它**。
         5) **为什么最终仍然可用**：`collect.py` 的 `search_grasp()` 会逐条真跑物理、
            在"扰动候选（roll±4°/gz±1.5mm）→ 纯同构解"里取第一个成功者，
            候选次数实测 1~13 次。**两轮 20 条验收都是 20/20 接受**（seed 91 / seed 3），
            抬升 102.1~104.7mm、时长 14.2~15.7s。失败样本按原设计落成负样本，
            不会污染正样本集。
         6) 基线复测（golden 起点）：demo 抬升 **+103.31mm** / 水平 **221.11mm**，
            `test_render.py all` **15/15 全绿 rc=0**；`run_pipeline.py` 4 段
            **ok 3 / warn 1 / fail 0**（warn 仍是步骤二自带的"约束1 ❌"，与起点无关）。

         7) **用户实看视频后的两处返工（2024-11，第四轮补）**：
             用户反馈"起初姿势不对 + 动作是先空抓一次再去抓"。
             · **空抓的成因**：`apply_start_pose("golden")` 给夹爪的是 `GRIP_CLOSED=0.20`，
               而整片夹爪脚本从 `GRIP_PRESET=0.0` 起步 → 录出来的前 0.5s 爪甲从 0.20
               合到 0.0，画面就是"在药盒边上空抓一把"（golden 下手指离盒 34mm、同高度）。
               改成落定前直接写 `qpos/ctrl = GRIP_PRESET`：第 0 帧起爪甲就不再动
               （实测 init 段爪甲漂移 0.0000、药盒 Δz 0.000mm）。
             · **起手抖动**：抬出用逐点笛卡尔 IK 时，零空间把 wrist_flex 从 55° 甩到 66.4°
               再回落（"抬手时手腕先翻一下"）。改成以 golden 为种子**解一次 IK 再关节插值**：
               抬出全程 wrist_flex **单调上升**，实测 55.03→59.97°（前 30 帧）/ 峰值 62.34°
               （2024-11 第五轮复核订正；本节原写的 57.6 与 ROUND_golden_start §3.2 写的
               "55.0→38.8 有 6° 回摆"都不是实测值）。
             · **代价（必须记住）**：这两处改动把 20 条的接受率从 20/20 变成 **18/20**
               （seed 3；失败的是 #9 #14，13 个候选全灭）。扫 lift/retr 步数配比
               (400/900、500/800、600/700，总步数恒 7500) 都是 18/20 → 说明失败的这两条
               本来就压在临界上，改任何 1 帧都可能翻。seed 91 仍是 20/20。
               ★ **2024-11 第五轮已从根上解决**：起点换成 old_ros 的 initial_pose 并把
               (init_lift, retract, above) 改成 (700,800,500) 后，timing_scale 0.90~1.10
               十档全过，20 条验收回到 **20/20**、候选次数 1~13 → 1~7。（见 ROUND_oldros_start.md）

     * MjSpec 内存加平垫实验（不落文件）：按闭合位姿给两指各贴 20×3×32 mm 平垫 →
       **压盒 6.07→24.19 mm、抬升 102.43→0.58 mm（彻底抓不起来）**。原因：固定指的平垫
       贴在盒侧面，下探时刮着盒壁把盒子压下去。=> 平垫必须按【张开位姿】设计并重扫 gz/roll，
       不是贴一片就完事，属于下一轮独立工作。
4. ~~**把负样本接进 SDG 批量生成**~~ → **2024-11 第三轮已做（sdg-v1.1）**
   先纠正一个此前的误判：`save_episode()` 本来就是**无条件调用**的，`ok=False` 的 episode
   早就在落 `npz + json`（只是不渲染视频），索引里也有 `paths`。真正的缺陷是这四条：
     ㋐ npz 的 `meta_json` 里**没有 `success` 字段**（只有 `accepted`），而且缺 `nq/nv/nu`
        → 导致 `render/test_render.validate_trajectory()` 对 collect 产物**直接报错**，
          也就是说"落盘后一次都没校验过"其实连校验器都还跑不通。
     ㋑ `renders=None` 时**不写 `cam_*_T_cw`** → 负样本和 `--no-render` 样本拿不到相机外参，
        事后无法投影 / 无法补渲染（这批数据等于半残）。
     ㋒ `spawn_off_pad` 那类拒绝**完全没有产物**（连 json 都没有），编号上是个洞。
     ㋓ 没有开关、文档没写、也没有"负样本要不要留"的决定记录。
   现在：meta 补齐 `nq/nv/nu/success/success_info/instruction/video_frames/intrinsics_by_cam`；
   `extrinsics_only()` 在无渲染路径单独算外参（与 CameraRig 公式逐位一致，实测差 0.0e+00，
   已由 `check_dataset.py --cross-extrinsics` 守住）；off-pad 落 `artifact="json_only"`；
   默认 `--keep-failed`（保留负样本）、`--no-keep-failed` 省盘、`--render-failed` 可渲染。
   验证：强制 `judge()->False` 跑一条 → npz+json 齐全、`success=false`、
   双相机外参 finite、无 AVI、`check_dataset` 判其"负样本一致性 ok"。
5. **缩小 `episode_0.npz` 体积**
   现在 640x480 × 455 帧 = 约 800 MB / episode（原样存 uint8 RGB）。
   改成存 JPEG 字节流可降到 ~1/10，且不影响出视频。批量采集前必须先解决。
6. ~~**数据集切分 / 数据卡**~~ → **2024-11 第三轮已做（sdg-v1.1）**
   新增 `collect/split.py`（random / theta-holdout / mod 三种策略，按 `success` 分层，
   输出 `scripts/splits/{train,val,test}.json` + `manifest.json`，并把摘要回填 `index.json.splits`），
   同一数据 + 同一 seed 重跑**哈希逐字节相同**（sha 不含 `created`）。
   新增 `collect/stats.py`（每维直方图 + 成功率-θ 曲线 + `timing_scale→sim_time` 回归 +
   `DATA_CARD.md` 数据卡），新增 `collect/check_dataset.py`（体检，失败退出码 1）。
   仍缺：真机一致性抽检（需要真机数据，暂无法做）。
7. **补偿 PD 稳态误差**
   IK 是纯运动学解，物理执行有约 0.075 rad 肘部偏差（TCP 偏低 10~16 mm）。
   正确做法是在 IK 目标里预加补偿，不是加大 kp（kp=600 会扎进桌面）。
8. **腕部相机构图**
   抓取中段药盒占腕部画面 68%，构图可能太满。需肉眼确认 `preview_wrist.png`
   后决定是否调 `wrist_camera` 的 `pos`/`fovy`。

---

## 7. 文件清单（已按步骤重组目录）

每个目录下都有 `README.txt` 说明该目录每个文件的作用，这里只给总览。

```
deepseek-harness/
├── README.md              项目总览（先读这个）
├── model/                 模型 + 贴图 + meshes/（原 mjcf/+meshes/）
├── ik/                    test_ik.py            DLS 逆运动学
├── grasp/                 test_grasp.py (主) + test_physics.py (探索版)
├── render/                test_render.py (主；只出 mp4，不再 AVI)
├── collect/                ★ SDG 生成 + 数据工程四件套
│     collect.py            批量随机生成
│     dataset.py            读数据的唯一入口（6 条数据契约）
│     split.py              train/val/test 划分（random / theta-holdout / mod）
│     check_dataset.py      批量体检（失败退出码 1）
│     stats.py              覆盖度统计 + DATA_CARD.md
├── scripts/                ★ 动作脚本（ep_*.npz|json）+ index/（片段）+ index.json
│                             + splits/（划分）+ report.json / stats.json / stats.png
├── videos/                 ★ 生成的视频（ep_*_env|wrist.mp4）
├── dataset/                ★ 导出的 lerobot 数据集（训练直接读；见 docs/MIGRATION_ACT.md）
├── inspect/                排查产物（按需生成，可随时删）
├── demo/                   单条 demo 产物（原 output/）
├── tools/                 run_pipeline.py（一条命令跑完三步） / digest.py（摘要代替整读）
│     probe_grasp_robustness.py  换起点/配比/节奏后还抓得住吗（30 s）
│     probe_contact_audit.py     ★ 接触真实性审计（下探戳盒/压盒/回弹/撞倒，0.5 s 一条，G1~G4）
├── debug/                 probe_topdown.py（俯抓可行性取证）/ pad_fit_try.py（指垫位置两趟反算）
├── old_ros/               send_trajectory.py    旧 ROS 版，禁止运行
├── docs/                  HANDOFF.md + README.txt + TOKEN_EFFICIENCY.md
├── logs/                  历史控制台输出（run_pipeline 每次刷新 run_<段名>.log）
└── logs/                  历史控制台输出
```

**工作区已于 2024-11 精简（956 MB → 57 MB）**：删掉 `assets_export/`（100 MB 模型导出）、
`demo/episode_0.npz`（800 MB 像素缓存，render 可重建）、`debug/` 的 25 个历史脚本、
`mjcf/_bak_*` 两个备份、`__pycache__`。删除清单与理由见根 `README.md` 第 1 节。

**运行路径已全部锚定到仓库根**（`Path(__file__).resolve().parents[1]`），
所以 `python3 ik/test_ik.py` 从任何工作目录运行都能找到模型。
唯一例外是 `debug/probe_topdown.py`，它写死相对路径，必须在仓库根运行。

### 曾经想删、最后决定保留的

| 内容 | 理由 |
|---|---|
| `logs/` 的历史控制台输出 | 改参数时可以直接对照历史数字，不用重跑 |
| `old_ros/send_trajectory.py` | 真机金标准关节角 / 标定数据的唯一原件，但**不要运行**（缺 rclpy） |
| `grasp/test_physics.py` | 步骤二的早期探索版，保留作任务书交付物；日常用 `test_grasp.py` |

已经删掉的（结论都留在文档里，不必怀念）：`debug/` 25 个历史取证脚本、
`assets_export/`、`demo/episode_0.npz`、`model/_bak_*` 备份。

---

## 8. 关于 skills / 配置 / API（换窗口须知）

### 8.1 Skills：没有任何需要迁移的

已确认本环境 **未安装任何 skill**：

- `~/.dsh/skills/` 不存在
- 工作区没有 `.dsh/` 目录
- 会话全程未调用过任何 skill

所以换窗口**不需要迁移 skill**。

### 8.2 本机配置位置（仅供参考，不通用）

| 内容 | 路径 |
|---|---|
| 配置根目录 | `~/.dsh/`（`DSH_HOME`） |
| 凭据 | `~/.dsh/.credentials.yaml` |
| 模型/供应商设置 | `~/.dsh/settings.yaml.imported` |

凭据里引用的是环境变量名：`DEEPSEEK_API_KEY`、`OPENAI_API_KEY`。

### 8.3 VS Code 的 DeepSeek 需要单独配 API

**需要。** 这是两个独立产品，配置不共享。

- 本工作区的密钥存在 `~/.dsh/`，VS Code 插件读不到
- 你需要在 VS Code 插件里**单独填一次 API Key**（和这里用的是同一个 DeepSeek 账号的 key 即可）
- 如果之前没保存过 key，去 DeepSeek 开放平台重新生成一个

### 8.4 换窗口要注意的三件事

1. **Python 环境前缀**：新窗口里如果也要跑代码，记住 `env -u PYTHONHOME -u PYTHONPATH`
2. **路径已做锚定**：`step1~step3` 的脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根，
   从任何工作目录运行都不会失效。但 `debug/probe_topdown.py` 仍写死相对路径，**必须在仓库根运行**
   （`cd /home/yyl/deepseek-harness && python3 debug/probe_topdown.py`）
3. **不要把第 3.5 节那个 `axis` 改动重做一遍** —— 已经证明是错的

---

## 9. 直接给新窗口的开场白（可复制）

```
我在做 SO-ARM101 的纯物理 MuJoCo 仿真数据合成，工作区 /home/yyl/deepseek-harness。

三步都已跑通：
  步骤一 IK        ik/test_ik.py              24 次迭代收敛，残差 0.000923 m
  步骤二 物理抓取   grasp/test_grasp.py        抬升 102.66mm、搬运 206.0mm、约束2 ✅（约束1 ❌ 已知）
  步骤三 渲染采集   render/test_render.py      15.0s / 455 帧 x 2 相机 640x480，自检 10 项全绿
  步骤四 SDG 采集   collect/collect.py         随机批量，接受率 87%，产物进 scripts/ 与 videos/

产物在 demo/：sim_env_640x480.avi、sim_wrist_640x480.avi、episode_0.npz、meta.json。
文件已按步骤分目录，每个目录有 README.txt，总览见根 README.md，物理细节见 docs/HANDOFF.md。

下一步想做：多 episode 域随机化批量采集（并先把 episode_0.npz 从 280MB 压下来）。

约束：
1. 禁止任何 ROS 代码（rclpy/moveit/launch）
2. 禁止用覆盖 qpos 假装抓取
3. 每次只给一个脚本

环境注意：本机 python3 的 PYTHONHOME 被 AppImage 污染，所有命令前面必须加
env -u PYTHONHOME -u PYTHONPATH
```
