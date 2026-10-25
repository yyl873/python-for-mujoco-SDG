---
name: so-arm101-sim
description: 本仓库（SO-ARM101 纯物理 MuJoCo SDG 流水线）的作业规范 —— 三条红线、本机环境陷阱、目录与文件真身、不可改参数、当前状态与未收口的数字矛盾；动手改代码或改物理参数前先读。
whenToUse: 在 /home/yyl/deepseek-harness 里改 Python 代码、调 MJCF/物理参数、跑 ik/grasp/render/collect 任一阶段、或核对"这条结论是不是实测"时。
---

# so-arm101-sim —— 本仓库作业规范

## 1. 先认清这是什么

零 ROS 的纯物理仿真：`import mujoco` + numpy (+ cv2)，自己做 IK、抓取、渲染、SDG 批量。
没有 `.git`，**所有改动不可回滚** —— 改物理参数前先把原文件复制成 `*.bak` 或记下旧值。

## 2. 三条红线（违反即返工，不是"注意一下"）

1. **零 ROS**：不引入 `rclpy` / `moveit` / `launch`。`old_ros/` 只作参考，**禁止运行**。
2. **不伪造抓取**：不许写 `data.qpos` 把药盒"焊"在夹爪上。握持必须来自 `mj_step` 的
   接触力 + 摩擦（实测合爪法向力 2.6~4 N，药盒重力仅 0.113 N，余量本来就够）。
3. **渲染与控制解耦**：`render/test_render.py` 的 render 阶段不写 `ctrl`、不调 `mj_step`，
   只对已录轨迹做 `mj_forward` 正运动学回放。

改行为后要**重新核验这三条**，不能只看数值变好。

## 3. 本机环境陷阱（踩过，会浪费你半小时）

- 前缀必须有：`env -u PYTHONHOME -u PYTHONPATH`。裸 `python3` 会命中 AppImage 挂载，
  报 `No module named 'encodings'`。
- **别把 Python 输出管道给 `head` / `tail`**：稳定触发 `Fatal Python error: Floating point
  exception`。要截取就 `> logs/x.log 2>&1` 再读文件（或用 `dsh-digest ... --tail`）。
- `mj_step` 偶发 SIGFPE（约 7%，进程级、与本项目逻辑无关）；崩了原地重跑，
  物理结果确定性可复现。`tools/run_pipeline.py` 已把重试包好。
- 渲染后端只认 `MUJOCO_GL=egl`（默认已设）。`glfw` → `gladLoadGL error`；
  `osmesa` → `bad marshal data`。
- 版本：Python 3.10.12 / mujoco 3.16.0 / numpy 1.21.5 / opencv 4.5.4。

## 4. 目录与文件真身（照这个，别照 HANDOFF 的旧路径）

```
ik/test_ik.py          IK 求解（24 次迭代，残差 0.000923 m）
grasp/test_grasp.py    真物理抓取探针（打印接触力/抬升/搬运）
render/test_render.py  录轨迹 + 渲染(mp4/h264) + 自检 + npz 导出（make_video.py 已删除）
collect/collect.py     SDG 批量生成（--jobs K 并行；--render-only 只补渲）
collect/export_lerobot.py  ★母本 -> lerobot v3.0 数据集（ACT 直接训练；用 lerobot312 解释器）
collect/dataset.py, check_dataset.py, split.py, stats.py   数据工程四件套
model/scene.xml, model/so_arm101.xml                      MJCF 真身
model/meshes/          14 个 STL
tools/run_pipeline.py  清单式流水线执行器（ik/grasp/render 三段；video 段已退役）
tools/inspect_episode.py  ★按需出排查产物（拼图/预览/体检；--render 可补渲）
tools/probe_grasp_robustness.py  抓取鲁棒性探针（换起点/配比/节奏后还抓得住吗，30 s）
tools/probe_contact_audit.py     接触真实性审计（下探是否戳盒/压盒/回弹/撞倒，0.5 s 一条）★先用它
tools/digest.py        摘要工具
demo/, demo_neg/, videos/, logs/, scripts/, debug/        产物
dataset/               导出的 lerobot 数据集（训练直接读）
inspect/               排查产物（按需生成，可删）
```

⚠️ `docs/HANDOFF.md` §3 的标题写的是 `mjcf/scene.xml`，**实际路径是 `model/scene.xml`** ——
文档是目录重组前写的。同理 `HANDOFF` 里的 `cd /home/yyl/deepseek-harness && python3 test_grasp.py`
也该是 `python3 grasp/test_grasp.py`。

## 5. 不许动的参数（动了会静默毁掉抓取）

- **接触段步数不许改**：`descend 400 / grasp 300 / lift 400`。实测把下探放慢到 1150 步，
  结构件在盒顶压得更久，药盒被推倒（z 0.0278 → 0.0041），抓取全废。
  要加时长只能加在自由空间段和停顿段。
- **渲染尺寸全程恒定**，否则 `VideoWriter` 静默丢帧（`VideoSink` 里有断言兜住）。
- **不要靠加大 `kp` 消除 PD 稳态误差**：`kp=600` 会冲过头 −29.5 mm 扎进桌子。
  正确做法是在 IK 目标里预加补偿。
- `wrist_roll` 取 **105°~120°**（120° 为已验证最优）；75° 只有 `jaw` 单侧接触。
- 药盒只有 15 mm 可夹，**搬运必须垂直于夹持轴** —— 这是物理约束，不是 bug。
- ★★ **指垫必须"贴合手指真实内表面"且不可见**：垫面位置用**干净参考位姿**（先把垫子碰撞临时关掉跑一遍）
  按**药盒足迹窗口**（x±37/z±26）取该指最内侧极点反算。贴在"药盒表面"会变成悬空板
  （实测 0/14 探针点在手指实体内 → 用户看到的"指尖多了个垫子 + 缝隙"）。
  工具：`debug/pad_flush_try.py`（含 `--emit-mjcf`）。
- ★★ **合爪量与盒厚是一一对应的**：指面间距 ≈ 7.6 mm / 0.1 rad；jaw=0.15 → 4.45 mm（已合拢），
  jaw=0.28 → 14.15 mm ≈ 盒厚 15 mm。**夹 15 mm 盒必须 0.28**，写 0.15 会把盒挤出去。
- ★ **`GRASP_XY_OFFSET`（dy=+17.5 mm）是必需项**：不加时固定指实体整块压在盒里（9814 顶点 / 深 7.5 mm）。
  步骤二/三与 `collect` 三处都要吃这个补偿（collect 里还要按 theta 旋转）。
- ★ **指垫位置与「起止姿态 + 接近路径 + 步数配比」是耦合的**：`pad_gripper_link` /
  `pad_jaw_link` 的局部 pos/quat 是**按 descend 段末的真实到达位姿**（TCP≈23.57 mm，不是 IK 目标的
  12 mm）反算出来的。改起点/路径/步数中任何一项，都必须用 `debug/pad_fit_try.py` 重算这两组数
  （两趟：先只关指壳碰撞跑一遍取真实位姿，再贴垫）。贴错的表现是：下探零接触、但抬升 0（夹空）。
- ★ **`roll` 现在是指垫对齐轴，不是可抖动的自由度**：实测只有 −90° 干净；−95° 下探 6/400 步接触、
  −100° 压盒 1.58 mm、−105/−110° 彻底抓不起来。`collect.py` 的 `ROLL_JITTER_DEG` 因此改成 0。
- ★ **`gz` 窗口 0.008~0.016**（0.012 居中）：0.020 起垫子会在下探段擦到盒顶。张开量 0.6~2.232
  现在**完全等价**（指垫在指内侧，开口不参与接触）。
- **起点姿态与三段自由空间步数是耦合的**：改 `start_pose`（oldros/golden/home）或改
  `(init_lift, approach_retract, approach_above)` 的任何一段，都要重扫另一个 ——
  实测 oldros 起点配 golden 的 (500,800,700) = 抓取全灭；golden 起点配
  oldros 的 (700,800,500) 也全灭。动之前先跑 `tools/probe_grasp_robustness.py`（30 s）。

## 6. 当前状态（实测，可直接引用）

| 阶段 | 状态 | 数字 |
|---|---|---|
| IK | ✅ | 24 次迭代收敛，残差 0.000923 m |
| 接触审计（现行闸门） | ✅ | G1 下探 **0/400** / G2 压盒 **0.39 mm** / G3 夹稳倾角 **1.06°** / G4 抬升 **+122.22 mm** / G5 指缝 **−0.85/−1.07 mm** / G6 穿模深度 **0.02 mm** |
| 渲染 | ✅ | 455 帧 × 2 相机 640×480，**16/16 自检绿**；抬升 **+122.22 mm**、水平 **218.06 mm** |
| 抓取（历史探针） | ⚠️ | 已退出流水线：序列与生产路径分叉（下探段压盒 21.85 mm），保留备查 |
| 视频 | ✅ | **MP4/h264**（455 帧 640×480：0.7 MB；旧 MJPG AVI 13 MB）；采集不再出 avi/预览/拼图 |
| 起止姿态 | ✅ | 录制开始/结束 = old_ros `initial_pose`（pan 0°/lift −100°/elbow 88.26°/wrist_flex 70.63°/wrist_roll −103.53°，爪甲收起） |
| SDG 数据集 | ✅ | **sdg-v6**（物理同 v5，视频改 mp4）；20/20，候选 1 次命中 3.5 s，十档 timing_scale 全过 |
| ACT 导出 | ✅ | `export_lerobot.py` → lerobot v3.0：6 维 state/action（**度**）+ 双相机视频，`robot_type=so_follower`；332 s/20 条（≈16.6 s/条）、121 MB / 9102 帧；回读校验过；`lerobot-train --policy.type=act` 3 步跑通（loss 90.2→61.7） |

全跑一次约 100 s。三态回执：**`ik ok / grasp ok / render ok`（ok 3 / warn 0 / fail 0）**。
渲染耗时取决于**能否看到 `/dev/nvidia*`**：有 GPU（用户终端）`rig.grab` 双相机 **1.7 ms/帧**、单条 **19.1 s**（含启动）、8 进程 8 条 **33.1 s**；
看不到设备节点时（AI 沙箱）退化为软件 llvmpipe **174~197 ms/帧 / 75~90 s/条**。
★ 不要把沙箱里的耗时当成用户机器的耗时。并行采集用 `collect.py --jobs K`。
★ 搬运途中药盒会歪 **27~30°**：本臂 lift/elbow/flex 三轴平行，工具俯仰随 TCP 高度线性变化
**0.22°/mm**（0.012 m→44.1°，0.16 m→76.3°），抬 100 mm 必然如此，**不是接触伪影**；
`ik_lock` 试过压不下来（位置收敛、指向残差 31.9° 且与权重无关）。要减小只能降搬运高度或换 6 轴。

### 6.1 两条执行路径（HANDOFF §5.2）—— sdg-v4 起都 ✅

| 项 | step2 `test_grasp.py` | step3 自身 |
|---|---|---|
| 抬升 | +120.75 mm | +125.10 mm（sdg-v4） |
| 水平 | 203.6 mm | 209.56 mm |
| 约束1 | ✅ | ✅（下探指↔盒 0/400 步） |

两条路径的阶段序列仍不同（step2 无 `settle_a`），所以数字差 4~6 mm 属口径差，不是矛盾。
**修前**这一步是 ❌ Δz −6.85 mm，根因是指尖碰撞体（C 形件的凸包，完全下探时穿进药盒 22.6 mm），
第六轮换成平面指垫 + 位掩码后两条路径同时变绿，详见 `docs/ROUND_contact_realism.md`。

**不要把 104.13 / 222.0 / 103.45 / 221.01 当现行实测引用** —— 前者是重标定前的历史值，
后者是 sdg-v3（无指垫）的实测，复 pipeline 都复现不出来。

## 7. 验证怎么跑（省 token 的姿势）

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py            # 全阶段，只回三态回执
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py render     # 单段
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --status   # 只看上次回执
```

- 回执落 `logs/run_all.json`（全跑）/ `run_last.json`（部分跑），日志落 `logs/run_<stage>.log`。
- 数据交付/训练/迁移：`docs/MIGRATION_ACT.md`（采集 → 导出 → lerobot-train 一条龙）。
- 排查视频**按需**：`python3 tools/inspect_episode.py --episodes 0,7,13`（或 `--all --check-only` 只体检）；
  采集阶段**不再**生成拼图/预览/avi（第八轮用户要求：不要每条都转格式）。
- 验收判据在 `render/test_render.py` 的 verify（**15 项**自检（默认不出图片数组 npz，故少 1 项），第 12 项是"接触真实性"：压盒 ≤0.5 mm
  且夹稳倾角 ≤5°）；负样本通路在 `demo_neg/`。
- ★ 动任何与抓取有关的参数前，先跑接触审计（0.5 s/条，纯内存不落盘）：
  `env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 tools/probe_contact_audit.py`
  判据 G1 下探零接触 / G2 压盒 ≤0.3 mm / G3 夹稳倾角 ≤5° / G4 抬升 ≥50 mm；退出码 1 = 踩线。
- 动起点/步数配比/roll/gz 之前，先 `env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl \
  python3 tools/probe_grasp_robustness.py`（纯内存、不落盘，十档节奏一次回执）。
- ⚠️ `--json` **不是只读**：它会真跑。只想看状态用 `--status`。
- 实时性检查用 `MUJOCO_GL=egl python3 -m mujoco.viewer --mjcf=model/scene.xml`。

## 8. 动手前的顺序

1. 读 `docs/HANDOFF.md` 对应 §（尤其 §3 的 4 处模型修改、§4 物理事实、§5.2 未裁决项）。
2. 查 `tools/digest.py` 骨架再看原文，别整文件灌进上下文。
3. 改参数 → 复制原件为 `*.bak` → 跑单段验证 → 数字写回文档 → 删 `*.bak`。
4. 结论必须带数字与单位；`warn` 不许写成 `ok`。

## 9. 反例（别这么干）

- ❌ 只跑 render 就说"抓取好了" —— 抓取在 step2/step3 是两条路径；还要看**下探段有没有碰盒**
  （`tools/probe_contact_audit.py`：抬升 103 mm 也可能只是"指尖掐着盒顶"，实测旧模型就是这样）。
- ❌ 看到"抬升变大"就当改好了 —— 抬升从 103 变 125 是因为盒子不再被按进桌面，不是新能力。
- ❌ 改 `roll` 做数据抖动 —— 指垫方案下 roll 是对齐轴，±5° 就开始在下探段蹭盒。
- ❌ 把 `set_gripper(0.15)` 直接当合爪量用 —— 本模型 0.15 rad 时两指面只差 4.45 mm（等于合拢）。
- ❌ 用"已经被接触推歪的那一帧"反算指垫位置 —— 实测偏 4~5 mm；必须用干净参考位姿。
- ❌ 只改 `GRASP_XY_OFFSET` 不改 `collect.py` 的 `waypoints_for()` —— 批量会 20/20 `no_lift`。
- ❌ 按 IK 目标位姿（TCP=12 mm）贴指垫 —— 必须按 descend 段末真实位姿（TCP≈23.6 mm，PD 滞后），
  否则垫子高 23 mm、只夹到盒顶，表现是"下探零接触但抬升 0"。
- ❌ 用 `qpos` 覆写让药盒跟着夹爪走 —— 红线 2。
- ❌ 把 `logs/` 里的旧数字抄进 README 当新实测。
- ❌ 为了"更平滑"把下探步数调大 —— 已验证会推倒药盒。
- ❌ 在 render 阶段加 `mj_step` 让画面更真实 —— 红线 3。
- ❌ 只换起点姿态、不重扫 (init_lift, retract, above) —— 抓取会全灭（实测抬升 0、盒被横甩 37 mm）。
- ❌ 照搬 old_ros 的 `safe_z=0.20` 当抬升高度 —— 本模型实测 0.20/0.22 抓取全灭，必须是 0.24。