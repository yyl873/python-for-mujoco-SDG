# 本轮交接：起始姿态改成真机 golden pose（sdg-v2）

> ⚠️ **已被 `docs/ROUND_oldros_start.md` 取代（2024-11 第五轮）**：录制起止姿态已换成
> old_ros 录制脚本的 `initial_pose`（折叠待机位），数据版本升到 `sdg-v3`，
> `(init_lift, approach_retract, approach_above)` 也从 (500,800,700) 改成 (700,800,500)。
> 本文件保留作为"golden 起点那一轮"的历史记录与踩坑证据；其中的数字不要再当现行实测引用。
>
> 这是 2024-11 第四轮的完整交接单。复制到新窗口后，这份文件 + `docs/HANDOFF.md` 足够继续。
> 代码真身：`render/test_render.py`、`grasp/test_grasp.py`、`collect/collect.py`。
> 本机环境：**所有 python 调用必须** `env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3`（否则报
> `No module named 'encodings'`），渲染需 `MUJOCO_GL=egl`。

---

## 1. 目标与结论

用户要求：episode 的**起始姿态**对齐他自己遥操训练数据的第 0 帧

```
golden_joints = {shoulder_pan −36°, shoulder_lift +4°, elbow_flex +14°,
                 wrist_flex +55°, wrist_roll −104°}   # 夹爪闭合
```

并且 **总时长不变、运动速度正常**。

结论：**已落地**（`DATASET_VERSION = "sdg-v2"`）。第 0 帧关节实测
`[-36.00, +4.24, +14.18, +55.02, -104.00]°`，总步数仍 **7500 步 = 15.0 s / 455 帧 @30fps**。
代价：抓取处在临界边缘，接受率从 20/20 降到 **18/20**（详见 §5）。

---

## 2. 实现方式（三个文件，全部只在代码里，**没改 MJCF**）

`grasp/test_grasp.py`
- `GOLDEN_JOINTS`（度）/ `GOLDEN_Q_DEG` / `GOLDEN_Q`（弧度，`ARM_JOINTS` 顺序）。
- `Arm.apply_start_pose(pose="home"|"golden")`：golden 模式下**同时写 `qpos` 和 `ctrl`**；
  只写 qpos 不写 ctrl 的话，落定期手臂会朝 `ctrl=0` 塌下去。
- `Arm.reset(settle=1200, pose="home")` 转发到上面（步骤二仍用 home，见 §6）。

`render/test_render.py`
- `run_episode(..., start_pose="golden")`（默认 golden，传 `"home"` 可回到旧预设做 A/B）。
- 相位表新增 `init_hold` / `init_lift`，`init` 段从 800 步拆成 **200 + 500**，
  `approach_retract` 800、`approach_above` 700（原 init800 + retract800 + above600 合计相同）。
- 抬出的帧标 `phase="approach"` 而不是 `init`：`init` 在数据契约里表示"静止的起点"
  （verify 的帧间亮度稳定、数据卡的"首帧画面"都取它）。

`collect/collect.py`
- `DATASET_VERSION = "sdg-v2"`（起始姿态变了 → 与 sdg-v1.1 **不能混训**；
  `mjcf_sha256` 不变，因为起点写在代码里而不是 keyframe）。
- `START_POSE = "golden"`；`record_episode` 传 `start_pose`，每条 meta / index keys / `_policy.json`
  都记录 `start_pose`。demo 的 `record_meta` 另记 `golden_joints_deg`。

### 动作时序（发给下游必须一致）

```
init_hold 200 (0.4s)  在 golden 静止，爪甲闭合（0.0）
init_lift 500 (1.0s)  竖直抬出到 z=0.24（手指就在药盒旁边，必须先抬离）
approach_retract 800  高位撤回点 → 途中张爪到 1.5
approach_above   700  药盒正上方 z=0.16
descend 400 / grasp 300 / settle_a 400 / lift 400   ← 接触段，锁定不许动
settle_b 300 / transport 600 / place 300 / pre_open 100 / release 300 / drop 300
retract 450 / home 1300 / tail 150
合计 7500 步 = 15.0 s
```

---

## 3. 用户实看视频后揪出的两个真问题（已修，有实测）

### 3.1 "在药盒边上空抓一把再真抓"

成因：`apply_start_pose("golden")` 把夹爪写成 `GRIP_CLOSED=0.20`，而整片夹爪脚本从
`GRIP_PRESET=0.0` 起步 → 录出来的**前 0.5 s 爪甲从 0.20 合到 0.0**。
golden 下手指离药盒只有 34 mm、且与盒同高，画面上就是一次空抓。

修复：`run_episode` 在落定前直接写 `qpos[gadr] = ctrl[cg] = GRIP_PRESET`。

实测（`scripts/ep_000000.npz`）：第 0 帧爪甲 `0.000/0.000`，`init` 段爪甲漂移 **0.0000**，
药盒 Δz **0.000 mm**、0 接触。

### 3.2 抬手时手腕先向上翻一下

成因：抬出原先用逐点笛卡尔 IK，零空间把 `wrist_flex` 从 55° 甩到 **66.4°** 再回落。

修复：以 golden 为种子 `ik(x, y, RETRACT_Z)` 解一次，再关节空间插值。

实测（**2024-11 第五轮复核后订正**）：抬出段 `wrist_flex` **单调上升**，`55.03° → 59.97°`
（前 30 帧）/ 全段峰值 62.34°（`approach` 第 54 帧），**没有回摆**。
本节原先写的"55.0° → 38.8°，开头约 6° 回摆"与 `HANDOFF §6.2-3)-7` 写的"55.0 → 57.6"
**都不是实测值**，已一并订正 → 以"单调上升、峰值 62.3°"为准。
复现：`render/test_render.py record --start-pose golden --out <tmpdir>` 后读 `qpos[:,3]`
（注意 v3 的步数配比与 golden 起点不配套，这条命令抓取会失败，只用来取关节轨迹）。

---

## 4. 验收（全部重跑过，三态结论）

| 条目 | 关键量 | 状态 |
|---|---|---|
| `render/test_render.py all` | 自检 **15/15 全绿**、`总判定 ✅`、rc=0 | **ok** |
| 轨迹一致性 | 抬升 **+103.31 mm**（基准 104.13 ±5）、水平 221.10 mm（基准 222.0） | **ok** |
| `tools/run_pipeline.py` | 4 段 **ok 3 / warn 1 / fail 0**，render 自检 15/15、success=true，rc=1（warn 属设计） | **ok** |
| `collect/check_dataset.py` | 20/20 通过 | **ok** |
| `collect/split.py` | random / seed 7 → **15 / 3 / 2**，`index.json.splits` 已回填 | **ok** |
| `collect/stats.py --card` | sdg-v2、接受率 90%、数据卡已重写 | **ok** |
| 数据集 | 20 条 / **接受 18** / 负样本 2（#9 #14） | **warn** |

warn 说明：`run_pipeline` 的 warn 来自步骤二自带的"约束1 ❌"（药盒被撞动），与起点改动无关；
数据集 warn 是接受率 90%（见 §5）。

产物：
- `videos/ep_000000_{env,wrist}.avi`（抬升 103.69 mm / 水平 187.3 mm / 14.96 s / 候选 3 次）
- `videos/ep_000001_{env,wrist}.avi`（101.12 / 202.1 / 14.70 s / 候选 1 次）
- `demo/env_camera.mp4`、`demo/wrist_camera.mp4`、`demo/sim_{env,wrist}_640x480.avi`
- `scripts/ep_000000..19.{npz,json}`、`scripts/index.json`、`scripts/splits/*`、`scripts/DATA_CARD.md`

---

## 5. 抓取的临界性（**下一轮必须记住**）

golden 起点下，"抓得住"依赖**下探结束时约 5° 的 PD 滞后**，处在临界点：
固定节奏单跑，`timing_scale = 0.90/0.92/0.95/0.98/1.03/1.06` 会失败（抬升 0），
`1.00/1.10` 成功；home 起点 0.90~1.10 全成功。
失败签名：合爪开始后 0.2 s 内药盒被横向甩出 29 mm 并躺平（z 27.77 → 9.27）。

**试过三种"消除临界"的改法，全部让抓取全灭（16 个节奏档全 0），不要重试：**
1. 合爪期间把 ctrl 保持在 `q_gsp`（不冻结当前 qpos）
2. 下探后停 0.4 s 收敛再合爪
3. 下探前在 `q_abv` 停 0.4 s 收敛

接受率现状：seed 91 → 20/20；seed 3 → **18/20**（#9 #14 的 13 个候选全灭 → 落负样本）。
扫 lift/retract 配比 400/900、500/800、600/700（总步数恒 7500）都是 18/20
→ **继续调步数没有意义**。要拉回来只能：
- (a) 改 `collect.py::search_grasp`：全候选失败后再叠一档 `timing_scale`（±5%）重试；
- (b) 做搁置的**平垫（TPU）几何**：按【张开位姿】设计并重扫 gz/roll，这是根因级修复。

---

## 6. 未收口 / 需要用户确认

1. **姿势是否真的对上了？** 关节值确实是用户给的五个数（±0.2°），第 0 帧 TCP=(231,149,37) mm。
   若用户仍说"起步姿势不对"，那就是**关节约定不一致**，最可能是 `shoulder_pan` 符号
   （本模型 pan 向左为负：药盒在 +17° 方位，够到它需要 pan ≈ −18°；`collect.py` 里标着
   `pan_sign=-1`）。需要用户给**一条遥操数据第 0 帧的 qpos 前 5 维**或该帧截图。
2. 用户看到的可能是旧片：`demo_neg/env_camera.mp4` 是 10-24 20:41 的（旧起点 + 字面空抓不抬升）。
   新片时间戳均为 10-25 14:38 之后。
3. **步骤二 `grasp/test_grasp.py` 起点仍是 home**（`REF_LIFT_MM=104.13`、"约束1/2"口径都基于它）。
   要不要也换成 golden？换了必须重标 约束1 阈值，并同步 `run_pipeline` 的一致性口径。
   → 已向用户提过，未答复。
4. 数据卡/覆盖度在 n=20 下报"空箱"（theta / yaw 各 2/10），属样本量问题，不是缺陷。
5. `demo/episode_0.npz` 存的是 **raw uint8**（419 MB × 2 相机 = 838 MB），HANDOFF §6.2-5
   的"改 JPEG 字节流"仍未做。

---

## 7. 复现命令

```bash
cd /home/yyl/deepseek-harness
P="env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3"

# 单条 demo + 15 项自检
MUJOCO_GL=egl $P render/test_render.py all

# 四段流水线（ik / grasp / render / video），rc=1 表示有 warn，属设计
$P tools/run_pipeline.py

# 20 条数据（不渲染，约 12 s）
$P collect/collect.py --start 0 --n 20 --seed 3 --no-render

# 前两条补 640x480 渲染（约 95 s/条）
$P collect/collect.py --start 0 --n 2 --seed 3

# 体检 / 切分 / 数据卡
$P collect/check_dataset.py
$P collect/split.py --policy random --seed 7
$P collect/stats.py --card
```

注意：进程级 `SIGFPE`（signal 8）约 7% 概率随机出现，出现在 `mj_step`。
**重试即可**，批量脚本要用"从下一条未完成 id 续跑"的循环（fragment 机制保证不重不漏）。
`collect.py --reset` **单独跑不会退出**，它会接着生成默认 5 条 —— 清索引请直接
`rm -rf scripts/index scripts/index.json`。

> ⚠️ **本文件记录的是历史（sdg-v2）。现行是 `sdg-v4`（2024-11 第六轮）：
> 指尖碰撞体改成两片平面指垫 + 碰撞位掩码后，下探段对药盒零接触、压盒 0.22 mm、
> 夹稳倾角 0.07°、抬升 +125.10 mm、水平 209.56 mm，20/20 接受。
> 现行结论一律看 `docs/ROUND_contact_realism.md`；本文件里的抓取数字（抬升 ~103 mm 那一套）
> 已被取代，不要当现行实测引用。**

