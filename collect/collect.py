#!/usr/bin/env python3
"""
collect.py —— SDG 批量数据生成器（随机刷数据）
==============================================

为什么需要它
------------
前面三步跑出来的只是**一条**写死的 episode（药盒固定在 (0.295, 0.090)）。
SDG 要的是成百上千条"同任务、不同摆放/不同节奏/不同噪声"的样本。
本脚本把「随机采样 -> 真物理录制 -> 自动验收 -> 渲染 -> 落盘」串成一条流水线。

目录分工（这是用户要求的）：
    scripts/   每条数据的【动作脚本】：ep_000000.npz（轨迹）+ ep_000000.json（标签）
                                 + index.json（清单）
    videos/    每条数据的【视频】：ep_000000_env.mp4 + ep_000000_wrist.mp4（h264 容器）
    demo/      单条 demo 的产物（跑 render/test_render.py 时的输出）
    logs/      运行日志

随机化什么（每条 episode 独立采样，参数全量写进 json）
----------------------------------------------------
  场景旋转       整体绕底座竖轴转 ±10°（绿垫硬约束反算的上限；同构解保证抓取几何不变）
  药盒出生位姿   在旋转后的位置落地，另加 +0~6mm 径向、±1.5° 额外偏航
  抓取参数       roll、gz 各带小扰动，逐条真跑物理挑一个"确实成功"的
  节奏           阶段步数在【自由空间/停顿段】上 ×0.90~1.10（接触段一步不动）
  环境相机       位置 ±1~3 cm、fovy ±3°、朝向 yaw/pitch ±2°（腕部相机随臂运动，不能动）
  传感器         模糊 σ 0.6~1.0、噪声 σ 2~4 DN、曝光抖动 0.5%~2%、
                 + 免费视觉随机化（运动虚影/暗角/慢漂色彩抖动，见 README）

每条的随机种子 = f(base_seed, episode_id)（SeedSequence([base_seed, eid])）。
★这条很重要：`--n 25` 与 `--n 20` + `--start 20 --n 5` 必须得到逐位相同的样本，
  否则"续跑"就是在换一套分布采样（sdg-v1 的老版本正是这个毛病）。

不做的事（数据卫生红线）
------------------------
  * 不随机摩擦系数 μ（TPU 对纸盒的实测值 1.2 是标定结果，改了就等于换了个机器人；
    真要随机也要单独一个 dataset_version 并在 json 里标出来）
  * 不写 qpos 伪造握持。唯一的 qpos 写入是【出生位姿】（落定前写一次，之后全交给物理）
  * 不随机相机到"看不见物体"的程度（env_camera 的抖动幅度是刻意的）

自动验收（不通过的不渲染，落成【负样本】并记进 index.json 的 rejected 段）
--------------------------------------------------------------------------
  1) IK 残差 < 2 mm（三个接近路点 + 放置路点）
  2) 落定后药盒还立着（z > 20 mm）
  3) 抬升 > 95 mm（被真夹起来了）
  4) 药盒最后落在木箱内腔里
  5) 松手后不应再有握持接触；臂不能刮到木箱
  6) 药盒整个 footprint 在绿垫上

  ★没有轨迹的拒绝（出生位姿反复重采样仍在垫外）：只落 ep_XXXXXX.json（artifact=json_only）。
  ★有轨迹但没通过验收的：照常落 ep_XXXXXX.npz + .json（success=false、has_video=false、
    相机外参照样齐），这就是负样本 —— 正样本谁都能造，真实的失败模式才是稀有资源。
    不想要就加 --no-keep-failed。

清单结构（v1.1 起）
------------------
    scripts/index/ep_000000.json   每完成一条就落一个片段（原子写，SIGFPE 也不丢）
    scripts/index/_policy.json     本次运行用的随机化策略快照（split/stats/rebuild 不必加载模型）
    scripts/index.json             聚合视图：扫全部片段生成（人读/训练侧读都够了）

用法
----
    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 5 --no-render   # 先只刷物理，秒级
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 5 --quick       # 320x240 快出片
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 20               # 正式 640x480
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 2 --keep-frames # 额外存图片数组
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --start 20 --n 5     # 续着刷（清单不再被冲掉）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --shard 1/4 --n 100  # 分片跑（外部并行）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --rebuild-index      # 分片后重建聚合视图
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --reset --n 2        # 清索引层重刷

下游四个脚本（都只读 scripts/，不需要模型）：
    collect/dataset.py       读数据的唯一入口（含数据契约 docstring）
    collect/split.py         生成 train/val/test 划分
    collect/check_dataset.py 批量体检（复用 test_render.validate_trajectory）
    collect/stats.py         覆盖度统计 + 数据卡
"""

import os
import sys
import json
import time
import argparse
import hashlib
import subprocess
from pathlib import Path

# ============================================================
# 第 0 段  环境前置（MUJOCO_GL 必须在 import mujoco 之前定好）
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grasp"))       # 复用步骤二的 Arm 封装
sys.path.insert(0, str(ROOT / "render"))      # 复用渲染器 / 记录器 / 动作序列

os.environ.setdefault("MUJOCO_GL", "egl")
sys.unraisablehook = lambda *a, **k: None     # EGL 析构噪音静音

import numpy as np                             # noqa: E402
import cv2                                     # noqa: E402
import mujoco                                  # noqa: E402

import test_render as R                        # noqa: E402  （渲染/录制全部复用，不重写）

SCRIPTS_DIR = ROOT / "scripts"                 # ★ 动作脚本（npz / json / index）
VIDEOS_DIR = ROOT / "videos"                   # ★ 视频（mp4 / h264，训练侧直接吃）
LOGS_DIR = ROOT / "logs"
INDEX_DIR = SCRIPTS_DIR / "index"              # ★ 每条一个片段（抗崩溃；index.json 只是聚合视图）
SPLITS_DIR = SCRIPTS_DIR / "splits"            # ★ split.py 产出的 train/val/test 划分

# 数据版本：改了【随机化范围 / 采样方式 / npz 元数据结构】就要动这个数字。
# sdg-v1   -> 初版（default_rng(base_seed) 顺序抽 + meta 无 nq/nv/nu/success）
# sdg-v1.1 -> 采样 RNG 改成按 episode_id 派生（续跑可复现）+ schema_version=2
#             （meta 新增 nq/nv/nu/success/success_info/instruction/video_frames/
#              intrinsics_by_cam，且无视频路径也写相机外参）
# sdg-v2   -> 起始姿态换成真机遥操的 golden pose（pan −36°/lift +4°/elbow +14°/
#             wrist_flex +55°/wrist_roll −104°，夹爪闭合），并把 init 段拆成
#             init_hold(静止) + init_lift(竖直抬出)；总步数仍 7500 = 15.0 s。
# sdg-v3   -> ★【录制起点/收尾姿态换成 old_ros 录制脚本的 initial_pose】
#             （pan 0°/lift −100°/elbow 88.26°/wrist_flex 70.63°/wrist_roll −103.53°，
#              夹爪收起 = 真机折叠待机位，开录前与录完后都是它），
#             并把 (init_lift, approach_retract, approach_above) 从 (500,800,700) 改成
#             (700,800,500) —— 换起点必须跟着重扫这三段，否则合爪把药盒横甩出去。
#             总步数仍 7500 = 15.0 s / 455 帧；前 1~1.6 s 与 v2 完全不同，**不能与 v2 混训**。
# 注意：mjcf_sha256 不变 = XML 没动；变的是"起始姿态"和"自由段步数分配"。
# sdg-v6   -> 第八轮：视频容器 AVI(MJPG) → MP4(h264)，且**不再顺手出任何观看用格式**
#             （avi / 预览 png / 拼图全部移出采集路径，改由 tools/inspect_episode.py 按需生成）。
#             paths 键 env_avi/wrist_avi → video_env/video_wrist（读侧保留旧键回退）。
#             物理与轨迹同 v5 逐位相同（只换容器与文件名），但训练侧读法不同，仍单独一代。
DATASET_VERSION = "sdg-v6"   # 物理同 v5（指垫贴指面 + dy=+17.5mm + 合爪 0.28），视频改 mp4
SCHEMA_VERSION = 2                             # npz/meta 结构版本，check_dataset 按它判代
TASK_ID = "pick_place_v1"                      # 任务标识（给下游按任务筛数据）
DEFAULT_INSTRUCTION = ("Pick up the medicine box and place it into the wooden box.")
# 起止姿态：oldros = old_ros initial_pose（默认）；golden = 上一轮遥操第 0 帧（A/B 用）；
#           home = 旧预设（塌到 q≈0，仅回归对比用）
START_POSE = "oldros"
# 起止姿态的关节目标（弧度，ARM_JOINTS 顺序）。姿态常量只在 grasp/test_grasp.py 里定义一次，
# 这里只做"名字 -> 姿态"的映射，避免第二份数值。改起点必须同时考虑：
#   START_POSE / DATASET_VERSION 各改一处 + 重跑 tools/probe_grasp_robustness.py
_START_Q_BY_POSE = {"oldros": R.OLDROS_Q_RAW, "golden": R.GOLDEN_Q,
                    "home": np.zeros(len(R.ARM_JOINTS))}


def _start_q():
    """当前 START_POSE 对应的 5 维关节目标（弧度）；未知名字退化成 home（全 0）。"""
    return np.asarray(_START_Q_BY_POSE.get(str(START_POSE).lower(),
                                          np.zeros(len(R.ARM_JOINTS))), float)
# 中文注释：instruction 是给 VLA 类模型用的自然语言指令。纯视觉策略/世界模型可以不用，
# 但字段先占位（固定字符串 + --instruction 可覆盖），免得以后要加就得重刷全批。
OBJECT_NOMINAL = np.array([0.295, 0.090])      # demo 里药盒的静止位置（也是抓取参数的标定点）
PICK_ZONE_XY = (0.31, 0.090)                   # 拾取垫中心（scene.xml 的 pick_zone），仅供画图/说明
# ★随机化的几何带宽由【绿垫】决定，不是拍脑袋写的（2024-11 用户要求：药盒不许离开绿垫）
#   实测换算：垫半径 96mm、竖立药盒 footprint 半对角 37.8mm、余量 2mm
#     -> 盒心离垫心最多 96-37.8-2 = 56.2mm
#   又因为抓取必须靠"绕 pan 轴的同构旋转"来保住工具姿态，旋转会让盒心沿半径 271.6mm 的
#   圆弧移动 -> 由弦长反解：|θ| <= 10° 时 footprint 最远点 92.9mm（在垫内），
#   11° 就顶到 96.0mm 贴边。所以带宽取 ±10°（PadZone 会在运行时再次确认，不硬信这个数）。
#   实测：垫内 |θ|<=10° + dr 0~6mm，配高度补偿与动作候选搜索，抓取 9/9 全成功。
THETA_MAX_DEG = 10.0    # 场景旋转角上限（PadZone 会按实际几何再收一次）
DR_LO, DR_HI = 0.0, 6.0 # 径向抖动（沿 pan 轴 -> 药盒方向，实测正向可用）
PAD_MARGIN = 0.002      # footprint 距垫边至少留 2mm，避免"半只脚在垫外"
YAW_EXTRA_DEG = 1.5     # 药盒自身额外偏航（±，叠在场景旋转之上）
N_JITTER_CANDS = 4      # 扰动候选个数；之后一定还有一个纯同构解兜底

# ---- 为什么这么随机化（实测依据，别再改回去）-----------------------------------
# 直接挪药盒位置(径向)或让它转角度，抓取会大面积失败：
#   径向 -2mm 就掉、+10mm 还行；偏航 -1° 就掉、+5° 还行；绕底座转 15° 也掉。
# 原因不是"抓不到"，而是【工具姿态】跟着 IK 的分支乱跑：Arm.ik() 只有位置 3 个约束，
# 5 自由度臂还剩 2 个自由度是"随手落到哪儿算哪儿"，工具轴倾角一变，
# 指垫就不再骑在药盒的两条窄端面上。
#
# 正确做法（本文件采用）：利用"整台机器人对底座竖轴是旋转对称的"这一事实，
# 把场景【整体】绕底座转 θ，同时把抓取路点的 pan 分量平移对应角度 ——
# 这给出一个【同构解】，抓取几何与 demo 逐位相同（相对药盒完全一致），
# 于是"抓得住"不再是运气，而几何多样性照样有了（画面、姿态、臂形全变了）。
#
# 在这个同构解之上再加【小扰动候选】（抓取高度 ±1.5mm），
# 让每条数据的接触细节有点差异；扰动失败就回退到纯同构解，
# 所以接受率接近 100%，而且每条数据都带"这个动作确实成功了"的标签。
#
# ★ 2024-11 第六轮：ROLL_JITTER_DEG 从 ±4° 改成 0。
#   原因：指垫改成"平面指垫 + 位掩码"之后，指垫面与药盒大面的平行度直接决定夹持，
#   实测 roll 偏离 −90° 就开始在【下探段】蹭到药盒：
#     −95° -> 下探接触 6/400、夹稳倾角 2.52°（还能夹起来）
#     −100° -> 9/400、压盒 1.58 mm（仍能夹起来，但已经不是干净下探）
#     −105°/-110° -> 8/400、压盒 26~28 mm、彻底抓不起来
#   也就是说 roll 现在是"指垫对齐轴"，不再是可抖动的自由度。
#   抓取高度 ±1.5mm 仍然可以抖（实测 0.008~0.016 m 全部干净）。
ROLL_JITTER_DEG = 0.0
GZ_JITTER_MM = 1.5
N_JITTER_CANDS = 4      # 先试几个扰动候选，最后一定有一个纯同构解兜底


class PadZone:
    """
    绿垫（pick_zone）的几何 + "药盒整个 footprint 是否还在垫上"的判据。

    为什么不把 96mm/37.8mm 写死在采样代码里：垫子和药盒尺寸都来自 model/scene.xml，
    写死就会在改模型后静默错位（我们已经在"绕原点转 vs 绕 pan 轴转"上栽过一次）。
    所以这里全部从模型读，采样范围也由它反算。
    """

    def __init__(self, model):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_zone")
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "pick_zone_geom")
        if bid < 0 or gid < 0:
            raise RuntimeError("模型里找不到 pick_zone / pick_zone_geom")
        self.center = np.asarray(model.body_pos[bid][:2], float)
        self.radius = float(model.geom_size[gid][0])          # cylinder 的半径
        self.top_z = float(model.body_pos[bid][2]) + float(model.geom_size[gid][1])
        # 药盒"竖立"时的底面半尺寸（碰撞体 box 的 x/y）
        ogid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_physics")
        self.half = np.asarray(model.geom_size[ogid][:2], float)
        self.footprint_r = float(np.hypot(self.half[0], self.half[1]))

    # ---------- 判据 ----------
    def corners(self, xy, yaw_rad):
        """药盒竖立时底面 4 个角的世界坐标（不含 z）。"""
        Rz = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)],
                       [np.sin(yaw_rad), np.cos(yaw_rad)]])
        loc = np.array([[sx * self.half[0], sy * self.half[1]]
                        for sx in (-1, 1) for sy in (-1, 1)])
        return np.asarray(xy, float) + loc @ Rz.T

    def max_radius_mm(self, xy, yaw_rad):
        """footprint 四角里离垫心最远的那个（mm）。这是"是否还在垫上"的唯一判据。"""
        c = self.corners(xy, yaw_rad)
        return float(np.max(np.linalg.norm(c - self.center, axis=1))) * 1000.0

    def limit_mm(self):
        return (self.radius - PAD_MARGIN) * 1000.0

    def on_pad(self, xy, yaw_rad):
        return self.max_radius_mm(xy, yaw_rad) <= self.limit_mm()

    def theta_limit_deg(self, nominal_xy, pan_center):
        """
        反算：绕 pan 轴转多大角度，footprint 还能整个留在垫上（留 margin）。
        用二分，比手推公式更抗模型改动。
        """
        lo, hi = 0.0, 90.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            th = np.radians(mid)
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            xy = pan_center + Rz @ (np.asarray(nominal_xy, float) - pan_center)
            if self.on_pad(xy, th):
                lo = mid
            else:
                hi = mid
        return lo

    def describe(self):
        return (f"垫心={np.round(self.center,3)} 半径={self.radius*1000:.1f}mm "
                f"顶面z={self.top_z:.4f} | 药盒footprint半对角={self.footprint_r*1000:.1f}mm "
                f"-> 盒心可偏 {self.limit_mm()-self.footprint_r*1000:.1f}mm, "
                f"判据上限 {self.limit_mm():.1f}mm")


class NominalGrasp:
    """
    标定基准：demo 里药盒在 (0.295,0.090) 时解出来的三个接近路点 + pan 符号。

    只算一次（约 0.1 s），之后每条 episode 都用它做"同构旋转"，不再逐步调参。
    """

    def __init__(self, arm):
        a = arm
        p0 = OBJECT_NOMINAL
        roll = np.radians(-90.0)
        seed = [-0.33, -0.08, 0.80, -0.02, roll]
        # pan 关节的旋转轴位置（世界系 xy）—— 场景旋转必须以它为圆心，
        # 否则"绕原点转"和"绕 pan 轴转"差 39mm，TCP 就对不上药盒了（踩过）。
        jid = mujoco.mj_name2id(a.m, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan_joint")
        self.center = np.asarray(a.m.body_pos[a.m.jnt_bodyid[jid]][:2], float)
        # ★ 第七轮：标定解必须带抓取点横向补偿，否则指垫与药盒在 y 向错开 ~17 mm（实测 20/20 全废）。
        self.off = np.asarray(getattr(R, "GRASP_XY_OFFSET", (0.0, 0.0)), float)
        gx, gy = p0[0] + self.off[0], p0[1] + self.off[1]
        self.q_ret, self.e1 = a.ik([p0[0], p0[1], R.RETRACT_Z], seed)
        self.q_abv, self.e2 = a.ik([gx, gy, R.APPROACH_Z], self.q_ret)
        self.q_gsp, self.e3 = a.ik([gx, gy, 0.012], self.q_abv)
        # pan 的旋转符号：绕底座转 +θ 时，pan 到底是 +θ 还是 -θ（不同批次装配方向可能相反）
        self.pan_sign = self._detect_sign(arm)
        # 同构解出来的指垫/工具姿态，仅用于记录
        arm.d.qpos[arm.adr] = self.q_gsp
        mujoco.mj_forward(arm.m, arm.d)
        gid = mujoco.mj_name2id(arm.m, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        self.tool_axis = -np.asarray(arm.d.xmat[gid]).reshape(3, 3)[:, 2]
        self.q_nominal = self.q_gsp.copy()

    def rotate_xy(self, xy, theta_rad):
        """把世界系 xy 点绕 pan 轴转 theta（弧度）。"""
        Rz = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                       [np.sin(theta_rad), np.cos(theta_rad)]])
        return self.center + Rz @ (np.asarray(xy, float) - self.center)

    def _detect_sign(self, arm):
        """
        判定"场景转 +θ 时 pan 要 ±θ 哪个方向"。

        用 θ=+35° 做一次刚体旋转测试：把标定 TCP 绕 pan 轴转过去当目标，
        对两个符号各做一次正运动学，残差小的那个就是对的（正确时残差 ≈ 0.00mm）。
        """
        th = np.radians(35.0)
        arm.d.qpos[arm.adr] = self.q_gsp
        mujoco.mj_forward(arm.m, arm.d)
        p_t = self.rotate_xy(arm.d.site_xpos[arm.tcp][:2], th)
        best, best_err = +1, 1e9
        for sign in (+1, -1):
            q = self.q_gsp.copy()
            q[0] = float(np.clip(self.q_gsp[0] + sign * th, arm.lo[0], arm.hi[0]))
            arm.d.qpos[arm.adr] = q
            mujoco.mj_forward(arm.m, arm.d)
            err = float(np.linalg.norm(arm.d.site_xpos[arm.tcp][:2] - p_t))
            if err < best_err:
                best, best_err = sign, err
        return best


def _PAN_CENTER():
    """pan 轴中心的模块级缓存（sample_episode 在还没有 arm 的地方也要用）。"""
    return _PAN_CENTER.value


_PAN_CENTER.value = np.zeros(2)


def congruent_waypoints(nom, theta_deg):
    """把标定好的三个路点整体绕底座旋转 θ（= 同构解）。"""
    _PAN_CENTER.value = np.asarray(nom.center, float)   # 保证采样与同构解用同一个圆心
    th = np.radians(theta_deg)
    q_ret, q_abv, q_gsp = nom.q_ret.copy(), nom.q_abv.copy(), nom.q_gsp.copy()
    for q in (q_ret, q_abv, q_gsp):
        q[0] = q[0] + nom.pan_sign * th
    return dict(q_ret=q_ret, q_abv=q_abv, q_gsp=q_gsp)
Z_SPAWN = 0.030                                # 出生高度（药盒静止时中心 z≈0.0278，只掉 2mm）
SETTLE_STEPS = 1200                            # 落定步数（与 demo 一致）
NOMINAL_P0_Z = 0.02777                         # demo 里药盒落定后的中心高度（抓取参数的标定基准）
# ↑ 为什么需要它：旋转后的药盒可能落在桌面(z=0)而不是拾取垫(顶面 z=0.002)上，
#   2mm 的高度差就足以让指垫错过窄端面（实测：不补偿时只有 θ∈{-75,-70,0,+5,+10} 能抓）。
#   所以每条 episode 先落定、量出 p0.z，再把抓取高度整体补偿 (p0.z - NOMINAL_P0_Z)。

# 验收阈值
TOL_IK_MM = 2.0
MIN_UPRIGHT_Z = 0.020
MIN_RISE_MM = 95.0
BOX_INNER_X, BOX_INNER_Y = 0.050, 0.062        # 木盒内腔（115x142 外径、壁 4mm）留的判据
BOX_DROP_Z = (0.008, 0.045)


# ============================================================
# 第 1 段  随机化：一条 episode 的全部随机量
# ============================================================


def sample_episode(rng, idx, pad, theta_max_deg):
    """
    采样一条 episode 的随机量（纯 dict，可直接 JSON 序列化）。

    ★几何硬约束：药盒必须整个英尺（footprint）留在绿垫上（用户明确要求）。
      做法是"绕 pan 轴转 theta + 径向 dr"，但 theta 的上限由 PadZone 反算（实测约 10°），
      不是随手给的 —— 因为只有"绕 pan 轴的同构旋转"才能保住工具姿态，
      而只要旋转，盒心就沿半径 271.6mm 的圆弧走，转多了必然离开垫子。
      采样完还会立刻用 footprint 判据夹一次，保证落点算出来就是合法的。
    """
    # 半径/角度都在垫内取；theta 用 PadZone 反算的上限（并受 THETA_MAX_DEG 约束）
    tmax = float(min(theta_max_deg, THETA_MAX_DEG))
    theta = float(rng.uniform(-tmax, tmax))
    dr = float(rng.uniform(DR_LO, DR_HI))
    yaw_extra = float(rng.uniform(-YAW_EXTRA_DEG, YAW_EXTRA_DEG))

    C = _PAN_CENTER()
    radial = OBJECT_NOMINAL - C
    radial = radial / np.linalg.norm(radial)

    def place(theta_deg, dr_mm):
        th = np.radians(theta_deg)
        Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return C + Rz @ (OBJECT_NOMINAL - C) + radial * (dr_mm / 1000.0)

    # 夹紧：越界就按比例回缩（先缩 dr，再缩 theta），保证采样结果本身就合规
    xy = place(theta, dr)
    yaw = np.radians(theta + yaw_extra)
    for _ in range(60):
        if pad.on_pad(xy, yaw):
            break
        dr *= 0.75
        theta *= 0.85
        xy = place(theta, dr)
        yaw = np.radians(theta + yaw_extra)

    return dict(
        episode_id=int(idx),
        seed=int(rng.integers(0, 2 ** 31 - 1)),
        spawn=dict(x=round(float(xy[0]), 5), y=round(float(xy[1]), 5), z=Z_SPAWN,
                   yaw_deg=round(theta + yaw_extra, 2), theta_deg=round(theta, 2),
                   dr_mm=round(dr, 2), yaw_extra_deg=round(yaw_extra, 2)),
        jitter=dict(roll_delta_deg=round(float(rng.uniform(-ROLL_JITTER_DEG, ROLL_JITTER_DEG)), 2),
                    gz_delta_mm=round(float(rng.uniform(-GZ_JITTER_MM, GZ_JITTER_MM)), 2)),
        timing_scale=round(float(rng.uniform(0.90, 1.10)), 3),
        camera=dict(env_pos_jitter_m=round(float(rng.uniform(0.01, 0.03)), 3),
                    env_fovy_delta_deg=round(float(rng.uniform(-3.0, 3.0)), 2),
                    env_yaw_delta_deg=round(float(rng.uniform(-2.0, 2.0)), 2),
                    env_pitch_delta_deg=round(float(rng.uniform(-2.0, 2.0)), 2)),
        sensor=dict(blur_sigma=round(float(rng.uniform(0.6, 1.0)), 3),
                    noise_sigma=round(float(rng.uniform(2.0, 4.0)), 3),
                    # 逐帧曝光抖动：真机 AGC 幅度很小，2%~5% 是"每帧重掷"时代的旧值
                    gain_jitter=round(float(rng.uniform(0.005, 0.02)), 4),
                    # 视觉随机化（与 render/test_render.SensorModel 同一套开关）
                    # 拖影：长度由像素速度决定，这里只随机"系数"和"上限"
                    motion_k=round(float(rng.uniform(0.45, 0.80)), 3),
                    motion_px_max=round(float(rng.uniform(3.0, 4.5)), 2),
                    vignette=round(float(rng.uniform(0.22, 0.38)), 3),
                    # 色彩抖动改成"每条视频基准 + 慢漂移"，幅度回到 ±2%~±5%
                    color_jitter=round(float(rng.uniform(0.02, 0.05)), 3),
                    flicker_tau_s=round(float(rng.uniform(1.2, 3.0)), 2)),
        light=dict(azim_deg=round(float(rng.uniform(0.0, 360.0)), 1),
                   tilt_deg=round(float(rng.uniform(5.0, 25.0)), 1),
                   diffuse=round(float(rng.uniform(0.35, 0.70)), 3)),
        material=dict(gain=round(float(rng.uniform(0.90, 1.10)), 3)),
        grasp=None,          # 由 search_grasp() 真跑物理后回填
        pad=None,            # 由 ensure_on_pad() 回填（落定后实测的 footprint 半径）
    )


def make_spawn_fn(cfg):
    """
    返回一个 spawn(arm) 回调：把药盒放到随机位置（立着、带一点偏航）。

    ★数据卫生：这是【出生位姿】。只在落定前写一次 qpos，随后 mj_step 落定，
      整个 episode 再不会有任何 qpos 写入 —— 握持仍然完全来自接触力。
    """
    x, y, z = cfg["spawn"]["x"], cfg["spawn"]["y"], cfg["spawn"]["z"]
    yaw = np.radians(cfg["spawn"]["yaw_deg"])

    def _spawn(arm):
        d, a = arm.d, arm
        d.qpos[a.oadr + 0:a.oadr + 3] = [x, y, z]
        # 绕 z 的偏航四元数（w, x, y, z），保持药盒直立
        d.qpos[a.oadr + 3:a.oadr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]

    return _spawn


# ============================================================
# 第 2 段  录制一条轨迹（纯物理，秒级）
# ============================================================


def record_episode(arm, cfg, nom, wp, roll):
    """
    跑一条真物理轨迹，返回 (recorder, info, plan)。

    复用 render/test_render.py 的 run_episode —— 阶段划分、步数口径、ctrl 写入顺序
    与 demo 完全同源，所以 demo 里验证过的东西在这里一样成立。
    wp 给的是【接近路点的同构解】，搬运/放置段仍由 run_episode 自己按木盒位置解 IK
    （木盒是固定道具，不跟着转）。

    ★每条都换一个全新 MjData：连续复用同一个 data 跑不同出生位姿时，
      本机偶发 SIGFPE（换新 data 后连续 20+ 条零崩溃，复用则几十条必崩）。
      代价只有几毫秒，直接规避。
    """
    arm.d = mujoco.MjData(arm.m)
    plan = R.make_step_plan(cfg["timing_scale"])
    rec = R.EpisodeRecorder(arm, fps=R.FPS)
    info = R.run_episode(arm, rec, roll=roll, gz=R.DEF_GZ, transport=True,
                         plan=plan, spawn=make_spawn_fn(cfg),
                         settle=SETTLE_STEPS, wp=wp, start_pose=START_POSE)
    cfg["_start_pose"] = info.get("start_pose", START_POSE)   # 下划线键不进 randomization
    # 起止姿态的实际关节值（度）。取 init 段落定后的 qpos，不是"请求的常量"。
    cfg["_start_joints_deg"] = [round(float(v), 3) for v in np.degrees(info["q_home"])]
    return rec, info, plan


def settled_p0(arm, cfg):
    """
    只做"出生 + 落定"，返回药盒落定后的中心位置（并缓存进 cfg）。

    为什么要单独跑一次：抓取高度必须按药盒【实际落在哪个面上】做补偿，
    否则拾取垫(2mm 厚)与桌面(0mm)的高度差会让指垫错过窄端面。
    这次落定是确定性的，和真正录制时那段落定逐位相同，所以量到的 p0 可以直接用。
    """
    if "_p0" in cfg:
        return np.asarray(cfg["_p0"], float)
    arm.d = mujoco.MjData(arm.m)
    mujoco.mj_resetDataKeyframe(arm.m, arm.d, 0)
    make_spawn_fn(cfg)(arm)
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(arm.m, arm.d)
    cfg["_p0"] = [float(v) for v in arm.obj_pos()]
    return np.asarray(cfg["_p0"], float)


def ensure_on_pad(arm, cfg, pad, rng, theta_max_deg, max_tries=12):
    """
    落定后校验"药盒是否整个站在绿垫上"，不满足就重新采样出生位姿（最多 max_tries 次）。

    为什么采样算了还要再查一次：采样用的是"几何预测"，而物理落定会有 1~2mm 的滑移/微转，
    再加上药盒可能落在垫沿上翻倒。只有真正 mj_step 落定、拿实际 qpos 算 footprint，
    才能证明"这条数据确实起点在垫内"。查不过就换一个位姿，绝不放行。

    返回 (ok, cfg)。cfg["pad"] 里记实测值：footprint 最大半径、是否在垫上、重试次数。
    """
    for attempt in range(1, max_tries + 1):
        cfg.pop("_p0", None)                      # 位姿变了，落定缓存必须失效
        p0 = settled_p0(arm, cfg)                 # 真落定一次（确定性，约 0.05s）
        quat = np.asarray(arm.d.qpos[arm.oadr + 3:arm.oadr + 7], float)
        # 只关心绕 z 的偏航（药盒竖立，roll/pitch 应为 0；翻倒由 p0.z 判据拦）
        yaw = 2.0 * np.arctan2(quat[3], quat[0])
        fp_mm = pad.max_radius_mm(p0[:2], yaw)
        on_pad = fp_mm <= pad.limit_mm()
        upright = p0[2] >= MIN_UPRIGHT_Z
        cfg["pad"] = dict(on_pad=bool(on_pad and upright),
                          footprint_max_mm=round(float(fp_mm), 2),
                          limit_mm=round(float(pad.limit_mm()), 2),
                          yaw_settled_deg=round(float(np.degrees(yaw)), 2),
                          obj_z_mm=round(float(p0[2] * 1000), 2),
                          retries=int(attempt - 1))
        if on_pad and upright:
            return True, cfg
        # 不满足：重抽出生位姿（theta/dr/yaw 一起重来）
        fresh = sample_episode(rng, cfg["episode_id"], pad, theta_max_deg)
        cfg.update({k: fresh[k] for k in ("spawn", "jitter", "pad")})
        cfg.pop("pad", None)
        cfg["pad_retry"] = attempt
    return False, cfg


def waypoints_for(arm, nom, cfg, roll_delta_deg=0.0, gz_delta_mm=0.0):
    """
    给定随机化 cfg，算出三个接近路点：
      1) 先拿"同构解"（场景旋转后的标定解，pan 平移 θ）
      2) 用真实落定的 p0 做高度补偿：药盒落在桌面还是垫子上，抓取高度差多少就补多少
      3) 叠上扰动（roll ±4°、gz ±1.5mm），再用位置 IK 重解抓取路点

    这一步是整批数据能"既随机又不失败"的关键：同构解保证姿态对，
    高度补偿保证指垫对准，扰动只改接触细节。
    """
    p0 = settled_p0(arm, cfg)
    theta = cfg["spawn"]["theta_deg"]
    wp = congruent_waypoints(nom, theta)

    q_abv = np.asarray(wp["q_abv"], float).copy()
    q_abv[4] = float(np.clip(q_abv[4] + np.radians(roll_delta_deg), arm.lo[4], arm.hi[4]))
    dz = (p0[2] - NOMINAL_P0_Z) + gz_delta_mm / 1000.0
    # ★ 第七轮：横向补偿（GRASP_XY_OFFSET）要跟着场景旋转一起转，否则每条 episode
    #   指垫都偏 ~17 mm —— 实测漏了这一步时 20/20 全部 no_lift。
    off = np.asarray(getattr(R, "GRASP_XY_OFFSET", (0.0, 0.0)), float)
    if off.any():
        th = np.radians(theta)
        c, sn = np.cos(th), np.sin(th)
        off = np.array([c * off[0] - sn * off[1], sn * off[0] + c * off[1]])
    q_gsp, _ = arm.ik([p0[0] + off[0], p0[1] + off[1], 0.012 + dz], q_abv)
    return dict(q_ret=np.asarray(wp["q_ret"], float), q_abv=q_abv, q_gsp=q_gsp)


def search_grasp(arm, cfg, nom, attempts=None):
    """
    动作候选搜索：先试带扰动的解，最后一次用"纯同构 + 高度补偿"兜底。
    每次都是【真跑物理 + 验收】，返回第一个通过的。

    这样每条数据的动作标签都是"确实完成了任务的那个动作"，
    json 里记下实际用的 roll / gz / 是否用了扰动 / 试了几次。
    """
    # 候选顺序：先试本条的随机扰动，再试一组固定的粗网格，最后用纯同构解兜底。
    # 每多一个候选 ≈ 0.1s，换来的是"这条数据不会因为 2mm 的位置差就废掉"。
    #
    # ★ 2024-11 第六轮：候选表按 ROLL_JITTER_DEG 过滤。
    #   指垫方案下 roll 是"指垫对齐轴"，±2° 就开始在【下探段】蹭盒（见 ROLL_JITTER_DEG 注释），
    #   所以这些 roll 扰动候选不是"备用成功路径"，而是"必然违反 G1 的路径"，白白耗时。
    #   实测：过滤前 seed 3 的第 1/3 条要试 5/7 次；过滤后一次命中。
    cands = [("jitter", cfg["jitter"]["roll_delta_deg"], cfg["jitter"]["gz_delta_mm"]),
             ("jitter", -cfg["jitter"]["roll_delta_deg"], -cfg["jitter"]["gz_delta_mm"])]
    for dr_roll, dz in ((2.0, 1.0), (-2.0, -1.0), (4.0, 1.5), (-4.0, -1.5),
                        (6.0, 0.0), (-6.0, 0.0), (8.0, 2.0), (-8.0, -2.0),
                        (8.0, -2.0), (-8.0, 2.0)):
        if abs(dr_roll) > abs(ROLL_JITTER_DEG) + 1e-9:
            continue                     # roll 抖动上限之外的候选：跳过（见上）
        cands.append(("jitter", dr_roll, dz))
    cands.append(("congruent", 0.0, 0.0))

    n = 0
    wp = None
    for kind, dr_roll, dz in cands:
        n += 1
        wp = waypoints_for(arm, nom, cfg, dr_roll, dz)
        roll = float(np.degrees(wp["q_gsp"][4]))
        rec, info, plan = record_episode(arm, cfg, nom, wp, roll)
        ok, reasons, _ = judge(rec, info, cfg)
        if ok:
            return rec, info, plan, dict(kind=kind, roll_deg=round(roll, 2),
                                         gz=round(R.DEF_GZ + (cfg["_p0"][2] - NOMINAL_P0_Z)
                                                  + (dz / 1000.0 if kind == "jitter" else 0.0), 5),
                                         attempt=n), n
        if attempts is not None and n >= attempts:
            break
    return None, None, plan, None, n


def judge(rec, info, cfg):
    """
    自动验收。返回 (ok, reasons, metrics)。

    reasons 为空才算通过；每条失败原因都写清楚，
    这样 index.json 里能直接统计"失败类型分布"（比如大多是 IK 不可达）。
    """
    m = rec.metrics()
    rep = R.episode_report(rec, rec.a, info)
    obj = np.asarray(rec._rows["obj_pos"])
    ph = np.asarray(rec._rows["phase"])
    cn = np.asarray(rec._rows["contact_n"])
    box = np.asarray(rec._rows["box_n"])

    reasons = []
    ik_err = [abs(v) for v in info["ik_err_mm"] if np.isfinite(v)]
    if ik_err and max(ik_err) > TOL_IK_MM:
        reasons.append(f"ik_unreachable({max(ik_err):.1f}mm)")
    if info["p0"][2] < MIN_UPRIGHT_Z:
        reasons.append(f"toppled_at_spawn(z={info['p0'][2]*1000:.1f}mm)")
    pad_info = cfg.get("pad") or {}
    if pad_info and not pad_info.get("on_pad", True):
        reasons.append(f"off_pad(footprint {pad_info.get('footprint_max_mm')}mm "
                       f">> {pad_info.get('limit_mm')}mm)")
    if m["rise_mm"] < MIN_RISE_MM:
        reasons.append(f"no_lift({m['rise_mm']:.1f}mm)")

    bx = rec.a.m.body_pos[rec.a.tbox]
    fin = np.asarray(m["obj_end"])
    if not (abs(fin[0] - bx[0]) <= BOX_INNER_X and abs(fin[1] - bx[1]) <= BOX_INNER_Y
            and BOX_DROP_Z[0] <= fin[2] <= BOX_DROP_Z[1]):
        reasons.append("not_in_box")

    rel = np.where(ph == "release")[0]
    if len(rel) and (cn[int(rel[-1]) + 1:] > 0).any():
        reasons.append("still_holding_after_release")
    if int((box > 0).sum()) > 0:
        reasons.append("arm_hit_box")

    metrics = dict(
        rise_mm=round(float(m["rise_mm"]), 3),
        hdisp_mm=round(float(m["hdisp_mm"]), 3),
        obj_start=[round(float(v), 5) for v in info["p0"]],
        obj_end=[round(float(v), 5) for v in fin],
        drop_mm=round(float(rep["drop_mm"]), 3) if np.isfinite(rep["drop_mm"]) else None,
        home_dq_max_deg=round(float(rep["home_dq_max_deg"]), 3),
        grip_ctrl_end=round(float(rep["grip_ctrl_end"]), 4),
        box_hit_frames=int((box > 0).sum()),
        frames=int(len(rec)),
        sim_time_s=round(float(rec._rows["time"][-1]) - float(rec._rows["time"][0]), 4),
        moving_ratio=round(float(rep["motion"]["moving_ratio"]), 3),
        longest_static_s=round(float(rep["motion"]["longest_static_sec"]), 3),
        ik_err_mm=[round(float(v), 3) for v in info["ik_err_mm"]],
        start_pose=info.get("start_pose"),
        pad=pad_info,
    )
    return (not reasons), reasons, metrics


# ============================================================
# 第 3 段  渲染成两条 MP4（不落 800MB 图片数组，除非 --keep-frames）
# ============================================================


def apply_light_and_material(model, cfg):
    """
    光照 + 材质反照率随机化（不动任何几何，所以不影响抓取）。

    为什么值得做：现在场景只有一盏固定方向、固定强度的定向光（(0,0,-1) / 0.5），
    材质全白。对视觉策略来说这是"最容易被记住、最不迁移"的东西 ——
    训练集里光方向一变，模型就懵。把它随机化是性价比最高的一步。
      · 光照：方位角 0~360°、俯仰 5~25°、强度 0.35~0.70
      · 材质：药盒/木盒/桌面/垫子的 rgba 整体 ×0.90~1.10（只改明暗，不改色相）
    返回实际使用的参数，写进 json。
    """
    cfg_l = cfg["light"]
    az = np.radians(cfg_l["azim_deg"])
    tilt = np.radians(cfg_l["tilt_deg"])
    # dir 是"光线传播方向"：从上方斜射下来
    d = np.array([-np.sin(tilt) * np.cos(az), -np.sin(tilt) * np.sin(az), -np.cos(tilt)])
    model.light_dir[0] = d
    model.light_diffuse[0] = [cfg_l["diffuse"]] * 3

    gain = float(cfg["material"]["gain"])
    touched = []
    for name in ("mat_obj_front", "mat_obj_back", "mat_obj_side", "mat_box", "mat_table"):
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, name)
        if mid < 0:
            continue
        rgba = np.asarray(model.mat_rgba[mid], float).copy()
        rgba[:3] = np.clip(rgba[:3] * gain, 0.0, 1.0)
        model.mat_rgba[mid] = rgba
        touched.append(name)
    return dict(light_dir=[round(float(v), 3) for v in d],
                diffuse=cfg_l["diffuse"], mat_gain=gain, mats=touched)


def apply_camera_jitter(model, cfg):
    """
    环境相机抖动：位置沿三个轴各偏移 ±jitter，fovy 加一个偏差。
    腕部相机（body 挂载）不能动，否则画面与真机分布不符。
    返回 (原值, 新值) 以便 json 记录实际用的参数。
    """
    cid = model.cam("env_camera").id
    base_pos = np.array([0.65, 0.0, 0.55])
    j = cfg["camera"]["env_pos_jitter_m"]
    rng = np.random.default_rng(cfg["seed"])
    pos = base_pos + rng.uniform(-j, j, 3)
    fovy = 30.0 + cfg["camera"]["env_fovy_delta_deg"]
    model.cam_pos[cid] = pos
    model.cam_fovy[cid] = fovy
    # 朝向抖动（2024-11 新增）：外参只抖位置不够，视线方向也要抖 ±2°，
    # 否则模型会学到"永远同一个灭点"。MuJoCo 3.16 的相机姿态字段是 cam_quat（四元数），
    # 没有 cam_xyaxes —— 所以绕世界 z 转 yaw、绕相机 x 轴转 pitch，再合成四元数。
    yaw = np.radians(cfg["camera"].get("env_yaw_delta_deg", 0.0))
    pitch = np.radians(cfg["camera"].get("env_pitch_delta_deg", 0.0))
    q0 = np.asarray(model.cam_quat[cid], float)          # 原来的相机朝向
    R0 = np.zeros(9)
    mujoco.mju_quat2Mat(R0, q0)
    R0 = R0.reshape(3, 3)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1.0]])
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]])
    R1 = Rx @ Rz @ R0
    q1 = np.zeros(4)
    mujoco.mju_mat2Quat(q1, R1.ravel())
    model.cam_quat[cid] = q1
    return dict(pos=[round(float(v), 4) for v in pos],
                fovy=round(float(fovy), 3),
                intrinsics=None)          # 内参在渲染时按实际 fovy 算，回填


class Mp4Sink:
    """
    MP4 / h264 写入器（30fps / 固定尺寸）—— ★采集只出这一种视频。

    为什么是 mp4+h264 而不是 AVI(MJPG)：
      * 训练侧（lerobot / ACT / ffmpeg / pyav / 浏览器）都直接吃得下，不用再转码；
      * 同长度同一帧，体积约为 MJPG AVI 的 1/10（实测 455 帧 640×480：0.9 MB vs 13 MB）。
    所以"生成数据"和"生成方便我肉眼看的格式"可以彻底分开：
    这里只写训练要用的那一份，拼图/预览图/帧号烧字全部搬到 tools/inspect_episode.py。

    burn=True 时在左上角烧 "Env F:123" —— 调试很好用，但【不能进训练数据】：
    那等于在像素里埋了一个帧号捷径（真机画面没有这行字）。
    所以批量采集默认 burn=False，只在 inspect 工具里用。
    """

    def __init__(self, path, fps, size, burn=False, codec="avc1"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*str(codec))
        self.vw = cv2.VideoWriter(str(self.path), fourcc, float(fps), size)
        if not self.vw.isOpened():
            raise RuntimeError(f"VideoWriter 打不开: {self.path}（codec={codec}）")
        self.size = size
        self.burn = bool(burn)
        self.codec = str(codec)
        self.n = 0

    def write(self, rgb, tag):
        frame = rgb.copy() if rgb.flags.writeable else rgb
        if self.burn:
            txt = f"{tag} F:{self.n}"
            cv2.putText(frame, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.vw.write(frame[:, :, ::-1])       # RGB -> BGR
        self.n += 1

    def close(self):
        self.vw.release()


# 老名字保留一行别名：老脚本/老日志里的 AviSink 现在其实就是 Mp4Sink（容器已换）。
AviSink = Mp4Sink


# ---------------------------------------------------------------------------
# 无渲染路径的相机参数（外参 / 内参）
# ---------------------------------------------------------------------------
# 为什么需要：负样本和 --no-render 样本【没有视频】，但下游仍然要能把像素投到 3D。
# 所以即使不渲染，也把逐帧外参算出来（纯粹是 mj_forward + 矩阵运算，无光栅化）。
# ★公式必须与 render/test_render.CameraRig.extrinsics/intrinsics 完全一致，
#   否则"有视频"和"没视频"两批数据的外参会系统性错开。check 由 check_dataset.py
#   的 test_render 交叉比对守（见 collect/check_dataset.py 的 --cross 选项）。

def _cam_math(model, width, height):
    """返回 (cam_ids, intrinsics(cam), extrinsics(data, cam)) 三个纯函数。"""
    cam_ids = {}
    for c in R.CAMS:
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
        if cid < 0:
            raise RuntimeError(f"模型里找不到相机 '{c}'")
        cam_ids[c] = cid

    def intrinsics(cam):
        fovy = float(model.cam_fovy[cam_ids[cam]])
        fy = height / (2.0 * np.tan(np.radians(fovy) / 2.0))
        return dict(fx=fy, fy=fy, cx=width / 2.0, cy=height / 2.0,
                    width=width, height=height, fovy=fovy)

    def extrinsics(data, cam):
        cid = cam_ids[cam]
        Rc = np.array(data.cam_xmat[cid], float).reshape(3, 3)
        p = np.array(data.cam_xpos[cid], float)
        T = np.eye(4)
        T[:3, :3] = Rc.T
        T[:3, 3] = -Rc.T @ p
        return T

    return cam_ids, intrinsics, extrinsics


_PRISTINE = {}


def _pristine_model():
    """
    复用同一个 MjModel，但每次先把它还原成"未经随机化"的原始值。

    为什么要复用：`MjModel.from_xml_path` 约 60~80 ms，逐条新建就是 0.1 s/条的纯开销
    （实测 20 条 --no-render 从 5.5 s 涨到 8.0 s，全是这个）。
    为什么必须还原：`apply_camera_jitter` 是**以当前 cam_quat 为基准**再叠一个旋转，
    `apply_light_and_material` 是**乘性**改 mat_rgba —— 不还原就会逐条累积漂移，
    第 20 条的画面/光照已经不是 json 里记的那个参数了。这种静默漂移最难查。
    """
    model = _PRISTINE.get("model")
    if model is None:
        model = mujoco.MjModel.from_xml_path(R.SCENE_XML)
        _PRISTINE["model"] = model
        _PRISTINE["snap"] = dict(cam_pos=model.cam_pos.copy(), cam_quat=model.cam_quat.copy(),
                                 cam_fovy=model.cam_fovy.copy(), light_dir=model.light_dir.copy(),
                                 light_diffuse=model.light_diffuse.copy(),
                                 mat_rgba=model.mat_rgba.copy())
    else:
        for k, v in _PRISTINE["snap"].items():
            getattr(model, k)[:] = v
    return model


def extrinsics_only(ep, cfg, width=640, height=480):
    """
    不渲染、只算逐帧相机外参 + 内参（负样本 / --no-render 用）。

    光照与相机抖动都从 cfg 派生（apply_light_and_material / apply_camera_jitter），
    所以这里的相机位姿与"真渲染那条路"逐位相同 —— 以后想补渲染这批负样本，
    直接拿这些外参就能对上画面。
    成本：453 帧 × mj_forward ≈ 0.2~0.4 s，不建 Renderer（所以没有 EGL 开销）。
    """
    model = _pristine_model()
    data = mujoco.MjData(model)
    light_info = apply_light_and_material(model, cfg)
    cam_info = apply_camera_jitter(model, cfg)
    _, intrin, extrin = _cam_math(model, width, height)

    n = len(ep["qpos"])
    T_env = np.empty((n, 16), np.float32)
    T_wrist = np.empty((n, 16), np.float32)
    for i in range(n):
        data.qpos[:] = ep["qpos"][i]
        mujoco.mj_forward(model, data)
        T_env[i] = extrin(data, R.CAM_ENV).ravel()
        T_wrist[i] = extrin(data, R.CAM_WRIST).ravel()

    cam_info["intrinsics"] = intrin(R.CAM_ENV)
    cam_info["intrinsics_by_cam"] = {c: intrin(c) for c in R.CAMS}
    cam_info["light"] = light_info
    return dict(cameras=cam_info, cam_env_T_cw=T_env, cam_wrist_T_cw=T_wrist,
                video_frames=None, width=int(width), height=int(height),
                rendered=False)


def render_episode(ep, cfg, out_env, out_wrist, width=640, height=480,
                   keep_frames=None, burn_counter=False):
    """
    纯正运动学回放 + 光栅化 + 传感器劣化，直接写两条 MP4（h264）。

    与 demo 的差别：不落 800MB 的图片数组（可选 --keep-frames），
    并且额外记录【逐帧相机外参】—— 训练时做像素↔3D 投影要用。
    也不出任何"为了肉眼好看"的产物（拼图/预览/帧号烧字）—— 那些在 tools/inspect_episode.py。
    """
    model = mujoco.MjModel.from_xml_path(R.SCENE_XML)
    data = mujoco.MjData(model)
    light_info = apply_light_and_material(model, cfg)   # 光照/材质随机化（不动几何）
    cam_info = apply_camera_jitter(model, cfg)
    rig = R.CameraRig(model, width=width, height=height)
    sensor = R.SensorModel(blur_sigma=cfg["sensor"]["blur_sigma"],
                           noise_sigma=cfg["sensor"]["noise_sigma"],
                           gain_jitter=cfg["sensor"]["gain_jitter"],
                           seed=cfg["seed"],
                           # 视觉随机化：老 cfg（没有这些键）走默认值
                           motion_prob=cfg["sensor"].get("motion_prob", 0.75),
                           motion_px=(1.0, cfg["sensor"].get("motion_px_max", 3.0)),
                           motion_k=cfg["sensor"].get("motion_k", 0.6),
                           vignette=cfg["sensor"].get("vignette", 0.30),
                           color_jitter=cfg["sensor"].get("color_jitter", 0.03),
                           flicker_tau_s=cfg["sensor"].get("flicker_tau_s", 2.0),
                           fps=R.FPS)
    sinks = {R.CAM_ENV: Mp4Sink(out_env, R.FPS, (width, height), burn=burn_counter),
             R.CAM_WRIST: Mp4Sink(out_wrist, R.FPS, (width, height), burn=burn_counter)}
    bufs = {c: [] for c in R.CAMS} if keep_frames else None

    # 运动虚影要跟着真实运动：用"上一帧 TCP 在图像平面的像素位置"算像素速度。
    # 静止段 speed≈0 -> 不糊；下探/搬运段 1.5~3 px。与 test_render.cmd_render 同口径。
    n = len(ep["qpos"])
    T_env, T_wrist = [], []
    prev_uv = {c: None for c in R.CAMS}

    def _speed(c, data, i):
        uv = R._project_px(rig, data, c, ep["tcp_pos"][i])
        sp = 0.0
        if uv is not None and prev_uv[c] is not None:
            sp = float(np.linalg.norm(uv - prev_uv[c]))
        prev_uv[c] = uv
        return sp

    for i in range(n):
        data.qpos[:] = ep["qpos"][i]
        mujoco.mj_forward(model, data)
        raw = rig.grab(data)
        speeds = {c: _speed(c, data, i) for c in R.CAMS}
        for c in R.CAMS:
            img = sensor.apply(raw[c], speed_px=speeds[c])
            sinks[c].write(img, "Env" if c == R.CAM_ENV else "Wrist")
            if bufs is not None:
                bufs[c].append(img)
        T_env.append(rig.extrinsics(data, R.CAM_ENV).ravel())
        T_wrist.append(rig.extrinsics(data, R.CAM_WRIST).ravel())

    for s in sinks.values():
        s.close()
    # ★帧数对齐断言：视频第 i 帧必须正好是 npz 第 i 行。MP4(h264) 编码不改帧数，
    #   但"没断言"就等于"没保证" —— 训练侧一旦错位一帧，动作-画面配对就是错的。
    video_frames = {c: int(sinks[c].n) for c in R.CAMS}
    bad = {c: v for c, v in video_frames.items() if v != n}
    if bad:
        raise RuntimeError(f"视频帧数与轨迹帧数不一致: {bad} != {n}（{out_env}）")

    cam_info["intrinsics"] = rig.intrinsics(R.CAM_ENV)
    cam_info["intrinsics_by_cam"] = {c: rig.intrinsics(c) for c in R.CAMS}
    cam_info["sensor_drawn"] = sensor.params()
    rig.close()

    if bufs is not None:
        np.savez(keep_frames, env=np.stack(bufs[R.CAM_ENV]),
                 wrist=np.stack(bufs[R.CAM_WRIST]), fps=np.float32(R.FPS))

    cam_info["light"] = light_info
    return dict(cameras=cam_info,
                cam_env_T_cw=np.asarray(T_env, np.float32),
                cam_wrist_T_cw=np.asarray(T_wrist, np.float32),
                video_frames=video_frames, n_frames=int(n),
                width=int(width), height=int(height), rendered=True)


# ============================================================
# 第 4 段  落盘：scripts/ep_xxxxxx.npz + .json，videos/ep_xxxxxx_*.mp4
# ============================================================


def _sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def _write_json_atomic(path, obj, indent=1):
    """
    原子写 JSON：先写 <path>.tmp，再 os.replace 换上去。

    为什么非原子不可：本机 mj_step 偶发 SIGFPE 会直接杀进程（不抛异常、不给 finally），
    要是正写到一半被杀，留下半截 JSON，整批清单就废了 —— 而清单是唯一能说明
    "哪条是 accepted、每条用了什么参数"的东西。os.replace 在同一文件系统上是原子的：
    要么还是旧文件，要么已经是完整新文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _savez_atomic(path, **arrays):
    """原子写 npz（同上）。注意 numpy 会给文件对象不加 .npz 后缀，所以传的是文件句柄。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)
    return path


def save_episode(rec, cfg, metrics, plan, art, accept, reasons,
                 instruction=DEFAULT_INSTRUCTION, keep_frames=None):
    """
    一条 episode 的全部产物：

        scripts/ep_000000.npz   轨迹（qpos/qvel/ctrl/phase/接触力/位姿/相机外参）
        scripts/ep_000000.json  标签（随机量 + 验收结果 + 指标 + 模型指纹）
        videos/ep_000000_env.mp4, _wrist.mp4     ← 只有正样本（除非 --render-failed）

    art 是"渲染产物或纯外参产物"（render_episode / extrinsics_only 的返回值）：
      art=None            → 连外参都没有（--no-extrinsics），npz 里不会有 cam_*_T_cw
      art.rendered=False  → 有外参没视频（负样本 / --no-render），has_video=false
    """
    eid = cfg["episode_id"]
    stem = f"ep_{eid:06d}"
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    arrays = {k: (np.array(v, dtype="<U16") if k == "phase" else np.array(v))
              for k, v in rec._rows.items()}
    if art is not None:
        arrays["cam_env_T_cw"] = np.asarray(art["cam_env_T_cw"], np.float32).ravel()   # (N*16,)
        arrays["cam_wrist_T_cw"] = np.asarray(art["cam_wrist_T_cw"], np.float32).ravel()

    # 成功标签用与 demo 完全同源的判据再算一遍（evaluate_success 只读轨迹数组）。
    # 存两份（accepted / success_info）不是为了冗余，是为了让 check_dataset 能守住
    # "自动验收"与"成功标签"永远一致；两个口径一旦分叉，训练侧的标签就不可信了。
    success_info = R.evaluate_success(dict(metrics=metrics), arrays)
    n_frames = int(len(rec))

    meta = dict(
        dataset_version=DATASET_VERSION,
        schema_version=SCHEMA_VERSION,
        task=TASK_ID,
        instruction=str(instruction),
        episode_id=eid,
        source="collect.py (record: mj_step 真实接触力 / render: 纯运动学回放)",
        mujoco=mujoco.__version__,
        model=str(Path(R.SCENE_XML).relative_to(ROOT)),
        mjcf_sha256={"scene.xml": _sha16(ROOT / "model" / "scene.xml"),
                     "so_arm101.xml": _sha16(ROOT / "model" / "so_arm101.xml")},
        randomization={k: v for k, v in cfg.items() if not k.startswith("_")},
        plan=plan, total_steps=int(sum(plan.values())),
        fps=float(R.FPS), timestep=float(rec.a.m.opt.timestep),
        # 维度写进 meta：validate_trajectory 靠它核对 (N, nq/nv/nu) 形状
        nq=int(rec.a.m.nq), nv=int(rec.a.m.nv), nu=int(rec.a.m.nu),
        n_frames=n_frames,
        width=int(art["width"]) if art is not None else None,
        height=int(art["height"]) if art is not None else None,
        has_video=bool(art is not None and art.get("video_frames")),
        video_frames=art.get("video_frames") if art is not None else None,
        rendered=bool(art is not None and art.get("rendered")),
        start_pose=cfg.get("_start_pose", START_POSE),
        start_joints_deg=cfg.get("_start_joints_deg"),
        accepted=bool(accept), success=bool(accept), success_info=success_info,
        reject_reasons=reasons,
        metrics=metrics,
        cameras=art["cameras"] if art is not None else None,
        extrinsics=("cam_env_T_cw/cam_wrist_T_cw: world->camera, 行主序展平 (N*16,), reshape(N,4,4)"
                    if art is not None else None),
    )

    arrays["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    npz_path = SCRIPTS_DIR / f"{stem}.npz"
    _savez_atomic(npz_path, **arrays)

    json_path = SCRIPTS_DIR / f"{stem}.json"
    _write_json_atomic(json_path, meta)

    paths = dict(npz=str(npz_path.relative_to(ROOT)), json=str(json_path.relative_to(ROOT)),
                 video_env=None, video_wrist=None, frames_npz=None)
    if art is not None and art.get("video_env"):
        paths["video_env"] = str(Path(art["video_env"]).relative_to(ROOT))
        paths["video_wrist"] = str(Path(art["video_wrist"]).relative_to(ROOT))
    if keep_frames:
        paths["frames_npz"] = str(Path(keep_frames).relative_to(ROOT))
    return paths


def save_offpad_stub(cfg, metrics, reasons, instruction=DEFAULT_INSTRUCTION):
    """
    "出生位姿反复重采样仍在绿垫外"这种拒绝：没有轨迹可落，但也要留一份 json。

    为什么值得留：它证明"这条编号被试过了、为什么被丢"，以及当时的 footprint 实测值。
    没有它，编号上就是一个洞，事后既不能复现也不能统计（只会在 index 里剩一行）。
    产物标记 artifact="json_only"，npz / 视频都是 null。
    """
    eid = cfg["episode_id"]
    meta = dict(
        dataset_version=DATASET_VERSION, schema_version=SCHEMA_VERSION,
        task=TASK_ID, instruction=str(instruction),
        episode_id=eid, artifact="json_only",
        success=False, accepted=False, reject_reasons=reasons,
        randomization={k: v for k, v in cfg.items() if not k.startswith("_")},
        metrics=metrics,
        mjcf_sha256={"scene.xml": _sha16(ROOT / "model" / "scene.xml"),
                     "so_arm101.xml": _sha16(ROOT / "model" / "so_arm101.xml")},
        note="出生位姿重采样后仍无法把整个 footprint 放进绿垫：未开始录制，没有轨迹/视频。",
    )
    json_path = SCRIPTS_DIR / f"ep_{eid:06d}.json"
    _write_json_atomic(json_path, meta)
    return dict(npz=None, json=str(json_path.relative_to(ROOT)),
                video_env=None, video_wrist=None, frames_npz=None)


def _keys_block(cfg, metrics, success, has_video, width, height):
    """
    写进索引条目的"紧凑随机量"。作用：split.py / stats.py 只读 index.json 就够了，
    不用为了取一个 θ 去打开 1000 个 per-episode json。
    （per-episode json 仍是权威来源；这里只是索引里的投影。）
    """
    sp = cfg.get("spawn") or {}
    cam = cfg.get("camera") or {}
    lit = cfg.get("light") or {}
    sen = cfg.get("sensor") or {}
    mat = cfg.get("material") or {}
    pad = cfg.get("pad") or {}
    gp = cfg.get("grasp") or {}
    return dict(
        success=bool(success), has_video=bool(has_video),
        theta_deg=sp.get("theta_deg"), dr_mm=sp.get("dr_mm"),
        yaw_extra_deg=sp.get("yaw_extra_deg"),
        spawn_xy=[sp.get("x"), sp.get("y")],
        spawn_yaw_deg=sp.get("yaw_deg"),
        timing_scale=cfg.get("timing_scale"),
        light_azim_deg=lit.get("azim_deg"), light_tilt_deg=lit.get("tilt_deg"),
        light_diffuse=lit.get("diffuse"), material_gain=mat.get("gain"),
        env_pos_jitter_m=cam.get("env_pos_jitter_m"),
        env_fovy_delta_deg=cam.get("env_fovy_delta_deg"),
        env_yaw_delta_deg=cam.get("env_yaw_delta_deg"),
        env_pitch_delta_deg=cam.get("env_pitch_delta_deg"),
        sensor={k: sen.get(k) for k in ("blur_sigma", "noise_sigma", "gain_jitter",
                                        "motion_k", "motion_px_max", "vignette",
                                        "color_jitter", "flicker_tau_s")},
        grasp_kind=gp.get("kind"), grasp_attempts=metrics.get("grasp_attempts"),
        grasp_roll_deg=gp.get("roll_deg"), grasp_gz=gp.get("gz"),
        pad_footprint_mm=pad.get("footprint_max_mm"), pad_on=pad.get("on_pad"),
        pad_yaw_settled_deg=pad.get("yaw_settled_deg"),
        rise_mm=metrics.get("rise_mm"), hdisp_mm=metrics.get("hdisp_mm"),
        drop_mm=metrics.get("drop_mm"), sim_time_s=metrics.get("sim_time_s"),
        frames=metrics.get("frames"), moving_ratio=metrics.get("moving_ratio"),
        home_dq_max_deg=metrics.get("home_dq_max_deg"),
        width=None if width is None else int(width),
        height=None if height is None else int(height),
    )


# ============================================================
# 第 5 段  索引层：每条的片段 + 聚合视图
# ============================================================


def _frag_path(eid):
    return INDEX_DIR / f"ep_{int(eid):06d}.json"


def _is_frag(name):
    return name.startswith("ep_") and name.endswith(".json") and not name.endswith(".tmp")


def load_fragments():
    """
    读 scripts/index/*.json，返回 {episode_id: entry}。

    片段是权威清单：一条 episode 跑完就先落片段，聚合视图（index.json）只是它的投影。
    好处有两个：
      1) 任何时刻被杀进程，已完成的条目都在，不会出现"npz 还在、清单没了"的孤儿；
      2) 续跑/分片/补刷都不用做"读改写整份清单"这种有竞态的操作。
    """
    out = {}
    if not INDEX_DIR.exists():
        return out
    for p in sorted(INDEX_DIR.iterdir()):
        if not p.is_file() or not _is_frag(p.name):
            continue
        try:
            e = json.loads(p.read_text(encoding="utf-8"))
            out[int(e["episode_id"])] = e
        except Exception as exc:                              # noqa: BLE001
            print(f"  ⚠️ 片段损坏，已跳过: {p.name}（{type(exc).__name__}: {exc}）")
    return out


def write_fragment(entry):
    return _write_json_atomic(_frag_path(entry["episode_id"]), entry)


def migrate_legacy_index():
    """
    sdg-v1 的清单是"一份 index.json 装全部条目"。第一次跑 v1.1 时把它拆成片段，
    这样老数据不会因为改了结构就失联。返回迁移条数（0 表示没有老清单）。
    """
    legacy = SCRIPTS_DIR / "index.json"
    if not legacy.exists():
        return 0
    # 已经有片段了就别再迁一遍（否则每次运行都会把老清单里的条目"复活"一遍）
    if load_fragments():
        return 0
    try:
        old = json.loads(legacy.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return 0
    n = 0
    for key in ("accepted", "rejected"):
        for e in old.get(key) or []:
            if "episode_id" not in e:
                continue
            e = dict(e)
            e.setdefault("success", bool(e.get("accepted")))
            e.setdefault("schema_version", old.get("schema_version", 1))
            e.setdefault("dataset_version", "sdg-v1")
            e["migrated_from"] = "index.json(sdg-v1)"
            write_fragment(e)
            n += 1
    return n


def _policy_path():
    return INDEX_DIR / "_policy.json"


def policy_block(pad, theta_cap):
    """随机化策略快照。单独存一份，让 rebuild-index / split / stats 不必加载 MuJoCo 模型。"""
    return dict(
        dataset_version=DATASET_VERSION, schema_version=SCHEMA_VERSION,
        mjcf_sha256={"scene.xml": _sha16(ROOT / "model" / "scene.xml"),
                     "so_arm101.xml": _sha16(ROOT / "model" / "so_arm101.xml")},
        randomization_policy=dict(
            scene_rotation_deg=[-round(min(theta_cap, THETA_MAX_DEG), 1),
                                round(min(theta_cap, THETA_MAX_DEG), 1)],
            pad_constraint=dict(center=[float(v) for v in pad.center],
                                radius_mm=round(pad.radius * 1000, 1),
                                footprint_limit_mm=round(pad.limit_mm(), 1),
                                margin_mm=round(PAD_MARGIN * 1000, 1),
                                policy="落定后按 footprint 四角实测，超出即重采样；全批不允许出现垫外样本"),
            radial_jitter_mm=[DR_LO, DR_HI],
            object_yaw_jitter_deg=[-YAW_EXTRA_DEG, YAW_EXTRA_DEG],
            roll_jitter_deg=ROLL_JITTER_DEG, gz_jitter_mm=GZ_JITTER_MM,
            n_jitter_candidates=N_JITTER_CANDS,
            grasp_policy="扰动候选优先，纯同构解兜底（同构解 ≈ demo 同款抓取几何）",
            timing_scale=[0.90, 1.10],
            env_cam_pos_jitter_m=0.03, env_cam_fovy_jitter_deg=3.0,
            env_cam_yaw_jitter_deg=2.0, env_cam_pitch_jitter_deg=2.0,
            light=dict(azim_deg=[0, 360], tilt_deg=[5, 25], diffuse=[0.35, 0.70]),
            material_gain=[0.90, 1.10],
            blur_sigma=[0.6, 1.0], noise_sigma=[2.0, 4.0], gain_jitter=[0.02, 0.05],
            # 免费视觉随机化（2024-11）：和 render/test_render.SensorModel 同一套开关
            sensor=dict(motion_k=[0.45, 0.80], motion_px_max=[3.0, 4.5],
                        vignette=[0.22, 0.38], color_jitter=[0.02, 0.05],
                        flicker_tau_s=[1.2, 3.0], gain_jitter=[0.005, 0.02],
                        note="全后处理：虚影长度随像素速度，色彩抖动为每条视频基准+慢漂移"
                             "（不再逐帧重掷）；不改几何与相机内外参；同 seed 逐像素可复现"),
            fixed=dict(friction=1.2, grip_preset=0.0, grip_approach=1.5,
                       grip_release=1.2, grip_shut=0.2),
            note="摩擦系数不随机：μ=1.2 是 TPU 对纸盒的实测标定值，改了等于换了台机器人",
        ),
        start_pose=START_POSE,
        # 起止姿态的关节目标（度，ARM_JOINTS 顺序）。每条 episode 的实测首帧在
        # npz 的 meta_json.start_joints_deg 里（落定期有微小 PD 偏差）。
        start_joints_deg=[round(float(v), 3) for v in np.degrees(_start_q())],
        seed_policy="per_episode SeedSequence([base_seed, episode_id])"
                    "（续跑/分片与一次性跑完的样本逐位相同）",
        instruction=DEFAULT_INSTRUCTION,
    )


def load_policy():
    p = _policy_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return {}


def _load_splits():
    """把 split.py 的 manifest 摘要挂到 index.json，训练侧读一处就知道划分从哪来。"""
    m = SPLITS_DIR / "manifest.json"
    if not m.exists():
        return None
    try:
        d = json.loads(m.read_text(encoding="utf-8"))
    except Exception:                                         # noqa: BLE001
        return None
    return {k: d.get(k) for k in ("policy", "seed", "ratios", "n", "created", "files_sha256")}


def rebuild_index(entries, policy=None, write=True):
    """
    扫片段 → 生成聚合视图 scripts/index.json（原子写）。返回 (path, index, 统计)。

    ★这是唯一写 index.json 的地方。它不接收"内存里的列表"，而是接收片段字典 ——
      所以不会出现"续跑时从空列表开始、把前几批全冲掉"那种灾难。
    """
    policy = policy if policy is not None else load_policy()
    rows = [entries[k] for k in sorted(entries)]
    acc = [e for e in rows if e.get("accepted")]
    rej = [e for e in rows if not e.get("accepted")]
    n_success = sum(1 for e in rows if e.get("success", e.get("accepted")))
    total = len(rows)

    runs, seen = [], set()
    for e in rows:
        r = e.get("run")
        if not r:
            continue
        key = json.dumps({k: r.get(k) for k in ("base_seed", "started_at", "argv", "shard")},
                         sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            runs.append(r)
    runs.sort(key=lambda r: str(r.get("started_at") or ""))

    # 模型指纹：同一批数据必须来自同一个 mjcf。变了就是两批数据，不能混切/混训。
    # 策略快照里没有就现算（XML 只有几 KB，sha256 可忽略），保证 --rebuild-index 自足。
    mjcf = policy.get("mjcf_sha256") or {
        "scene.xml": _sha16(ROOT / "model" / "scene.xml"),
        "so_arm101.xml": _sha16(ROOT / "model" / "so_arm101.xml")}
    shas = {json.dumps(r.get("mjcf_sha256"), sort_keys=True) for r in runs
            if r.get("mjcf_sha256")}
    mixed_model = len(shas) > 1
    index = dict(
        dataset_version=DATASET_VERSION,
        schema_version=SCHEMA_VERSION,
        mjcf_sha256=mjcf,
        mjcf_mixed=bool(mixed_model),
        created=str(np.datetime64("now")),
        n_episodes=total,
        n_requested=total,                       # 向后兼容 v1 的字段名
        n_accepted=len(acc), n_rejected=len(rej), n_success=int(n_success),
        accept_rate=round(len(acc) / max(1, total), 3),
        success_rate=round(n_success / max(1, total), 3),
        runs=runs,
        randomization_policy=policy.get("randomization_policy"),
        seed_policy=policy.get("seed_policy"),
        # 起点姿态提到顶层：sdg-v3 之后"换起点 = 换一批数据"，下游看 index.json 第一眼
        # 就该知道这批是从哪个姿态起步的（每条 episode 的实测值在 metrics.start_pose /
        # npz 的 meta_json.start_joints_deg 里）。
        start_pose=policy.get("start_pose", START_POSE),
        # 老策略快照（sdg-v2 时代写的 _policy.json）没有这个字段，这里现算兜底，
        # 保证 index.json 里的起点姿态永远与代码一致。
        start_joints_deg=policy.get("start_joints_deg")
        or [round(float(v), 3) for v in np.degrees(_start_q())],
        splits=_load_splits(),
        accepted=acc,
        rejected=rej,
        note="本文件由片段 scripts/index/*.json 聚合生成，不要手改；"
             "重新聚合用 collect.py --rebuild-index",
    )
    if mixed_model:
        print(f"  ⚠️ 这批数据的 mjcf 指纹有 {len(shas)} 种：模型中途被改过，"
              "不同版本的样本不能混切/混训（split.py 会拦）")
    path = SCRIPTS_DIR / "index.json"
    if write:
        _write_json_atomic(path, index)
    return path, index


def _reset_index(talk=True):
    """
    --reset：清【索引层】+ 全部**由它派生的报告**，但不删任何 npz/视频。

    为什么连 splits/report/stats 一起删：它们描述的是"上一批数据"。重刷之后 id 会被
    复用而内容全变，留着这些文件就是**静默过期**——最坏情况是有人拿旧 split 训新数据。
    删掉之后 split.py/check_dataset.py/stats.py 会明确报"文件不存在"，逼你重新生成。
    """
    n = 0
    if INDEX_DIR.exists():
        for p in list(INDEX_DIR.iterdir()):
            if p.is_file() and (_is_frag(p.name) or p.name == "_policy.json"
                                or p.name.endswith(".tmp")):
                p.unlink()
                n += 1
    for name in ("index.json", "report.json", "stats.json", "stats.png", "DATA_CARD.md"):
        f = SCRIPTS_DIR / name
        if f.exists():
            f.unlink()
            n += 1
    for f in sorted(SPLITS_DIR.glob("*.json")) if SPLITS_DIR.exists() else []:
        f.unlink()
        n += 1
    if talk:
        print(f"  --reset  : 已清空索引层与派生报告共 {n} 个文件"
              "（产物 npz/视频未删，重刷同编号会覆盖；切分与体检需要重新生成）")
    return n


# ============================================================
# 第 6 段  主流程
# ============================================================


def _jobs_env(k):
    """给每个渲染子进程分配 llvmpipe 线程数：总线程别超过核数，否则互相抢。"""
    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    n_cpu = os.cpu_count() or 8
    env["LP_NUM_THREADS"] = str(max(2, n_cpu // max(1, k)))
    return env


def _spawn_workers(argv_wo_jobs, k):
    """
    把 --jobs K 摊成 K 个子进程（用已有的 --shard i/K 机制），等全部结束再聚合。

    为什么用子进程而不是线程/多进程池：EGL 上下文 + MuJoCo 模型在 fork 之后容易踩坑，
    而 --shard 这条路径本来就有（分片不写 index.json，产物互不覆盖）。
    """
    procs = []
    env = _jobs_env(k)
    for i in range(k):
        cmd = [sys.executable, str(Path(__file__).resolve())] + argv_wo_jobs + ["--shard", f"{i}/{k}"]
        print(f"  ▸ worker {i}/{k}: {' '.join(cmd[1:])}", flush=True)
        procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
    codes = [p.wait() for p in procs]
    return 0 if all(c == 0 for c in codes) else (2 if any(c not in (0, 2) for c in codes) else 2)


def _strip_jobs(argv):
    """从原始 argv 里摘掉 --jobs/--no-rebuild，其余原样传给子进程。"""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--jobs":
            skip = True
            continue
        if a.startswith("--jobs="):
            continue
        if a == "--no-rebuild":
            continue
        out.append(a)
    return out


def render_only(eids, width=640, height=480, burn=False, rebuild=True):
    """
    不跑物理，把【已存在】的 npz 按它自己记录的随机化参数重渲成 mp4。

    用途：采集时用了 --no-render（省时间），事后只想看某几条 → 走这条路补片。
    同源性：轨迹来自 npz（就是当初真物理跑出来的那份），这里只做纯运动学回放
    （mj_forward + 光栅化），一行 ctrl 都不写，符合"渲染与控制解耦"。
    """
    n_ok = 0
    for eid in eids:
        npz_p = SCRIPTS_DIR / f"ep_{eid:06d}.npz"
        js_p = SCRIPTS_DIR / f"ep_{eid:06d}.json"
        if not npz_p.exists() or not js_p.exists():
            print(f"  ! ep_{eid:06d} 缺 npz/json，跳过（先跑一次采集）")
            continue
        meta = json.loads(js_p.read_text(encoding="utf-8"))
        rand = meta.get("randomization") or {}
        sen = dict(rand.get("sensor") or {})
        sen.setdefault("blur_sigma", 0.8)
        sen.setdefault("noise_sigma", 3.0)
        sen.setdefault("gain_jitter", 0.01)
        cfg = dict(seed=int(rand.get("seed") or 0),
                   camera=dict(rand.get("camera") or {}),
                   sensor=sen,
                   light=dict(rand.get("light") or {}),
                   material=dict(rand.get("material") or {}))
        with np.load(npz_p, allow_pickle=False) as z:
            ep = {k: np.array(z[k]) for k in
                  ("qpos", "tcp_pos", "time", "phase") if k in z.files}
        out_env = VIDEOS_DIR / f"ep_{eid:06d}_env.mp4"
        out_wrist = VIDEOS_DIR / f"ep_{eid:06d}_wrist.mp4"
        art = render_episode(ep, cfg, out_env, out_wrist,
                             width=width, height=height, burn_counter=burn)
        # 回填 meta：既然有视频了，json 里就别再写 has_video=false 自欺
        meta.update(has_video=True, rendered=True,
                    video_frames=art["video_frames"],
                    width=art["width"], height=art["height"],
                    cameras=art["cameras"],
                    dataset_version=DATASET_VERSION,      # 容器换代 = 另一代数据
                    video_format="mp4/h264",
                    rerendered_at=str(np.datetime64("now")),
                    rerender_note="--render-only：同一份 npz 轨迹重渲为 mp4（未重跑物理）")
        _write_json_atomic(js_p, meta)

        frag_p = INDEX_DIR / f"ep_{eid:06d}.json"
        frag = (json.loads(frag_p.read_text(encoding="utf-8"))
                if frag_p.exists()
                else dict(episode_id=eid, accepted=bool(meta.get("accepted")),
                          success=bool(meta.get("success"))))
        paths = frag.setdefault("paths", {})
        paths["video_env"] = str(out_env.relative_to(ROOT))
        paths["video_wrist"] = str(out_wrist.relative_to(ROOT))
        keys = frag.setdefault("keys", {})
        keys.update(has_video=True, width=int(art["width"]), height=int(art["height"]))
        _write_json_atomic(frag_p, frag)
        print(f"  ✅ ep_{eid:06d} 重渲 {art['video_frames']} 帧 → "
              f"{out_env.relative_to(ROOT)} / {out_wrist.name}")
        n_ok += 1

    if n_ok and rebuild:
        rebuild_index(load_fragments())
    return 0 if n_ok else 2


def main(argv=None):
    # 行缓冲：mj_step 里偶发 SIGFPE 是"没有任何输出就死"，重定向到文件时
    # 默认全缓冲会把已经跑完的进度一起吞掉，崩了都看不出崩在第几条。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:                                         # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="SDG 批量数据生成（随机摆放 + 真物理 + 渲染）")
    ap.add_argument("--n", type=int, default=5, help="生成多少条（默认 5）")
    ap.add_argument("--start", type=int, default=0, help="起始编号（默认 0，接着刷就填上次的 n）")
    ap.add_argument("--no-render", action="store_true",
                    help="只刷物理+动作脚本（秒级），不渲染视频")
    ap.add_argument("--quick", action="store_true", help="320x240 快出片")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--keep-frames", action="store_true",
                    help="额外存图片数组 scripts/ep_xxxxxx_frames.npz（约 800MB/条，注意磁盘）")
    ap.add_argument("--seed", type=int, default=None, help="全局随机种子（默认按时间）")
    ap.add_argument("--max-reject", type=int, default=None,
                    help="本轮最多允许丢多少条就停（默认不设限，只在最后汇报）")
    ap.add_argument("--keep-failed", dest="keep_failed", action="store_true", default=True,
                    help="负样本也落 npz+json（默认开；只落轨迹不渲染视频）")
    ap.add_argument("--no-keep-failed", dest="keep_failed", action="store_false",
                    help="不落负样本（只留索引里的一行，省盘）")
    ap.add_argument("--render-failed", action="store_true",
                    help="连负样本也渲染视频（每条 +100s，默认关）")
    ap.add_argument("--extrinsics", dest="extrinsics", action="store_true", default=True,
                    help="无渲染时也算逐帧相机外参（默认开，约 +0.3s/条）")
    ap.add_argument("--no-extrinsics", dest="extrinsics", action="store_false")
    ap.add_argument("--burn-counter", action="store_true",
                    help="视频左上角烧 'Env F:n' 帧号（调试用；训练数据不要开）")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                    help="写进每条 meta 的自然语言指令（给 VLA 用；纯视觉可以无视）")
    ap.add_argument("--reset", action="store_true",
                    help="先清空索引层（片段 + index.json）再继续生成 --n 条；"
                         "注意它【不会】单独退出，只清索引不删产物")
    ap.add_argument("--rebuild-index", action="store_true",
                    help="只从片段重建 index.json 后退出（分片跑完收尾用）")
    ap.add_argument("--shard", default=None, metavar="I/K",
                    help="只跑 eid%%K==I 的条目（配合外部并行；分片进程不写 index.json）")
    ap.add_argument("--render-only", action="store_true",
                    help="不跑物理：把【已存在】的 npz 按原随机化参数重渲成 mp4 后退出"
                         "（配合 --episodes；排查/补片用，纯运动学回放不 mj_step）")
    ap.add_argument("--episodes", default=None,
                    help="配合 --render-only：要重渲的编号，如 0,3,7")
    ap.add_argument("--jobs", type=int, default=1,
                    help="并行进程数（默认 1）。渲染是软件 GL(llvmpipe)，实测 8 进程约 3x；"
                         "内部按 --shard 切分给子进程，结束后统一聚合 index.json")
    ap.add_argument("--no-rebuild", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.shard:
        try:
            sh_i, sh_k = (int(v) for v in str(args.shard).split("/"))
        except ValueError:
            raise SystemExit("--shard 要写成 I/K，例如 --shard 1/4")
        if sh_k <= 0 or not (0 <= sh_i < sh_k):
            raise SystemExit(f"--shard {args.shard} 不合法：要求 0 <= I < K 且 K > 0")
        args.shard = (sh_i, sh_k)

    width = args.width or (320 if args.quick else 640)
    height = args.height or (240 if args.quick else 480)
    base_seed = args.seed if args.seed is not None else int(time.time()) % (2 ** 31 - 1)

    LOGS_DIR.mkdir(exist_ok=True)
    SCRIPTS_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)
    VIDEOS_DIR.mkdir(exist_ok=True)

    if args.reset:
        _reset_index()

    # ---- 只重建清单就退出（分片跑完的收尾动作，无需加载模型）----
    if args.rebuild_index:
        entries = load_fragments()
        n_mig = migrate_legacy_index()
        if n_mig and not entries:
            entries = load_fragments()
        path, index = rebuild_index(entries)
        print(f"  片段 {len(entries)} 条 → {path.relative_to(ROOT)}"
              f"（收 {index['n_accepted']} / 丢 {index['n_rejected']}"
              f"，接受率 {index['accept_rate']*100:.0f}%"
              + (f"，迁移老清单 {n_mig} 条" if n_mig else "") + "）")
        return 0

    # ---- 只重渲：不跑物理，把已存在的 npz 重新出成 mp4（补片 / 排查用）----
    if args.render_only:
        if not args.episodes:
            raise SystemExit("--render-only 必须配 --episodes（如 --episodes 0,3,7）")
        eids = [int(x) for x in str(args.episodes).replace(" ", "").split(",") if x != ""]
        if args.jobs and args.jobs > 1 and len(eids) > 1:
            # 渲染是软件 GL，并行才有救：按子集切给 K 个 worker，最后统一聚合清单
            k = min(int(args.jobs), len(eids))
            base = _strip_jobs(argv if argv is not None else sys.argv[1:])
            env = _jobs_env(k)
            procs = []
            for i in range(k):
                sub = [str(e) for e in eids[i::k]]
                cmd = ([sys.executable, str(Path(__file__).resolve()), "--render-only",
                        "--episodes", ",".join(sub), "--width", str(width),
                        "--height", str(height), "--no-rebuild"]
                       + (["--burn-counter"] if args.burn_counter else []))
                print(f"  ▸ worker {i}/{k}: episodes {'/'.join(sub)}", flush=True)
                procs.append(subprocess.Popen(cmd, cwd=str(ROOT), env=env))
            codes = [p_.wait() for p_ in procs]
            rebuild_index(load_fragments())
            ok = sum(1 for c in codes if c == 0)
            print(f"  重渲完毕: {ok}/{k} 个 worker 正常退出（清单已聚合）")
            return 0 if ok == k else 2
        return render_only(eids, width=width, height=height,
                           burn=args.burn_counter, rebuild=not args.no_rebuild)

    # ---- 并行生成：摊成 K 个 --shard 子进程，跑完统一聚合 index.json ----
    if args.jobs and args.jobs > 1 and not args.shard and not args.rebuild_index:
        base = _strip_jobs(argv if argv is not None else sys.argv[1:])
        print(f"  并行      : {args.jobs} 进程（每个再按 --shard 只跑 1/{args.jobs} 的编号；"
              f"LLVM 线程 {_jobs_env(args.jobs)['LP_NUM_THREADS']}）")
        rc = _spawn_workers(base, int(args.jobs))
        entries = load_fragments()
        path, index = rebuild_index(entries)
        print()
        print(f"  数据集: {index['n_episodes']} 条（收 {index['n_accepted']} / 丢 "
              f"{index['n_rejected']}，接受率 {index['accept_rate']*100:.0f}%）")
        return rc

    arm = R.Arm(R.SCENE_XML)          # 模型只加载一次，每条 episode 复用（省 0.2s/条）
    nom = NominalGrasp(arm)           # 标定基准：三个接近路点 + pan 符号（只算一次）
    pad = PadZone(arm.m)              # 绿垫几何 + "整个 footprint 在垫上"的判据
    _PAN_CENTER.value = np.asarray(nom.center, float)   # 采样与同构解共用同一个旋转圆心
    theta_cap = pad.theta_limit_deg(OBJECT_NOMINAL, nom.center)
    policy = policy_block(pad, theta_cap)
    _write_json_atomic(_policy_path(), policy)

    print("=" * 78)
    print(" collect  随机刷数据（SDG）")
    print("=" * 78)
    print(f"  绿垫约束  : {pad.describe()}")
    print(f"              场景旋转上限（由垫子几何反算）= ±{theta_cap:.1f}°"
          f"（代码里再取 min(±{THETA_MAX_DEG:.0f}°)）")
    print(f"  标定基准  : pan_sign={nom.pan_sign:+d}  "
          f"工具轴=({nom.tool_axis[0]:+.3f},{nom.tool_axis[1]:+.3f},{nom.tool_axis[2]:+.3f})  "
          f"IK 残差 {max(nom.e1, nom.e2, nom.e3)*1000:.2f}mm")

    # ---- 续跑：从片段里恢复已有条目（这是"不冲掉历史"的关键）----
    entries = load_fragments()
    n_mig = migrate_legacy_index()
    if n_mig:
        entries = load_fragments()
        print(f"  老清单    : 已把 sdg-v1 的 index.json 迁成 {n_mig} 个片段")
    if entries:
        n_acc_old = sum(1 for e in entries.values() if e.get("accepted"))
        print(f"  已有数据  : {len(entries)} 条（收 {n_acc_old} / 丢 "
              f"{len(entries)-n_acc_old}），本轮按编号追加/覆盖")

    t0 = time.time()
    n_acc_run = n_rej_run = 0
    RUN = dict(base_seed=int(base_seed), started_at=str(np.datetime64("now")),
               argv=[str(a) for a in (argv if argv is not None else sys.argv[1:])],
               shard=None if not args.shard else list(args.shard),
               pid=os.getpid(), width=int(width), height=int(height),
               no_render=bool(args.no_render), keep_failed=bool(args.keep_failed),
               instruction=str(args.instruction),
               mjcf_sha256=policy.get("mjcf_sha256"))

    print(f"  条数      : {args.n}（编号 {args.start:06d}~{args.start + args.n - 1:06d}）"
          + (f"  分片 {args.shard[0]}/{args.shard[1]}（只跑 eid%{args.shard[1]}=={args.shard[0]}）"
             if args.shard else ""))
    print(f"  全局种子  : {base_seed}（每条再派生 SeedSequence([{base_seed}, eid])）")
    print(f"  输出      : scripts/ep_*.npz|json    videos/ep_*_env|wrist.mp4")
    print(f"  渲染      : {'关（只刷物理）' if args.no_render else f'{width}x{height} @ {R.FPS:.0f}fps'}"
          f"   负样本: {'落轨迹' if args.keep_failed else '不落'}"
          f"{'（含视频）' if args.render_failed else '（不渲染）'}")
    print(f"  随机范围  : 场景旋转 ±{min(theta_cap, THETA_MAX_DEG):.0f}°（垫内）/ 径向 "
          f"{DR_LO:.0f}~{DR_HI:.0f}mm / 偏航 ±{YAW_EXTRA_DEG:.1f}° / "
          f"节奏 ×0.90~1.10 / 相机 ±3cm / 噪声 2~4DN")
    print(f"  抓取搜索  : 扰动候选 ≤{N_JITTER_CANDS} 次（roll±{ROLL_JITTER_DEG:.0f}° / "
          f"gz±{GZ_JITTER_MM:.1f}mm）+ 纯同构解兜底，逐条真跑物理取第一个成功")
    print()

    for k in range(args.n):
        eid = args.start + k
        if args.shard and (eid % args.shard[1]) != args.shard[0]:
            continue
        # ★每条一个独立 rng（只由 base_seed 与 eid 决定）：
        #   这样 --n 25 与 --n 20 后再 --start 20 --n 5 抽到的随机量逐位相同。
        rng_ep = np.random.default_rng([int(base_seed), int(eid)])
        cfg = sample_episode(rng_ep, eid, pad, theta_cap)
        t_ep = time.time()

        # ---- 绿垫硬约束：真落定一次，确认整个 footprint 在垫上，不行就换位姿 ----
        pad_ok, cfg = ensure_on_pad(arm, cfg, pad, rng_ep, theta_cap)
        if not pad_ok:
            reasons = ["spawn_off_pad"]
            metrics = dict(pad=cfg.get("pad"))
            paths = save_offpad_stub(cfg, metrics, reasons, args.instruction)
            entries[eid] = dict(episode_id=eid, accepted=False, success=False,
                                reasons=reasons, metrics=metrics, paths=paths,
                                steps=0, sim_s=0.0, schema_version=SCHEMA_VERSION,
                                dataset_version=DATASET_VERSION,
                                keys=_keys_block(cfg, metrics, False, False, width, height),
                                run=RUN)
            write_fragment(entries[eid])
            if not args.shard:
                rebuild_index(entries, policy)
            n_rej_run += 1
            print(f"  [{k+1:>3}/{args.n}] ❌ ep_{eid:06d} 出生位姿{cfg['pad']['retries']}次重试"
                  f"仍在垫外（{cfg['pad']['footprint_max_mm']}mm），跳过（只落 json）")
            continue

        rec, info, plan, grasp, n_att = search_grasp(arm, cfg, nom)
        if rec is None:
            # 连同构解都没过验收（正常不该发生，说明模型/参数被改坏了）
            wp_fb = waypoints_for(arm, nom, cfg)
            rec, info, plan = record_episode(arm, cfg, nom, wp_fb,
                                             float(np.degrees(wp_fb["q_gsp"][4])))
            ok, reasons, metrics = judge(rec, info, cfg)
            reasons = [f"no_grasp_candidate({n_att}次)"] + reasons
            cfg["grasp"] = dict(kind="none", roll_deg=round(float(np.degrees(nom.q_gsp[4])), 2),
                                gz=0.012, attempt=n_att)
        else:
            cfg["grasp"] = grasp
            ok, reasons, metrics = judge(rec, info, cfg)
        metrics["grasp_attempts"] = int(n_att)
        metrics["grasp_kind"] = cfg["grasp"]["kind"]

        # ---- 渲染（正样本默认渲染；负样本要 --render-failed 才渲）----
        art = None
        frames_path = None
        if not args.no_render and (ok or args.render_failed):
            ep = {k2: np.array(v) for k2, v in rec._rows.items()}
            video_env = VIDEOS_DIR / f"ep_{eid:06d}_env.mp4"
            video_wrist = VIDEOS_DIR / f"ep_{eid:06d}_wrist.mp4"
            if args.keep_frames:
                frames_path = SCRIPTS_DIR / f"ep_{eid:06d}_frames.npz"
            art = render_episode(ep, cfg, video_env, video_wrist, width=width, height=height,
                                 keep_frames=frames_path, burn_counter=args.burn_counter)
            art.update(video_env=video_env, video_wrist=video_wrist)
        elif args.extrinsics:
            # 没视频也要外参：负样本/--no-render 的数据照样能投到 3D
            ep = {k2: np.array(v) for k2, v in rec._rows.items()}
            art = extrinsics_only(ep, cfg, width, height)

        do_save = ok or args.keep_failed
        if do_save:
            paths = save_episode(rec, cfg, metrics, plan, art, ok, reasons,
                                 instruction=args.instruction, keep_frames=frames_path)
        else:
            paths = dict(npz=None, json=None, video_env=None, video_wrist=None, frames_npz=None)
        has_video = bool(art is not None and art.get("video_frames"))

        entries[eid] = dict(
            episode_id=eid, accepted=bool(ok), success=bool(ok), reasons=reasons,
            metrics=metrics, paths=paths, schema_version=SCHEMA_VERSION,
            dataset_version=DATASET_VERSION,
            keys=_keys_block(cfg, metrics, ok, has_video, width, height),
            steps=int(sum(plan.values())),
            sim_s=round(float(sum(plan.values()) * arm.m.opt.timestep), 3),
            run=RUN)
        write_fragment(entries[eid])
        if not args.shard:
            rebuild_index(entries, policy)

        if ok:
            n_acc_run += 1
        else:
            n_rej_run += 1
        flag = "✅" if ok else "❌"
        why = "" if ok else "  <- " + ",".join(reasons)
        extra = "" if ok or do_save else "  （未落盘：--no-keep-failed）"
        print(f"  [{k+1:>3}/{args.n}] {flag} ep_{eid:06d}  "
              f"抬升 {metrics['rise_mm']:+7.2f}mm  水平 {metrics['hdisp_mm']:6.1f}mm  "
              f"{entries[eid]['steps']}步/{entries[eid]['sim_s']:.2f}s  "
              f"试{n_att}次 {time.time()-t_ep:.1f}s{why}{extra}")

        if args.max_reject is not None and n_rej_run > args.max_reject:
            print(f"\n  ⚠️ 本轮已丢弃 {n_rej_run} 条超过 --max-reject，提前停止")
            break

    # ---- 收尾：聚合视图（分片模式不写，避免多进程抢同一个文件）----
    dt = time.time() - t0
    path, index = rebuild_index(entries, policy, write=not args.shard)
    print()
    print(f"  本轮  : 收 {n_acc_run} / 丢 {n_rej_run}   耗时 {dt:.1f}s")
    print(f"  数据集: {index['n_episodes']} 条（收 {index['n_accepted']} / 丢 "
          f"{index['n_rejected']}，接受率 {index['accept_rate']*100:.0f}%）")
    if args.shard:
        print("  ⚠️ 分片模式未写 index.json：全部跑完后执行 "
              "`collect.py --rebuild-index` 统一聚合")
    else:
        print(f"  清单  : {path.relative_to(ROOT)}（权威来源是 scripts/index/*.json）")
    print(f"  动作脚本: scripts/ep_*.npz + .json   视频: videos/ep_*_env|wrist.mp4")
    if n_rej_run:
        from collections import Counter
        cnt = Counter(r for e in entries.values() if not e.get("accepted")
                      for r in (e.get("reasons") or []))
        print("  丢弃原因分布（全数据集）: "
              + ", ".join(f"{k}×{v}" for k, v in cnt.most_common()))
    return 0 if n_acc_run else 2


if __name__ == "__main__":
    sys.exit(main())
