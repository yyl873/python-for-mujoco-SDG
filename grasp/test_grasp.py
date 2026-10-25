"""
★★★ 注意：本脚本已【不是】抓取判据（第八轮实测：抬升 +4.51 mm、握持接触 0 个 = 抓空）。
    它的动作序列与生产路径分叉（缺 settle_a、下探段压盒 21.85 mm），保留仅作历史对照。
    要看"现在到底抓不抓得起来"，用这两个：
        tools/probe_contact_audit.py    六项闸门 G1~G6，0.9 s/条（下探零接触/压盒/倾角/抬升/缝隙/穿模）
        render/test_render.py           录制 + 渲染 + 15 项自检 + success 标签

test_grasp.py — 纯物理抓取 + 搬运完整闭环 (药盒保持原始直立)
==========================================================
不覆盖 qpos 假装抓取。所有判定来自 mj_step 与 mj_contactForce。

================================================================
一、模型事实 (实测得出, 不是猜的)
================================================================
1) 夹爪是带铰链的平行夹爪:
       gripper_joint = 2.232 (127.9°) -> 张开最大
       gripper_joint = 0.00 -> 闭合 (留微小间隙)
   两指相对而立, 中间形成夹口。

2) wrist_roll 关节限位 ±2.3 rad (约 ±131.8°), 超出会被 clip,
   所以 135°/180°/270° 这些角度是无效的 (都会落到 131.8°)。

3) 夹口张开方向在世界系里不天然对准药盒 15mm 厚度向(y),
   必须靠腕部滚转把它掰正。扫描确认 105° 最好:
       75°  -> 只夹到 jaw 单侧 (grip 0 接触), 抬升 112~116mm 但边缘易滑
       82~105° -> jaw 4 + grip 2 双侧接触, 抬升 107~118mm, 稳定
       105° -> 最优 (抬升 117.7mm, 合爪 Fn 13.8N)

4) 从 home 姿态直接插值到目标会让手臂"扫倒"药盒。
   home 关键帧的 ctrl 默认为 0, 落定后手臂塌到 q≈0, TCP 在 (0.39, 0, 0.22)。
   直接平移过去会横穿药盒。必须先"高位撤回"(z=0.24) 再移到正上方。

================================================================
二、路径 (对应你提的两条约束)
================================================================
约束1 去抓取途中夹爪不能碰药盒
    home -> 高位撤回(z=0.24) -> 药盒正上方(z=0.16) -> 竖直下探
    实测药盒 z 全程保持 0.0278, 直到被抓走。

约束2 搬运途中夹住的药盒不能碰木盒(target_box)
    先竖直抬升到 z≈0.148, 再高空水平移动到木盒上方(z=0.16)。
    木盒上沿仅 0.05, 药盒全程在 0.15 以上, 零接触。

================================================================
三、运行
================================================================
    python test_grasp.py                  # 完整流程 (含搬运)
    python test_grasp.py --no-transport   # 只抓不搬
    python test_grasp.py --roll 90 --gz 0.032
"""

import argparse
from pathlib import Path

import numpy as np
import mujoco

# 路径锚定到"仓库根目录/"：本文件在 <root>/grasp/test_grasp.py，
# parents[1] 即 <root>。这样从任意工作目录运行都能找到 model/scene.xml。
ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = str(ROOT / "model" / "scene.xml")

ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_flex_joint",
              "wrist_flex_joint", "wrist_roll_joint"]

# ---------------------------------------------------------------------------
# 真机遥操的【初始姿态】（golden pose）—— 用户提供的遥操起点，度
#   pan -36° / lift +4° / elbow +14° / wrist_flex +55° / wrist_roll -104°
#
# 为什么要有它：训练数据的第 0 帧姿态必须和真机录制时一致。旧预设（keyframe home
# + ctrl=0）会让手臂塌到 q≈0、TCP 到 (0.39, 0, 0.22) —— 真机待机位不是那样。
#
# ★ 不能直接从 golden pose 做关节空间插值去抓：手指就在药盒旁边（jaw 34mm /
#   gripper 49mm，且与药盒同高），关节插值会把手指横着扫过药盒、把它推倒。
#   干净的路是【先笛卡尔竖直抬离台面 -> 高位 -> 再接原来的接近段】
#   （见 render/test_render.run_episode 的 init_hold / init_lift 两段）。
# ---------------------------------------------------------------------------
GOLDEN_JOINTS = {
    "shoulder_pan_joint": -36.0,
    "shoulder_lift_joint": 4.0,
    "elbow_flex_joint": 14.0,
    "wrist_flex_joint": 55.0,
    "wrist_roll_joint": -104.0,
}
# 按 ARM_JOINTS 顺序排好再转弧度：顺序写错就是另一个姿态，所以只在这里排一次
GOLDEN_Q_DEG = np.array([GOLDEN_JOINTS[n] for n in ARM_JOINTS], float)
GOLDEN_Q = np.radians(GOLDEN_Q_DEG)

# ---------------------------------------------------------------------------
# old_ros 的【录制起点 / 收尾姿态】—— 真机 ROS 录制脚本的初始预设
#   old_ros/send_trajectory.py:369（开始录制前）与 :552（结束录制后）用的是同一个
#   initial_pose = [0.0, -1.74535, 1.54036, 1.23277, -1.80689, 0.0]   # 弧度 + 夹爪
#   换成分度：pan 0° / lift −100° / elbow 88.26° / wrist_flex 70.63° /
#             wrist_roll −103.53° / 夹爪 0（收起）
# 这是真机"折叠待机"位（lift 顶下限、elbow 顶上限），与 golden pose（遥操第 0 帧、
# 手指贴在药盒旁边）是两个完全不同的位置：这里的 TCP 离药盒 151 mm。
#
# 可行性（2024-11 第五轮实测，见 docs/ROUND_oldros_start.md）：
#   · 姿态本身合法：只有 lift 超限 2.07e-05 rad、elbow 超限 3.60e-04 rad（都是标定值
#     的舍入，≈0.02°），写 qpos 时统一 np.clip 到限位内；FK 得 TCP=(168.1, 7.6, 21.6)mm，
#     最低臂部 geom z=62.4mm，零接触 —— 起点比 golden 更安全。
#   · 只换起点、不重扫自由段步数 = 抓取全灭（合爪把药盒横甩 37mm 并躺平）。
#     必须同时把 (init_lift, approach_retract, approach_above) 改成 (700, 800, 500)，
#     实测 timing_scale 0.90~1.10 十档全过（抬升 103.4~103.9mm / 水平 221.0mm），
#     比 golden 起点（0.95/1.05/1.10 全灭）更稳。
#   · old_ros 的 safe_z=0.20【不能照搬】：抬升高度改 0.20/0.22 实测抓取全灭，
#     RETRACT_Z 必须保持 0.24。
#   · 起手爪甲：old_ros 是 set_gripper(0.8) 先张开再动；本片按用户要求保持
#     GRIP_PRESET 收起态（张开只发生在 approach 段）—— 这条偏离是有意的，别"对齐"回去。
# ---------------------------------------------------------------------------
OLDROS_JOINTS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -100.0,
    "elbow_flex_joint": 88.26,
    "wrist_flex_joint": 70.63,
    "wrist_roll_joint": -103.53,
}
OLDROS_Q_DEG = np.array([OLDROS_JOINTS[n] for n in ARM_JOINTS], float)
OLDROS_Q_RAW = np.radians(OLDROS_Q_DEG)   # 原样（含 0.02° 级超限），写 qpos 前 clip

RETRACT_Z = 0.24     # 高位撤回高度

# ★ 抓取点的横向补偿（2024-11 第七轮）：dy=+17.5 mm。
#   这个 5 自由度手的两个指面在 jaw=0.28 时相距 14.15 mm，而手在 y 向上偏了约 17 mm ——
#   不补偿的话"固定指的实体"会压在药盒体积里（实测夹稳时固定指 9814 个顶点在盒内、
#   深度 7.5 mm = 用户看到的穿模），同时活动指离盒面 4.8 mm（= 用户看到的缝隙）。
#   dy=+17.5 mm 之后：两指面都贴住盒（缝 −0.85/−1.07 mm）、穿模深度 0.02 mm。
#   ⚠️ 改这个数必须同时重算指垫局部坐标（debug/pad_flush_try.py）并重扫 GRIP_CLOSED。
GRASP_XY_OFFSET = (0.0, 0.0175)

# 夹爪全开角度：真机标定 1580~3035 counts = 127.9° = 2.232 rad
# (原模型是 1.70 rad / 97.4°，比真机少 24%，两指张不到 153mm)
GRIP_MAX = 2.232
# 闭合目标不用 0.00：实测合到 0 会持续挤压把药盒挤出去（静止 1s 滑 116mm 直接掉）。
# 0.20 约 11.5°，指尖刚好贴住纸盒，静止滑移 5.6mm 且能搬完全程。
# 下探时的张开量（不是全开 2.232）。第七轮实测：全开 2.232 下探时活动指本体就会扫到药盒
# （43 个碰撞步、药盒被压 21.85 mm），1.5 下探全程零接触。步骤二/三共用这一个值。
GRIP_APPROACH = 1.5
GRIP_CLOSED = 0.28   # 第七轮：与 render 的 GRIP_SHUT 同步（指面间距 14.15mm ≈ 盒厚 15mm）
APPROACH_Z = 0.16    # 药盒正上方
TRANSPORT_Z = 0.16   # 搬运高度

DEF_ROLL = -90.0     # 腕部滚转 (度), 2024-10 按真机尺寸重新标定后的最优
# 抓取时 TCP 高度。2024-11 第二轮做过一次 0.012 -> 0.014 的尝试，**结论：回退到 0.012**。
# 实测扫描（同一条动作脚本，只改 gz，指标取"压盒/回弹/双侧接触"）：
#   0.004/0.008/0.010 -> 抬升 0 mm（抓不住，指垫够不到窄端面）
#   0.012 -> 压盒 6.30 回弹 1.72 抬升 103.47  Fn_jaw 1.30   ★ 保持这个
#   0.014 -> 压盒 6.07 回弹 1.57 抬升 102.43  Fn_jaw 1.83   ✗ 被下面这条否决
#   0.016 -> 压盒 5.69 回弹 1.47 抬升 101.83  Fn_jaw 1.58
#   0.020 及以上 -> 抬升 4.5 mm（抓不住）
# 为什么否决 0.014：步骤二（本文件的 main）**没有 settle_a 停顿**，合爪后立刻抬升。
#   0.014 时合爪末帧 jaw 侧还没吃上力（双侧靠 settle_a 0.8s 才平衡），于是：
#     0.012 -> 末端握持接触 8 个(jaw4/grip4)，约束2 ✅
#     0.014 -> 末端握持接触 0 个，约束2 ❌（搬运途中滑落，药盒 z 掉到 0.0405）
#   => 参数不能只对"带停顿的步骤三"好，必须对没有停顿的步骤二也一样好。
# ★ 上面这组数是【指壳当碰撞体】时代（sdg-v3 及更早）。2024-11 第六轮已按这条提示修掉：
#   两指各贴一枚 34×2×34 mm 平面指垫 + 碰撞位掩码（指壳不再与药盒碰撞）后，
#   实测 压盒 0.22 mm、回弹 0.22 mm、下探段指↔盒接触 0/400 步、抬升 +125.10 mm。
#   见 docs/ROUND_contact_realism.md；本文件下方的 gz 窗口（0.008~0.016）在指垫方案下仍然成立
DEF_GZ = 0.012       # 抓取时 TCP 高度 (指垫比 TCP 高约 22mm, 必须压低)

# 支撑面: 与药盒本就常接触, 不计入"碰撞"
SUPPORT_BODIES = {"pick_zone", "table", "mousepad", "floor", "world"}


class Arm:
    """封装模型的索引与基本操作, 避免全局变量。"""

    def __init__(self, xml=SCENE_XML):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        m = self.m
        nid = lambda objtype, n: mujoco.mj_name2id(m, objtype, n)

        self.adr = np.array([m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, n)]
                             for n in ARM_JOINTS], dtype=int)
        self.ctrl = np.array([nid(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS],
                             dtype=int)
        self.cg = nid(mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint")
        self.tcp = nid(mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.obj = nid(mujoco.mjtObj.mjOBJ_BODY, "object")
        self.grip = nid(mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
        self.jaw = nid(mujoco.mjtObj.mjOBJ_BODY, "jaw_link")
        self.tbox = nid(mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self.obj_geom = nid(mujoco.mjtObj.mjOBJ_GEOM, "object_physics")
        self.oadr = m.jnt_qposadr[m.body_jntadr[self.obj]]
        # 夹爪关节的 qpos 地址（reset 时要按 keyframe 的值保持，不能写死 0）
        self.gadr = int(m.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")])

        self.lo = np.array([m.jnt_range[nid(mujoco.mjtObj.mjOBJ_JOINT, n)][0]
                            for n in ARM_JOINTS])
        self.hi = np.array([m.jnt_range[nid(mujoco.mjtObj.mjOBJ_JOINT, n)][1]
                            for n in ARM_JOINTS])
        self._J = np.zeros((3, m.nv))

        self.name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        self.reset()

    # ---------- 状态 ----------
    def apply_start_pose(self, pose="home"):
        """
        把手臂摆到指定【起始姿态】（只动手臂 + 夹爪，不动药盒），返回该姿态的关节向量。

          pose="home"   : 沿用 keyframe（ctrl 全 0 -> 落定期手臂塌到 q≈0）。旧行为。
          pose="golden" : 真机遥操起点 GOLDEN_Q，夹爪闭合；ctrl 也写这个姿态，
                          落定期手臂【保持】它，不会塌下去。
          pose="oldros" : old_ros 录制脚本的 initial_pose（折叠待机位，见 OLDROS_JOINTS），
                          默认起点。与 golden 同一套写法，只是关节值不同；两处
                          0.02° 级超限在这里 np.clip 到 jnt_range 内。
            注意 "home" 返回的是【keyframe 里的 qpos】，而真正落定后手臂会塌到 q≈0
            （因为 ctrl=0），所以 home 模式下 run_episode 的 q_home 要在落定后才取。
        """
        d = self.d
        name = str(pose).lower()
        if name in ("golden", "oldros"):
            q = GOLDEN_Q if name == "golden" else np.clip(OLDROS_Q_RAW, self.lo, self.hi)
            d.qpos[self.adr] = q
            d.qpos[self.gadr] = GRIP_CLOSED
            d.ctrl[:] = 0.0
            d.ctrl[self.ctrl] = q          # 保持住，别让 PD 把它拉回 0
            d.ctrl[self.cg] = GRIP_CLOSED
            return np.array(q, float)
        return np.array(d.qpos[self.adr], float)

    def reset(self, settle=1200, pose="home"):
        """
        复位到 keyframe 并让药盒自然落定。
        pose="home"（默认）：ctrl 全 0 -> 手臂塌到 q≈0，这是历史行为，步骤二用它。
        pose="golden"      ：手臂保持在真机遥操起点，夹爪闭合。
        pose="oldros"      ：手臂保持在 old_ros 的 initial_pose（折叠待机位）。
        """
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self.apply_start_pose(pose)
        self.step(settle)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.m, self.d)

    def obj_pos(self):
        return self.d.qpos[self.oadr:self.oadr + 3].copy()

    def tcp_pos(self):
        return self.d.site_xpos[self.tcp].copy()

    def grip_ctrl(self):
        """只改夹爪目标, 保持手臂当前指令不变。"""
        q_now = np.array(self.d.qpos[self.adr], float)
        self.d.ctrl[self.ctrl] = q_now

    # ---------- IK ----------
    def ik(self, target, seed, tol=1e-4):
        """
        阻尼最小二乘 IK。只用于产生关节目标。
        注意: 内部会写 d.qpos 做正运动学, 返回前必须恢复真实状态,
        否则调用者拿到的 q_from 会等于 IK 解, 导致插值失效 (手臂瞬弹)。
        """
        m, d = self.m, self.d
        q_save = d.qpos[self.adr].copy()
        vel_save = d.qvel.copy()

        q = np.asarray(seed, float).copy()
        err_n = 1e9
        for _ in range(600):
            d.qpos[self.adr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.asarray(target, float) - d.site_xpos[self.tcp]
            err_n = float(np.linalg.norm(err))
            if err_n < tol:
                break
            self._J.fill(0.0)
            mujoco.mj_jacSite(m, d, self._J, None, self.tcp)
            Ja = self._J[:, self.adr]
            dq = Ja.T @ np.linalg.solve(Ja @ Ja.T + 0.05 ** 2 * np.eye(3), err)
            n = np.linalg.norm(dq)
            if n > 0.15:
                dq *= 0.15 / n
            q = np.clip(q + dq, self.lo, self.hi)

        # 恢复真实物理状态
        d.qpos[self.adr] = q_save
        d.qvel[:] = vel_save
        mujoco.mj_forward(m, d)
        return q, err_n

    def ik_lock(self, target, seed, ax_ref, tol=1e-4, iters=150,
                damp=0.02, eps=1e-3, w_axis=0.05, w_rest=0.3):
        """
        位置为主、工具指向为辅的 IK（2024-11 第六轮新增）。

        为什么需要：位置 IK 的 2 个零空间自由度会把姿态随手丢给解出来的分支 ——
        实测同一套位置 IK，"药盒正上方"与"抓取位"两个解的工具指向相差约 40°。
        抬升/搬运走关节空间插值经过这段，握在手里的药盒被带着转（0.3° -> 26° -> 30°）。

        怎么解（第一版写成 5 约束方阵，**崩了**：解出的 TCP 跑到 (171,6,160)，残差 0.34）：
          位置硬约束(3 行) + 指向软约束(2 行, w_axis) + 零空间把关节拉回种子(w_rest)。
          零空间投影 N = I - Jp⁺Jp 是关键 —— 姿态主要由关节本身决定，
          让解"贴着种子走"，指向就不会被甩到另一个分支去。

        返回 (q, 位置残差 m)。指向偏差要自己按需再量。
        """
        m, d = self.m, self.d
        q_save = d.qpos[self.adr].copy()
        vel_save = d.qvel.copy()
        q = np.clip(np.asarray(seed, float).copy(), self.lo, self.hi)
        q0 = q.copy()
        ax_ref = np.asarray(ax_ref, float)
        ax_ref = ax_ref / max(np.linalg.norm(ax_ref), 1e-12)
        target = np.asarray(target, float)
        I3, I5 = np.eye(3), np.eye(5)

        def fk(qv):
            # ★ 必须用 mj_fwdPosition：只调 mj_kinematics+mj_comPos 时 site_xmat 不更新
            #   （实测轴向数值导数全是 0.000，指向约束等于没加 —— 这就是"位置收敛了、
            #   指向还是偏 31°"的原因）。
            d.qpos[self.adr] = qv
            mujoco.mj_fwdPosition(m, d)
            R = d.site_xmat[self.tcp].reshape(3, 3)
            return d.site_xpos[self.tcp].copy(), R

        err_pos = 1e9
        for _ in range(iters):
            p, R = fk(q)
            e_pos = target - p
            err_pos = float(np.linalg.norm(e_pos))
            if err_pos < tol:
                break
            e = np.concatenate([e_pos, w_axis * (ax_ref - R[:, 0])[:2]])
            J = np.zeros((5, 5))
            for jj in range(5):
                qp = q.copy()
                qp[jj] += eps
                p2, R2 = fk(qp)
                J[:3, jj] = (p2 - p) / eps
                J[3:, jj] = w_axis * (R2[:2, 0] - R[:2, 0]) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + damp ** 2 * np.eye(5), e)
            # ★ 零空间投影必须用【精确】伪逆：带阻尼的伪逆不是投影矩阵，
            #   种子拉力会漏进位置方向，导致位置永远差几十毫米收不回来
            #   （第一版就是这么错的：残差 23 mm、指向仍偏 25°）。
            N = I5 - np.linalg.pinv(J[:3]) @ J[:3]
            dq = dq + N @ (w_rest * (q0 - q))          # 贴着种子走，别换分支
            nrm = float(np.linalg.norm(dq))
            if nrm > 0.08:
                dq *= 0.08 / nrm
            q = np.clip(q + dq, self.lo, self.hi)

        d.qpos[self.adr] = q_save
        d.qvel[:] = vel_save
        mujoco.mj_forward(m, d)
        return q, err_pos

    # ---------- 运动 ----------
    def move(self, q_target, gctrl, n_steps, watch=True):
        """关节空间线性插值移动, 返回 (碰撞步数, 碰到的 body 名集合)。"""
        m, d = self.m, self.d
        q_from = np.array(d.qpos[self.adr], float)
        n_hit, hit_bodies = 0, set()
        for k in range(n_steps):
            f = (k + 1) / n_steps
            d.ctrl[self.ctrl] = q_from + (np.asarray(q_target, float) - q_from) * f
            d.ctrl[self.cg] = gctrl
            mujoco.mj_step(m, d)
            if watch:
                hb = self.arm_hits()
                if hb:
                    n_hit += 1
                    hit_bodies |= hb
        return n_hit, hit_bodies

    def move_cartesian(self, p_to, gctrl, n_steps, knots=20, watch=True):
        """
        TCP 沿【世界直线】走到 p_to，返回 (碰撞步数, 碰到的 body 名集合)。

        为什么需要它（2024-11 预设姿态改成真机 golden pose 之后）：
            golden pose 下手指就在药盒旁边（jaw 最近 34mm、gripper 49mm，且与药盒同高）。
            关节空间直线插值会让手指"横着"扫过药盒 —— 实测把药盒直接推倒
            （药盒 z 27.8mm -> 9.3mm，整个 episode 报废）。
            竖直方向的笛卡尔直线（先把自己抬离台面）才是干净的路。

        实现：把整段切成 knots 个结点，结点处用位置 IK 求关节角（上一结的解当种子，
        保持同一支解），段内关节线性插值。knots=20 时 800 步只需 20 次 IK，开销可忽略。
        """
        m, d = self.m, self.d
        n_hit, hit_bodies = 0, set()
        p_start = np.array(self.d.site_xpos[self.tcp], float)
        p_to = np.asarray(p_to, float)
        q_prev = np.array(d.qpos[self.adr], float)
        seg = max(1, int(n_steps) // max(1, int(knots)))
        done = 0
        while done < n_steps:
            k = min(seg, n_steps - done)
            frac = (done + k) / float(n_steps)
            target = p_start + (p_to - p_start) * frac
            q_next, _ = self.ik(list(target), q_prev)
            q_next = np.asarray(q_next, float)
            q_cur = np.array(d.qpos[self.adr], float)
            for j in range(k):
                f = (j + 1) / k
                d.ctrl[self.ctrl] = q_cur + (q_next - q_cur) * f
                d.ctrl[self.cg] = gctrl
                mujoco.mj_step(m, d)
                if watch:
                    hb = self.arm_hits()
                    if hb:
                        n_hit += 1
                        hit_bodies |= hb
            q_prev = q_next
            done += k
        return n_hit, hit_bodies

    # ---------- 接触 ----------
    def _contacts_with(self, target_bodies):
        out6 = np.zeros(6)
        res = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = int(self.m.geom_bodyid[c.geom1])
            b2 = int(self.m.geom_bodyid[c.geom2])
            if b1 == self.obj:
                other = b2
            elif b2 == self.obj:
                other = b1
            else:
                continue
            if other not in target_bodies:
                continue
            mujoco.mj_contactForce(self.m, self.d, i, out6)
            f = np.asarray(out6[:3])
            nrm = np.asarray(c.frame[:3])
            res.append(dict(body=other,
                            F=float(np.linalg.norm(f)),
                            Fn=float(abs(f @ nrm)),
                            dist=float(c.dist)))
        return res

    def grip_contacts(self):
        r = self._contacts_with((self.grip, self.jaw))
        for x in r:
            x["label"] = "jaw" if x["body"] == self.jaw else "grip"
        return r

    def arm_hits(self):
        """药盒与机械臂(含夹爪)的接触, 排除支撑面。"""
        s = set()
        for x in self._contacts_with(set(range(self.m.nbody))):
            nm = self.name(x["body"])
            if nm in SUPPORT_BODIES:
                continue
            s.add(nm)
        return s

    def tbox_contacts(self):
        """药盒与木盒的接触 —— 约束2 的检查项。"""
        return self._contacts_with((self.tbox,))

    def box_hits(self):
        """
        机械臂(含夹爪) 与 target_box(木盒) 的接触 —— 2024-11 新增，返回 [(body, geom1, geom2)]。

        为什么需要它：尾段要把药盒"放进木盒"，指尖会伸进盒口再竖着提出来，
        一旦腕部/指尖刮到木盒侧壁，45 g 的木盒会被推着跑（真机上同样如此）。
        `_contacts_with()` 只筛与药盒(object)相关的接触，口径不对，所以这里单写：
        逐个接触对判断"一侧是木盒、另一侧既不是药盒也不是支撑面"。
        """
        res = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1 = int(self.m.geom_bodyid[c.geom1])
            b2 = int(self.m.geom_bodyid[c.geom2])
            if (b1 == self.tbox) == (b2 == self.tbox):
                continue                      # 与木盒无关的接触对，跳过
            other = b2 if b1 == self.tbox else b1
            if other == self.obj or self.name(other) in SUPPORT_BODIES:
                continue                      # 药盒落在木盒里 / 木盒贴桌面，都正常
            res.append((other, int(c.geom1), int(c.geom2)))
        return res


# ============================================================
#  主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roll", type=float, default=DEF_ROLL, help="腕部滚转(度)")
    ap.add_argument("--gz", type=float, default=DEF_GZ, help="抓取 TCP 高度")
    ap.add_argument("--no-transport", action="store_true")
    ap.add_argument("--settle", type=int, default=1200)
    args = ap.parse_args()

    a = Arm()
    m, d = a.m, a.d

    print("=" * 80)
    print("阶段 0  药盒自然落定 (原始直立, 零 qpos 覆盖)")
    print("=" * 80)
    a.reset(args.settle)
    p0 = a.obj_pos()
    print(f"  药盒 = [{p0[0]:.5f} {p0[1]:.5f} {p0[2]:.5f}]")
    print(f"  高 52mm, 厚 15mm(y向), 顶面 z={p0[2]+0.026:.5f}")
    print(f"  wrist_roll 限位 = {a.lo[4]:.4f} ~ {a.hi[4]:.4f} rad "
          f"({np.degrees(a.lo[4]):.1f}° ~ {np.degrees(a.hi[4]):.1f}°)")
    roll_in = float(np.clip(np.radians(args.roll), a.lo[4], a.hi[4]))
    print(f"  使用 roll = {args.roll}° -> 实际 {np.degrees(roll_in):.1f}°")

    print("\n" + "=" * 80)
    print("阶段 1  IK 求解路点")
    print("=" * 80)
    seed = [-0.33, -0.08, 0.80, -0.02, roll_in]
    q_ret, e1 = a.ik([p0[0], p0[1], RETRACT_Z], seed)
    print(f"  高位撤回  z={RETRACT_Z}   err={e1*1000:.2f}mm")
    gx, gy = p0[0] + GRASP_XY_OFFSET[0], p0[1] + GRASP_XY_OFFSET[1]
    q_abv, e2 = a.ik([gx, gy, APPROACH_Z], q_ret)
    print(f"  正上方    z={APPROACH_Z}   err={e2*1000:.2f}mm  (横向补偿 dx={GRASP_XY_OFFSET[0]*1000:+.1f} dy={GRASP_XY_OFFSET[1]*1000:+.1f} mm)")
    q_gsp, e3 = a.ik([p0[0], p0[1], args.gz], q_abv)
    print(f"  抓取点    z={args.gz}   err={e3*1000:.2f}mm")
    print(f"  q_grasp = {np.round(q_gsp, 5).tolist()}")

    print("\n" + "=" * 80)
    print("阶段 2  高位撤回 -> 正上方  (夹爪全开, 约束1 的准备工作)")
    print("=" * 80)
    d.ctrl[a.cg] = GRIP_MAX
    h1, b1 = a.move(q_ret, GRIP_MAX, 500)
    print(f"  home -> 撤回   TCP={np.round(a.tcp_pos(),4)}  药盒z={a.obj_pos()[2]:.5f}  "
          f"碰撞步={h1} {sorted(b1) if b1 else ''}")
    h2, b2 = a.move(q_abv, GRIP_MAX, 400)
    print(f"  撤回 -> 正上方 TCP={np.round(a.tcp_pos(),4)}  药盒z={a.obj_pos()[2]:.5f}  "
          f"碰撞步={h2} {sorted(b2) if b2 else ''}")

    print("\n" + "=" * 80)
    print("阶段 3  竖直下探  —— 约束1")
    print("=" * 80)
    h3, b3 = a.move(q_gsp, GRIP_APPROACH, 400)
    p_after = a.obj_pos()
    print(f"  下探完成 TCP={np.round(a.tcp_pos(),4)}  目标 z={args.gz}")
    print(f"  药盒z={p_after[2]:.5f} (起始 {p0[2]:.5f}, Δ={(p_after[2]-p0[2])*1000:+.2f}mm)")
    print(f"  下探碰撞步={h3} {sorted(b3) if b3 else '(无)'}")
    ok1 = abs(p_after[2] - p0[2]) < 0.002
    print(f"  约束1: {'✅ 通过 (药盒未被碰倒)' if ok1 else '❌ 药盒被撞动了'}")

    print("\n" + "=" * 80)
    print(f"阶段 4  合爪 (ctrl {GRIP_APPROACH} -> {GRIP_CLOSED})")
    print("=" * 80)
    print(f"{'step':>5} {'ctrl':>6} {'obj_z':>8} {'接触':>5} {'Fn(N)':>9} {'jaw':>4} {'grip':>5}")
    q_hold = np.array(d.qpos[a.adr], float)
    for k in range(300):
        c = GRIP_APPROACH + (GRIP_CLOSED - GRIP_APPROACH) * (k + 1) / 300
        d.ctrl[a.ctrl] = q_hold
        d.ctrl[a.cg] = c
        mujoco.mj_step(m, d)
        if (k + 1) % 50 == 0:
            cs = a.grip_contacts()
            print(f"{k+1:5d} {c:6.3f} {a.obj_pos()[2]:8.4f} {len(cs):5d} "
                  f"{sum(x['Fn'] for x in cs):9.4f} "
                  f"{sum(1 for x in cs if x['label']=='jaw'):4d} "
                  f"{sum(1 for x in cs if x['label']=='grip'):5d}")
    cs = a.grip_contacts()
    print()
    for x in cs:
        print(f"    {x['label']:6s} |F|={x['F']:.4f}N  Fn={x['Fn']:.4f}N  "
              f"穿透={x['dist']*1000:+.2f}mm")
    mg = m.body_mass[a.obj] * abs(m.opt.gravity[2])
    mu = float(m.geom_friction[a.obj_geom][0])
    fn_sum = sum(x["Fn"] for x in cs)
    print(f"\n  药盒重力 {mg:.4f}N  静摩擦 μ={mu}  需单侧法向 ≥ {mg/(2*mu):.5f}N")
    print(f"  实测法向合计 {fn_sum:.4f}N  (裕度 {fn_sum/max(mg/(2*mu),1e-9):.1f}x)")

    print("\n" + "=" * 80)
    print("阶段 5  竖直抬升")
    print("=" * 80)
    z_before = a.obj_pos()[2]
    # ★ 第七轮修正：这里原来写 0.0（完全合拢）。指垫贴齐指面之后，抬升途中继续合到 0.0
    #   等于在夹住盒子的同时继续加压 —— 实测药盒被挤出去（横移 36 mm、抬升 5 mm）。
    #   与合爪段保持一致用 GRIP_CLOSED。
    h5, _ = a.move(q_abv, GRIP_CLOSED, 400)
    z_lift = a.obj_pos()[2]
    cs_l = a.grip_contacts()
    print(f"  药盒 z: {z_before:.4f} -> {z_lift:.4f}  (Δ={(z_lift-z_before)*1000:+.2f}mm)")
    print(f"  握持接触 {len(cs_l)} 个, Fn={sum(x['Fn'] for x in cs_l):.4f}N")

    box_hit = 0
    p_final = a.obj_pos()
    if not args.no_transport:
        print("\n" + "=" * 80)
        print("阶段 6  搬运到木盒上方 —— 约束2")
        print("=" * 80)
        bx = m.body_pos[a.tbox]
        print(f"  木盒 = [{bx[0]:.4f} {bx[1]:.4f} {bx[2]:.4f}]  上沿 z≈{bx[2]+0.05:.3f}")
        q_over, e4 = a.ik([float(bx[0]), float(bx[1]), TRANSPORT_Z], q_abv)
        print(f"  IK 到木盒上方 err={e4*1000:.2f}mm")
        print(f"\n{'step':>5} {'TCPy':>8} {'objy':>8} {'objz':>8} {'握持':>5} {'Fn':>7} {'木盒接触':>8}")
        q_from = np.array(d.qpos[a.adr], float)
        for k in range(600):
            f = (k + 1) / 600
            d.ctrl[a.ctrl] = q_from + (q_over - q_from) * f
            d.ctrl[a.cg] = 0.0
            mujoco.mj_step(m, d)
            if a.tbox_contacts():
                box_hit += 1
            if (k + 1) % 100 == 0:
                t = a.tcp_pos(); p = a.obj_pos(); gc = a.grip_contacts()
                print(f"{k+1:5d} {t[1]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {len(gc):5d} "
                      f"{sum(x['Fn'] for x in gc):7.4f} {len(a.tbox_contacts()):8d}")
        p_final = a.obj_pos()
        print(f"\n  约束2: 药盒-木盒接触步数 = {box_hit}/600 -> "
              f"{'✅ 通过' if box_hit == 0 else '❌ 碰撞'}")

    print("\n" + "=" * 80)
    print("最终结果")
    print("=" * 80)
    cs_end = a.grip_contacts()
    nj = sum(1 for x in cs_end if x["label"] == "jaw")
    ng = sum(1 for x in cs_end if x["label"] == "grip")
    rise = (z_lift - z_before) * 1000
    print(f"  药盒: [{p0[0]:.4f} {p0[1]:.4f} {p0[2]:.4f}] -> "
          f"[{p_final[0]:.4f} {p_final[1]:.4f} {p_final[2]:.4f}]")
    print(f"  抬升 {rise:+.2f} mm   水平位移 {np.linalg.norm(p_final[:2]-p0[:2])*1000:.1f} mm")
    print(f"  末端握持接触 {len(cs_end)} 个 (jaw {nj} / grip {ng})")
    print(f"  约束1 {'✅' if ok1 else '❌'}   约束2 {'✅' if box_hit == 0 else '❌'}")

    if rise > 50 and nj >= 1:
        print("\n  ✅ 抓取成功: 药盒被抬离, 两指均有接触")
    elif rise > 50:
        print("\n  ✅ 抓取成功 (单侧接触, 稳定性较低)")
    elif rise > 5:
        print("\n  ⚠️  部分成功")
    else:
        print("\n  ❌ 抓取失败")


if __name__ == "__main__":
    main()
