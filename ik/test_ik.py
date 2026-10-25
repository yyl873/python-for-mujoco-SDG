"""
test_ik.py — 纯 MuJoCo 阻尼最小二乘(DLS)逆运动学测试
=====================================================
不依赖 ROS / MoveIt，只用 `mujoco` 绑定 + numpy。

运行:
    python test_ik.py                 # 默认目标 [0.20, 0.10, 0.05]
    python test_ik.py 0.25 -0.05 0.10 # 自定义目标 (x y z)
"""

import sys
from pathlib import Path

import numpy as np
import mujoco

# 路径锚定到"仓库根目录/"：本文件在 <root>/ik/test_ik.py，parents[1] 即 <root>。
# 这样无论从哪个目录运行都不会找不到 model/scene.xml。
ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = str(ROOT / "model" / "scene.xml")

# 参与 IK 的 5 个关节 (不含夹爪 gripper_joint，夹爪开合与末端定位无关)
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# IK 迭代超参数
MAX_ITER = 200      # 最大迭代次数
TOL = 0.001         # 收敛阈值 (米)，任务书要求 < 0.001
LAMBDA = 0.05       # 阻尼系数，抑制奇异位形下的关节速度爆炸
MAX_DQ = 0.2        # 单步关节增量上限 (rad)，防止一次跳太远导致线性化失效


def load_model():
    """加载场景，返回 (model, data, 5 个关节的 qpos 地址, 末端 site 的 id)。"""
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)

    # 用名字查表拿地址，绝不硬编码下标——改 XML 顺序也不会崩
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id < 0:
        raise RuntimeError("模型里找不到 site 'gripperframe'")

    return model, data, qpos_adr, site_id


def solve_ik_to_xyz(model, data, qpos_adr, site_id, target_xyz,
                    q_seed=None, max_iter=MAX_ITER, tol=TOL, lam=LAMBDA):
    """
    阻尼最小二乘 (DLS) 逆运动学。

    ## 数学原理

    **1. 雅可比矩阵 J —— 把"关节速度"翻译成"末端速度"**

    对末端位置 p(q) 做一阶泰勒展开：
        p(q + dq) ≈ p(q) + J·dq   ,   J = ∂p/∂q ∈ R^{3×5}

    J 的第 i 列 = 只让第 i 个关节转一点点时，末端点朝哪个方向动、动多快。
    MuJoCo 用 `mj_jacSite` 一次性给出平移雅可比 jacp (3×nv)；
    nv 里还包含物块自由关节的 6 维，所以只切前 5 列 :5，得到机械臂自己的 J。

    于是「我要末端移动 err」变成「我要解 J·dq = err」，这是个超定/欠定的线性最小二乘。

    **2. 什么时候必须加阻尼 λ**

    J 是 3×5，方阵 JᵀJ 奇异时（机械臂伸直、腕部共线等奇异位形），
    伪逆解 q̇ = J⁺·err 的分量会飙到几百 rad/s，仿真里就是关节瞬间抽搐、跳飞。

    DLS 把问题改成带正则项的最小化：
        min ‖J·dq − err‖² + λ²‖dq‖²
    正规方程 (JᵀJ + λ²I)·dq = Jᵀ·err，用 Sherman–Morrison 恒等式改写为
        dq = Jᵀ · (J·Jᵀ + λ²I)⁻¹ · err          ← 代码里就是这个式子
    好处：J·Jᵀ 只是 3×3，求逆几十纳秒；λ 越大越"保守"，末端跟踪慢但绝不会跳飞。

    直观理解：λ² 是在 J 的每个奇异值上加了一层"地板"。奇异值 σ 对应的增益
    从 1/σ 变成 σ/(σ²+λ²)，σ→0 时增益→0 而不是 →∞。
    λ=0.05 时，增益上限约为 1/(2λ) = 10 rad/m，够快也够稳。

    **3. 为什么要迭代**

    泰勒展开只对局部线性成立。解出 dq 后把关节推到 q+dq，末端真实到达位置
    与线性预测有偏差（曲率误差），所以循环"测误差 → 解 dq → 前进"直到 err < tol。
    这本质上是带阻尼的高斯-牛顿法 (Levenberg–Marquardt 的纯梯度版本)。
    """
    target = np.asarray(target_xyz, dtype=float)

    # 用 home 关键帧做种子，并把种子灌进 qpos
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q = np.array(data.qpos[qpos_adr], dtype=float)
    if q_seed is not None:
        q[:] = np.asarray(q_seed, dtype=float)

    # 关节软限位：只允许在 range 内搜索，否则解出的姿态物理上不可达
    jnt_lo, jnt_hi = [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        jnt_lo.append(lo)
        jnt_hi.append(hi)
    jnt_lo = np.array(jnt_lo)
    jnt_hi = np.array(jnt_hi)

    J = np.zeros((3, model.nv))
    J_arm = np.zeros((3, len(qpos_adr)))

    def fk():
        """只做正运动学刷新，不步进物理——IK 阶段不想让重力把臂拉下来。"""
        data.qpos[qpos_adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

    print(f"目标末端位置: {target}")
    print(f"{'iter':>4} {'err(m)':>10} {'|dq|':>9}   q(rad)")
    print("-" * 78)

    err = target.copy()
    for it in range(max_iter):
        fk()
        p = np.array(data.site_xpos[site_id])
        err = target - p
        err_norm = float(np.linalg.norm(err))

        # --- 计算 J (只取机械臂 5 列) ---
        J.fill(0.0)
        mujoco.mj_jacSite(model, data, J, None, site_id)
        J_arm = J[:, qpos_adr]

        # --- DLS 求解: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ err ---
        A = J_arm @ J_arm.T + (lam ** 2) * np.eye(3)
        dq = J_arm.T @ np.linalg.solve(A, err)

        print(f"{it:4d} {err_norm:10.6f} {np.linalg.norm(dq):9.5f}   "
              + " ".join(f"{v:+.3f}" for v in q))

        if err_norm < tol:
            print("-" * 78)
            print(f"✅ 收敛于第 {it} 次迭代，误差 {err_norm:.6f} m < {tol} m")
            break

        # 步长裁剪 + 限位裁剪，双重保险
        scale = min(1.0, MAX_DQ / (np.linalg.norm(dq) + 1e-12))
        q = np.clip(q + dq * scale, jnt_lo, jnt_hi)
    else:
        print("-" * 78)
        print(f"⚠️  达到最大迭代次数 {max_iter} 仍未收敛，最终误差 {err_norm:.6f} m")

    # 用最终解刷新一次，回报真实末端位置与误差
    fk()
    p_final = np.array(data.site_xpos[site_id])
    return q.copy(), float(np.linalg.norm(target - p_final)), p_final


def main():
    target = [0.20, 0.10, 0.05]
    if len(sys.argv) == 4:
        target = [float(v) for v in sys.argv[1:4]]

    model, data, qpos_adr, site_id = load_model()
    q_sol, err, p_final = solve_ik_to_xyz(model, data, qpos_adr, site_id, target)

    print()
    print("求解结果")
    print(f"  关节角(rad): "
          + ", ".join(f"{n.split('_joint')[0]}={v:+.6f}" for n, v in zip(JOINT_NAMES, q_sol)))
    print(f"  末端实测位置: [{p_final[0]:.6f} {p_final[1]:.6f} {p_final[2]:.6f}]")
    print(f"  末端残留误差: {err:.6f} m")
    print()
    print("可直接复制到步骤二/三的下探姿态:")
    print("  DIVE_QPOS = [" + ", ".join(f"{v:.6f}" for v in q_sol) + ", 0.0]  # 末位为夹爪张开")


if __name__ == "__main__":
    main()
