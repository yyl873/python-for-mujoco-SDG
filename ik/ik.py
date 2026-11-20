# ==========================================
# 导入工具包
# ==========================================
import sys
from pathlib import Path
import numpy as np
import mujoco

# ==========================================
# 路径与常量配置
# ==========================================
# 找到当前文件所在目录的上一级（也就是项目根目录），防止路径找不到
ROOT = Path(__file__).resolve().parents[1]
# 拼出模型的完整路径
SCENE_XML = str(ROOT / "model" / "scene.xml")

# 定义参与IK的5个关节名字（注意：故意剔除了夹爪，因为夹爪不影响手的位置）
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# IK算法的4个“刹车和油门”参数
MAX_ITER = 200      # 最多试算200步，算不出来就拉倒
TOL = 0.001         # 目标误差：小于1毫米就算成功
LAMBDA = 0.05       # DLS阻尼系数：防止靠近奇异点时关节速度爆炸（除以0）
MAX_DQ = 0.2        # 单步极限：一次最多只能转0.2弧度，步子太大容易扯到蛋（线性化失效）

# ==========================================
# 函数1：加载模型和基础数据
# ==========================================
def load_model():
    # 把XML图纸变成可以计算的模型
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    # 给模型分配一块内存，用来存它的动态数据（角度、速度、受力等）
    data = mujoco.MjData(model)

    # 拿这5个关节的“名字”去引擎里查它们的“编号”
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
    # 根据编号，找到这5个关节在全局角度数组 qpos 里的“排位（索引）”
    qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids], dtype=int)

    # 查末端夹爪标记点（gripperframe）的编号
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    if site_id < 0:
        raise RuntimeError("模型里找不到 site 'gripperframe'") # 找不到就直接报错

    return model, data, qpos_adr, site_id

# ==========================================
# 函数2：核心 IK 求解器（DLS算法）
# ==========================================
def solve_ik_to_xyz(model, data, qpos_adr, site_id, target_xyz,
                    q_seed=None, max_iter=MAX_ITER, tol=TOL, lam=LAMBDA):
    
    target = np.asarray(target_xyz, dtype=float) # 把目标坐标转成浮点数数组

    # 把模型重置到初始状态（第一帧）
    mujoco.mj_resetDataKeyframe(model, data, 0)
    # 把当前的5个关节角度抓出来，存到变量 q 里（这是我们的起点）
    q = np.array(data.qpos[qpos_adr], dtype=float)
    if q_seed is not None:
        q[:] = np.asarray(q_seed, dtype=float) # 如果你传了初始角，就覆盖掉（引导多解方向）

    # ----- 获取关节物理限位 -----
    jnt_lo, jnt_hi = [], []
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = model.jnt_range[jid]
        jnt_lo.append(lo)
        jnt_hi.append(hi)
    jnt_lo = np.array(jnt_lo)
    jnt_hi = np.array(jnt_hi)

    # 预先分配好雅可比矩阵的内存（3行代表xyz，nv列代表所有自由度）
    J = np.zeros((3, model.nv))
    # 专门切出给这5个臂关节的雅可比矩阵（3x5）
    J_arm = np.zeros((3, len(qpos_adr)))

    # 定义一个快速刷新位置的小工具（FK）
    def fk():
        data.qpos[qpos_adr] = q # 把现在的角度塞进引擎
        mujoco.mj_kinematics(model, data) # 算正运动学（手在哪）
        mujoco.mj_comPos(model, data) # 更新质心等物理量

    # 打印一个漂亮表头
    print(f"目标末端位置: {target}")
    print(f"{'iter':>4} {'err(m)':>10} {'|dq|':>9}   q(rad)")
    print("-" * 78)

    err = target.copy() # 给个初始误差，保证能进循环
    
    # ======== 开始死循环逼近 ========
    for it in range(max_iter):
        fk() # 1. FK算当前位置
        p = np.array(data.site_xpos[site_id]) # 2. 读出当前末端坐标
        err = target - p # 3. 算误差：目标 - 当前
        err_norm = float(np.linalg.norm(err)) # 4. 算绝对距离

        # 5. 让引擎算雅可比（只要位置不要姿态，所以第二路传None）
        J.fill(0.0) # 先把旧数据清零
        mujoco.mj_jacSite(model, data, J, None, site_id)
        J_arm = J[:, qpos_adr] # 6. 切出 3x5 矩阵，把夹爪和其他物体排除

        # 7. 核心DLS公式：dq = J^T * (J*J^T + lam^2*I)^-1 * err
        A = J_arm @ J_arm.T + (lam ** 2) * np.eye(3) # 加阻尼的安全垫
        dq = J_arm.T @ np.linalg.solve(A, err) # 算出每个关节该微调多少 (dq)

        # 打印当前这一步的日志
        print(f"{it:4d} {err_norm:10.6f} {np.linalg.norm(dq):9.5f}   "
              + " ".join(f"{v:+.3f}" for v in q))

        # 8. 如果误差小于1毫米，成功！
        if err_norm < tol:
            print("-" * 78)
            print(f"✅ 收敛于第 {it} 次迭代，误差 {err_norm:.6f} m < {tol} m")
            break # 跳出循环

        # 9. 步长限制（防止一次转太多）
        # 如果算出的 dq 超过了 MAX_DQ，就按比例缩小
        scale = min(1.0, MAX_DQ / (np.linalg.norm(dq) + 1e-12))
        
        # 10. 更新角度并做限位裁剪（强行卡在物理范围内）
        q = np.clip(q + dq * scale, jnt_lo, jnt_hi)
        
    else:
        # 如果循环200次全部跑完还没 break，说明没算出来
        print("-" * 78)
        print(f"⚠️  达到最大迭代次数 {max_iter} 仍未收敛，最终误差 {err_norm:.6f} m")

    # 拿到最终结果，再算一次FK，看看到底停在哪了
    fk()
    p_final = np.array(data.site_xpos[site_id])
    # 返回：最终的5个角度、最终误差、最终坐标
    return q.copy(), float(np.linalg.norm(target - p_final)), p_final

# ==========================================
# 函数3：程序入口
# ==========================================
def main():
    # 默认目标点
    target = [0.20, 0.10, 0.05]
    
    # 如果你在命令行敲了3个参数，就用你敲的参数当目标点
    if len(sys.argv) == 4:
        target = [float(v) for v in sys.argv[1:4]]

    # 加载模型
    model, data, qpos_adr, site_id = load_model()
    # 执行IK求解
    q_sol, err, p_final = solve_ik_to_xyz(model, data, qpos_adr, site_id, target)

    # ----- 打印最终结果 -----
    print()
    print("求解结果")
    print(f"  关节角(rad): "
          + ", ".join(f"{n.split('_joint')[0]}={v:+.6f}" for n, v in zip(JOINT_NAMES, q_sol)))
    print(f"  末端实测位置: [{p_final[0]:.6f} {p_final[1]:.6f} {p_final[2]:.6f}]")
    print(f"  末端残留误差: {err:.6f} m")
    print()
    print("可直接复制到步骤二/三的下探姿态:")
    # 输出一个带括号的列表，方便你复制去别的脚本用
    print("  DIVE_QPOS = [" + ", ".join(f"{v:.6f}" for v in q_sol) + ", 0.0]  # 末位为夹爪张开")

# 如果是直接运行这个脚本，就执行 main()
if __name__ == "__main__":
    main()