"""
test_physics.py — 纯物理接触与抓取测试
=====================================
不覆盖任何 qpos 来"假装抓住"。所有判断都来自 mj_contactForce / mj_step。

运行:
    python test_physics.py
    python test_physics.py --ctrl-max 0.35 --close-steps 300 --lift-steps 600

几何背景 (从 scene.xml 实测):
    pick_zone 顶面 z = 0.002
    药盒 x 向半长 0.0365 / y 向半厚 0.0075 / z 向半高 0.026
    药盒静止于 (0.295, 0.090, 0.02789)
    药盒顶面 ≈ 0.0539；抓顶部下方 15 mm => TCP z ≈ 0.0389
    夹爪从 ±y 对夹，所以是夹 y 向的 15 mm 厚度

阶段:
    A  抬到药盒正上方 7 cm，再竖直下探到抓取高度
    B  ctrl 从 0 线性爬到 ctrl_max（缓慢合爪），观察接触力建立
    C  ctrl 保持不变，把 TCP 抬高，看药盒是否跟着走
"""

import argparse
import numpy as np
import mujoco

SCENE_XML = "model/scene.xml"

ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_flex_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
]

# ---- 用 test_ik.py 的 DLS 解出、对准药盒的姿态 (TCP 误差 < 0.1 mm) ----
ABOVE_QPOS = [-0.338638, -0.466592, 0.815831, 0.122751, -0.012054]  # TCP (0.295,0.090,0.1089)
GRASP_QPOS = [-0.338642, -0.085574, 0.799667, -0.021875, -0.011928]  # TCP (0.295,0.090,0.0389)

OBJ_XYZ = np.array([0.295, 0.090, 0.02789])   # 实测静止位置
DT = 0.002


# ============================================================
#  0. 基础设施
# ============================================================
def build():
    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    return m, d


def ids(m):
    g = {}
    g["tcp"]  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    g["obj"]  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    g["grip"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")
    g["jaw"]  = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw_link")
    g["obj_geom"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_physics")

    g["qpos"] = np.array([
        m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ARM_JOINTS
    ], dtype=int)
    g["obj_adr"] = m.jnt_qposadr[m.body_jntadr[g["obj"]]]
    g["ctrl"] = np.array([
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in ARM_JOINTS
    ], dtype=int)
    g["ctrl_g"] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint")
    return g


def set_ctrl(d, g, q5, gctrl):
    """只写 data.ctrl —— 交给 PD 位置伺服，绝不碰 qpos。"""
    d.ctrl[g["ctrl"]] = np.asarray(q5, dtype=float)
    d.ctrl[g["ctrl_g"]] = float(gctrl)


def obj_z(d, g):
    return float(d.qpos[g["obj_adr"] + 2])


def obj_xyz(d, g):
    a = g["obj_adr"]
    return d.qpos[a:a + 3].copy()


def grip_contacts(m, d, g):
    """
    遍历 mj_data.contact，挑出「药盒 vs 夹爪两指」的真实接触。

    要点:
      - d.ncon 每个 step 都重算，不能跨 step 缓存
      - 用 geom_bodyid 过滤。药盒 body 有 6 个视觉 geom（contype=0 不参与碰撞），
        真正参与物理的是 object_physics
      - mujoco.mj_contactForce(m, d, i, out6) 返回 6 维:
            out6[:3] = 接触作用力   out6[3:] = 力矩
        函数内部已把 solver 的 impulse 换算成 N
      - 法向分量 = |F · n|，n = contact.frame[:3]
    """
    out6 = np.zeros(6)
    res = []
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = int(m.geom_bodyid[c.geom1])
        b2 = int(m.geom_bodyid[c.geom2])
        if b1 == g["obj"]:
            other = b2
        elif b2 == g["obj"]:
            other = b1
        else:
            continue
        if other not in (g["grip"], g["jaw"]):
            continue

        mujoco.mj_contactForce(m, d, i, out6)
        f = np.asarray(out6[:3])
        n = np.asarray(c.frame[:3])
        res.append({
            "geom1": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
            "geom2": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2),
            "dist": float(c.dist),
            "force": float(np.linalg.norm(f)),
            "normal": float(abs(f @ n)),
            "body": "jaw" if other == g["jaw"] else "gripper",
        })
    return res


def move_to(m, d, g, q_target, gctrl, n_steps, label=""):
    """PD 位置伺服跟踪一条直线关节插值，返回过程中 TCP 是否越过目标。"""
    q_now = np.array(d.qpos[g["qpos"]], dtype=float)
    for k in range(n_steps):
        frac = (k + 1) / n_steps
        q_cmd = q_now + (np.asarray(q_target, float) - q_now) * frac
        set_ctrl(d, g, q_cmd, gctrl)
        mujoco.mj_step(m, d)
    return np.array(d.qpos[g["qpos"]])


# ============================================================
#  1. 主实验
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl-max", type=float, default=0.35,
                    help="夹爪闭合目标 ctrl (0=张开, 1.70=完全闭合)")
    ap.add_argument("--close-steps", type=int, default=250)
    ap.add_argument("--lift-steps", type=int, default=500)
    ap.add_argument("--lift-z", type=float, default=0.09,
                    help="上提阶段 TCP 目标高度")
    ap.add_argument("--hold-steps", type=int, default=300,
                    help="提起后悬停步数")
    args = ap.parse_args()

    m, d = build()
    g = ids(m)

    print("=" * 78)
    print("模型信息")
    print("=" * 78)
    print(f"  nu / nq / nv   = {m.nu} / {m.nq} / {m.nv}")
    print(f"  timestep       = {m.opt.timestep} s")
    print(f"  重力           = {m.opt.gravity}")
    print(f"  积分器         = {m.opt.integrator} (0=Euler 1=RK4 2=implicit 3=implicitfast)")
    print(f"  药盒质量       = {m.body_mass[g['obj']]:.4f} kg")
    print(f"  药盒 qposadr   = {g['obj_adr']}")
    print(f"  夹爪 ctrl 范围 = {m.actuator_ctrlrange[g['ctrl_g']]}")
    print(f"  夹爪 kp/kv     = {m.actuator_gainprm[g['ctrl_g']][0]:.1f} / "
          f"{-m.actuator_biasprm[g['ctrl_g']][2]:.1f}")
    print(f"  夹爪 force范围 = {m.actuator_forcerange[g['ctrl_g']]}")

    # 先让药盒落定
    for _ in range(1500):
        mujoco.mj_step(m, d)
    print(f"\n药盒静止位置 = {np.round(obj_xyz(d, g), 5)}  (期望 {np.round(OBJ_XYZ, 5)})")

    half_thick = float(m.geom_size[g["obj_geom"]][1])
    print(f"药盒 y 向半厚 = {half_thick*1000:.1f} mm -> 合爪净开口约需 {2*half_thick*1000:.0f} mm")

    # =========================================================
    # Phase A: 抬到上方 -> 竖直下探
    # =========================================================
    print("\n" + "=" * 78)
    print("Phase A  抬到上方再下探")
    print("=" * 78)
    print(f"{'step':>5} {'TCP_x':>8} {'TCP_y':>8} {'TCP_z':>8} {'obj_z':>8} {'ncon':>5}")

    move_to(m, d, g, ABOVE_QPOS, 0.0, 400)
    t = d.site_xpos[g["tcp"]]
    print(f"{'above':>5} {t[0]:8.4f} {t[1]:8.4f} {t[2]:8.4f} {obj_z(d,g):8.4f} {d.ncon:5d}")

    for seg in range(8):
        move_to(m, d, g, GRASP_QPOS, 0.0, 50)
        t = d.site_xpos[g["tcp"]]
        print(f"{'dive%+d'%((seg+1)*50):>5} {t[0]:8.4f} {t[1]:8.4f} {t[2]:8.4f} "
              f"{obj_z(d,g):8.4f} {d.ncon:5d}")

    tcp_now = d.site_xpos[g["tcp"]].copy()
    grasp_target = np.array([0.295, 0.090, 0.0389])
    print(f"\n下探结束 TCP = {np.round(tcp_now, 5)}")
    print(f"     抓取目标点 = {np.round(grasp_target, 5)}")
    print(f"     XY 偏差 = {np.linalg.norm(tcp_now[:2]-grasp_target[:2])*1000:.1f} mm")
    print(f"     Z  偏差 = {(tcp_now[2]-grasp_target[2])*1000:+.1f} mm")
    cs = grip_contacts(m, d, g)
    print(f"     下探期间指尖-药盒接触 = {len(cs)} 个")

    # 夹爪开到最大再合，确保有闭合行程
    for _ in range(100):
        set_ctrl(d, g, GRASP_QPOS, 0.0)
        mujoco.mj_step(m, d)

    # =========================================================
    # Phase B: 缓慢合爪
    # =========================================================
    print("\n" + "=" * 78)
    print(f"Phase B  合爪 (ctrl 0 -> {args.ctrl_max}, {args.close_steps} 步 = "
          f"{args.close_steps*DT:.2f} s)")
    print("=" * 78)
    print(f"{'step':>5} {'ctrl':>6} {'obj_z':>8} {'Δz(mm)':>8} {'ncon':>5} "
          f"{'F_tot(N)':>9} {'F_norm(N)':>10} {'grip_q':>8}")
    print("-" * 78)

    z_start = obj_z(d, g)
    max_force = 0.0
    for k in range(args.close_steps):
        ctrl = args.ctrl_max * (k + 1) / args.close_steps   # 线性爬升 = 缓慢闭合
        set_ctrl(d, g, GRASP_QPOS, ctrl)
        mujoco.mj_step(m, d)

        if (k + 1) % 25 == 0 or k == args.close_steps - 1:
            cs = grip_contacts(m, d, g)
            ftot = sum(c["force"] for c in cs)
            fnor = sum(c["normal"] for c in cs)
            max_force = max(max_force, ftot)
            gj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")
            print(f"{k+1:5d} {ctrl:6.3f} {obj_z(d,g):8.4f} "
                  f"{(obj_z(d,g)-z_start)*1000:+8.2f} {len(cs):5d} "
                  f"{ftot:9.4f} {fnor:10.4f} {d.qpos[m.jnt_qposadr[gj]]:8.4f}")

    print("\n合爪结束 —— 逐个接触点详情")
    cs = grip_contacts(m, d, g)
    if not cs:
        print("  ⚠️  没有任何指尖-药盒接触！")
        print("     检查: 夹爪开口是否已经被 v 向薄，或 TCP 高度不对")
    for c in cs:
        print(f"  {c['body']:8s} |F|={c['force']:.4f} N  F_n={c['normal']:.4f} N  "
              f"dist={c['dist']*1000:+.2f} mm  ({c['geom1']} <-> {c['geom2']})")

    mg = m.body_mass[g["obj"]] * abs(m.opt.gravity[2])
    mu = float(m.geom_friction[g["obj_geom"]][0])
    f_norm_end = sum(c["normal"] for c in cs)
    print(f"\n力学判据")
    print(f"  药盒重力 mg            = {mg:.4f} N")
    print(f"  静摩擦系数 μ(药盒)      = {mu}")
    print(f"  单侧法向力需 ≥          = {mg/(2*mu):.5f} N (两侧对称夹持)")
    print(f"  合爪末法向力合计        = {f_norm_end:.4f} N")
    print(f"  峰值接触力(合爪期间)    = {max_force:.4f} N")

    # =========================================================
    # Phase C: 上提 + 悬停
    # =========================================================
    print("\n" + "=" * 78)
    print(f"Phase C  上提 TCP -> z={args.lift_z} ({args.lift_steps} 步) + 悬停 {args.hold_steps} 步")
    print("=" * 78)
    print(f"{'step':>7} {'tcp_z':>8} {'obj_z':>8} {'Δobj(mm)':>9} {'ncon':>5} {'F_tot(N)':>9}")
    print("-" * 78)

    z_at_grasp = obj_z(d, g)
    # 关节空间插值到"竖直上升"的解：直接复用 ABOVE_QPOS 的 z，但保持 grasp 的 pan/roll
    lift_q = np.array(ABOVE_QPOS, dtype=float)

    q_from = np.array(d.qpos[g["qpos"]], dtype=float)
    for k in range(args.lift_steps):
        frac = (k + 1) / args.lift_steps
        set_ctrl(d, g, q_from + (lift_q - q_from) * frac, args.ctrl_max)
        mujoco.mj_step(m, d)
        if (k + 1) % 50 == 0:
            cs = grip_contacts(m, d, g)
            print(f"{'L%+d'%((k+1)):>7} {d.site_xpos[g['tcp']][2]:8.4f} {obj_z(d,g):8.4f} "
                  f"{(obj_z(d,g)-z_at_grasp)*1000:+9.2f} {len(cs):5d} "
                  f"{sum(c['force'] for c in cs):9.4f}")

    print(f"\n悬停 {args.hold_steps} 步（{args.hold_steps*DT:.2f} s），观察是否缓慢滑落")
    for k in range(args.hold_steps):
        set_ctrl(d, g, lift_q, args.ctrl_max)
        mujoco.mj_step(m, d)
        if (k + 1) % 50 == 0:
            cs = grip_contacts(m, d, g)
            print(f"{'H%+d'%((k+1)):>7} {d.site_xpos[g['tcp']][2]:8.4f} {obj_z(d,g):8.4f} "
                  f"{(obj_z(d,g)-z_at_grasp)*1000:+9.2f} {len(cs):5d} "
                  f"{sum(c['force'] for c in cs):9.4f}")

    # =========================================================
    # 诊断
    # =========================================================
    print("\n" + "=" * 78)
    print("诊断结论")
    print("=" * 78)
    p_end = obj_xyz(d, g)
    rise = (p_end[2] - z_at_grasp) * 1000
    cs_end = grip_contacts(m, d, g)
    ftot_end = sum(c["force"] for c in cs_end)

    print(f"  药盒 Z: 抓取时 {z_at_grasp:.4f} -> 结束时 {p_end[2]:.4f}  (Δ={rise:+.2f} mm)")
    print(f"  药盒 XY 位移  = {np.linalg.norm(p_end[:2]-OBJ_XYZ[:2])*1000:+.2f} mm")
    print(f"  结束接触点    = {len(cs_end)}, 接触力合计 = {ftot_end:.4f} N")
    print(f"  TCP 终点 z    = {d.site_xpos[g['tcp']][2]:.4f}")

    if not cs_end:
        print("  ❌ 抓取失败：夹爪与药盒完全脱离")
        if rise < 1.0:
            print("     -> 药盒一直没离开桌面，说明夹爪全程没夹住（开口/高度问题）")
        else:
            print("     -> 药盒曾抬起又滑脱（摩擦不足）")
    elif rise > 5.0:
        print("  ✅ 抓取成功：药盒随夹爪上升，接触力持续存在")
    elif ftot_end > 1e-6:
        print("  ⚠️  有接触但药盒没跟上：正在打滑（摩擦不足或夹持点太低）")
    else:
        print("  ⚠️  接触力已归零：药盒滑脱")

    a = g["obj_adr"]
    print(f"\n  药盒末态 qpos = {np.round(d.qpos[a:a+7], 5)}")


if __name__ == "__main__":
    main()
