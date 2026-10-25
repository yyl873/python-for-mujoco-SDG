import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import JointState
from moveit_msgs.msg import PlanningScene, CollisionObject
from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header
from builtin_interfaces.msg import Duration
import mujoco
import numpy as np
import time
import random
import math

JOINT_LIMITS_LOWER = np.array([-1.91986, -1.74533, -1.69, -1.6, -2.3])
JOINT_LIMITS_UPPER = np.array([ 1.91986,  1.74533,  1.54,  1.6,  2.3])

class SDGRecorder(Node):
    JOINT_NAMES = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_flex_joint',
        'wrist_flex_joint', 'wrist_roll_joint', 'gripper_joint',
    ]

    def __init__(self):
        super().__init__('sdg_recorder_node')
        # 只保留场景发布器，去掉了没用的 MoveGroup
        qos = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', qos)
        self.traj_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        self.joint_positions_dict = {n: 0.0 for n in self.JOINT_NAMES}
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 10)

        self.get_logger().info("加载 MuJoCo 模型...")
        self.mj_model = mujoco.MjModel.from_xml_path(
            "install/so_arm101_bringup/share/so_arm101_bringup/mjcf/scene.xml"
        )
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model, height=480, width=640)

        self.obj_qpos_adr = self.mj_model.jnt_qposadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        ]
        self.obj_dof_adr = self.mj_model.jnt_dofadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "obj_free")
        ]
        self.gripper_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.jaw_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "jaw_link")
        self.gripper_link_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link")

        obj_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "object")
        box_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        self.obj_pos = self.mj_model.body_pos[obj_body_id].copy()
        self.box_pos = self.mj_model.body_pos[box_body_id].copy()

        self.is_attached = False
        self.attach_local_offset = np.zeros(3)
        self.attach_rel_pos = np.zeros(3)
        self.attach_rel_mat = np.eye(3)
        self.attach_follow_gripper = False
        self.prev_obj_quat = None
        self.released = False  # 💡 新增：是否已经释放

        self.images_env, self.images_wrist, self.actions = [], [], []
        self.recording = False
        self.timer = None

    def joint_cb(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joint_positions_dict[n] = float(p)

    def get_arm_q(self):
        return [self.joint_positions_dict[n] for n in self.JOINT_NAMES]

    def add_box_to_scene(self, name, x, y, z, size_xyz, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        scene = PlanningScene()
        scene.is_diff = True
        obj = CollisionObject()
        obj.id = name
        obj.header = Header(frame_id="base_link")
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x = float(qx)
        pose.orientation.y = float(qy)
        pose.orientation.z = float(qz)
        pose.orientation.w = float(qw)
        primitive = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=size_xyz)
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        scene.world.collision_objects.append(obj)
        self.scene_pub.publish(scene)

    def set_gripper(self, val):
        goal = FollowJointTrajectory.Goal()
        traj = JointTrajectory()
        traj.joint_names = ["gripper_joint"]
        point = JointTrajectoryPoint()
        point.positions = [float(val)]
        point.time_from_start = Duration(sec=1, nanosec=0)
        traj.points.append(point)
        goal.trajectory = traj
        self.traj_client.wait_for_server()
        fut = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh and gh.accepted:
            rf = gh.get_result_async()
            rclpy.spin_until_future_complete(self, rf)

    def start_recording(self):
        self.recording = True
        self.timer = self.create_timer(1.0 / 30.0, self.record_frame)

    def stop_recording(self):
        self.recording = False
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    def record_frame(self):
        if not self.recording:
            return

        # 1. 同步机械臂状态
        for i, name in enumerate(self.JOINT_NAMES):
            self.mj_data.qpos[i] = self.joint_positions_dict[name]
        
                # 💡 修改：只有在“没抓取”且“没释放”的时候，才把它锁在原地
        if not self.is_attached and not self.released and hasattr(self, 'initial_obj_qpos') and self.initial_obj_qpos is not None:
            self.mj_data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 7] = self.initial_obj_qpos
            self.mj_data.qvel[self.obj_dof_adr : self.obj_dof_adr + 6] = 0.0
            
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # =========================================================================

        # 3. 逻辑层：状态机（Kinematic 覆盖）
        if self.is_attached:
            # ... 你原来的相对绑定代码保持不变 ...
            grip_pos = self.mj_data.site_xpos[self.gripper_site_id].copy()
            grip_mat = self.mj_data.site_xmat[self.gripper_site_id].reshape(3, 3)
            new_obj_pos = grip_pos + grip_mat @ self.attach_rel_pos
            new_obj_mat = grip_mat @ self.attach_rel_mat
            
            self.mj_data.qpos[self.obj_qpos_adr : self.obj_qpos_adr + 3] = new_obj_pos
            
            if self.attach_follow_gripper:
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, new_obj_mat.flatten())
                if self.prev_obj_quat is not None and np.dot(quat, self.prev_obj_quat) < 0:
                    quat = -quat
                self.prev_obj_quat = quat.copy()
                self.mj_data.qpos[self.obj_qpos_adr + 3 : self.obj_qpos_adr + 7] = quat
            
            self.mj_data.qvel[self.obj_dof_adr : self.obj_dof_adr + 6] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)

        # 4. 渲染并隐藏碰撞体
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0 
        self.renderer.update_scene(self.mj_data, camera="env_camera", scene_option=opt)
        self.images_env.append(self.renderer.render().copy())
        self.renderer.update_scene(self.mj_data, camera="wrist_camera", scene_option=opt)
        self.images_wrist.append(self.renderer.render().copy())
        self.actions.append(np.rad2deg(self.get_arm_q()))

    def send_joint_waypoints(self, start_pos, end_pos, duration_sec=2.5, steps=80):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.JOINT_NAMES)
        t_vals = np.linspace(0.0, duration_sec, steps + 1)
        s, e = np.array(start_pos, dtype=float), np.array(end_pos, dtype=float)
        for t in t_vals:
            a = t / duration_sec
            pt = JointTrajectoryPoint()
            pt.positions = ((1 - a) * s + a * e).tolist()
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t - int(t)) * 1e9))
            goal.trajectory.points.append(pt)
        self.traj_client.wait_for_server()
        fut = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()
        if gh is None or not gh.accepted:
            return False
        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        return True

    def move_to_xyz_ik(self, tx, ty, tz, duration=3.0):
        """纯 Python IK 极速移动，彻底抛弃 MoveIt"""
        q_now = self.get_arm_q()
        q_ik, err, _ = self.solve_ik_to_xyz([tx, ty, tz], np.array(q_now[:5]))
        
        if err > 0.03:
            self.get_logger().error(f"❌ IK 无法到达: ({tx:.3f},{ty:.3f},{tz:.3f}), 残差={err*1000:.1f}mm")
            return False
            
        q_target = list(q_ik) + [q_now[5]]
        self.get_logger().info(f"⚡ IK 到达: ({tx:.3f},{ty:.3f},{tz:.3f})，耗时 {duration}秒")
        self.send_joint_waypoints(q_now, q_target, duration_sec=duration, steps=80)
        return True

    def _set_qpos_for_eval(self, q5):
        for i in range(5):
            self.mj_data.qpos[i] = float(np.clip(q5[i], JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]))
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _site_axes_world(self):
        gmat = self.mj_data.site_xmat[self.gripper_site_id].reshape(3, 3)
        return gmat[:, 0], gmat[:, 1], gmat[:, 2]

    def solve_ik_3dof_position(self, target_xyz, q_seed_5, max_iter=100, tol=0.003):
        qpos_backup = self.mj_data.qpos.copy()
        qvel_backup = self.mj_data.qvel.copy()
        self._set_qpos_for_eval(q_seed_5)
        target = np.array(target_xyz, dtype=float)
        for _ in range(max_iter):
            grip_pos = self.mj_data.site_xpos[self.gripper_site_id].copy()
            err = target - grip_pos
            if np.linalg.norm(err) < tol:
                break
            jacp = np.zeros((3, self.mj_model.nv))
            jacr = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.gripper_site_id)
            J = jacp[:, :3]
            lam = 0.05
            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            dq = np.clip(J.T @ np.linalg.solve(JJt, err), -0.2, 0.2)
            for i in range(3):
                self.mj_data.qpos[i] = float(np.clip(self.mj_data.qpos[i] + dq[i], JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]))
            mujoco.mj_forward(self.mj_model, self.mj_data)
        q_result = self.mj_data.qpos[:5].copy()
        final_pos = self.mj_data.site_xpos[self.gripper_site_id].copy()
        err = float(np.linalg.norm(target - final_pos))
        self.mj_data.qpos[:] = qpos_backup
        self.mj_data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return q_result, err

    def solve_grasp_pose(self, target_xyz, obj_y_dir, q_pre):
        obj_y = np.array(obj_y_dir, dtype=float) / np.linalg.norm(obj_y_dir)
        candidates = []
        for wf_deg in [30, 45, 60, 75, 90, 100]:
            for wr_deg in [-180, -135, -90, -45, 0, 45, 90, 135, 180]:
                q_seed = np.array(q_pre[:5], dtype=float)
                q_seed[3] = np.deg2rad(wf_deg)
                q_seed[4] = np.deg2rad(wr_deg)
                q_ik, p_err = self.solve_ik_3dof_position(target_xyz, q_seed)
                if p_err > 0.025:
                    continue

                qpos_backup = self.mj_data.qpos.copy()
                self._set_qpos_for_eval(q_ik)
                sx, sy, sz = self._site_axes_world()
                x_align = float(abs(np.dot(sz, obj_y)))
                z_align = float(abs(sx[2]))
                if z_align < 0.90:
                    continue
                self.mj_data.qpos[:] = qpos_backup
                mujoco.mj_forward(self.mj_model, self.mj_data)

                candidates.append({'q': q_ik.copy(), 'p_err': p_err, 'x_align': x_align, 'z_align': z_align, 'wf': wf_deg, 'wr': wr_deg})

        candidates.sort(key=lambda c: (1.2 * c['z_align'] + c['x_align']) / (1 + 10 * c['p_err']), reverse=True)
        return candidates

    def solve_ik_to_xyz(self, target_xyz, q_seed, max_iter=200, tol=0.003):
        qpos_backup = self.mj_data.qpos.copy()
        qvel_backup = self.mj_data.qvel.copy()
        self._set_qpos_for_eval(q_seed[:5])
        target = np.array(target_xyz, dtype=float)
        final_err, final_pos = 1e9, None
        for _ in range(max_iter):
            grip_pos = self.mj_data.site_xpos[self.gripper_site_id].copy()
            err = target - grip_pos
            final_err = float(np.linalg.norm(err))
            final_pos = grip_pos
            if final_err < tol:
                break
            jacp = np.zeros((3, self.mj_model.nv))
            jacr = np.zeros((3, self.mj_model.nv))
            mujoco.mj_jacSite(self.mj_model, self.mj_data, jacp, jacr, self.gripper_site_id)
            J = jacp[:, :5]
            lam = 0.05
            JJt = J @ J.T + (lam ** 2) * np.eye(3)
            dq = np.clip(J.T @ np.linalg.solve(JJt, err), -0.2, 0.2)
            for i in range(5):
                self.mj_data.qpos[i] = float(np.clip(self.mj_data.qpos[i] + dq[i], JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]))
            mujoco.mj_forward(self.mj_model, self.mj_data)
        q_result = self.mj_data.qpos[:5].copy()
        self.mj_data.qpos[:] = qpos_backup
        self.mj_data.qvel[:] = qvel_backup
        mujoco.mj_forward(self.mj_model, self.mj_data)
        return q_result, final_err, final_pos


def main():
    import math
    rclpy.init()
    node = SDGRecorder()
    
        # ================= 1. 场景与全姿态随机化 (确保直立) =================
    pad_id = mujoco.mj_name2id(node.mj_model, mujoco.mjtObj.mjOBJ_BODY, "pick_zone")
    pad_x = float(node.mj_model.body_pos[pad_id][0])
    pad_y = float(node.mj_model.body_pos[pad_id][1])
    
    # 垫子内极坐标随机位置
    obj_x = pad_x + random.uniform(-0.03, 0.03)
    obj_y = pad_y + random.uniform(-0.03, 0.03)
    
    # 💡 生成绕 Z 轴随机旋转的四元数 (w, x, y, z)
    obj_yaw = random.uniform(-np.pi, np.pi)
    qw = np.cos(obj_yaw / 2.0)
    qz = np.sin(obj_yaw / 2.0)
    
    box_x = float(node.box_pos[0])
    box_y = float(node.box_pos[1])

    # 强行赋予物理引擎：指定位置 + 强制直立
    node.mj_data.qpos[node.obj_qpos_adr : node.obj_qpos_adr + 3] = [obj_x, obj_y, 0.028]
    # ⚠️ 修复：MuJoCo 的四元数顺序是 [w, x, y, z]！之前写反了导致物块扭曲翻倒！
    node.mj_data.qpos[node.obj_qpos_adr + 3 : node.obj_qpos_adr + 7] = [qw, 0.0, 0.0, qz]
    
    mujoco.mj_forward(node.mj_model, node.mj_data)
    
    # 💡 保存初始绝对直立的姿态
    node.initial_obj_qpos = node.mj_data.qpos[node.obj_qpos_adr : node.obj_qpos_adr + 7].copy()

    # 4. 更新 MoveIt 场景（MoveIt 的四元数顺序是 x,y,z,w）
    node.add_box_to_scene("table", 0.18, -0.005, 0.0, [0.5, 0.7, 0.04])
    node.add_box_to_scene("object_1", obj_x, obj_y, 0.028, [0.073, 0.052, 0.015], 0.0, 0.0, qz, qw)
    # node.add_box_to_scene("target_box", box_x, box_y, 0.045, [0.115, 0.14, 0.05])
    
    time.sleep(1.0)

            # ================= 雷达测距：找药盒最高点 =================
    from_pt = np.array([obj_x, obj_y, 0.30]) 
    vec = np.array([0.0, 0.0, -1.0])
    geom_id = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(node.mj_model, node.mj_data, from_pt, vec, None, 1, -1, geom_id)
    
    if dist > 0:
        surface_z = 0.30 - dist  # 测到药盒最顶端
    else:
        surface_z = 0.028 + 0.026 # 如果没测到，用药盒中心0.028 + 半高0.026

    # 你的想法：抓取点设在顶部偏下一点点，刚好包住上部
    # 药盒总高 5.2cm，顶面往下 1.5cm 处抓取最稳
    obj_grasp_z = surface_z - 0.015 
    
    # 平移的安全高度：比最顶端再高 8 厘米，保证平移时绝对不碰
    safe_z = surface_z + 0.08  
    
    node.get_logger().info(f"🎯 雷达测出药盒顶端 Z={surface_z:.4f}，平移高度 Z={safe_z:.4f}，抓取高度 Z={obj_grasp_z:.4f}")
    
    # ================= 3. 强制开机初始姿态 =================
    node.get_logger().info("🤖 正在移动到预设初始姿态...")
    initial_pose = [0.0, -1.74535, 1.54036, 1.23277, -1.80689, 0.0] 
    q_now = node.get_arm_q()
    node.send_joint_waypoints(q_now, initial_pose, duration_sec=3.0, steps=60)
    time.sleep(1.0) 
    
    node.start_recording()
    node.set_gripper(0.8)
    time.sleep(0.5)

    # ================= 动作 1：先垂直抬升到绝对安全高度 =================
    safe_z = 0.20  # 桌面 20cm 以上的绝对安全高度
    node.get_logger().info(f"动作1: 原地垂直抬升到安全高度 Z={safe_z}...")
    
    q_now = node.get_arm_q()
    current_pos = node.mj_data.site_xpos[node.gripper_site_id].copy()
    
    # 计算当前 XY 位置下，Z=0.20 的关节角
    q_ik, _, _ = node.solve_ik_to_xyz([current_pos[0], current_pos[1], safe_z], np.array(q_now[:5]))
    # 平滑抬升
    node.send_joint_waypoints(q_now, list(q_ik) + [q_now[5]], duration_sec=2.0, steps=60)
    time.sleep(0.5)

        # ================= 动作 2：求解姿态 + 高空平移 =================
    node.get_logger().info(f"动作2: 在高空对齐物块旋转姿态...")
    import math
    target_dir = [-math.sin(obj_yaw), math.cos(obj_yaw), 0.0]
    
    q_now = node.get_arm_q()
    # 1. 先算出最终抓取点（较低位置）的完美姿态 q_grasp
    candidates = node.solve_grasp_pose([obj_x, obj_y, obj_grasp_z], target_dir, q_now)
    if len(candidates) == 0:
        node.get_logger().error("❌ 找不到垂直解！"); return
    best = candidates[0]
    q_grasp = list(best['q'])
    
    # 2. 💡 核心大招：用抓取姿态作为种子，去反推安全高度 safe_z 的关节角
    # 这样夹爪飞到天上时，就已经是对准物块的角度了！
    q_above_ik, _, _ = node.solve_ik_to_xyz([obj_x, obj_y, safe_z], np.array(q_grasp))
    node.send_joint_waypoints(q_now, list(q_above_ik) + [q_now[5]], duration_sec=2.5, steps=80)
    time.sleep(0.5)



        # ================= 动作 3：精准下探（完美局部偏移补偿） =================
    node.get_logger().info("动作3: 计算完美补偿点并垂直下探...")
    import math
    
    # 💡 这是你之前测试的最完美值！
    perfect_local_offset = np.array([-0.004, 0.0, 0.01])
    
    # 构造世界坐标系下的旋转矩阵（因为物块随机 Yaw 旋转）
    cos_yaw = math.cos(obj_yaw)
    sin_yaw = math.sin(obj_yaw)
    
    # 把局部偏移转换成世界坐标偏移
    offset_x_world = perfect_local_offset[0] * cos_yaw - perfect_local_offset[2] * sin_yaw
    offset_y_world = perfect_local_offset[0] * sin_yaw + perfect_local_offset[2] * cos_yaw
    offset_z_world = perfect_local_offset[1]  # 因为只有平面旋转，Y 轴（局部Z轴）保持不变
    
    # 💡 核心：机械臂下潜目标 = 药盒中心 - 世界偏移
    # 这样当 gripperframe 到达该点时，夹爪的完美区域正好包住药盒！
    target_x = obj_x - offset_x_world
    target_y = obj_y - offset_y_world
    target_z = obj_grasp_z - offset_z_world
    
    # 计算物块旋转后的朝向
    target_dir = [-math.sin(obj_yaw), math.cos(obj_yaw), 0.0]
    
    q_now = node.get_arm_q()
    # 用补偿后的目标点去解算 IK
    candidates = node.solve_grasp_pose([target_x, target_y, target_z], target_dir, q_now)
    
    if len(candidates) == 0:
        node.get_logger().error("❌ 找不到垂直解！")
        return
        
    best = candidates[0]
    q_grasp = list(best['q']) + [q_now[5]]
    
    # 平滑下探
    node.send_joint_waypoints(q_now, q_grasp, duration_sec=2.0, steps=60)
    time.sleep(0.5)

        # ================= 动作 4：闭合夹爪 =================
    node.get_logger().info("动作4: 闭合夹爪并动态绑定")
    node.set_gripper(0.15)
    time.sleep(0.5)
    
    grip_pos = node.mj_data.site_xpos[node.gripper_site_id].copy()
    grip_mat = node.mj_data.site_xmat[node.gripper_site_id].reshape(3, 3)
    obj_pos = node.mj_data.qpos[node.obj_qpos_adr : node.obj_qpos_adr + 3].copy()
    obj_quat = node.mj_data.qpos[node.obj_qpos_adr + 3 : node.obj_qpos_adr + 7].copy()
    obj_mat = np.zeros(9)
    mujoco.mju_quat2Mat(obj_mat, obj_quat) 
    obj_mat = obj_mat.reshape(3, 3)
    
    # 此时的动态偏移，会极其接近你的完美值！
    node.attach_rel_pos = grip_mat.T @ (obj_pos - grip_pos)
    node.attach_rel_mat = grip_mat.T @ obj_mat
    
    node.attach_follow_gripper = True
    node.is_attached = True

    # ================= 动作 5：垂直拔起 =================
    node.get_logger().info("动作5: 垂直拔起...")
    q_now = node.get_arm_q()
    # 保持当前 XY 不动，直接拔高到 safe_z
    q_lift_ik, _, _ = node.solve_ik_to_xyz([obj_x, obj_y, safe_z], np.array(q_now[:5]))
    node.send_joint_waypoints(q_now, list(q_lift_ik) + [q_now[5]], duration_sec=2.0, steps=60)
    time.sleep(0.5)

        # 直接定义目标盒子的下放高度
    place_above_z = 0.15  # 木盒上方的安全平移高度
    place_z = 0.08        # 放入木盒内部的高度（防止撞侧壁）
            # ================= 动作 6：飞到木盒正上方高空，并下降到释放高度 =================
    place_above_z = 0.15  # 高空平移高度
    place_z = 0.08        # 释放高度（木盒内部上方）
    
    node.get_logger().info("动作6: 飞到木盒正上方高空...")
    q_now = node.get_arm_q()
    q_box_above_ik, _, _ = node.solve_ik_to_xyz([box_x, box_y, place_above_z], np.array(q_now[:5]))
    node.send_joint_waypoints(q_now, list(q_box_above_ik) + [q_now[5]], duration_sec=2.0, steps=60)
    time.sleep(0.3)

    node.get_logger().info("动作6.5: 下降到释放高度...")
    q_now = node.get_arm_q()
    q_box_place_ik, _, _ = node.solve_ik_to_xyz([box_x, box_y, place_z], np.array(q_now[:5]))
    node.send_joint_waypoints(q_now, list(q_box_place_ik) + [q_now[5]], duration_sec=1.5, steps=45)
    time.sleep(0.3)

    # ================= 动作 7：释放，触发真实物理自由落体 =================
    node.get_logger().info("动作7: 高空释放，让物块物理自由落体...")
    
    # 1. 解除视觉绑定（让物理引擎接管）
    node.is_attached = False
    node.attach_follow_gripper = False
    node.released = True  # 告诉 record_frame 不要再锁定物体
    
    # 2. 💡 核心修复：强行把物理坐标精准同步到夹爪释放点
    node.mj_data.qpos[node.obj_qpos_adr] = box_x
    node.mj_data.qpos[node.obj_qpos_adr + 1] = box_y
    node.mj_data.qpos[node.obj_qpos_adr + 2] = place_z  # 之前漏了这句！高度必须在 place_z
    node.mj_data.qpos[node.obj_qpos_adr + 3 : node.obj_qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0] # 绝对水平
    
    # 3. 赋予一个微小的向下初速度，模拟机械臂松手的惯性
    node.mj_data.qvel[node.obj_dof_adr : node.obj_dof_adr + 3] = [0.0, 0.0, -0.1]
    node.mj_data.qvel[node.obj_dof_adr + 3 : node.obj_dof_adr + 6] = [0.0, 0.0, 0.0]
    
    # 4. 张开夹爪
    node.set_gripper(0.8)
    
    # 5. 💡 极致操作：用物理引擎推进下落，并配合 ROS 2 定时器录制
    node.get_logger().info("等待物块真实物理掉落...")
    drop_steps = 45  # 模拟 1.5 秒的下落与反弹动画
    # 计算每一帧需要执行多少次物理步长 (MuJoCo 默认 timestep 是 0.002s，30fps 需要步进约 16 次)
    steps_per_frame = max(1, int(1.0 / (30.0 * node.mj_model.opt.timestep)))
    
    for i in range(drop_steps):
        # 必须强制同步机械臂的关节，否则物理引擎会让机械臂垮掉
        for j, name in enumerate(node.JOINT_NAMES):
            node.mj_data.qpos[j] = node.joint_positions_dict[name]
            
        # 推进物理引擎，让重力和碰撞发力（产生真实的掉落在盒子里弹跳的效果）
        for _ in range(steps_per_frame):
            mujoco.mj_step(node.mj_model, node.mj_data)
            
        # 💡 这一步极其关键！用 spin_once 触发 record_frame 定时器，确保每一帧都被录进视频
        rclpy.spin_once(node, timeout_sec=1.0 / 30.0)

    # ================= 动作 7.5：原地垂直撤离木盒 =================
    node.get_logger().info("动作7.5: 原地垂直抬升撤离")
    
    current_site_pos = node.mj_data.site_xpos[node.gripper_site_id].copy()
    current_x = current_site_pos[0]
    current_y = current_site_pos[1]
    target_z = current_site_pos[2] + 0.10  # 抬升 10cm
    
    q_now = node.get_arm_q()
    q_retreat, _, _ = node.solve_ik_to_xyz([current_x, current_y, target_z], np.array(q_now[:5]))
    node.send_joint_waypoints(q_now, list(q_retreat) + [q_now[5]], duration_sec=1.0, steps=30)
    time.sleep(0.5)
    # ================= 动作 8：恢复到指定的结束姿态 =================
    node.get_logger().info("动作8: 退回预设结束姿态")
    end_pose = [0.0, -1.74535, 1.54036, 1.23277, -1.80689, 0.0] 
    q_now = node.get_arm_q()
    node.send_joint_waypoints(q_now, end_pose, duration_sec=3.0, steps=60)
    time.sleep(1.0)

    node.stop_recording()

    if len(node.actions) > 0:
        np.savez("episode_0.npz", env=np.array(node.images_env), wrist=np.array(node.images_wrist), action=np.array(node.actions))
        node.get_logger().info(f"✅ 数据录制成功！共 {len(node.actions)} 帧。")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
