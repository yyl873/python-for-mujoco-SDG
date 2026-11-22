# ================= 1. 依赖库导入区 =================
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (MotionPlanRequest, Constraints, RobotState, PositionConstraint, AttachedCollisionObject, CollisionObject, OrientationConstraint)
from geometry_msgs.msg import Pose, Vector3
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import BoundingVolume
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.duration import Duration

import mujoco
import numpy as np
import cv2
import os
import math
import random
import time

# ================= 2. 配置区 =================
FPS = 30
WIDTH, HEIGHT = 640, 480
# [⚠️需验证] 这里的顺序必须和你的 ros2_control 里配置的 joints 顺序绝对一致！
JOINT_NAMES = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_flex_joint', 
               'wrist_flex_joint', 'wrist_roll_joint', 'gripper_joint']
# ==========================================

class SDGGenerator(Node):
    def __init__(self):
        super().__init__('sdg_generator')
        # [🔒固定写法] 建立与 MoveIt 通信的 Action 客户端
        self.action_client = ActionClient(self, MoveGroup, '/move_action')
        self.js = None
        self.create_subscription(JointState, '/joint_states', self._cb, 10)
        
        # [🔒固定写法] 发布控制指令的 Publisher
        self.gripper_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.attached_pub = self.create_publisher(AttachedCollisionObject, '/attached_collision_object', 10)
        self.scene_pub = self.create_publisher(CollisionObject, '/collision_object', 10)

    # [🔒固定写法] 不断更新获取当前最新的关节角
    def _cb(self, msg):
        self.js = msg

    # [⚙️核心逻辑] 构建 MoveIt 的位姿约束
    def create_constraint(self, x, y, z):
        c = Constraints()
        pc = PositionConstraint()
        pc.header.frame_id = 'base_link'
        pc.link_name = 'gripper_link'
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        
        # [⚠️需验证] 这里的尺寸是允许的位置误差容差 (1cm)
        region = BoundingVolume()
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [0.01, 0.01, 0.01] 
        
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        region.primitives.append(primitive)
        region.primitive_poses.append(pose)
        pc.constraint_region = region
        c.position_constraints.append(pc)

        # ================= 姿态约束 (强制夹爪朝下) =================
        oc = OrientationConstraint()
        oc.header.frame_id = 'base_link'
        oc.link_name = 'gripper_link'
        # [⚠️需验证] 这个四元数 (w:0.707, x:0.707) 对应绕 X 轴转 90度，具体对不对取决于你的 TF 树！
        # 如果你发现机械臂倒立去抓，你要把这个值改成 (w:1, x:0, y:0, z:0) 或者别的。
        oc.orientation.w = 0.707
        oc.orientation.x = 0.707
        oc.orientation.y = 0.0
        oc.orientation.z = 0.0
        # [⚠️需验证] 允许的旋转误差 (弧度)，0.2 大约等于 11度
        oc.absolute_x_axis_tolerance = 0.2
        oc.absolute_y_axis_tolerance = 0.2
        oc.absolute_z_axis_tolerance = 0.2
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        # ========================================================
        return c

    # [⚙️核心逻辑] 发送规划请求并执行
    def plan_and_execute(self, constraint, label):
        req = MotionPlanRequest()
        req.group_name = 'manipulator' # [⚠️需验证] 必须和你的 SRDF 里的 group 名字一致
        req.num_planning_attempts = 10
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = 0.3
        
        rs = RobotState()
        rs.joint_state = self.js
        req.start_state = rs
        req.goal_constraints = [constraint]

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options.plan_only = False

        print(f"⏳ 规划 [{label}] ...")
        self.action_client.wait_for_server()
        
        # [🔒固定写法] 异步发送 Goal 并阻塞等待结果
        fut = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, fut)
        gh = fut.result()

        if not gh.accepted:
            print(f"❌ [{label}] Goal 被拒绝")
            return False

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        res = rf.result().result

        # [🔒固定写法] 1 代表 SUCCESS
        if res.error_code.val == 1:
            print(f"✅ [{label}] 成功!")
            return True
        else:
            print(f"❌ [{label}] 规划失败, code: {res.error_code.val}")
            return False

    # [⚙️核心逻辑] 控制夹爪开合
    def control_gripper(self, pos, label):
        print(f"🦾 夹爪动作: {label} ...")
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = ['gripper_joint'] # [⚠️需验证] 确认关节名字
        pt = JointTrajectoryPoint()
        pt.positions = [pos] # [⚠️需验证] pos=1.0 是张开，0.0 是闭合，确认你的配置是否如此
        pt.time_from_start = Duration(seconds=1.0).to_msg() # 动作耗时 1 秒
        traj.points.append(pt)
        self.gripper_pub.publish(traj)
        # [🔒固定写法] 必须睡一会儿，等物理引擎把夹爪收拢
        time.sleep(1.5) 

    # [⚙️核心逻辑] 更新 MoveIt 虚拟场景中的物体
    def update_scene(self, name, x, y, z, attach=False):
        aco = AttachedCollisionObject()
        aco.link_name = 'gripper_link' if attach else ''
        aco.object.header.frame_id = 'base_link'
        aco.object.id = name
        aco.object.operation = CollisionObject.ADD
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        # [⚠️需验证] 这是物体的长宽高 (7.3 x 5.2 x 1.5 cm)
        box.dimensions = [0.073, 0.052, 0.015] 
        
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = x, y, z
        pose.orientation.w = 1.0
        aco.object.primitives.append(box)
        aco.object.primitive_poses.append(pose)
        
        if attach:
            # [⚠️需验证] 允许碰撞的连杆列表，防止 MoveIt 以为夹爪在撞自己
            aco.touch_links = ['gripper_link', 'jaw_link', 'wrist_link']
            self.attached_pub.publish(aco)
            print(f"📎 已将物体附着到夹爪")
        else:
            self.scene_pub.publish(aco.object)


# ================= 3. MuJoCo 渲染与状态同步区 =================

# [⚙️核心逻辑] 将当前真实的关节角，强行喂给 MuJoCo 模型
def update_mujoco_from_js(m, d, js):
    for i, name in enumerate(JOINT_NAMES):
        # [🔒固定写法] 通过名字找索引
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid != -1:
            d.qpos[m.jnt_qposadr[jid]] = js.position[i]
    mujoco.mj_forward(m, d) # [🔒固定写法] 让物理引擎计算更新坐标

# [⚙️核心逻辑] 截图并保存
def render_snapshot(m, d, episode_id, step_name, is_attached, obj_qpos_adr):
    renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)
    # [⚠️需验证] 这里的相机名 'env_cam' 必须和你的 scene.xml 里写的一模一样！
    env_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'env_cam') 
    wrist_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    
    # ================= 核心作弊逻辑：物体吸附 =================
    # 这是为了暂时解决 MuJoCo 不知道物体被抓起来了的问题
    if is_attached:
        grip_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'gripper_link')
        grip_pos = d.xpos[grip_bid]
        # [🚧待优化] 这里直接把物体塞到夹爪下方 5cm 处，非常粗暴。
        # 完美做法是计算相对变换矩阵。
        d.qpos[obj_qpos_adr:obj_qpos_adr+3] = [grip_pos[0], grip_pos[1], grip_pos[2] - 0.05]
        d.qpos[obj_qpos_adr+3:obj_qpos_adr+7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
    # ========================================================

    os.makedirs(f"dataset/episode_{episode_id}", exist_ok=True)
    
    # 渲染环境相机
    renderer.update_scene(d, camera=env_cam_id)
    img_env = renderer.render()
    img_env = cv2.flip(img_env, 1) # [⚠️需验证] 环境相机是否必须镜像？需要你看图验证
    cv2.imwrite(f"dataset/episode_{episode_id}/{step_name}_env.jpg", cv2.cvtColor(img_env, cv2.COLOR_RGB2BGR))
    
    # 渲染腕部相机
    renderer.update_scene(d, camera=wrist_cam_id)
    img_wrist = renderer.render()
    cv2.imwrite(f"dataset/episode_{episode_id}/{step_name}_wrist.jpg", cv2.cvtColor(img_wrist, cv2.COLOR_RGB2BGR))
    print(f"📸 已生成截图: {step_name}")


def main():
    rclpy.init()
    node = SDGGenerator()
    
    print("等待 /joint_states ...")
    while node.js is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    # 加载 MuJoCo
    xml_path = '/home/yyl/mujoco_ws/src/so_arm101_bringup/mjcf/scene.xml'
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # 获取物体的 freejoint 索引 (名字必须和 XML 里一致)
    obj_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'obj_free')
    obj_qpos_adr = m.jnt_qposadr[obj_jid] if obj_jid != -1 else 7 

    # ================= 1. 物品位置采样 =================
    # 物品1：在半径 7.5cm (0.075m) 的圆内随机
    r = random.uniform(0, 0.075)
    theta = random.uniform(0, 2*math.pi)
    obj1_x = 0.260 + r * math.cos(theta)
    obj1_y = 0.150 + r * math.sin(theta)
    
    # 物品2：目标盒子的固定位置
    obj2_x = 0.212 
    obj2_y = -0.212

    # ================= 2. 关键 Z 轴高度设定 =================
    obj1_center_z = 0.0075 # MuJoCo 里物品1的物理中心高度 (高1.5cm)
    grasp_z = 0.035        # MoveIt 规划抓取时的末端 Z 高度
    hover_z = 0.13         # 抓取悬停高度
    
    place_z = 0.045        # 放入盒子内部的末端 Z 高度
    box_hover_z = 0.15     # 盒子上方悬停高度

    # ================= 3. 同步场景 =================
    # 写入 MuJoCo 物理世界
    # 在 main 函数里：
    d.qpos[obj_qpos_adr:obj_qpos_adr+3] = [obj1_x, obj1_y, 0.0075] 
    mujoco.mj_forward(m, d)
    node.update_scene("target_obj", obj1_x, obj1_y, 0.0075) 
    time.sleep(0.5) 
    node.control_gripper(1.0, "张开")

    # ================= 4. 运动规划与执行状态机 =================
    # 去抓取上方
    if node.plan_and_execute(node.create_constraint(obj1_x, obj1_y, hover_z), "抓取上方"):
        update_mujoco_from_js(m, d, node.js)
        render_snapshot(m, d, 0, "01_approach", False, obj_qpos_adr)
        
        # 下探
        if node.plan_and_execute(node.create_constraint(obj1_x, obj1_y, grasp_z), "下探抓取"):
            update_mujoco_from_js(m, d, node.js)
            render_snapshot(m, d, 0, "02_descend", False, obj_qpos_adr)
            
            # 闭合夹爪并吸附
            node.control_gripper(0.0, "闭合")
            node.update_scene("target_obj", obj1_x, obj1_y, obj1_center_z, attach=True)
            
            # 抬起
            if node.plan_and_execute(node.create_constraint(obj1_x, obj1_y, grasp_z + 0.15), "抬起"):
                update_mujoco_from_js(m, d, node.js)
                render_snapshot(m, d, 0, "03_lift", True, obj_qpos_adr) 
                
                # 飞向盒子上方
                if node.plan_and_execute(node.create_constraint(obj2_x, obj2_y, box_hover_z), "搬运到上方"):
                    update_mujoco_from_js(m, d, node.js)
                    render_snapshot(m, d, 0, "04_box_above", True, obj_qpos_adr)
                    
                    # 下放
                    if node.plan_and_execute(node.create_constraint(obj2_x, obj2_y, place_z), "下放"):
                        update_mujoco_from_js(m, d, node.js)
                        render_snapshot(m, d, 0, "05_place", True, obj_qpos_adr)
                        
                        # 释放
                        node.control_gripper(1.0, "释放")
                        node.update_scene("target_obj", obj2_x, obj2_y, 0.025, attach=False) 
                        
                        # 撤离
                        node.plan_and_execute(node.create_constraint(obj2_x, obj2_y, box_hover_z), "撤离")
                        print("🎉 抓放闭环成功！去 dataset 文件夹看截图！")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
