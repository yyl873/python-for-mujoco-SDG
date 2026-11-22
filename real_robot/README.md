# real_robot —— 真机侧 ROS2 bringup（本仓库唯一碰 ROS 的地方）

这个目录放的是 **SO-ARM101 真机的 ROS2 bringup 包**（URDF/xacro + MoveIt2 配置 + ros2_control
+ RViz + 一个 MoveIt2 动作客户端），用于在真机上部署与验证本仓库训练出来的策略。

```
real_robot/
└── so_arm101_bringup/       # ROS2 包（Apache-2.0，保留原 LICENSE）
    ├── package.xml          # 依赖：rclpy / moveit_py / controller_manager_msgs / xacro / rviz2 …
    ├── urdf/  meshes/       # 真机 URDF + 13 个 STL（package://so_arm101_bringup/meshes/…）
    ├── launch/              # control.launch.py（控制）/ display.launch.py（只显示）
    ├── config/              # joint_limits / kinematics / moveit / ros2_controllers / srdf / ompl
    ├── rviz/                # 可视化配置
    └── test/                # ROS 模板 lint 测试 + MoveIt2 客户端原型（现名 test/aaa.py）
```

## 与仿真管线的关系（重要）

本仓库主线的**红线 #1「零 ROS」只约束仿真与数据链路**（`ik/ grasp/ render/ collect/ tools/ rl/`）：
那条链路里不出现 `rclpy`/`moveit`，是因为它必须是可复现、无中间件的纯物理管线。

真机侧必然是 ROS 的——所以它被**隔离在 `real_robot/`**，与仿真管线零耦合：仿真不 import 它，
它也不 import 仿真。数据流是单向的：

```
仿真/训练（零 ROS）→ ckpt → 真机部署（real_robot/，ROS2）→ 真机数据回流（可选）→ 再训练
```

## 怎么用

```bash
# 把包放进你的 ROS2 工作区
mkdir -p ~/ros2_ws/src && cp -r real_robot/so_arm101_bringup ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select so_arm101_bringup && source install/setup.bash

# 只看模型（不连硬件）
ros2 launch so_arm101_bringup display.launch.py

# 连硬件跑控制
ros2 launch so_arm101_bringup control.launch.py
```

依赖：ROS2（MoveIt2 生态）+ `moveit_py` / `controller_manager_msgs` / `xacro` / `rviz2`。

## 已知待办（发布前）

- [ ] `test/aaa.py` 是一个真实的 MoveIt2 动作客户端（302 行），文件名是临时名 → 建议改成
      `scripts/moveit_client.py` 并移出 `test/`（`test/` 在 ROS 约定里只放 lint/单测）。
- [ ] 原工作区的 `mjcf/` 目录**没有**复制进来：它不被任何 launch/config 引用，且仿真侧的
      MJCF 已在主仓 `model/`。需要的话一条命令补回：
      `cp -r ~/mujoco_ws/src/so_arm101_bringup/mjcf real_robot/so_arm101_bringup/`
- [ ] 真机闭环评测（20 次）还没有做——这是主仓 README「下一步」第 5 条。

## 许可

`so_arm101_bringup/` 保留原 **Apache-2.0** LICENSE（ROS 包模板许可），与主仓 MIT 并存，
以各自目录内的 LICENSE 为准。
