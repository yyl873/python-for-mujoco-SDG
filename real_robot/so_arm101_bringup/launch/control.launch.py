import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch import LaunchDescription
import xacro

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return {}

def generate_launch_description():
    pkg_name = 'so_arm101_bringup'
    pkg_path = get_package_share_directory(pkg_name)
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    # hardware_type = LaunchConfiguration('ros2_control_hardware_type', default='mujoco')
    xacro_file = os.path.join(pkg_path, 'urdf', 'so_arm101.urdf.xacro')
    controllers_yaml = os.path.join(pkg_path, 'config', 'ros2_controllers.yaml')
    srdf_file = os.path.join(pkg_path, 'config', 'so_arm101.srdf')
    rviz_config_file = os.path.join(pkg_path, 'rviz', 'config.rviz')

    robot_description_raw = xacro.process_file(
        xacro_file,
        mappings={'ros2_control_hardware_type': 'mujoco'}
    ).toxml()

    with open(srdf_file, 'r') as f:
        robot_description_semantic = f.read()

    kinematics_dict = load_yaml(pkg_name, 'config/kinematics.yaml')
    joint_limits_dict = load_yaml(pkg_name, 'config/joint_limits.yaml')
    moveit_controllers_dict = load_yaml(pkg_name, 'config/moveit_controllers.yaml')
    trajectory_execution_dict = load_yaml(pkg_name, 'config/trajectory_execution.yaml')
    moveit_dict = load_yaml(pkg_name, 'config/moveit.yaml')
    ompl_dict = load_yaml(pkg_name, 'config/ompl_planning.yaml')

    # 注意：不再设置 ompl_dict['planning_plugin']
    # ompl_planning.yaml 里已有 planning_plugins 列表
    ompl_pipeline_params = {'ompl': ompl_dict}

    moveit_params = [
        {'robot_description': robot_description_raw},
        {'robot_description_semantic': robot_description_semantic},
        kinematics_dict,
        joint_limits_dict,
        moveit_controllers_dict,
        trajectory_execution_dict,
        moveit_dict,
        {'default_planning_pipeline': 'ompl'},
        {'planning_pipelines': ['ompl']},
        ompl_pipeline_params,
        {'use_sim_time': use_sim_time}
    ]

    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description_raw, 'use_sim_time': use_sim_time}]
    )

    robot_control_node = Node(
        package='mujoco_ros2_control',
        executable='ros2_control_node',
        parameters=[
            {'robot_description': robot_description_raw},
            controllers_yaml,
            {'use_sim_time': use_sim_time}
        ],
        output='screen'
    )

    jsb_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
        output='screen'
    )

    arm_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller'],
        output='screen'
    )

    # 不再 spawn gripper_controller：夹爪已并入 joint_trajectory_controller

    move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        parameters=moveit_params,
        output='screen'
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
        parameters=moveit_params,
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', description='Use simulation time'),
        robot_state_publisher_node,
        DeclareLaunchArgument('ros2_control_hardware_type', default_value='mujoco', description='Hardware type: mock_components, mujoco, real, gazebo'),
        TimerAction(period=2.0, actions=[robot_control_node]),
        TimerAction(period=4.0, actions=[jsb_spawner, arm_spawner]),
        TimerAction(period=7.0, actions=[move_group_node]),
        TimerAction(period=9.0, actions=[rviz_node])
    ])