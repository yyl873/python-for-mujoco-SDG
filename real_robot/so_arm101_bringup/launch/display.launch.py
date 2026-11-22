import os #拼接文件路径
from ament_index_python.packages import get_package_share_directory #找share目录的绝对路径
from launch import LaunchDescription #一个launch文件的大容器，打包所有启动节点
from launch_ros.actions import Node #导入Node动作类，向ros2发送启动节点的指令
import xacro #ros2只认urdf，解析器


def generate_launch_description():#必须是这个名字
    pkg_name = 'so_arm101_bringup'
    pkg_path = get_package_share_directory(pkg_name)#找到我的文件在install里的
    xacro_file = os.path.join(pkg_path, 'urdf', 'so_arm101.urdf.xacro')
    robot_descrition_raw = xacro.process_file(xacro_file).toxml()#读取xacro文件并执行宏计算，身成标准的URDF字符串


    robot_state_publisher_node = Node(
        package = 'robot_state_publisher',#节点所在的ros2包名
        executable = 'robot_state_publisher',#运行的可执行文件
        parameters = [{'robot_description':robot_descrition_raw,
                      'use_sim_time':False}]#不启用仿真时间
    )

    joint_state_publisher_gui_node = Node(
        package = 'joint_state_publisher_gui',#包名
        executable = 'joint_state_publisher_gui',#可执行文件
        parameters = [{'use_sim_time':False}]
    )#弹出一个gui的窗口去拖动模块，向/joint_state发送角度数据
     #robot_state_publisher收到数据更新/tf，Rviz2里机械臂就可以动了。真机就可以不用这个节点
    rviz2_node = Node(
        package = 'rviz2',#包名
        executable = 'rviz2',#可执行文件
        name = 'rviz2',
        output = 'screen'
    )
    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,#接入一个虚拟的控制器去验证之前的urdf是否正确
        rviz2_node
    ])