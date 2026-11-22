from setuptools import find_packages, setup
import os
from glob import glob


def share(sub, pattern):
    """收集 share/<pkg>/<sub> 的文件；目录不存在时返回空列表（不报错）。

    ★ 本仓库里 mjcf/ 未随包：其中的 scene.xml 引用了 lab_tex.png / mousepad_tex.png，
      这两个贴图在主仓 model/ 里，仿真侧 MJCF 也以 model/ 为准。ROS 的 launch/config
      不引用 mjcf，所以留空不影响 colcon build。
    """
    return (os.path.join('share', package_name, sub), sorted(glob(pattern)))

package_name = 'so_arm101_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        share('launch', 'launch/*.launch.py'),
        share('urdf', 'urdf/*'),
        share('meshes', 'meshes/*'),
        share('config', 'config/*'),   # 去处 yaml，否则编译不了 srdf
        share('rviz', 'rviz/*.rviz'),
        share('mjcf', 'mjcf/*.xml'),   # 未随包时为空，不报错
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yyl',
    maintainer_email='yyl@todo.todo',
    description='SO-ARM101 真机 bringup：URDF/xacro + MoveIt2 + ros2_control + MuJoCo SDG 数据生成节点',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 真机数据生成节点（ROS2 ↔ MuJoCo）：需 /joint_states 与 MoveIt2
            'sdg_generator = so_arm101_bringup.sdg_generator:main',
        ],
    },
)
