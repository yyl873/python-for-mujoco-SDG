================================================================================
 meshes/  —— 机械臂 STL 几何体（13 个模型 + 许可证）
================================================================================
【这个目录是干什么的】
    SO-ARM101 机械臂的 CAD 网格。mjcf/so_arm101.xml 用
    <compiler meshdir="../meshes"/> 从这里加载。

    ★ 不要移动或重命名本目录，也不要改目录名 ——
      XML 里的 meshdir 是相对路径，改了模型就加载不了。

【文件清单（13 个 STL）】

  底座 / 固定
    base_so101_v2.stl                 底座主体
    base_motor_holder_so101_v1.stl    底座电机支架
    motor_holder_so101_base_v1.stl    底座电机夹具
    waveshare_mounting_plate_so101_v2.stl   WaveShare 安装板

  大臂 / 小臂
    upper_arm_so101_v1.stl            大臂
    under_arm_so101_v1.stl            小臂（下臂）
    rotation_pitch_so101_v1.stl       俯仰旋转件

  腕部
    wrist_roll_follower_so101_v1.stl  腕部滚转随动件（wrist_camera 就挂在这条链上）
    wrist_roll_pitch_so101_v2.stl     腕部滚转+俯仰件
    motor_holder_so101_wrist_v1.stl   腕部电机夹具

  夹爪
    moving_jaw_so101_v1.stl           活动指（jaw_link，合爪时运动的那一侧）

  舵机
    sts3215_03a_v1.stl                STS3215 舵机外壳（★★ 见下方注意）
    sts3215_03a_no_horn_v1.stl        同款舵机，无输出轴（用于不需要转动的关节）

  LICENSE                             Apache License 2.0（原始模型许可）

【★★ 关于 sts3215_03a_v1.stl 的碰撞设置（重要）】
    这个舵机壳在 so_arm101.xml 里出现了多次。其中 upper_arm_link 和
    lower_arm_link 下的两个实例已经被改成【只保留 visual、退出碰撞】。

    原因：舵机壳从连杆侧面凸出约 2 cm，机械臂下探时会它先于指尖撞到桌面，
          把整条手臂顶偏，造成 TCP 跟踪误差 12.5 mm（IK 算的位姿和物理执行
          的位姿对不上，后面所有仰赖精确到位的动作都会偏）。
    效果：碰撞 geom 总数 23 -> 21。

    这个改动只影响碰撞，不影响画面（网格还在，只是不参与碰撞检测）。

【怎么改几何体参数】
    尺寸/质量/惯量不在 STL 里，在 mjcf/so_arm101.xml 的 <default> 和
    各 <body> 的 <inertial> 里。改之前先读 ../docs/HANDOFF.md §3。

【来源】
    SO-ARM101（TheRobotStudio）开源机械臂，Apache 2.0。
