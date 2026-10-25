================================================================================
 model/  —— 模型定义（场景 + 机械臂 + 贴图）
================================================================================
【这个目录是干什么的】
    整个项目的"物理世界"就定义在这两个 XML 里。改物理行为只动这里，
    Python 脚本一律按名字查表，不硬编码 id。

【文件清单】

  scene.xml          125 行   ★ 场景主文件，所有脚本都加载它
        结构：
            <include file="so_arm101.xml"/>     把机械臂塞进来
            <option iterations=100 tolerance=1e-10 cone=elliptic impratio=10/>
                                                提高物理精度
            <asset>      9 个贴图材质（实验室背景/桌面/鼠标垫/药盒/木盒）
            <worldbody>
                floor           地面平面，pos z=-0.1
                lab_wall        实验室背景墙（contype=0，纯视觉）
                table           桌子，中心 -0.02 使桌面顶面恰好在 Z=0
                pick_zone       圆形拾取垫（蓝色圆柱，药盒初始位置）
                mousepad        鼠标垫（纯装饰）
                object          ★ 药盒，11.5 g，带 freejoint
                target_box      ★ 目标木盒，搬运终点
            <keyframe name="home">
                初始姿态 qpos = 机械臂 6 关节 + 药盒 freejoint 7 维

  so_arm101.xml      461 行  机械臂本体（含 env_camera / wrist_camera）
        结构：
            <compiler angle="radian" meshdir="../meshes"/>   ← STL 从 ../meshes 读
            <default class="sts3215">     舵机参数（kp=150 kv=10 forcerange=±3.35N）
            <worldbody>
                <camera name="env_camera"  pos="0.65 0 0.55" fovy=30/>   固定全景
                base_link -> shoulder_link -> upper_arm_link -> lower_arm_link
                          -> wrist_link -> gripper_link -> jaw_link
                    <camera name="wrist_camera" .../>  挂在 wrist_roll 链上
            <actuator>   6 个 position 舵机
            <contact>    6 条 <exclude>，只排除相邻连杆之间的自碰撞

  *.png              9 张贴图
        lab_tex.png       实验室背景墙
        table_tex.png     桌面
        mousepad_tex.png  鼠标垫
        object_tex1/2/3   药盒背面/正面/侧面
        box_tex2.png      目标木盒

【对模型做过的 5 处修改（已验证，别改回去）】

  ① scene.xml：删掉 5 条 <exclude>
        原本用 <exclude> 把夹爪和药盒之间的碰撞关掉——那是视觉欺骗，
        夹爪根本没碰到药盒也能"抓住"。删掉后才是纯物理。
        现在 scene.xml 里没有 <contact> 块；
        so_arm101.xml 里剩的 6 条 exclude 只排除相邻连杆自碰撞，属于正常建模。

  ② scene.xml 第 79-81 行：软化药盒接触
        solref="0.04 1"（原 0.02 1）、solimp="0.9 0.95 0.001"、condim="4"
        原因：药盒只有 11.5 g，默认接触刚度下夹爪一碰就产生 20.89 N 冲量
              （重力的 185 倍），药盒直接弹飞。
        效果：峰值接触力 20.89 N -> 8.39 N，法向夹持力 0.005 N -> 0.42 N。

  ③ scene.xml 第 101 行：目标木盒从 (0.290,-0.132,0.025) 挪到 (0.440,0.090,0.025)
        原因：夹爪沿 y 夹住药盒的 15mm 厚度。若搬运方向也是 y，药盒会被顺着
              夹口挤出去（实测沿 y 搬运第 334 步掉落，沿 x 全程不掉）。
              把木盒挪到 +x 方向，让搬运变成"垂直于夹持轴"的移动。

  ④ so_arm101.xml：两个电机壳退出碰撞
        upper_arm_link 和 lower_arm_link 下的 sts3215_03a_v1 电机壳只保留
        <geom class="visual">，删掉了对应的 <geom class="collision">。
        原因：电机壳从连杆侧面凸出约 2cm，下探时会先于指尖撞桌面，把手臂顶偏
              （造成 TCP 跟踪误差 12.5 mm）。
        碰撞 geom 总数：23 -> 21。

  ⑤ （2024-11 第六/七轮）指尖碰撞体：加两片平面指垫 + 碰撞位掩码；第七轮改成"贴齐指面 + 不可见"
        so_arm101.xml：gripper_link / jaw_link 下各加一枚 **50x2x40 mm、rgba 全 0（不可见）**的
                       box 指垫（pad_gripper_link / pad_jaw_link），垫面**贴合该指真实内表面**
                       （按药盒足迹窗口取该指最内侧极点反算，参考位姿必须是"没接触污染"的那一帧），
                       并给这两个 body 的**指壳碰撞 mesh** 加上 contype="8" conaffinity="1"。
        scene.xml：药盒 object_physics 加 contype="2" conaffinity="5"。
        位定义：bit0=通用件(臂/桌/垫/木盒/地面) bit1=药盒 bit2=指垫 bit3=指壳。
        效果（实测，tools/probe_contact_audit.py）：
            下探段 指壳/指垫↔药盒接触 92/400 步 -> **0/400 步**
            压盒 6.87 mm -> **0.39 mm**；夹稳倾角 10.95° -> **1.06°**
            夹稳时两指缝 -> **-0.85 / -1.07 mm**（贴合，不再有 4.8~8.5 mm 的悬空缝）
            手指实体在药盒内的最大深度 7.5 mm -> **0.02 mm**（之前的"穿模"）
            抬升 +103.45 mm -> **+122.22 mm**（盒子不再被按进桌面）
        原因：指壳是 C 形件，MuJoCo 用它的**凸包**做碰撞，凸包把 C 的开口填满 ——
              实测完全下探位姿下固定指与药盒距离 **−22.6 mm**，两指之间在碰撞里是实心的，
              所以张开量、抓取高度怎么调都躲不开"先戳盒顶"。详见 docs/ROUND_contact_realism.md
        ⚠️ 指垫的 pos/quat 是按【descend 段末真实到达位姿】反算的；改起点/接近路径/步数配比
           必须用 debug/pad_fit_try.py 重算，否则表现是"下探零接触、但抬升 0（夹空）"。

  ⚠️ 唯一一个"改了会坏"的地方（已回退，别重犯）：
        so_arm101.xml 第 113 行 gripper_joint 的 axis 必须保持 "0 0 1"。
        曾误改成 "0.98384 0.17903 0"，结果夹爪变成水平旋转、完全无法闭合。

【怎么可视化】
    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 -m mujoco.viewer --mjcf=model/scene.xml

    左侧 Control 面板最后一个滑块是 gripper_joint：
        0.00 -> 闭合（留微小间隙）
        1.70 -> 张开最大

【注意】
    * 本目录的 XML 和 ../model/meshes/ 的相对位置不能变
      （so_arm101.xml 里写的是 meshdir="../meshes"）。
    * 详细物理常数与踩坑记录见 ../docs/HANDOFF.md。

================================================================================
 附：2024-10 按真机实测数据校准的记录（备份目录 （已删） 已于 2024-11 精简时删除，
     改动表就是下面这些数字，够回滚）
================================================================================

【为什么要校准】
    原模型的道具尺寸和关节限位是"为了能抓起来"手工调的，与真机不符。
    真机实测数据来自用户：SO-ARM101 + 3D 打印件 + 舵机线束 + 主板 + 摄像头。

【道具尺寸修正（全部为用户实测）】

  项目            改前                        改后              依据
  --------------  --------------------------  ----------------  ------------------
  药盒尺寸        73 x 15 x 52 mm（竖立）      74 x 52 x 15 mm   "长7.4 宽5.2 高1.5cm"
  药盒姿态        立在 73x15 窄边上            平躺              高=1.5cm 即厚度
  药盒初始 z      0.0265                      0.0095            底面贴绿垫(2mm)
  药盒惯量        2.8/7.7/5.3e-6              2.81/5.46/7.84e-6 按平放重算
  药盒视觉薄板    6 片按竖立摆放               6 片按平躺摆放    跟随姿态修正
  木盒外尺寸      115 x 140 x 50 mm           142 x 115 x 50 mm "长14.2 宽11.5 高5cm"
  绿垫直径        190 mm                      192 mm            "直径19.2cm"
  鼠标垫          310 x 242 mm                300 x 245 mm      "长30cm 宽24.5cm"
  药盒质量        11.5 g                      11.5 g  (未变)    实测一致
  木盒质量        45 g                        45 g    (未变)    实测一致

【关节限位修正（依据 ~/.cache/huggingface/lerobot/.../yyl_follower_arm.json）】
    STS3215: 4096 counts / 360°, 1 count = 0.0879°

  关节              真机 counts   真机行程    改前      改后
  ----------------  ------------  ----------  --------  ----------------
  shoulder_pan      778 ~ 3477    237.2°      220.0°    未改（差17°，可接受）
  shoulder_lift     787 ~ 3147    207.4°      200.0°    未改（差7°）
  elbow_flex        984 ~ 3182    193.2°      185.0°    未改（差8°）
  wrist_flex        830 ~ 3157    204.5°      183.4°    未改（差21°）
  wrist_roll        0 ~ 4095      359.9°      263.6°    **±180°（整圈）**
  gripper           1580 ~ 3035   127.9°      97.4°     **0 ~ 127.9°**

  影响：wrist_roll 原来比真机少 96°，gripper 张开幅度少 24%。
        这两项直接决定了"能不能把腕子拧到位 / 指间能不能跨过药盒"。

【质量修正（真机整臂实测 860 g，含线束/主板/摄像头）】

  部件              改前      改后     说明
  ----------------  --------  -------  --------------------
  base_link         147.0 g   253.0 g  主板 + 底座线束
  shoulder_link     100.0 g   118.0 g  线束
  upper_arm_link    103.0 g   123.0 g  线束
  lower_arm_link    104.0 g   124.0 g  线束
  wrist_link         79.0 g    96.0 g  线束
  gripper_link       87.0 g   132.0 g  **摄像头（装在爪甲，随臂运动）**
  jaw_link           12.0 g    14.0 g
  合计              632.0 g   860.0 g  与实测一致

  惯量按各自 mass 比例等比缩放（保持物理自洽）。

【2024-11 修正：木盒长宽朝向】

  用户实测指出：木盒摆在这个位置时，长 142mm 是 **y 向**、宽 115mm 是 **x 向**，
  旧模型写反了（写成 x 半宽 71 / y 半宽 57.5）。已改成：

      box_bottom  size="0.0575 0.071 0.002"   外 115(x) x 142(y) x 4
      box_left/right  size="0.002 0.071 0.025"  pos="±0.0555 0 0"
      box_front/back  size="0.0535 0.002 0.025" pos="0 ±0.069 0"
      inertial    diaginertia="6e-05 8e-05 1.2e-04"   （Ix/Iy 随长宽一起对调）

  内腔随之变成 x ±53.5mm / y ±67mm（壁厚 4mm、底板顶面 z=0.004 不变）。
  同时把 render 的 verify 的"药盒落在木盒里"判定改成 x≤50mm、y≤62mm。
  （2024-11 精简时已删除该备份；改动就上面 5 行，可直接改回）

【⚠️ 这次修正会让旧的抓取参数全部失效】
    grasp/test_grasp.py 里的 roll=120° / gz=0.046 是针对
    "药盒竖立"调的，现在药盒平躺（只有 15mm 高），必须重新设计抓取动作。
