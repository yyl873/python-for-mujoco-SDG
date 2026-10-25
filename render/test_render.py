"""
test_render.py — 纯净渲染采集管线（步骤三）
================================================================
把步骤二（test_grasp.py）已验证的纯物理抓取轨迹，转成带标签的
双相机视频数据集（MP4 + 可选序列帧 + meta.json）。

一、为什么切成 record / render 两段
----------------------------------------------------------------
任务书要求「剥离所有控制逻辑，只负责渲染」。用"数据契约"来切，而不是
靠约定或注释：

    record : 跑真物理（mj_step），把每帧状态 + 真实接触力写进 episode.npz
             （唯一包含 ctrl 写入与动作序列的代码段）
    render : 只读 npz，用 mj_forward 做纯正运动学回放，再渲染 → 视频
             （没有 ctrl、没有 mj_step、没有抓取判定）

好处有三个：
  1. render 阶段物理上不可能"作弊"——它拿不到 ctrl，也推不动仿真；
  2. 改噪声/分辨率/相机不需要重跑物理，npz 是缓存的真相；
  3. 同一份 npz 可以渲染任意多次、任意视角，天然适合 SDG 扩样。

二、与红线的关系（重要）
----------------------------------------------------------------
红线 2 禁止「覆盖 qpos 假装抓住物块」。本脚本的 render 阶段确实会写
data.qpos，但写的是 record 阶段由 mj_step + 真实接触力算出来的轨迹，
本质是"录制→回放"，不是伪造。证据留在 meta.json 里：每一帧都有
fn_jaw / fn_grip / contact_n，握持是否真实可逐帧审计。

抓取本身发生在 record 阶段：合爪时实测法向力 34.70 N，药盒重力
0.1128 N，μ=10，裕度 6151 倍（见 HANDOFF.md）。

三、与步骤二的同参保证
----------------------------------------------------------------
动作序列的常量（roll=120°、gz=0.046、retract z=0.24、approach z=0.16、
transport z=0.16、各段步数）全部从 test_grasp 导入或逐字复刻，
不重新发明参数。verify() 会断言抬升 122.2±5 mm、水平位移 218.1±5 mm（sdg-v5 基准），
一旦序列被改动会立刻报错。

六、数据工程（2024-11 新增：只加"导出/校验"辅助代码，不碰物理与渲染）
----------------------------------------------------------------
  1. success 标签：按"抬升高度 < 50 mm 判失败"自动判定，写进 meta.json 顶层
     "success": true/false；轨迹 npz 的 meta_json 里也留一份，这样只拿到
     episode.npz 的下游也能直接读标签。
  2. 失败样本照样导出：抓取失败（没抬起来 / 中途滑落）时视频、npz、meta.json
     全部照常落盘，只是 success=false。verify() 对负样本里"预期就不该成立"
     的两项（轨迹一致性、落在木盒里）降级为 ⚠️，不算成管线故障。
  3. schema 校验：第 10 段的 validate_*() 严格核对两份 npz 的键/形状/dtype，
     不一致直接 SystemExit —— 不匹配就别喂给训练，免得下游静默错位。

四、运行（本机 python3 的 PYTHONHOME 被 AppImage 污染，前缀不能省）
----------------------------------------------------------------
    # ★ 最快看到画面：双相机小窗 + 8 格拼图，约 10~20 秒
    env -u PYTHONHOME -u PYTHONPATH python3 test_render.py record --quick
    env -u PYTHONHOME -u PYTHONPATH python3 test_render.py render --quick

    # 正式出片（640x480，约 173 帧 x 2 相机 ≈ 30 秒）
    env -u PYTHONHOME -u PYTHONPATH python3 test_render.py

    # 只重渲染，不重跑物理（改噪声参数时用）
    env -u PYTHONHOME -u PYTHONPATH python3 test_render.py render --noise-sigma 5

    # 出干净图，不做传感器劣化
    env -u PYTHONHOME -u PYTHONPATH python3 test_render.py render \
        --blur-sigma 0 --noise-sigma 0

五、本机渲染环境实测结论（踩过的坑，别再重复）
----------------------------------------------------------------
  MUJOCO_GL=egl    ✅ 可用（mesa 软件光栅）640x480 约 70~90 ms/帧
  MUJOCO_GL=glfw   ❌ gladLoadGL error（DISPLAY=:1 建不出 GLX context）
  MUJOCO_GL=osmesa ❌ bad marshal data（本机 pyc/库不匹配）
  EGL 退出噪音    必有 EGLError ... __del__，用 sys.unraisablehook 静音
  GPU             当前 nvidia-smi 不可用，全程软件渲染，靠 --width 提速
"""

import os
import sys
import json
import time
import argparse
import hashlib
from pathlib import Path

# ============================================================
# 第 0 段  环境前置 —— 必须在 import mujoco 之前完成
# ============================================================


def _prescan(flag):
    """在 argparse 之前预扫描命令行，因为 MUJOCO_GL 必须在 import mujoco 前设好。"""
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


_GL = _prescan("--gl")
if _GL:
    os.environ["MUJOCO_GL"] = _GL
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

# EGL 上下文析构时 OpenGL 绑定会抛 EGLError，Python 以 "Exception ignored in"
# 打到 stderr。它发生在解释器退出阶段，不影响任何产物，纯噪音，直接静音。
sys.unraisablehook = lambda *a, **k: None

import numpy as np          # noqa: E402
import cv2                  # noqa: E402
import mujoco               # noqa: E402

# 路径锚定到仓库根：本文件在 <root>/render/test_render.py，parents[1] 即 <root>。
# 步骤二的 Arm 封装在 <root>/grasp/，运行时动态加进 sys.path，
# 这样从任何工作目录运行都能 import 到，且不依赖 pip 安装。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grasp"))

# 复用步骤二：模型路径、Arm 封装、已验证的动作常量
# （test_grasp.SCENE_XML 也已经是基于它自己 __file__ 的绝对路径，直接沿用）
from test_grasp import (    # noqa: E402
    Arm, SCENE_XML, ARM_JOINTS, DEF_ROLL, DEF_GZ, GRASP_XY_OFFSET, GRIP_APPROACH,
    RETRACT_Z, APPROACH_Z, TRANSPORT_Z,
    GOLDEN_JOINTS, GOLDEN_Q,          # 真机 golden pose 起点（2024-11）
    OLDROS_JOINTS, OLDROS_Q_RAW,      # old_ros initial_pose 起止姿态（2024-11 第五轮，默认）
)

# 「需要在落定后先竖直抬出」的起点集合：这两种起点手臂都不在 q≈0 的塌陷位，
# 且落定后可能离药盒/台面很近，关节插值直接去 q_ret 会横着扫过东西。
POSE_WITH_INIT_LIFT = ("golden", "oldros")

# 录制起止姿态的默认值（demo 与批量采集共用这一个口径；改这里 = 改数据集的起点分布）。
# "oldros" = old_ros 录制脚本 initial_pose（折叠待机位）
# "golden" = 上一轮的真机遥操第 0 帧姿态（留作 A/B）
# "home"   = 旧预设（ctrl=0 塌到 q≈0，仅回归对比）
START_POSE_DEFAULT = "oldros"

OUT_DIR = ROOT / "demo"        # 单条 demo 的产物落点（批量数据走 scripts/ 与 videos/）


def _sha256(path: Path):
    """
    模型文件指纹。写进 meta.json 是为了"数据卫生"：
    以后有人改了 MJCF（质量/摩擦/指垫），从 meta 就能看出这批数据不是同一个模型的，
    不会出现"同一个 dataset_version 混进两种物理"的情况。
    """
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return None
    h.update(p.read_bytes())
    return h.hexdigest()[:16]

# ============================================================
# 第 1 段  常量区（全部来自步骤二/任务书，不新造）
# ============================================================

CAMS = ("env_camera", "wrist_camera")   # 模型里已定义的两个相机
CAM_ENV, CAM_WRIST = CAMS

SETTLE = 1200          # 阶段 0 落定步数（不录制）

# ---------------------------------------------------------------------------
# 时长对齐真机数据：
#   真机 151 条 episode / 67649 帧 / 30fps  ->  平均 448 帧 ≈ 14.93 s
#   本管线 timestep=0.002 s，要凑 15.0 s 就是 15.0/0.002 = 7500 步。
#   下面各段合计恰好 7500 步；录帧按"时间栅格"触发（见 EpisodeRecorder.push），
#   所以帧数 = 15.0 x 30 = 450 帧，与真机 448 帧差 0.4%。
# ---------------------------------------------------------------------------
FPS = 30.0             # 录制帧率（与真机一致）
TARGET_SEC = 15.0      # 目标时长
# ⚠️ 不能均匀放慢：实测把下探从 400 步拉到 1150 步，结构件在盒顶压得更久，
#    药盒直接被推倒(z 0.0278 -> 0.0041)，抓取全废。
#    所以【接触段(下探/合爪/抬升)保持步骤二验证过的步数】，
#    时间只加在【自由空间段(撤回/正上方/搬运)】和【保持段】上。
# 各段的实测速度上限（再慢就翻车，见 debug/ 与 HANDOFF 的扫描记录）：
#   接近：retract<=900, above<=700        超过 -> 合爪时药盒被推倒
#   搬运：transport<=600                  超过 -> 中途滑脱
#
# 2024-11 改版：干掉原来的 N_HOLD=2000（4.0 s 纯静止，占全片 27%），
# 换成一条【真动作尾段】——下放 -> 开爪让药盒掉进木盒 -> 竖直提出 -> 回初始姿态。
# 总步数仍锁死 7500（15.0 s / 450 帧 @30fps），接触段(下探/合爪/抬升)步数一个都没动，
# 所以抬升/水平位移基准值不变，verify 的"与步骤二一致"依然有效。
# 静止占比：27% -> 24.7%；扣掉开头的 init(数据集起点必需)，只剩 2.1 s(14%)，
# 且最长连续静止段 0.8 s（原来是 4.0 s）。
# 2024-11（起止姿态 = old_ros 的 initial_pose，第五轮）：init 段拆成两段，总步数不变：
#   init_hold  在起始姿态静止（落定/PD 收敛，也是一条数据"从初始位开始"的证据）
#   init_lift  关节空间竖直把自己抬到 RETRACT_Z —— 真机录制脚本里开录后第一件事也是
#              垂直抬到安全高度（old_ros/send_trajectory.py 动作1），只是它用 0.20，
#              实测本模型 0.20/0.22 会让抓取全灭，所以仍用 0.24。
# 之后 approach_retract 才是原来的"高位撤回"（从抬出的位置走到药盒上方高位）。
N_INIT_HOLD = 200      # 在起始姿态静止（0.4 s）
N_INIT_LIFT = 700      # 关节空间竖直抬出到高位（1.4 s，约 0.156 m/s）
N_INIT = N_INIT_HOLD + N_INIT_LIFT     # 900
N_RETRACT = 800        # 抬出 -> 药盒上方高位撤回（1.6 s，实测 <=900 都不翻车）
N_ABOVE = 500          # 撤回 -> 药盒正上方（1.0 s）
# ★ (init_lift, retract, above) = (700, 800, 500) 不是拍的：
#   old_ros 起点下固定这三段合计 2200 步扫了 41 组，只有 5 组抓得住；其中
#   (700,800,500)（以及 800/700/500、700/1000/300、800/900/300）在
#   timing_scale 0.90~1.10 十档全部通过（抬升 103.4~103.9mm / 水平 221.0mm）。
#   golden 起点用的 (500,800,700) 在 old_ros 起点下是 5/5 全灭 —— 起点换了，
#   这三段必须跟着重扫，扫法见 tools/probe_grasp_robustness.py。
N_DESCEND = 400        # 竖直下探（★接触段, 0.8 s，锁定）
# ★ 下探后、合爪前的"停一下"：不是装饰，是把"合爪起点的状态"变成确定的。
#   下探结束时手臂还落后目标约 5°（PD 稳态误差），残留速度 0.03~0.5 rad/s（随节奏变），
#   直接合爪时这个差值会决定药盒是被夹住还是被推倒。
#   2024-11 第五轮实测（单次单候选口径，0.90/0.95/1.00/1.05/1.10 五档）：
#     golden 起点 (500,800,700)：0.90 ✅ / 0.95 ❌ / 1.00 ✅ / 1.05 ❌ / 1.10 ❌
#     old_ros 起点 (700,800,500)：五档全 ✅（另三组候选组合同样十档全过）
#   ⚠️ 这张表是"单次单候选"的，不等于批量接受率 —— collect.py 的 search_grasp 会再叠
#      roll/gz 扰动重试 13 次，所以 golden 时代照样能到 18/20。两者不要混着引用。
#   先停 0.4 s 让手臂收敛到 q_gsp，合爪就从同一个确定状态出发（步数从 home 段挪来，
#   总步数仍 7500 = 15.0 s，时长不变）。
# ★ 试过"下探前/后各停 0.4 s 让 PD 收敛"，**两种都让抓取全灭**（16 个节奏档全 0）：
#   停在下探目标上 -> 手臂多压 5°，药盒被挤出去；停在 q_abv 上 -> 收敛后的下探
#   又落在另一种接触状态上。结论：这个"抓得住"依赖下探结束时那点 PD 滞后，
#   不要去消除它。下面两行保留常量只为记录，实际不参与 STEP_PLAN。
N_SETTLE_D = 0         # （已废弃，勿启用）
N_CLOSE = 300          # 合爪（★接触段, 0.6 s，锁定）
N_PAUSE_A = 400        # 合爪后停顿"确认夹稳"（0.8 s）
N_LIFT = 400           # 抬升（★接触段, 0.8 s，锁定）
N_PAUSE_B = 300        # 抬升后停顿（0.6 s）
N_TRANSPORT = 600      # 水平搬运（1.2 s）
# ---- 尾段（新）：下放 -> 释放 -> 掉落 -> 提出 -> 归位 ----
N_PLACE = 300          # 保持夹持下放到木盒口（0.6 s）
N_PRE_OPEN = 100       # 释放前静止，消掉残余摆动（0.2 s）
N_RELEASE = 300        # 开爪释放，药盒自由下落（0.6 s）
N_DROP = 300           # 等它落进木盒并静止（0.6 s）
N_RETRACT_OUT = 450    # 从木盒里竖直提出到高位（0.9 s）
N_HOME = 1300          # 关节空间回初始姿态（2.6 s）
N_TAIL = 150           # 收尾静止（0.3 s）
# 合计 = 200+700+800+500+400+300+400+400+300+600+300+100+300+300+450+1300+150
#      = 7500 步 = 15.0 s
# （起点换成 old_ros initial_pose 后 init_lift 500->700、approach_above 700->500，
#   两段互相抵消，总步数/时长/帧数一个都没变。）

# 阶段 -> 步数。全片只有这一份加法表，cmd_record 会断言总和 == 7500，
# 免得改了某一段忘了配平、帧数从 450 漂到别的值。
STEP_PLAN = {
    "init_hold": N_INIT_HOLD, "init_lift": N_INIT_LIFT,
    "approach_retract": N_RETRACT, "approach_above": N_ABOVE,
    "descend": N_DESCEND, "grasp": N_CLOSE, "settle_a": N_PAUSE_A,
    "lift": N_LIFT, "settle_b": N_PAUSE_B, "transport": N_TRANSPORT,
    "place": N_PLACE, "pre_open": N_PRE_OPEN, "release": N_RELEASE,
    "drop": N_DROP, "retract": N_RETRACT_OUT, "home": N_HOME, "tail": N_TAIL,
}
TOTAL_STEPS = 7500     # = TARGET_SEC / timestep，硬约束

# 可以随机缩放的段（自由空间 + 纯停顿）。★接触/夹持段一律不动：
#   descend / grasp / lift / transport / place / release / drop / pre_open / retract
#   —— 这些步数是实测速度包线的上界，改一点就会推倒药盒或中途滑脱。
SCALABLE_PHASES = ("init_hold", "init_lift", "approach_retract", "approach_above",
                   "settle_a", "settle_b", "home", "tail")


def make_step_plan(scale=1.0):
    """
    生成一份阶段步数表。scale 只作用在 SCALABLE_PHASES 上（批量 SDG 用来做时长抖动）。

    scale=1.0 时与 STEP_PLAN 逐位相同（demo 走这条路径，cmd_record 会断言 7500）。
    批量生成时 scale 取 0.90~1.10 -> 总时长 14.1~15.9 s。

    ⚠️ 别再写"接触段步数不变，所以抓得住不受影响"了 —— 这句已被证伪（2024-11 第五轮）：
      settle_a 本身就在 SCALABLE_PHASES 里，而更关键的是【进入 descend 时的状态】
      （残留速度/PD 滞后）由前面几段自由段的节奏决定。实测同一套接触段
      golden 起点在 0.95/1.05/1.10 三档全灭、old_ros 起点十档全过 —— 节奏是真的会翻抓取的，
      只是不同起点对它的敏感度差很多。改动前先跑 tools/probe_grasp_robustness.py。
    """
    plan = {}
    for k, v in STEP_PLAN.items():
        plan[k] = int(round(v * scale)) if k in SCALABLE_PHASES else int(v)
    return plan

# 抓取点的横向补偿（2024-11 第六轮）。
# ★ 为什么需要它：`gripperframe` 这个 site **不在两指正中** —— 实测（oldros 起点、
#   q_gsp 位姿、grip=1.5）固定指（gripper_link）的最低顶点落在 (296.0, 92.5, 6.6) mm，
#   而药盒中心是 (295.0, 90.0)、盒体占 x[258,332] y[82.5,97.5] z[2,53.8]：
#   也就是说"指尖默认落在盒体正中央"，只靠压低 TCP（gz=0.012）根本绕不开，
#   下探必然先戳在盒顶上（实测 92/400 步接触、峰值 2.73 N、盒被压进桌面 5.1 mm）。
#   张开量 1.5 与 2.232 的下探接触步数**一模一样**（92 对 92），说明这不是"张不够"，
#   而是整个手在横向偏了 ~13 mm。
# 补偿量由 tools/probe_contact_audit.py --dx/--dy 扫出来（见 logs/audit_b1b.jsonl），
# 这里写的是扫描结论，不是拍的。
# 抓取点横向补偿：常量与说明统一放在 grasp/test_grasp.py 的 GRASP_XY_OFFSET（步骤二/三共用一份，
# 免得两条路径各写一个数、其中一条忘了改）。第七轮的实测依据见那里的注释。     # (dx, dy) 米；作用在"药盒正上方"和"竖直下探"两个 IK 目标上

# 释放高度：药盒要从木盒口上方落下，落地前不能撞盒沿。
# 实测 TCP 比药盒中心高 33.2 mm、药盒半高 26 mm、木盒上沿 z=0.050：
#   PLACE_Z=0.115 -> 药盒底面 0.0558（离盒沿 6 mm），落地落差 26 mm（自由落体 0.07 s）
PLACE_Z = 0.115

GRIP_OPEN, GRIP_SHUT = 2.232, 0.28  # 夹爪 ctrl 语义（2.232=127.9° 张开 / 0.28 夹住 15mm 盒）
# ★ 2024-11 第七轮：指垫改成"贴合手指真实内表面"之后，合爪量必须重定。
#   old_ros 的 set_gripper(val) 是【直接把 val 当弧度】下发给 gripper_joint，所以 0.15 就是 0.15 rad。
#   但实测两指真实内表面间距随闭合量变化约 7.6 mm / 0.1 rad：
#       jaw=0.15 -> 指面间距 4.45 mm（等于手指已合拢，夹 15 mm 盒会把盒挤出去）
#       jaw=0.20 -> 8.8 mm
#       jaw=0.28 -> 14.15 mm  ★ 与盒厚 15 mm 匹配
#   所以本模型夹这个盒子的正确闭合量是 **0.28**，不是 0.15。
#   用户要的"看起来合到 0.15 的贴合感"由"指垫贴齐指面"实现（夹稳时两指面都贴住盒子，
#   实测缝 −0.72 / −1.08 mm），而不是靠把 ctrl 写小。
#   实测（tools/probe_contact_audit.py）：0.28 -> 下探 0/400、压盒 0.37、夹稳倾角 1.1°、抬升 +122.33。
                                   # 2.232 来自真机标定 1580~3035 counts
                                   # 闭合目标 0.20 而不是 0.00：实测合到 0 会持续挤压，
                                   # 把药盒当西瓜籽一样慢慢挤出去（静止 1s 滑 116mm 直接掉）；
                                   # 合到 0.20（约 11.5°，指尖刚好贴住）滑移降到 5.6mm 且能搬完

# ---- 夹爪开合脚本（2024-11 改版）------------------------------------------
# 用户要求：起手和收尾时爪甲【不是全开】，而是"收起的预设姿态"；
#          张开只在抓取前后出现，而且不必每次开到最大。
# 于是把张合拆成四个量：
GRIP_PRESET = 0.0       # 预设姿态 = 夹爪收起（与 keyframe 里 gripper qpos 一致）
                        # init / tail 两段就停在这，首末帧都是"爪子收起"的样子
# 数值来自 --grip-open / --grip-release 的实测扫描（当时在临时目录里扫过，这里只留结论）：
#   下探张开量 0.6 / 0.8 / 0.9 / 1.0 / 1.2 / 1.5 / 1.8 / 2.232 全部能抓起来
#       （抬升 103.5~103.8mm、水平 221.0mm、落点都在木盒里）——说明张开量只要有
#       余量跨过 74mm 盒宽就行，没必要顶到最大。取 1.5（≈86°）留足余量。
#   释放张开量 0.8 / 1.4 / 2.232 也都满足"松手后零接触"。取 1.2 留余量。
#   所以这里【不再是全开】：全开 2.232 只在"真需要"时才用（比如以后夹更大的东西）。
# GRIP_APPROACH 现在统一从 grasp/test_grasp.py 导入（步骤二/三同值），这里不再重复定义
GRIP_RELEASE = 1.2      # 释放时张开量（够指尖离开药盒即可）
# 张开/收起的动作都放在【移动段】里做完（approach 边移动边张开，
# retract + home 边后退边收起），所以不会额外多出静止帧。

PHASES = ("init", "approach", "descend", "grasp", "settle_a", "lift",
          "settle_b", "transport", "place", "pre_open", "release", "drop",
          "retract", "home", "tail")

# 步骤二实测基准值，用于 verify() 断言
# 历史：92.55 / 155.1（木盒在 0.440 时代）-> 104.13 / 222.0（2024-10 重标定）
#      -> 125.10 / 209.6（2024-11 第六轮：指垫改真之后）
#      -> 122.22 / 218.06（2024-11 第七轮：指垫改成"贴合手指真实内表面"+ dy=+17.5mm + 合爪 0.28）
# 2024-11 第二轮试过把 gz 提到 0.014（基准会变成 102.43 / 221.31），但它在**没有停顿的
# 步骤二**里会滑落（约束2 ❌、末端握持接触 0），所以 gz 回退 0.012，基准也跟着回退。
REF_LIFT_MM = 122.22
REF_HDISP_MM = 218.06
# 接触真实性的两道闸（第六轮）。旧指壳方案实测 压盒 6.87 mm / 夹稳倾角 10.95°；
# 指垫方案 0.22 mm / 0.07°。阈值取"肉眼不可辨"的量级，不是"跟实测一样"。
PRESS_MAX_MM = 0.5          # 下探~夹稳期间药盒相对静置位最多沉这么多
GRASP_TILT_MAX_DEG = 5.0    # 夹稳那一刻药盒相对竖直的最大倾角
TOL_MM = 5.0

# ---- 成功标签判据（数据工程用；只读轨迹数组，不影响物理与渲染）-------------
# 任务书口径："最后的抬升高度 < 50mm 视为失败"。
# 50mm 不是拍脑袋：药盒半高 26mm、木盒口 z=50mm，抬升 50mm 已明确越过
# "只是被指垫顶了一下 / 蹭了一下"的量级；正样本实测 103mm，离阈值 51mm，余量足。
SUCCESS_LIFT_MM = 50.0
# 中途滑落：抬起来之后，在【搬运段末】塌回峰值的一半以下 = 抓起来又掉了。
# 只看 transport 段末，是因为 place/release/drop 本来就是"把它放下"，
# 拿全片末帧的抬升判滑落会把正常放件全判成失败。
SUCCESS_HOLD_FRAC = 0.5


# ============================================================
# 第 2 段  EpisodeRecorder —— 把物理轨迹写进数据契约
# ============================================================


class EpisodeRecorder:
    """
    逐帧快照器。push() 只在 record 阶段被调用，每 decimate 个 sim step 存一帧。

    存什么、为什么：
      qpos/qvel   render 阶段回放所需的最小充分状态（nq=13: 6 臂关节 + 物块 free joint 7）
      ctrl        留档：证明动作是 position 控制下达的，不是直接写 qpos
      phase       阶段标签，SDG 做阶段采样/分层抽样的关键字段
      fn_jaw/grip 真实接触力（mj_contactForce），握持真假的审计证据
      lift_mm     相对初始位的抬升，抓取成功的量化标签
    """

    def __init__(self, arm: Arm, fps: float = FPS):
        self.a = arm
        self.fps = float(fps)
        self.dt = float(arm.m.opt.timestep)
        # 兼容旧字段：等价步数（只用于打印/元数据）
        self.decimate = max(1, int(round(1.0 / (self.fps * self.dt))))
        self._next_t = 0.0
        self._rows = {k: [] for k in (
            "time", "qpos", "qvel", "ctrl", "phase",
            "tcp_pos", "obj_pos", "obj_quat",
            "contact_n", "fn_jaw", "fn_grip", "fn_sum",
            "lift_mm", "hdisp_mm",
            "box_n", "obj_speed",       # 2024-11 新增：臂↔木盒接触数 / 药盒线速度
        )}
        self._obj0 = None
        self._n_all = 0
        self._last_n = -1        # 上一次真正落帧时的 _n_all，避免 force 重复落帧

    def begin(self, obj0):
        """记录初始物体位置，作为 lift / hdisp 的基线。"""
        self._obj0 = np.asarray(obj0, float).copy()

    def push(self, phase: str, force=False):
        """
        每调用一次 mj_step 后调用一次；内部按 decimate 抽样。

        force=True：忽略抽样间隔，强制落一帧。用于阶段末帧——否则 34 步的
        抽样会把"阶段结束那一瞬间"漏掉，导致与 test_grasp 的数值对不上
        （实测：漏掉后抬升从 92.55 变成 109.70 的假象）。
        """
        self._n_all += 1
        # 时间栅格触发：仿真时间每跨过 1/fps 就录一帧。
        # 用时间而不是"每 N 步"，是因为 fps 和 timestep 往往不是整数倍关系
        # （0.002 s 步长下 30 fps 需要 16.67 步），按步取整会累积成 29.41 fps。
        if not force and self.a.d.time < self._next_t:
            return
        if self._last_n == self._n_all:
            return
        self._last_n = self._n_all
        # 对齐到下一个时间栅格（用当前时间推进，避免累积漂移）
        self._next_t = (np.floor(self.a.d.time * self.fps) + 1) / self.fps
        a, d = self.a, self.a.d

        obj = a.obj_pos()
        quat = d.qpos[a.oadr + 3: a.oadr + 7].copy()
        cs = a.grip_contacts()                 # 复用步骤二的接触读取逻辑
        # ★ float() 不能省：全片没有任何接触时 sum([]) 是 Python int 0，
        #   np.array 会把整列存成 int64，而 schema 要求 float64。
        #   这是 2024-11 加校验时抓出来的真实缺陷（负样本里 100% 会踩到）：
        #   同一份数据契约的 dtype 竟然是"看数据长什么样"决定的，下游读进去就是一个坑。
        fn_jaw = float(sum(x["Fn"] for x in cs if x["label"] == "jaw"))
        fn_grip = float(sum(x["Fn"] for x in cs if x["label"] == "grip"))

        r = self._rows
        r["time"].append(float(d.time))
        r["qpos"].append(d.qpos.copy())
        r["qvel"].append(d.qvel.copy())
        r["ctrl"].append(d.ctrl.copy())
        r["phase"].append(phase)
        r["tcp_pos"].append(a.tcp_pos())
        r["obj_pos"].append(obj)
        r["obj_quat"].append(quat)
        r["contact_n"].append(len(cs))
        r["fn_jaw"].append(fn_jaw)
        r["fn_grip"].append(fn_grip)
        r["fn_sum"].append(fn_jaw + fn_grip)
        r["lift_mm"].append((obj[2] - self._obj0[2]) * 1000.0)
        r["hdisp_mm"].append(float(np.linalg.norm(obj[:2] - self._obj0[:2])) * 1000.0)
        # 2024-11 新增两列：前者是"手臂有没有刮到木盒"的审计证据（应为 0），
        # 后者是"药盒是否已经落稳"的判据（掉落段末尾应 < 0.05 m/s）。
        # 注意 qvel 的地址和 qpos 的地址不是一回事（自由关节 qpos 7 个数 / qvel 6 个数），
        # 所以这里从 jnt_dofadr 查，不能拿 oadr 直接用。
        r["box_n"].append(len(a.box_hits()))
        odof = a.m.jnt_dofadr[a.m.body_jntadr[a.obj]]
        r["obj_speed"].append(float(np.linalg.norm(d.qvel[odof:odof + 3])))

    def __len__(self):
        return len(self._rows["time"])

    def phase_series(self):
        """按阶段取序列，便于用与 test_grasp 相同的口径复算指标。"""
        ph = np.array(self._rows["phase"])
        return dict(
            phase=ph,
            z=np.array(self._rows["obj_pos"])[:, 2],
            hdisp=np.array(self._rows["hdisp_mm"]),
            fn=np.array(self._rows["fn_sum"]),
            contact=np.array(self._rows["contact_n"]),
            t=np.array(self._rows["time"]),
        )

    def _phase_last(self, series, ph_name):
        idx = np.where(series["phase"] == ph_name)[0]
        return int(idx[-1]) if len(idx) else None

    def metrics(self):
        """
        按 test_grasp.py 的口径复算指标（口径不同会得到完全不同的数字）：
          抬升      = z(阶段5 lift 末) - z(阶段4 grasp 末)   -> test_grasp 的 z_lift - z_before
          水平位移  = hdisp(阶段 transport 末)              -> test_grasp 的 |p_final[:2]-p0[:2]|
        再附一个"相对初始位"的整体抬升作参考。

        2024-11 注意：水平位移必须取 transport 末而不是全序列末 —— 加了尾段之后，
        药盒会掉进木盒并在里面挪一点，拿末帧取 hdisp 会把它算成"搬运更远了"，
        与 test_grasp 的基准值（在搬运结束时刻取的）不再同一口径，会假报警。
        """
        s = self.phase_series()
        i_g = self._phase_last(s, "grasp")
        i_l = self._phase_last(s, "lift")
        # 搬运末帧；没开 transport 时退回全序列末（与改动前行为一致）
        i_t = self._phase_last(s, "transport")
        i_t = len(s["phase"]) - 1 if i_t is None else i_t
        rise = (float(s["z"][i_l]) - float(s["z"][i_g])) * 1000.0 if \
            (i_g is not None and i_l is not None) else float("nan")
        overall = (float(s["z"][-1]) - self._obj0[2]) * 1000.0 \
            if self._obj0 is not None else float("nan")
        return dict(rise_mm=rise, overall_rise_mm=overall,
                    hdisp_mm=float(s["hdisp"][i_t]),
                    obj_end=[float(v) for v in self._rows["obj_pos"][-1]],
                    obj_at_transport_end=[float(v) for v in self._rows["obj_pos"][i_t]])

    def summary(self):
        """按阶段汇总，打印用。"""
        ph = np.array(self._rows["phase"])
        z = np.array(self._rows["obj_pos"])[:, 2]
        fn = np.array(self._rows["fn_sum"])
        out = []
        for p in PHASES:
            idx = np.where(ph == p)[0]
            if len(idx) == 0:
                continue
            out.append(dict(phase=p, frames=len(idx),
                            z0=float(z[idx[0]]), z1=float(z[idx[-1]]),
                            fn_max=float(fn[idx].max()),
                            held=int((np.array(self._rows["contact_n"])[idx] > 0).sum())))
        return out

    def arrays(self) -> dict:
        """
        把内部 list-of-rows 转成 numpy 数组（save() 与 success 判定共用同一份）。

        单独抽出来是为了让"落盘的数组"和"判标签用的数组"是同一份构造逻辑：
        如果两个地方各拼一遍，改了一个忘改另一个，标签就会和实际写进 npz 的数据对不上。
        phase 存定长 unicode 而不是 object 数组：npz 里 object 数组需要
        allow_pickle=True 才能读，既慢又不利于跨版本读取。
        """
        return {k: (np.array(v, dtype="<U16") if k == "phase" else np.array(v))
                for k, v in self._rows.items()}

    def save(self, path: Path, meta: dict):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrs = self.arrays()
        arrs["meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
        np.savez_compressed(path, **arrs)
        return path


def load_episode(path: Path):
    """npz -> (dict of arrays, meta dict)。render 阶段唯一的输入。"""
    z = np.load(Path(path))
    ep = {k: z[k] for k in z.files if k != "meta_json"}
    meta = json.loads(str(z["meta_json"]))
    return ep, meta


# ---------------------------------------------------------------------------
# 成功标签：只读轨迹数组，用来区分正/负样本（不参与任何控制或渲染）
# ---------------------------------------------------------------------------

def _last_phase_index(ph, name):
    """某阶段最后出现的那一帧下标；没有该阶段返回 None。"""
    idx = np.where(np.asarray(ph) == name)[0]
    return int(idx[-1]) if len(idx) else None


def contact_audit_from_arrays(arrays: dict):
    """
    从录好的轨迹数组里算"接触真实性"三项（2024-11 第六轮新增）。

    为什么不只靠 tools/probe_contact_audit.py：那个是"改参数时的仪器"，
    跑不进产物；这里是"每条数据出厂自带的体检结果"，写进 meta.json 与 npz 的
    meta_json，下游拿到一条数据就能知道它有没有"下探戳盒 / 压进桌面 / 夹的时候把盒子撞歪"。
    30 fps 抽样对这三项够用：压盒过程持续 0.2~0.8 s，倾角是连续量，不会漏量级。

    口径：
      press_mm              descend~settle_a 之间药盒相对静置位的最大下沉（mm，正=下沉）
      rebound_mm            压到最深之后又回升多少（mm）
      tilt_at_grasp_end_deg 夹稳那一帧（settle_a 末）的药盒倾角（相对世界竖直）
    """
    ph = [str(x) for x in arrays.get("phase", [])]
    lift = np.asarray(arrays.get("lift_mm", []), float)
    quat = np.asarray(arrays.get("obj_quat", []), float).reshape(-1, 4)
    empty = dict(press_mm=None, rebound_mm=None, tilt_at_grasp_end_deg=None)
    if len(ph) == 0 or len(lift) != len(ph):
        return empty
    idx = [i for i, x in enumerate(ph) if x in ("descend", "grasp", "settle_a")]
    if not idx:
        return empty
    seg = lift[idx]
    press = -float(seg.min())
    i_min = idx[int(np.argmin(seg))]
    after = [i for i in idx if i > i_min]
    rebound = float(lift[after].max() - lift[i_min]) if after else 0.0
    last_sa = [i for i, x in enumerate(ph) if x == "settle_a"]
    tilt = None
    if last_sa:
        q = quat[last_sa[-1]]
        upz = 1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2)
        tilt = float(np.degrees(np.arccos(np.clip(upz, -1.0, 1.0))))
    return dict(press_mm=round(press, 3), rebound_mm=round(rebound, 3),
                tilt_at_grasp_end_deg=None if tilt is None else round(tilt, 2))


def evaluate_success(rec_meta: dict, ep: dict) -> dict:
    """
    判定一条 episode 是不是"抓起来了"（正样本），返回可直接写进 JSON 的 dict。

    判据（任务书口径）：
      success = 抬升 >= SUCCESS_LIFT_MM 且 没有中途滑落
      抬升    = z(阶段 lift 末) - z(阶段 grasp 末)     <- record 的 metrics.rise_mm，同源
      滑落    = 有 transport 段，且 transport 末帧"相对初始位的抬升"塌回峰值的一半以下

    为什么用 metrics.rise_mm 而不是自己再算一遍：demo/test_grasp/render 三方
    必须同一口径，否则"标签说成功、verify 说不一致"自相矛盾。
    失败原因用机器可读的短码，方便下游分层抽样：
      lift_too_low -> 药盒没离开桌面（最常见：夹空 / 夹偏 / 压翻）
      slipped      -> 抬起来了但在搬运途中掉了
      no_metrics   -> 轨迹里没有 record 阶段算出的指标（旧版 npz）
    """
    metrics = (rec_meta or {}).get("metrics") or {}
    try:
        rise = float(metrics.get("rise_mm"))
    except (TypeError, ValueError):
        rise = float("nan")

    out = dict(success=False, threshold_mm=SUCCESS_LIFT_MM,
               lift_mm=None if not np.isfinite(rise) else round(rise, 4),
               hold_mm=None, reasons=[])
    if not np.isfinite(rise):
        out["reasons"].append("no_metrics")
        return out

    if rise < SUCCESS_LIFT_MM:
        out["reasons"].append("lift_too_low")

    ph = np.asarray(ep.get("phase", []))
    lift = np.asarray(ep["lift_mm"], float) if "lift_mm" in ep else np.array([])
    i_t = _last_phase_index(ph, "transport") if len(ph) else None
    if i_t is not None and len(lift) == len(ph):
        hold = float(lift[i_t])            # 搬运末帧：药盒相对初始位的抬升
        out["hold_mm"] = round(hold, 4)
        # 只在"确实抬起来过"的前提下判滑落，否则 rise 已经报 lift_too_low 了
        if rise >= SUCCESS_LIFT_MM and hold < SUCCESS_HOLD_FRAC * rise:
            out["reasons"].append("slipped")

    out["success"] = not out["reasons"]
    return out


# ============================================================
# 第 3 段  动作序列（唯一含 ctrl 的代码，与 test_grasp 逐段同参）
# ============================================================


def _move(a: Arm, rec: EpisodeRecorder, phase: str, q_target, gctrl, n_steps,
          force=True, g_from=None):
    """
    关节空间线性插值移动 —— 与 test_grasp.Arm.move() 完全同构。

    为什么不用 Arm.move()：它内部 mj_step 后没有回调点，无法逐帧记录。
    这里把它的 8 行 loop 复刻一遍并插入 rec.push()，插值公式、ctrl 写入
    顺序、步数都与原实现一致，因此物理结果逐位相同（verify 会核对）。

    force=False：段末不再强制落盘。时间栅格本身已经把整条时间线切成等长的
    1/30 s，段末多补一帧会让总帧数比 15.0 s x 30 多出十几帧（实测 466 vs 450），
    与真机 448 帧的口径就飘了。所以只有"要当作基准用的那几个时刻"才 force。

    g_from=None：夹爪固定写 gctrl（原行为）；给了 g_from 就让夹爪在这段里
    【从 g_from 线性过渡到 gctrl】—— 张开/收起都借用已有的移动段完成，
    既不额外占时间，也不会像"先在 init 段张到最大"那样在起手就摆出张开姿态。
    """
    m, d = a.m, a.d
    q_from = np.array(d.qpos[a.adr], float)
    q_target = np.asarray(q_target, float)
    for k in range(n_steps):
        f = (k + 1) / n_steps
        d.ctrl[a.ctrl] = q_from + (q_target - q_from) * f
        d.ctrl[a.cg] = gctrl if g_from is None else (g_from + (gctrl - g_from) * f)
        mujoco.mj_step(m, d)
        rec.push(phase)
    rec.push(phase, force=force)     # 见 docstring：只有基准时刻才强制落帧


def _move_cartesian(a: Arm, rec: EpisodeRecorder, phase: str, p_to, gctrl,
                    n_steps, knots=20, force=True, g_from=None):
    """
    TCP 沿世界直线走到 p_to（笛卡尔），每 n/knots 步解一次位置 IK，段内关节线性插值。
    与 Arm.move_cartesian() 同一套算法，只是多了一个 rec.push() 回调点。

    为什么需要它（2024-11，预设姿态改成真机姿态之后）：
        非 home 起点（oldros / golden）下，落定后手指可能就贴在药盒旁边
        （golden 实测 jaw 最近 34mm、gripper 49mm、与药盒同高）。
        关节空间直线插值会让手指"横着"扫过药盒 —— 实测把药盒直接推倒
        （z 27.8mm -> 9.3mm）。竖直方向的笛卡尔直线（先把自己抬离台面）才是干净的路。
        而且这段里夹爪必须保持【闭合】，张爪的瞬间同样会把药盒推倒。
    """
    m, d = a.m, a.d
    p_start = np.array(d.site_xpos[a.tcp], float)
    p_to = np.asarray(p_to, float)
    q_prev = np.array(d.qpos[a.adr], float)
    seg = max(1, int(n_steps) // max(1, int(knots)))
    done = 0
    while done < n_steps:
        k = min(seg, n_steps - done)
        frac = (done + k) / float(n_steps)
        q_next, _ = a.ik(list(p_start + (p_to - p_start) * frac), q_prev)
        q_next = np.asarray(q_next, float)
        q_cur = np.array(d.qpos[a.adr], float)
        for j in range(k):
            f = (j + 1) / k
            d.ctrl[a.ctrl] = q_cur + (q_next - q_cur) * f
            d.ctrl[a.cg] = gctrl if g_from is None else (g_from + (gctrl - g_from) * f)
            mujoco.mj_step(m, d)
            rec.push(phase)
        q_prev = q_next
        done += k
    rec.push(phase, force=force)


def _ramp_gripper(a: Arm, rec: EpisodeRecorder, phase: str, g0, g1, n_steps,
                  force=True, q_arm=None):
    """
    手臂指令不动，只让夹爪从 g0 线性变到 g1。

    合爪(GRIP_APPROACH->GRIP_SHUT)和开爪(GRIP_SHUT->GRIP_RELEASE)都走这一个函数：
    物理上"释放"就是"合爪的逆过程"，同一个写法、只有起止值相反，
    不引入任何新控制机制（任务书要的纯 position 控制）。

    q_arm：合爪期间手臂 ctrl 写什么。默认 None = 冻结在当前 qpos（hack，用来"只看夹爪"）。
           传了就用这条目标（= 下探段的目标 q_gsp）—— 这才是真机控制器在合爪时的行为：
           目标不变，只改夹爪。2024-11 实测：golden 预设姿态下"冻结当前 qpos"会把药盒
           推倒（抬升 0mm、药盒 z 27.8->9.3），改成"保持下探目标"就稳（+115mm）。
           原因是冻结会让手臂停在一个 PD 未收敛的姿势上，夹爪合拢时的反作用力
           把盒子往一边推；保持目标则反作用力被 PD 吸收。
    """
    m, d = a.m, a.d
    q_cmd = np.array(d.qpos[a.adr], float) if q_arm is None else np.asarray(q_arm, float)
    for k in range(n_steps):
        d.ctrl[a.ctrl] = q_cmd
        d.ctrl[a.cg] = g0 + (g1 - g0) * ((k + 1) / n_steps)
        mujoco.mj_step(m, d)
        rec.push(phase)
    rec.push(phase, force=force)     # 段末帧是握持/释放判定的基准，必须精确


def _hold(a: Arm, rec: EpisodeRecorder, phase: str, n_steps, gctrl, force=True,
          q_arm=None):
    """
    静止保持：夹爪锁在 gctrl，手臂 ctrl 默认锁在当前 qpos。

    q_arm：给了就用这条关节目标（而不是当前 qpos）。合爪后的 settle_a 必须传下探目标
           q_gsp —— 与 _ramp_gripper 同一个理由：冻结在"PD 未收敛的姿势"上，
           夹爪反作用力会把药盒推倒（实测抬升 0mm）。保持目标则被 PD 吸收。
    """
    m, d = a.m, a.d
    q_cmd = np.array(d.qpos[a.adr], float) if q_arm is None else np.asarray(q_arm, float)
    for _ in range(n_steps):
        d.ctrl[a.ctrl] = q_cmd
        d.ctrl[a.cg] = gctrl
        mujoco.mj_step(m, d)
        rec.push(phase)
    rec.push(phase, force=force)


def run_episode(a: Arm, rec: EpisodeRecorder, roll=DEF_ROLL, gz=DEF_GZ, transport=True,
                grip_approach=GRIP_APPROACH, grip_release=GRIP_RELEASE,
                plan=None, spawn=None, settle=None, ik_seed=None, wp=None,
                start_pose="oldros", grasp_xy_offset=None):
    """
    复刻 test_grasp.main() 的阶段 0~6，再补一条真动作尾段，每个 sim step 记一帧。

    路径要点（HANDOFF 第 4 节）：直接插值到目标会横穿药盒把它扫倒，
    所以必须【起始位 -> 高位撤回(z=0.24) -> 正上方(z=0.16) -> 竖直下探】。

    ★ 2024-11 第五轮：起始姿态默认是【old_ros 录制脚本的 initial_pose】（折叠待机位）
        pan 0° / lift −100° / elbow 88.26° / wrist_flex 70.63° / wrist_roll −103.53°，
        夹爪收起。它同时是录制【结束】姿态 —— old_ros/send_trajectory.py:369/:552
        开录前去这个位、录完回这个位；本片的 home 段回 q_home（首帧 qpos）即等价。
        此时 TCP 离药盒 151mm，比 golden 更安全，但落定后仍要先【竖直抬出】
        （init_lift，夹爪保持收起），再走接近段。
    ★ 上一轮的默认起点是 golden pose（真机遥操第 0 帧：pan −36°/lift +4°/elbow +14°/
        wrist_flex +55°/wrist_roll −104°），手指贴在药盒旁（jaw 34mm、同高）。
        传 start_pose="golden" 可回到它做 A/B；传 "home" 回旧预设（塌到 q≈0）。

    全片阶段一览（步数见 STEP_PLAN，合计 7500 步 = 15.0 s）：
      init      起始位【静止】（init_hold），爪甲收起（数据集起点；不在这里张爪）
      approach  init_lift 笛卡尔竖直抬出 + 两段接近：撤回高位 -> 药盒正上方
                （边走边张开到 grip_approach）
      descend   竖直下探            ★接触段
      grasp     合爪                ★接触段
      settle_a  夹稳确认
      lift      竖直抬升            ★接触段
      settle_b  抬升后停
      transport 水平搬到木盒上方
      place     保持夹持下放到盒口
      pre_open  松手前停稳
      release   开爪到 grip_release，药盒自由下落
      drop      落进木盒并静止
      retract   竖直提出木盒（边退边把爪甲收到预设姿态）
      home      回初始姿态（与首帧同一臂姿态 + 同一收起爪姿）
      tail      收尾静止

    --- 批量 SDG 用的两个注入口（demo 都不传，行为与之前完全一致）---
      plan   : 阶段步数表（make_step_plan(scale) 生成），None 则用 STEP_PLAN
      spawn  : spawn(arm) 回调。只在落定【之前】调用一次，用于随机摆放药盒。
               ★这是"出生位姿"，不是伪造抓取：写一次 qpos 后立刻交给物理落定，
                 之后全程只读不写，握持依旧来自 mj_step 的接触力。
      settle : 落定步数，None 则用 SETTLE
      ik_seed: 接近路点的 IK 初始猜测（5 维）。批量生成绕底座旋转摆放时，
               把 pan 分量按旋转角平移，DLS 会立刻收敛到"同构解"，
               抓取几何与原始 demo 逐位相同（这是零成本的场景旋转随机化）。
      wp     : 直接给定接近路点的关节解 {"q_ret","q_abv","q_gsp"}（缺的键照旧用 IK 求）。
               批量生成用它注入"场景旋转后的同构解"，保证抓取几何与 demo 逐位相同。
      start_pose : 起止姿态。默认 "oldros" = old_ros 录制脚本的 initial_pose
               （pan 0°/lift −100°/elbow 88.26°/wrist_flex 70.63°/wrist_roll −103.53°，
               夹爪收起），开录前 / 录完后都是它。
               "golden" = 真机遥操起点（pan −36°/lift +4°/elbow +14°/wrist_flex +55°/
               wrist_roll −104°），上一轮的默认值，留作 A/B。
               "home" = 旧预设（ctrl=0 让手臂塌到 q≈0），只用于回归对比。
    """
    m, d = a.m, a.d
    p = dict(STEP_PLAN) if plan is None else dict(plan)
    settle = SETTLE if settle is None else int(settle)

    # ---- 阶段 0：自然落定（不录，静置画面没有 SDG 价值，且占帧） ----
    mujoco.mj_resetDataKeyframe(m, d, 0)
    if spawn is not None:
        spawn(a)                     # 出生位姿：只写这一次，随后完全交给物理
    # 起始姿态（只写 qpos/ctrl，不碰药盒；落定仍由物理完成）
    a.apply_start_pose(start_pose)
    for _ in range(settle):
        mujoco.mj_step(m, d)
    p0 = a.obj_pos()

    # ---- 阶段 1：IK 求解路点（纯运动学，不步进物理） ----
    roll_in = float(np.clip(np.radians(roll), a.lo[4], a.hi[4]))
    seed = ([-0.33, -0.08, 0.80, -0.02, roll_in] if ik_seed is None
            else [float(v) for v in ik_seed])
    seed[4] = roll_in                               # roll 永远以传进来的为准
    seed = list(np.clip(seed, a.lo, a.hi))
    wp = wp or {}
    q_ret = np.asarray(wp["q_ret"], float) if "q_ret" in wp else None
    q_abv = np.asarray(wp["q_abv"], float) if "q_abv" in wp else None
    q_gsp = np.asarray(wp["q_gsp"], float) if "q_gsp" in wp else None
    if q_ret is None:
        q_ret, e1 = a.ik([p0[0], p0[1], RETRACT_Z], seed)
    else:
        e1 = 0.0
    # 抓取点横向补偿：见 GRASP_XY_OFFSET 的注释（TCP 不在两指正中）
    off = np.asarray(GRASP_XY_OFFSET if grasp_xy_offset is None else grasp_xy_offset, float)
    gx, gy = float(p0[0] + off[0]), float(p0[1] + off[1])
    if q_abv is None:
        q_abv, e2 = a.ik([gx, gy, APPROACH_Z], q_ret)
    else:
        e2 = 0.0
    if q_gsp is None:
        q_gsp, e3 = a.ik([gx, gy, gz], q_abv)
    else:
        e3 = 0.0

    rec.begin(p0)

    # ---- 阶段 0.5 "init"：起始姿态（这是录制的起点，与真机第 0 帧同姿态） ----
    # 真机数据的第一帧就是机械臂静止在初始位，所以这里也要录进来，否则仿真的第 0 帧
    # 已经是"手臂在动"的状态，与真机数据分布对不上。
    # ⚠️ init 段夹爪【保持收起的预设姿态 GRIP_PRESET】，不在起手张到最大：
    #    "张开"这件事挪到 approach 段边移动边做 —— 起手画面就像真机待机时的样子。
    #
    # ★ 非 home 起点（oldros / golden）：顺序必须是【静止 -> 竖直抬离台面 -> 再走接近段】。
    #   两种起点落定后手臂都停在"被 ctrl 保持住"的不塌陷位，且离药盒/台面都比塌陷位近，
    #   抬出这一步同时给了 PD 收敛时间，并且让"起手 → 抬离"这段与真机动作同形
    #   （old_ros 开录后动作1 就是垂直抬到安全高度）。
    # ★起始帧的爪甲必须**已经**在预设姿态 GRIP_PRESET 上，不能再动一下：
    #   apply_start_pose 给的是 GRIP_CLOSED=0.20，而本片的夹爪脚本从 0.0 开始，
    #   于是录出来的前 0.5 s 爪甲会从 0.20 "啪"地合到 0.0 —— 画面看起来就是
    #   【在药盒边上空抓了一把】（2024-11 用户实看反馈），然后再抬走、张开、回来真抓。
    #   直接写 0.0：第 0 帧起爪甲就是静止的（接触判定不受影响）。
    d.qpos[a.gadr] = GRIP_PRESET
    d.ctrl[a.cg] = GRIP_PRESET
    q_home = np.array(d.qpos[a.adr], float)
    has_init_lift = str(start_pose).lower() in POSE_WITH_INIT_LIFT
    if has_init_lift and p.get("init_lift", 0) > 0:
        _hold(a, rec, "init", p["init_hold"], GRIP_PRESET, force=False, q_arm=q_home)
        tcp0 = np.array(d.site_xpos[a.tcp], float)
        # 抬出这一段的 phase 标 "approach" 而不是 "init"：
        #   "init" 在数据里必须表示【静止的起点】（verify 的"帧间稳定"用它算亮度抖动、
        #   数据卡用它界定"第一条数据的起点画面"）。手臂正在动的帧混进去会把这两个口径搞脏。
        #
        # ★用【关节空间】插值到"同一个 x,y 抬到 RETRACT_Z"的 IK 解，而不是逐点笛卡尔：
        #   逐点位置 IK 把姿态让给了零空间，实测抬出过程中 wrist_flex 先从 55° 甩到 66.4°
        #   再回落 —— 画面是"抬手时手腕先向上翻一下"，与真机直上直下的抬离不一致。
        #   以起始姿态为种子解一次 IK 再关节插值，手腕就单调走，抬出是一条直线。
        q_lift, e_lift = a.ik(np.array([tcp0[0], tcp0[1], RETRACT_Z], float), q_home)
        _move(a, rec, "approach", q_lift, GRIP_PRESET, p["init_lift"],
              force=False, g_from=GRIP_PRESET)
    else:
        # 旧起点（home）：手臂已经塌在 q≈0，垂直抬出没有意义，那几百步就在原地保持
        # （总步数不变，所以时长/帧数两者一致，可以做 A/B 对比）。
        _hold(a, rec, "init", p["init_hold"] + p.get("init_lift", 0),
              GRIP_PRESET, force=False, q_arm=q_home)

    # ---- 阶段 2~3：接近 + 下探（夹爪在这两段里边走边张开）----
    # 张开只做到 GRIP_APPROACH=1.5（≈86°，够跨过 74mm 盒宽，不必顶到全开 2.232；
    # 前面注释里有 0.6~2.232 的实测扫描结论）。到"药盒正上方"那一帧张开量正好到位。
    #
    # 这一段从【已经抬到 RETRACT_Z 的姿态】走到药盒上方的高位撤回点 q_ret，
    # 全程在高处、且夹爪一边走一边从 GRIP_PRESET 张到 GRIP_APPROACH ——
    # 张爪放在抬出之后，就不会出现"在药盒旁边张爪把它推倒"。
    _move(a, rec, "approach", q_ret, grip_approach, p["approach_retract"],
          force=False, g_from=GRIP_PRESET)
    _move(a, rec, "approach", q_abv, grip_approach, p["approach_above"])      # 段末落帧
    _move(a, rec, "descend", q_gsp, grip_approach, p["descend"], force=False)

    # ---- 阶段 4：合爪（真实接触力，红线 2 的合规处） ----
    # 2024-11 实测（两种起点都试过）：合爪期间手臂【冻结在当前 qpos】才抓得住；
    # 把 ctrl 保持在下探目标 q_gsp 上反而全灭（抬升 0，8 个节奏 × 2 种起点全失败）——
    # 因为下探结束时手臂落后目标约 5°，继续朝目标压会把已经压进垫子 4.5mm 的药盒挤走。
    # 冻结则让那 5° 的 PD 余力自然释放、手指轻轻贴住盒面，反而落在能夹住的接触状态上。
    _ramp_gripper(a, rec, "grasp", grip_approach, GRIP_SHUT, p["grasp"])

    # ---- 阶段 4.5：停顿，确认夹稳（真机遥操里操作者也会停一下） ----
    _hold(a, rec, "settle_a", p["settle_a"], GRIP_SHUT, force=False)

    # ---- 阶段 5：竖直抬升 ----
    # ⚠️ 第六轮查证结论（别再试图"锁住工具姿态"）：本臂 lift/elbow/flex 三轴平行，
    #   工具俯仰由三者之和决定，实测俯仰随 TCP 高度【线性】增长 0.22°/mm
    #   （0.012 m -> 44.1°，0.16 m -> 76.3°）。所以抬 100 mm 必然让手腕俯仰 ~22°，
    #   握在手里的盒子跟着俯仰 —— 实测搬运段倾角 27°、抬升峰 30°。
    #   这不是接触伪影，是这条 5 自由度链在"从正上方捏薄盒再竖直抬升"任务上的固有行为；
    #   试过 ik_lock（位置+指向约束 + 零空间拉回种子）也压不下来：位置与固定指向在该高度
    #   不可同时满足（残差 31.9° 且与权重无关）。真要减小只能降低搬运高度（抬升量随之变小）
    #   或换 6 轴。这里保持原来的 q_abv 目标，动作序列与已验收的口径一致。
    _move(a, rec, "lift", q_abv, GRIP_SHUT, p["lift"])          # 段末落帧（基准）

    # ---- 阶段 5.5：抬升后停顿 ----
    _hold(a, rec, "settle_b", p["settle_b"], GRIP_SHUT, force=False)

    # ---- 阶段 6：水平搬运到木盒上方（搬运轴必须垂直于夹持轴） ----
    q_over = q_place = q_out = None
    bx = m.body_pos[a.tbox]
    if transport:
        bx0, by0 = float(bx[0]), float(bx[1])
        q_over, _ = a.ik([bx0, by0, TRANSPORT_Z], q_abv)
        _move(a, rec, "transport", q_over, GRIP_SHUT, p["transport"])  # 段末落帧（基准）

        # ---- 阶段 7 "place"：还夹着药盒，竖直下放到木盒口 ----
        # 为什么不在 z=0.16 直接松手：落差 97mm 会砸下去、横向残余速度可能把药盒
        # 甩到盒沿上翻出去。放到 PLACE_Z=0.115 时药盒底面离盒沿只剩 6mm，
        # 落差 26mm，落地速度 ≈0.7m/s，是"放进去"而不是"扔进去"。
        q_place, e4 = a.ik([bx0, by0, PLACE_Z], q_over)
        _move(a, rec, "place", q_place, GRIP_SHUT, p["place"], force=False)

        # ---- 阶段 7.5 "pre_open"：松手前静止 0.4s ----
        # 搬运末段多少还有残余摆动，趁它还连着夹爪先停稳，再松手，
        # 这样药盒的下落方向基本就是纯竖直（位置可复现，不是碰运气）。
        _hold(a, rec, "pre_open", p["pre_open"], GRIP_SHUT, force=False)

        # ---- 阶段 8 "release"：开爪，药盒靠重力掉进木盒（真物理，无 qpos 伪造）----
        # 只张开到 GRIP_RELEASE=1.2：够指尖离开药盒即可，不需要一直顶到最大。
        _ramp_gripper(a, rec, "release", GRIP_SHUT, grip_release, p["release"], force=False)

        # ---- 阶段 8.5 "drop"：等它落进木盒并静止 ----
        # 夹爪保持释放张开量、手臂不动，这段是纯粹的自由落体 + 碰撞反弹，
        # 药盒最终躺在木盒底板上（哪里、什么姿态由物理决定，脚本不干预）。
        _hold(a, rec, "drop", p["drop"], grip_release, force=False)

        # ---- 阶段 9 "retract"：竖直从木盒里提出（指尖先离开盒子内部）----
        # 边上提边收爪：从 GRIP_RELEASE 一路收到 GRIP_PRESET，退出木盒时爪子已经合上。
        q_out, e5 = a.ik([bx0, by0, RETRACT_Z], q_place)
        _move(a, rec, "retract", q_out, GRIP_PRESET, p["retract"],
              force=False, g_from=grip_release)

        # ---- 阶段 10 "home"：回初始姿态 ----
        # q_home 是 init 段采到的起始 qpos（oldros 起点时就是 OLDROS_Q，golden 时是 GOLDEN_Q），
        # 所以首末帧完全同姿态 —— 录制的开始位与结束位就是同一个预设姿态，
        # 与 old_ros 录制脚本"开录前 initial_pose、录完回 initial_pose"一致。
        _move(a, rec, "home", q_home, GRIP_PRESET, p["home"], force=False)

        # ---- 阶段 11 "tail"：收尾静止（终点帧是静止画面，便于做"回到初始位"的标签）----
        # 收尾的爪甲同样是【收起的预设姿态】，与首帧一致。
        _hold(a, rec, "tail", p["tail"], GRIP_PRESET, force=False)
    else:
        e4 = e5 = float("nan")

    return dict(p0=p0, q_ret=q_ret, q_abv=q_abv, q_gsp=q_gsp, q_over=q_over,
                start_pose=str(start_pose),
                grasp_xy_offset=[float(off[0]), float(off[1])],
                q_place=q_place, q_out=q_out, q_home=q_home,
                plan=dict(p), total_steps=int(sum(p.values())),
                grip_approach=float(grip_approach), grip_release=float(grip_release),
                ik_err_mm=(e1 * 1000, e2 * 1000, e3 * 1000, e4 * 1000, e5 * 1000),
                roll_deg=float(np.degrees(roll_in)),
                sim_time=float(d.time))


# ============================================================
# 第 4 段  CameraRig —— 相机与光栅化
# ============================================================


class CameraRig:
    """
    一个 mujoco.Renderer 轮流服务两台相机。

    为什么不用两个 Renderer：每个 Renderer 都会建一个独立的 GL 上下文，
    本机只有 mesa 软件光栅，多上下文容易在析构阶段互相踩。串行切换
    update_scene(camera=...) 的画面与两个 Renderer 等价，风险更低。
    """

    def __init__(self, model, cams=CAMS, width=640, height=480,
                 depth=False, seg=False):
        self.m = model
        self.cams = tuple(cams)
        self.w, self.h = int(width), int(height)
        self.depth_on = bool(depth)
        self.seg_on = bool(seg)

        # 相机名查表校验：名字写错就立刻炸，而不是渲染出黑图
        self.cam_ids = {}
        for c in self.cams:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c)
            if cid < 0:
                raise RuntimeError(f"模型里找不到相机 '{c}'")
            self.cam_ids[c] = cid

        try:
            self.renderer = mujoco.Renderer(model, self.h, self.w)
        except Exception as exc:
            raise SystemExit(
                f"\n[渲染后端创建失败] MUJOCO_GL={os.environ.get('MUJOCO_GL')}\n"
                f"  {type(exc).__name__}: {exc}\n"
                "本机实测三种后端:\n"
                "  egl    ✅ 可用（默认，推荐）\n"
                "  glfw   ❌ gladLoadGL error —— DISPLAY 在但建不出 GLX context\n"
                "  osmesa ❌ bad marshal data —— 本机 pyc/库版本不匹配\n"
                "显式换后端: python test_render.py render --gl egl\n")

    # ---------- 内参 ----------
    def intrinsics(self, cam):
        """
        针孔模型内参。MuJoCo 的 fovy 是垂直视场角：
            fy = H / (2 * tan(fovy/2)) ,  fx = fy（方形像素）,  cx=W/2, cy=H/2
        """
        fovy = float(self.m.cam_fovy[self.cam_ids[cam]])
        fy = self.h / (2.0 * np.tan(np.radians(fovy) / 2.0))
        return dict(fx=fy, fy=fy, cx=self.w / 2.0, cy=self.h / 2.0,
                    width=self.w, height=self.h, fovy=fovy)

    # ---------- 外参 ----------
    def extrinsics(self, data, cam):
        """
        world -> camera 的 4x4 齐次矩阵。
        MuJoCo 的 cam_xmat 是 camera->world 的旋转 R，故
            R_cw = R^T,   t_cw = -R^T · p_cam
        """
        cid = self.cam_ids[cam]
        R = np.array(data.cam_xmat[cid], float).reshape(3, 3)
        p = np.array(data.cam_xpos[cid], float)
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ p
        return T

    # ---------- 取图 ----------
    def grab(self, data):
        """
        返回 {cam: rgb(H,W,3 uint8)}，可选 {"<cam>_depth": HxW float32,
        "<cam>_seg": HxWx2 int32}。
        render() 返回的是内部缓冲视图，必须 copy，否则下一帧就被覆盖。
        """
        out = {}
        for cam in self.cams:
            if self.depth_on:
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(data, camera=cam)
                out[f"{cam}_depth"] = self.renderer.render().copy()
                self.renderer.disable_depth_rendering()
            if self.seg_on:
                self.renderer.enable_segmentation_rendering()
                self.renderer.update_scene(data, camera=cam)
                out[f"{cam}_seg"] = self.renderer.render().copy()
                self.renderer.disable_segmentation_rendering()
            self.renderer.update_scene(data, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def _project_px(rig, data, cam, P):
    """
    世界点 P -> 该相机图像平面上的像素坐标 (u, v)。点跑到相机背后时返回 None。

    约定与 CameraRig.intrinsics/extrinsics 严格一致：
      外参是 world->cam 的 4x4（R_cw, t_cw），内参是 fy=H/(2tan(fovy/2))、cx=W/2、cy=H/2。
      MuJoCo 相机看向 -z、x 向右、y 向上，所以
          u = cx + fx * Xc / (-Zc),  v = cy - fy * Yc / (-Zc)
    自检依据：把绿垫圆盘（半径 96mm）反投影回 z=0 平面，半径实测 98mm（差 2mm 是像素量化）。

    用途：运动虚影的长度要正比于"图像平面上的像素速度"，这就是那把尺子。
    """
    T = rig.extrinsics(data, cam)
    pc = T[:3, :3] @ np.asarray(P, float) + T[:3, 3]
    depth = -float(pc[2])
    if depth <= 1e-6:
        return None
    K = rig.intrinsics(cam)
    return np.array([K["cx"] + K["fx"] * pc[0] / depth,
                     K["cy"] - K["fy"] * pc[1] / depth], float)


# ============================================================
# 第 5 段  SensorModel —— 模拟真机传感器劣化
# ============================================================


class SensorModel:
    """
    相机/传感器劣化模型（2024-11 第二轮：帧间一致 + 速度驱动虚影）。

    五层劣化，顺序按物理链路从上到下排（别随手交换，交换后就不再是"从光到电"）：

        1. 运动模糊   motion blur   光学：曝光时间内物体的拖影（长度由像素速度决定）
        2. 暗角       vignette      光学：廉价镜头的渐晕（四周变暗）
        3. 曝光增益   gain          光电：整帧亮度抖动
        4. 传感器噪声 gaussian      电学：读出噪声（DN）
        5. 色彩抖动   color jitter   ISP：亮度/对比度/饱和度微调

    为什么第一轮"逐帧独立重掷"是错的（用户反馈：视频里在闪）：
        真机的自动曝光/自动白平衡是【慢收敛】的闭环，时间常数在几百毫秒到几秒；
        每帧独立 uniform(±8%) 等于把 AGC 换成了随机数发生器 —— 实测 demo 视频静止段
        （init 48 帧，机械臂不动）相邻帧平均亮度差 6.25 DN、峰值 21 DN，肉眼就是"闪"。
        现在的做法：每条视频抽一次【基准值】，帧间用一阶低通（时间常数 flicker_tau_s）
        慢慢漂，幅度整体缩到 ±2.5%~±3%。静止时相邻帧差 < 1 DN，运动/换镜头时才慢慢变。

    运动模糊为什么要"速度驱动"：
        真机（100 万像素腕带 / 200 万像素环境相机）只在【物体或镜头相对运动】时才有虚影，
        静止对着不动的机械臂拍，画面是干净的（用户在真机录像里确认过）。
        第一轮是每帧独立抽长度（75% 的帧都有），静止段也在糊，看起来像相机在"抖"。
        现在：调用方把 TCP 在图像平面上的像素速度传进来（speed_px），
        拖影长度 = clip(k*speed_px, 0, max)，静止帧为 0。

    可复现性（任务书硬要求）：
        所有随机量都来自同一个 np.random.default_rng(seed)，按固定顺序抽样；
        慢漂移也是同一 RNG 驱动 -> 同 seed + 同输入帧 -> 逐像素相同。

    注意：motion blur 用的是离散化的直线核（cv2.filter2D），不是高斯；
          高斯是"失焦"，直线才是"运动"，两者视觉上完全不同，别拿 GaussianBlur 凑。
    """

    # 运动模糊的候选角度（方向在曝光时间内基本固定，取 8 档够用）
    _MOTION_ANGLES = 8

    def __init__(self, blur_sigma=0.8, noise_sigma=3.0, gain_jitter=0.01, seed=0,
                 motion_blur=True, motion_prob=0.75, motion_px=(1.0, 3.0),
                 motion_k=0.6, motion_max_px=4.0, motion_deadzone_px=0.2,
                 vignette=0.30, vignette_pow=2.0, color_jitter=0.03,
                 flicker_tau_s=2.0, fps=30.0):
        self.blur_sigma = float(blur_sigma)
        self.noise_sigma = float(noise_sigma)
        self.gain_jitter = float(gain_jitter)
        self.motion_blur = bool(motion_blur)
        self.motion_prob = float(np.clip(motion_prob, 0.0, 1.0))
        self.motion_px = (float(motion_px[0]), float(motion_px[1]))
        self.motion_k = float(motion_k)
        self.motion_max_px = float(motion_max_px)
        self.motion_deadzone_px = float(motion_deadzone_px)
        self.vignette = float(max(0.0, vignette))
        self.vignette_pow = float(max(0.5, vignette_pow))
        self.color_jitter = float(max(0.0, color_jitter))
        self.flicker_tau_s = float(max(0.05, flicker_tau_s))
        self.fps = float(fps)
        # 一阶低通系数：alpha = 1 - exp(-dt/tau)，30fps/tau=2s -> 每帧只走 1.6% 的路
        self._alpha = float(1.0 - np.exp(-1.0 / (self.fps * self.flicker_tau_s)))

        self.rng = np.random.default_rng(int(seed))
        # 高斯核尺寸按 3σ 原则取，且强制为奇数
        if self.blur_sigma > 0:
            self.ksize = max(3, int(round(self.blur_sigma * 6)) | 1)
        else:
            self.ksize = 0
        self._kernels = {}            # (长度, 角度索引) -> 归一化直线核
        self._vig = None              # 暗角掩膜缓存
        self._vig_key = None
        self._luma = np.array([0.299, 0.587, 0.114], np.float32)

        # ---- 每条视频抽一次的基准值（不是每帧重掷！）----
        j = self.color_jitter
        self.base = dict(
            brightness=1.0 + float(self.rng.uniform(-j * 0.75, j * 0.75)),
            contrast=1.0 + float(self.rng.uniform(-j, j)),
            saturation=1.0 + float(self.rng.uniform(-j, j)),
            vignette=float(self.rng.uniform(-0.015, 0.015)) if self.vignette > 0 else 0.0,
        )
        self.cur = dict(self.base)    # 当前帧的慢漂移值（低通后的状态）
        # 每帧抽到的随机量落到这里，meta.json 里能查到"这一条视频是怎么劣化的"
        self._drawn = []

    # ---------- 运动模糊核：直线上的权重做成核 ----------
    def _motion_kernel(self, length, angle_idx):
        key = (round(float(length), 2), int(angle_idx) % self._MOTION_ANGLES)
        if key in self._kernels:
            return self._kernels[key]
        L = max(1.0, float(length))
        half = max(1, int(np.ceil(L / 2.0)))
        k = 2 * half + 1
        ker = np.zeros((k, k), np.float32)
        ang = 2.0 * np.pi * key[1] / self._MOTION_ANGLES
        ca, sa = float(np.cos(ang)), float(np.sin(ang))
        c = half
        # 沿方向取 2*half+1 个采样点，双线性累加（比整数取整更不容易出现断点）
        for t in np.linspace(-L / 2.0, L / 2.0, 2 * half + 1):
            x, y = c + t * ca, c + t * sa
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            fx, fy = x - x0, y - y0
            for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                              (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                xi, yi = x0 + dx, y0 + dy
                if 0 <= xi < k and 0 <= yi < k:
                    ker[yi, xi] += w
        s = ker.sum()
        ker = ker / s if s > 0 else ker
        self._kernels[key] = ker
        return ker

    # ---------- 暗角掩膜：只跟分辨率有关，缓存起来 ----------
    def _vignette(self, h, w):
        if self._vig_key != (h, w):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
            # 归一化半径：中心 0，四角 1；幂次越大暗角越"收在边上"
            r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2.0)
            self._vig = (1.0 - self.vignette * (r ** self.vignette_pow))[..., None]
            self._vig_key = (h, w)
        return self._vig

    def reset(self):
        """清空统计（不会重置 RNG 状态与基准值；换 seed 请新建实例）。"""
        self._drawn.clear()

    def params(self):
        """
        本次用到的参数（写进 meta.json 的 sensor 段）。
        逐帧随机量只回统计（均值/最小/最大），不逐帧列 —— meta.json 已经 400 KB 了，
        再把每帧的抖动塞进去会把文件撑大一个量级，而这些量本来就由 seed 决定。
        """
        names = ("blur_px", "motion_angle_deg", "exposure", "brightness",
                 "contrast", "saturation")
        out = dict(motion_blur=bool(self.motion_blur), motion_prob=self.motion_prob,
                   motion_px=list(self.motion_px), motion_k=self.motion_k,
                   motion_max_px=self.motion_max_px,
                   vignette=self.vignette, vignette_pow=self.vignette_pow,
                   color_jitter=self.color_jitter, flicker_tau_s=self.flicker_tau_s,
                   base={k: round(float(v), 4) for k, v in self.base.items()},
                   n_calls=len(self._drawn))     # = 帧数 × 相机数（每台相机各自抽一次）
        if self._drawn:
            arr = np.array(self._drawn, float)
            for i, k in enumerate(names):
                out[k] = [round(float(arr[:, i].mean()), 4),
                          round(float(arr[:, i].min()), 4),
                          round(float(arr[:, i].max()), 4)]
        return out

    def _slow_drift(self):
        """
        一阶低通推进"慢漂移"：每帧朝基准值附近的一个小扰动走 alpha 步。

        为什么不是随机游走：随机游走会累积漂到很远（0.9 或 1.1），
        真机 AGC 是【围绕工作点】收敛的。所以这里每帧抽一个目标，再低通追过去，
        幅度由 color_jitter 控制，时间常数由 flicker_tau_s 控制。
        """
        a = self._alpha
        j = self.color_jitter
        tgt = dict(
            brightness=1.0 + float(self.rng.uniform(-j * 0.75, j * 0.75)),
            contrast=1.0 + float(self.rng.uniform(-j, j)),
            saturation=1.0 + float(self.rng.uniform(-j, j)),
            vignette=float(self.rng.uniform(-0.015, 0.015)) if self.vignette > 0 else 0.0,
        )
        for k in self.cur:
            self.cur[k] += (tgt[k] - self.cur[k]) * a

    def apply(self, rgb, speed_px=None):
        """
        rgb: HxWx3 uint8 -> 劣化后的 HxWx3 uint8。

        speed_px: 该帧在【图像平面】上的像素速度（mag，px/帧），由调用方从轨迹算出来。
                  None 时退回旧的"随机长度"行为（兼容 collect.py 等旧调用）。
                  给了就按 blur_px = clip(motion_k*speed_px, 0, motion_max_px)，
                  小于 motion_deadzone_px 视为静止 -> 完全不糊。
        """
        img = rgb
        rng = self.rng
        h, w = img.shape[:2]

        # ---- 1) 运动模糊（光学）：长度跟像素速度走，方向随机 ----
        mp = 0.0
        ang_deg = 0.0
        if self.motion_blur:
            if speed_px is None:
                if rng.random() < self.motion_prob:
                    lo, hi = self.motion_px
                    mp = float(rng.uniform(lo, hi))
            else:
                sp = float(speed_px)
                mp = min(self.motion_max_px, max(0.0, self.motion_k * sp))
                if mp < self.motion_deadzone_px:
                    mp = 0.0
            if mp > 0.0:
                ai = int(rng.integers(0, self._MOTION_ANGLES))
                ang_deg = float(ai) * 360.0 / self._MOTION_ANGLES
                ker = self._motion_kernel(round(mp * 2) / 2, ai)   # 量化到 0.5 px，命中缓存
                img = cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REPLICATE)
        # 镜头/像素的轻微失焦（任务书原要求的高斯模糊），与运动模糊是两回事，都保留
        if self.ksize:
            img = cv2.GaussianBlur(img, (self.ksize, self.ksize), self.blur_sigma)

        # ---- 2) 慢漂移（取代第一轮的逐帧重掷）----
        self._slow_drift()
        cur = self.cur

        # ---- 3) 暗角（光学）+ 暗角强度的慢漂移 ----
        if self.vignette > 0:
            vig = self._vignette(h, w)
            out = img.astype(np.float32) * (vig * (1.0 + cur["vignette"]))
        else:
            out = img.astype(np.float32)

        # ---- 4) 曝光增益（光电）：真机 AGC 逐帧微动，幅度小 ----
        exposure = 1.0
        if self.gain_jitter > 0:
            exposure = 1.0 + float(rng.uniform(-self.gain_jitter, self.gain_jitter))
            out *= exposure

        # ---- 5) 传感器噪声（电学）----
        if self.noise_sigma > 0:
            out += rng.normal(0.0, self.noise_sigma, out.shape).astype(np.float32)

        # ---- 6) 色彩抖动（ISP）：饱和度 -> 对比度 -> 亮度（都是慢漂移值）----
        sat, con, bri = cur["saturation"], cur["contrast"], cur["brightness"]
        if sat != 1.0:
            gray = out @ self._luma                       # HxW
            out = gray[..., None] + sat * (out - gray[..., None])
        if con != 1.0:
            out = (out - 127.5) * con + 127.5
        if bri != 1.0:
            out *= bri

        self._drawn.append((mp, ang_deg, exposure, bri, con, sat))
        return np.clip(out, 0, 255).astype(np.uint8)


# ============================================================
# 第 6 段  VideoSink / 图片落盘
# ============================================================


class VideoSink:
    """
    cv2.VideoWriter 封装（MP4 / h264）。注意 MuJoCo 给 RGB，OpenCV 要 BGR。

    容器用 mp4+h264 而不是 AVI(MJPG)：训练侧（lerobot/pyav/ffmpeg/浏览器）直接吃得下，
    不需要"再转一次格式才能看/才能训"。所以本脚本不再有 avi 产物，
    `render/make_video.py`（npz -> AVI）随之退役。
    """

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = (int(size[0]), int(size[1]))          # (W, H)
        self.fps = float(fps)
        fourcc = cv2.VideoWriter_fourcc(*"avc1")          # h264；不改回 mp4v/MJPG
        self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"VideoWriter 打不开: {self.path} (size={self.size}, fps={self.fps})")
        self.count = 0

    def write(self, rgb):
        h, w = rgb.shape[:2]
        if (w, h) != self.size:
            raise RuntimeError(f"帧尺寸 {w}x{h} 与 VideoWriter {self.size} 不符，会静默丢帧")
        self.writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))
        self.count += 1

    def close(self):
        self.writer.release()


def save_png(path: Path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def make_contact_sheet(tiles: dict, times, path: Path):
    """
    拼图：每台相机一行，每行挑 4 个时刻，左上角打 ASCII 标签。
    （cv2.putText 不支持中文，标签一律英文）
    """
    rows = []
    for cam, imgs in tiles.items():
        line = []
        for img, t in zip(imgs, times):
            t_img = img.copy()
            cv2.rectangle(t_img, (0, 0), (t_img.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(t_img, f"{cam}  t={t:.2f}s", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            line.append(t_img)
        rows.append(np.hstack(line))
    sheet = np.vstack(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet[:, :, ::-1])


# ============================================================
# 第 7 段  record 子命令
# ============================================================


def motion_stats(rec: EpisodeRecorder, a: Arm, static_eps=1e-3, skip=("init",)):
    """
    动作密度统计 —— 用来客观回答"还剩下多少静止帧"。

    判据：相邻两帧"5 个臂关节 + 夹爪关节"的最大变化 < static_eps rad 就算静止帧。
    ★ 夹爪必须算进去：开爪/合爪时臂关节是不动的，但画面明明在动——
      不把夹爪算进来会把"释放"误判成静止（第一版就踩了这个坑：
      测出 1.37 s 静止段，其实是 pre_open 0.4 + release 0.6 + drop 0.8 被并成了一段）。
      moving_ratio   非静止帧占比（越大越"密"）
      longest_static 最长连续静止段（帧数 / 秒）；默认跳过 init 段
                     （数据集起点，真机每条 episode 开头本来就是静止在初始位）
    真机 15 s 遥操里也有停顿，但不会有 4 s 长的死寂，所以用 1.0 s 作验收线。
    """
    q = np.asarray(rec._rows["qpos"])
    ph = np.asarray(rec._rows["phase"])
    gadr = a.m.jnt_qposadr[mujoco.mj_name2id(
        a.m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")]
    cols = np.concatenate([a.adr, [gadr]])
    dq = np.abs(np.diff(q[:, cols], axis=0)).max(axis=1) if len(q) > 1 else np.zeros(0)
    static = dq < static_eps                     # len = N-1，第 k 个元素描述"k -> k+1"
    keep = ~np.isin(ph[1:], skip)
    longest, cur = 0, 0
    for s, k in zip(static, keep):
        if not k:
            cur = 0
            continue
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    n_keep = int(keep.sum())
    n_static = int((static & keep).sum())
    return dict(frames=int(len(q)), moving_ratio=float(1.0 - n_static / max(1, n_keep)),
                longest_static_frames=int(longest),
                longest_static_sec=float(longest / rec.fps),
                static_frames=n_static, counted_frames=n_keep)


def episode_report(rec: EpisodeRecorder, a: Arm, info: dict):
    """尾段的一次性体检：释放高度、落点、归位误差、有没有刮到木盒。"""
    r = rec._rows
    ph = np.asarray(r["phase"])
    obj = np.asarray(r["obj_pos"])
    q = np.asarray(r["qpos"])
    box_n = np.asarray(r["box_n"])
    spd = np.asarray(r["obj_speed"])
    i_rel = np.where(ph == "release")[0]
    i_rel = int(i_rel[-1]) if len(i_rel) else None
    i_pre = np.where(ph == "pre_open")[0]
    i_pre = int(i_pre[-1]) if len(i_pre) else i_rel      # 松手前一刻的高度
    q0 = q[0, a.adr]
    q1 = q[-1, a.adr]
    tail = spd[-10:] if len(spd) >= 10 else spd
    return dict(
        obj_pre_release=[float(v) for v in obj[i_pre]] if i_pre is not None else None,
        tcp_pre_release=[float(v) for v in r["tcp_pos"][i_pre]] if i_pre is not None else None,
        obj_at_release_end=[float(v) for v in obj[i_rel]] if i_rel is not None else None,
        obj_final=[float(v) for v in obj[-1]],
        drop_mm=float((obj[i_pre, 2] - obj[-1, 2]) * 1000.0) if i_pre is not None else float("nan"),
        obj_speed_end=float(np.abs(tail).max()) if len(tail) else float("nan"),
        home_dq_deg=[float(v) for v in np.degrees(np.abs(q1 - q0))],
        home_dq_max_deg=float(np.degrees(np.abs(q1 - q0)).max()),
        grip_ctrl_end=float(np.asarray(r["ctrl"])[-1][a.cg]),
        box_hit_frames=int((box_n > 0).sum()),
        motion=motion_stats(rec, a),
    )


def cmd_record(args):
    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 时间账只允许有一份加法表：改了某一段忘了配平会立刻炸，不会静默改帧数
    _total = sum(STEP_PLAN.values())
    if _total != TOTAL_STEPS:
        raise SystemExit(f"STEP_PLAN 合计 {_total} 步 != {TOTAL_STEPS} 步，"
                         f"请重新配平（差 {_total - TOTAL_STEPS:+d} 步）")

    a = Arm(SCENE_XML)
    rec = EpisodeRecorder(a, fps=FPS)

    print("=" * 78)
    print("record  真物理轨迹录制（mj_step，零 qpos 伪造）")
    print("=" * 78)
    print(f"  模型      : {SCENE_XML}   (mujoco {mujoco.__version__})")
    print(f"  时间步长  : {a.m.opt.timestep} s   录制帧率: {rec.fps:.0f} fps（时间栅格）")
    print(f"  计划时长  : {_total} 步 x {a.m.opt.timestep}s = "
          f"{_total*a.m.opt.timestep:.2f} s -> 约 {int(_total*a.m.opt.timestep*rec.fps)} 帧")
    print(f"  roll      : {args.roll}°   gz: {args.gz} m   place_z: {PLACE_Z} m   "
          f"搬运: {'是' if not args.no_transport else '否'}")
    print(f"  夹爪      : 预设 {GRIP_PRESET} / 下探 {args.grip_open} / "
          f"释放 {args.grip_release} / 夹住 {GRIP_SHUT}")

    info = run_episode(a, rec, roll=args.roll, gz=args.gz,
                       transport=not args.no_transport,
                       grip_approach=args.grip_open, grip_release=args.grip_release,
                       start_pose=args.start_pose)

    print()
    print(f"  {'phase':<11}{'frames':>7}{'z_start':>10}{'z_end':>10}{'Fn_max':>9}{'握持帧':>8}")
    print("  " + "-" * 60)
    for s in rec.summary():
        print(f"  {s['phase']:<11}{s['frames']:>7}{s['z0']:>10.4f}{s['z1']:>10.4f}"
              f"{s['fn_max']:>9.4f}{s['held']:>8}")

    rep = episode_report(rec, a, info)
    print()
    print("  尾段体检:")
    if rep["obj_pre_release"]:
        print(f"    松手前药盒 z    : {rep['obj_pre_release'][2]:.4f} m   "
              f"(TCP z={rep['tcp_pre_release'][2]:.4f})")
        print(f"    掉落落差        : {rep['drop_mm']:.1f} mm   "
              f"(释放结束帧药盒 z={rep['obj_at_release_end'][2]:.4f})")
    print(f"    药盒终点        : [{rep['obj_final'][0]:.4f} {rep['obj_final'][1]:.4f} "
          f"{rep['obj_final'][2]:.4f}]")
    print(f"    落地速度        : {rep['obj_speed_end']:.3f} m/s (需 < 0.05)")
    print(f"    归位误差(各关节): {' '.join(f'{v:.1f}°' for v in rep['home_dq_deg'])}"
          f"   最大 {rep['home_dq_max_deg']:.1f}°")
    print(f"    末帧夹爪 ctrl   : {rep['grip_ctrl_end']:.3f} (预设收起={GRIP_PRESET})")
    print(f"    臂↔木盒接触帧   : {rep['box_hit_frames']} (需 0)")
    mv = rep["motion"]
    print(f"    动作密度        : 非静止帧 {mv['moving_ratio']*100:.0f}%   "
          f"最长连续静止 {mv['longest_static_frames']} 帧 = "
          f"{mv['longest_static_sec']:.2f} s（不含 init，需 <= 1.0）")

    meta = dict(
        model=SCENE_XML, mujoco=mujoco.__version__,
        nq=int(a.m.nq), nv=int(a.m.nv), nu=int(a.m.nu),
        timestep=float(a.m.opt.timestep), decimate=int(rec.decimate),
        fps=float(rec.fps), step_counts=dict(STEP_PLAN),
        start_pose=str(info.get("start_pose", "home")),
        # 起止姿态的实际关节值（度，ARM_JOINTS 顺序）。取的是 init 段落定后的 qpos，
        # 不是"请求的常量" —— 落定期有微小 PD 偏差，下游要的是真实第 0 帧姿态。
        start_joints_deg=[round(float(v), 3) for v in np.degrees(info["q_home"])],
        # 旧字段：只有 golden 起点才有，保留是为了让 sdg-v2 时代的老数据/脚本还能读。
        golden_joints_deg=dict(GOLDEN_JOINTS) if str(info.get("start_pose")) == "golden" else None,
        dataset_version="sdg-v3-demo",
        mjcf_sha256={Path(p).name: _sha256(Path(p))
                     for p in (SCENE_XML, Path(SCENE_XML).parent / "so_arm101.xml")},
        roll_deg=info["roll_deg"], gz=float(args.gz), place_z=PLACE_Z,
        grip=dict(preset=GRIP_PRESET, approach=float(args.grip_open),
                  release=float(args.grip_release), shut=GRIP_SHUT),
        retract_z=RETRACT_Z, approach_z=APPROACH_Z, transport_z=TRANSPORT_Z,
        release_grip=[GRIP_SHUT, float(args.grip_release)],
        ik_err_mm=[round(v, 4) for v in info["ik_err_mm"]],
        obj_start=[round(float(v), 6) for v in info["p0"]],
        sim_time=info["sim_time"],
        tail_report={k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in rep.items() if k != "motion"},
        motion=mv,
        source="record (mj_step, 真实接触力)",
        # 指标口径在此固化，render/verify 都读这里，避免两处各算一遍算错
        metrics=dict(rise_mm=round(rec.metrics()["rise_mm"], 4),
                     hdisp_mm=round(rec.metrics()["hdisp_mm"], 4),
                     overall_rise_mm=round(rec.metrics()["overall_rise_mm"], 4)),
        # 接触真实性（第六轮）：从轨迹数组算，写进 meta 供 verify 与下游体检用
        contact=dict(contact_audit_from_arrays(rec.arrays()),
                     contact_mode=("pad-pinch" if mujoco.mj_name2id(
                         a.m, mujoco.mjtObj.mjOBJ_GEOM, "pad_gripper_link") >= 0
                         else "mesh-hull")),
    )
    # ---- 成功标签（只读轨迹数组，不参与物理/控制）----
    # 同时写进 meta（-> meta.json 顶层 success）和 npz 里的 meta_json，
    # 这样"只拿到 episode.npz 的下游"和"只拿到 meta.json 的下游"看到的是同一个标签。
    # 注意：抓取失败时这里不会中止任何东西，npz 照常落盘（负样本也是数据）。
    succ = evaluate_success(meta, rec.arrays())
    meta["success"] = bool(succ["success"])
    meta["success_info"] = succ
    path = rec.save(out_dir / args.npz_name, meta)

    # ---- 落盘后立刻按数据契约核对一遍：形状/dtype 不对就别交付 ----
    problems = validate_trajectory(path, nq=meta["nq"], nv=meta["nv"], nu=meta["nu"])
    if problems:
        for x in problems:
            print(f"  [schema] ❌ {x}")
        raise SystemExit(f"轨迹 npz 契约校验失败（{len(problems)} 项）: {path}")
    print(f"  [schema] ✅ 轨迹契约通过  {len(rec.arrays())} 键  {path.name}")

    mt = rec.metrics()
    print()
    print(f"  抬升(阶段5)  : {mt['rise_mm']:+.2f} mm   "
          f"(test_grasp 基准 {REF_LIFT_MM} ± {TOL_MM})")
    print(f"  水平位移     : {mt['hdisp_mm']:.2f} mm   (基准 {REF_HDISP_MM} ± {TOL_MM})")
    print(f"  整体抬升     : {mt['overall_rise_mm']:+.2f} mm   (相对初始位, 仅供参考)")
    print(f"  药盒终点     : [{mt['obj_end'][0]:.4f} {mt['obj_end'][1]:.4f} {mt['obj_end'][2]:.4f}]")
    ok = (abs(mt["rise_mm"] - REF_LIFT_MM) <= TOL_MM
          and abs(mt["hdisp_mm"] - REF_HDISP_MM) <= TOL_MM)
    print(f"  与步骤二一致性: {'✅ 通过' if ok else '❌ 不一致（动作序列被改动？）'}")
    # 成功/失败标签：这是数据工程要的那个顶层字段，物理失败也照样打印、照样导出
    print(f"  success   : {str(succ['success']).lower()}   "
          f"抬升 {mt['rise_mm']:+.1f} mm (判据 >= {SUCCESS_LIFT_MM:.0f} mm)"
          f"{'' if succ['success'] else '   reason=' + ','.join(succ['reasons'])}")
    print(f"  帧数      : {len(rec)}")
    print(f"  已保存    : {path}  ({path.stat().st_size/1024:.1f} KB)")
    print(f"  耗时      : {time.time()-t_start:.2f} s")
    return 0 if ok else 1


# ============================================================
# 第 8 段  render 子命令 —— 纯正运动学回放 + 渲染
# ============================================================


def cmd_render(args):
    t_start = time.time()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ep, rec_meta = load_episode(Path(args.npz))
    n_all = len(ep["qpos"])
    idx = np.arange(0, n_all, max(1, int(args.stride)))
    n_frames = len(idx)

    m = mujoco.MjModel.from_xml_path(SCENE_XML)
    d = mujoco.MjData(m)
    rig = CameraRig(m, width=args.width, height=args.height,
                    depth=args.depth, seg=args.seg)
    fps = args.fps if args.fps else rec_meta["fps"] / max(1, int(args.stride))
    sensor = SensorModel(blur_sigma=args.blur_sigma, noise_sigma=args.noise_sigma,
                         gain_jitter=args.gain_jitter, seed=args.seed,
                         motion_blur=args.motion_blur,
                         motion_prob=args.motion_prob,
                         motion_px=(args.motion_px_min, args.motion_px_max),
                         motion_k=args.motion_k, motion_max_px=args.motion_max_px,
                         vignette=args.vignette, vignette_pow=args.vignette_pow,
                         color_jitter=args.color_jitter,
                         flicker_tau_s=args.flicker_tau, fps=fps)

    width, height = int(args.width), int(args.height)

    print("=" * 78)
    print("render  纯运动学回放 + 光栅化 + 传感器劣化")
    print("=" * 78)
    print(f"  轨迹      : {args.npz}  ({n_all} 帧, step={args.stride} -> {n_frames} 帧)")
    print(f"  分辨率    : {width}x{height}  fps={fps:.2f}  "
          f"GL={os.environ.get('MUJOCO_GL')}")
    print(f"  传感器    : blur={args.blur_sigma} noise={args.noise_sigma} "
          f"gain±{args.gain_jitter} seed={args.seed}")
    print(f"  视觉随机化: 运动虚影 {'开' if args.motion_blur else '关'}"
          f"(随速度, k={args.motion_k}, 上限 {args.motion_max_px}px)  "
          f"暗角 {args.vignette:.2f}  "
          f"慢漂移 ±{args.color_jitter:.2f} @tau={args.flicker_tau}s")
    print(f"  相机      : {', '.join(CAMS)}")

    # ---- 成功标签（只读轨迹数组；不参与回放、不参与渲染）----
    # 负样本（没抓起来 / 中途滑落）同样走完整条渲染+导出流程，
    # 只是 meta.json 里 success=false —— 训练侧拿它当负样本，不当故障。
    succ = evaluate_success(rec_meta, ep)
    print(f"  success   : {str(succ['success']).lower()}   "
          f"抬升 {succ['lift_mm']} mm (判据 >= {SUCCESS_LIFT_MM:.0f} mm)"
          f"{'' if succ['success'] else '   reason=' + ','.join(succ['reasons'])}"
          f"{'' if succ['success'] else '   -> 负样本：照常渲染并导出'}")

    sinks = {c: VideoSink(out_dir / f"{c}.mp4", fps, (width, height)) for c in CAMS}
    sheet_tiles = {c: [] for c in CAMS}
    sheet_idx = np.unique(np.linspace(0, n_frames - 1, 4).astype(int))
    sheet_pick = set(sheet_idx.tolist())

    frames_meta = []
    frames_dir = out_dir / "frames"
    if args.frames:
        for c in CAMS:
            (frames_dir / c).mkdir(parents=True, exist_ok=True)
            if args.depth:
                (out_dir / "depth" / c).mkdir(parents=True, exist_ok=True)
            if args.seg:
                (out_dir / "seg" / c).mkdir(parents=True, exist_ok=True)

    # 图片数组缓冲：导出 episode_0.npz（键名 env / wrist），供 make_video.py 消费。
    # 只存 RGB uint8 原图，不存状态 —— 视频脚本不需要物理，只需要像素。
    img_buf = {c: [] for c in CAMS} if args.export_npz else None

    # 运动虚影要"跟着真实运动"，所以每帧先算 TCP 在各相机图像平面上的像素速度：
    #   speed_px = |uv(t) - uv(t-1)|，静止帧为 0 -> 不糊；下探/搬运 -> 1.5~3 px。
    # 相机给的是 world->cam（rig.extrinsics），投影约定与 verify/诊断脚本同口径：
    #   相机前向 -z、x 右、y 上 -> u = cx + fx*X/(-Z), v = cy - fy*Y/(-Z)
    prev_uv = {c: None for c in CAMS}

    def _tcp_speed_px(c, data):
        uv = _project_px(rig, data, c, ep["tcp_pos"][fi])
        sp = 0.0
        if uv is not None and prev_uv[c] is not None:
            sp = float(np.linalg.norm(uv - prev_uv[c]))
        prev_uv[c] = uv
        return sp

    t_render0 = time.time()
    for k, fi in enumerate(idx):
        # ---- 纯正运动学回放：只写 qpos，然后 mj_forward 刷新位姿 ----
        # 不写 ctrl、不调 mj_step —— 这就是"剥离控制逻辑"的物理保证。
        d.qpos[:] = ep["qpos"][fi]
        mujoco.mj_forward(m, d)

        raw = rig.grab(d)
        speeds = {c: _tcp_speed_px(c, d) for c in CAMS}
        for c in CAMS:
            noisy = sensor.apply(raw[c], speed_px=speeds[c])
            sinks[c].write(noisy)
            if args.frames:
                save_png(frames_dir / c / f"{k:05d}.png", noisy)
            if img_buf is not None:
                img_buf[c].append(noisy)
            if k in sheet_pick:
                sheet_tiles[c].append(noisy.copy())
            if args.raw:
                save_png(out_dir / "raw" / c / f"{k:05d}.png", raw[c])

        # 深度 / 分割只在 --frames 时落盘：单帧是 npy 数组，单帧百 KB 量级，
        # 不开 --frames 就没有逐帧目录可放，也避免无意义的磁盘开销
        if args.frames:
            for c in CAMS:
                if args.depth:
                    np.save(out_dir / "depth" / c / f"{k:05d}.npy", raw[f"{c}_depth"])
                if args.seg:
                    np.save(out_dir / "seg" / c / f"{k:05d}.npy", raw[f"{c}_seg"])

        # ---- 逐帧标签 ----
        fm = dict(
            i=int(k), src=int(fi), t=round(float(ep["time"][fi]), 4),
            phase=str(ep["phase"][fi]),
            tcp=[round(float(v), 5) for v in ep["tcp_pos"][fi]],
            obj_pos=[round(float(v), 5) for v in ep["obj_pos"][fi]],
            obj_quat=[round(float(v), 5) for v in ep["obj_quat"][fi]],
            lift_mm=round(float(ep["lift_mm"][fi]), 3),
            hdisp_mm=round(float(ep["hdisp_mm"][fi]), 3),
            contact_n=int(ep["contact_n"][fi]),
            fn_jaw=round(float(ep["fn_jaw"][fi]), 4),
            fn_grip=round(float(ep["fn_grip"][fi]), 4),
            fn_sum=round(float(ep["fn_sum"][fi]), 4),
            ctrl=[round(float(v), 5) for v in ep["ctrl"][fi]],
            cam_T_cw={c: [round(float(v), 6) for v in rig.extrinsics(d, c).ravel()]
                      for c in CAMS},
        )
        frames_meta.append(fm)

        if (k + 1) % max(1, n_frames // 10) == 0 or k == n_frames - 1:
            print(f"  [{k+1:>4}/{n_frames}] t={fm['t']:.2f}s "
                  f"{fm['phase']:<9} obj_z={fm['obj_pos'][2]:.4f} "
                  f"lift={fm['lift_mm']:>7.2f}mm  握持={fm['contact_n']} "
                  f"Fn={fm['fn_sum']:.4f}N  ({(time.time()-t_render0)/(k+1)*1000:.0f} ms/帧)")

    for c in CAMS:
        sinks[c].close()

    # ---- 导出 episode_0.npz（键: env / wrist，RGB uint8，N x H x W x 3）----
    # 这是给 make_video.py 用的数据契约：只要像素，不要物理状态。
    # 键名固定为 env/wrist（不是 env_camera/wrist_camera），与视频脚本一致。
    img_npz_path = out_dir / args.export_name
    if img_buf is not None:
        np.savez(img_npz_path,
                 env=np.stack(img_buf[CAM_ENV]),
                 wrist=np.stack(img_buf[CAM_WRIST]),
                 fps=np.float32(fps))

    # 抓取瞬间的下标：meta 的外参要用它，拼图/预览也要用它 —— 所以**总是**算（纯下标运算，不渲染）
    ph = np.array([str(p) for p in ep["phase"]])
    grasp_end = int(np.where(ph == "grasp")[0][-1]) if (ph == "grasp").any() else n_frames - 1
    pick = int(np.argmin(np.abs(idx - grasp_end)))

    # ---- 拼图 + 预览单帧（只给肉眼排查用；默认不出，--inspect 才出）----
    if args.inspect:
        times = [float(ep["time"][idx[i]]) for i in sheet_idx]
        make_contact_sheet(sheet_tiles, times, out_dir / "contact_sheet.png")

        d.qpos[:] = ep["qpos"][idx[pick]]
        mujoco.mj_forward(m, d)
        prev = rig.grab(d)
        for c in CAMS:
            save_png(out_dir / f"preview_{c.replace('_camera', '')}.png", prev[c])

    # ---- meta.json ----
    meta = dict(
        generator="test_render.py (render)",
        model=SCENE_XML, mujoco=mujoco.__version__,
        source_npz=str(args.npz), record_meta=rec_meta,
        width=width, height=height, fps=round(float(fps), 3),
        n_frames=int(n_frames), stride=int(args.stride),
        timestep=float(m.opt.timestep),
        # 传感器段：既有"配置"（args），也有"这一条视频实际抽到的随机量"（sensor.params()）
        sensor=dict(blur_sigma=args.blur_sigma, noise_sigma=args.noise_sigma,
                    gain_jitter=args.gain_jitter, seed=int(args.seed),
                    motion_blur=bool(args.motion_blur),
                    motion_prob=float(args.motion_prob),
                    motion_px=[float(args.motion_px_min), float(args.motion_px_max)],
                    motion_k=float(args.motion_k),
                    motion_max_px=float(args.motion_max_px),
                    vignette=float(args.vignette),
                    vignette_pow=float(args.vignette_pow),
                    color_jitter=float(args.color_jitter),
                    flicker_tau_s=float(args.flicker_tau),
                    # 2024-11 第二轮：慢漂移取代逐帧重掷，静止段帧间亮度差从 6.25 DN 降到 <1 DN
                    drift=dict(algo="一阶低通（目标=基准值附近的小扰动）",
                               alpha=round(sensor._alpha, 5),
                               tau_s=sensor.flicker_tau_s),
                    drawn=sensor.params()),
        cameras={c: dict(name=c, intrinsics=rig.intrinsics(c)) for c in CAMS},
        phases=list(PHASES),
        # ★ 顶层成功标签（任务书要求）：抓取失败时是 false，而不是"没有这条数据"。
        #   判据与原因见 success_info（含 threshold_mm / lift_mm / hold_mm / reasons）。
        success=bool(succ["success"]),
        success_info=succ,
        success_threshold_mm=SUCCESS_LIFT_MM,
        frames=frames_meta,
    )
    # 相机外参：取抓取瞬间那一帧的 world->cam，转成列主序以便下游按惯例读取
    for c in CAMS:
        T = np.array(frames_meta[pick]["cam_T_cw"][c], float).reshape(4, 4)
        meta["cameras"][c]["extrinsics_cw_col_major"] = [round(float(v), 6) for v in T.T.ravel()]

    # ---- 数据契约校验（放在写 meta.json 之前）----
    # 校验不通过直接 SystemExit：宁可这条 episode 报错，也不要一条形状/dtype 错位
    # 的数据悄悄流进训练集（下游通常不会报错，只会训出一个莫名其妙的模型）。
    if args.no_schema:
        meta["schema_ok"] = None          # 主动跳过就如实记 null，不伪装成通过
        print("\n  [schema] ⏭  已用 --no-schema 跳过契约校验")
    else:
        meta["schema"] = validate_npz_schema(
            Path(args.npz),
            img_npz=(img_npz_path if img_buf is not None else None),
            n_frames=n_frames, rec_meta=rec_meta)
        meta["schema_ok"] = True

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

    rig.close()

    print()
    print(f"  视频      : {out_dir/'env_camera.mp4'} ({sinks[CAM_ENV].count} 帧)")
    print(f"              {out_dir/'wrist_camera.mp4'} ({sinks[CAM_WRIST].count} 帧)")
    if img_buf is not None:
        mb = img_npz_path.stat().st_size / 1024 / 1024
        print(f"  图片数组  : {img_npz_path}  ({mb:.1f} MB, 键 env/wrist)  [--export-npz]")
    print(f"  排查产物  : " + (f"拼图 {out_dir/'contact_sheet.png'} + 预览 preview_*.png（--inspect）"
                            if args.inspect else "未生成（要排查请跑 tools/inspect_episode.py，或加 --inspect）"))
    # 这条视频实际抽到的视觉随机量（均值/最小/最大）——审数据时先看这行
    sp = sensor.params()
    if sp.get("n_calls"):
        print(f"  随机化实测: 拖影 {sp['blur_px'][0]:.2f}px "
              f"[{sp['blur_px'][1]:.2f}~{sp['blur_px'][2]:.2f}]  "
              f"曝光 {sp['exposure'][1]:.3f}~{sp['exposure'][2]:.3f}  "
              f"亮度 {sp['brightness'][1]:.3f}~{sp['brightness'][2]:.3f}  "
              f"对比 {sp['contrast'][1]:.3f}~{sp['contrast'][2]:.3f}  "
              f"饱和 {sp['saturation'][1]:.3f}~{sp['saturation'][2]:.3f}")
    print(f"  标签      : {meta_path}  ({meta_path.stat().st_size/1024:.1f} KB)   "
          f"success={str(succ['success']).lower()}")
    print(f"  耗时      : {time.time()-t_start:.1f} s")

    if not args.no_verify:
        ok = verify(out_dir, meta, n_frames, npz_path=args.npz,
                    img_npz=img_npz_path if img_buf is not None else None,
                    require_success=bool(getattr(args, "require_success", False)))
        if not ok:
            return 1
    return 0


# ============================================================
# 第 9 段  verify —— 自动自检（不靠眼睛的部分）
# ============================================================


def verify(out_dir: Path, meta: dict, n_frames: int, npz_path=None, img_npz=None,
           require_success=False):
    """
    自检，全部客观可判：
      0. 成功标签（meta.success；负样本里 2/6 两项"预期不成立"，降级为 ⚠️）
      1. 视频可回读，帧数与 meta 一致
      2. 轨迹与步骤二实测一致（抬升 / 水平位移）
      3. 视频非静止（首帧 vs 中段帧像素差）
     3b. 帧间亮度稳定性（init 静止段帧间 |Δ| p95 < 1.5 DN，防"色彩抖动闪"回归）
      4. 物块确实出现在画面里（分割渲染统计 geom 像素）
      5. 导出的 episode_0.npz 形状/dtype 符合 make_video.py 的期望
      6. 药盒最后落在木盒内腔里（2024-11 新增：尾段"放进木盒"的客观判据）
      7. 松手之后真的没了握持（release 之后 contact_n 全 0）
      8. 落地后静止（末 10 帧药盒线速度 < 0.05 m/s）
      9. 回到初始姿态（末帧 5 个臂关节 vs 首帧；爪甲收到预设姿态 GRIP_PRESET）
     10. 没刮到木盒 + 动作密度（臂↔木盒接触 0；除 init 外最长静止段 ≤ 1.0 s）
     11. 数据契约（meta.schema_ok，形状/dtype 已由第 10 段校验过）

    负样本（success=false）的处理原则：**导出是否健康**与**抓取是否成功**是两件事。
    前者必须绿（视频/npz 都写对了），后者可以是红——所以 2/6 两项在负样本下跌级
    成 ⚠️（预期不符，不算故障）；需要"必须成功"时传 require_success=True。
    """
    print()
    print("=" * 78)
    print("verify  自检")
    print("=" * 78)
    ok = True

    # --- 0. 成功标签 ---
    succ = bool(meta.get("success", False))
    s_info = meta.get("success_info") or {}
    if require_success and not succ:
        good = False
        ok = False
        print(f"  [❌] 成功标签     success=False（reason={s_info.get('reasons')}），"
              f"但 --require-success 要求成功")
    else:
        # 负样本不是故障：它是我们要拿来当负样本的数据，所以只打 ⚠️ 不计入判定
        print(f"  [{'✅' if succ else '⚠️'}] 成功标签     success={str(succ).lower()}   "
              f"抬升 {s_info.get('lift_mm')} mm (判据 >= {SUCCESS_LIFT_MM:.0f} mm)"
              f"{'' if succ else '   reason=' + ','.join(s_info.get('reasons') or [])}"
              f"{'' if succ else '  （负样本：下面 2/6 两项预期不符）'}")

    # --- 1. 回读视频 ---
    for c in CAMS:
        p = out_dir / f"{c}.mp4"
        cap = cv2.VideoCapture(str(p))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        good = (n == n_frames) and (abs(fps - meta["fps"]) < 0.5) and (w == meta["width"])
        ok &= good
        print(f"  [{'✅' if good else '❌'}] {c:14s} {n:>4} 帧  {fps:5.2f} fps  "
              f"{w}x{h}  {p.stat().st_size/1024:.0f} KB")
        if n != n_frames:
            print(f"       期望 {n_frames} 帧，实际 {n} 帧 —— 视频写入丢帧了")

    # --- 2. 与步骤二一致性（口径见 record 的 metrics）---
    mrec = meta.get("record_meta", {}).get("metrics", {})
    rise, hdisp = mrec.get("rise_mm"), mrec.get("hdisp_mm")
    if rise is None or hdisp is None:
        consistent = False
    else:
        consistent = abs(rise - REF_LIFT_MM) <= TOL_MM and abs(hdisp - REF_HDISP_MM) <= TOL_MM
    if consistent:
        print(f"  [✅] 轨迹一致性  抬升 {rise:+.2f}mm (基准 {REF_LIFT_MM})  "
              f"水平 {hdisp:.2f}mm (基准 {REF_HDISP_MM})")
    elif not succ:
        # 抓取失败 -> 抬升/搬运必然偏离基准，这是负样本的定义，不是故障
        r_txt = "None" if rise is None else f"{rise:+.2f}mm"
        h_txt = "None" if hdisp is None else f"{hdisp:.2f}mm"
        print(f"  [⚠️] 轨迹一致性  抬升 {r_txt} 水平 {h_txt} —— 负样本预期不符，跳过")
    else:
        ok = False
        r_txt = "None" if rise is None else f"{rise:+.2f}mm"
        h_txt = "None" if hdisp is None else f"{hdisp:.2f}mm"
        print(f"  [❌] 轨迹一致性  抬升 {r_txt} (基准 {REF_LIFT_MM})  "
              f"水平 {h_txt} (基准 {REF_HDISP_MM})")

    # --- 2b. 接触真实性（★2024-11 第六轮新增）---
    # 用户反馈："下落抓取必然撞倒或戳到药盒，产生回弹，但现实没有这种机制。"
    # 旧实现确实是那样的：下探段 92/400 步有指尖↔药盒接触、峰值 2.73 N、
    # 把药盒压到桌面以下 5.1 mm（相对静置 6.87 mm），松压后回弹 2.6 mm，夹稳时歪 11°。
    # 现在夹持由两片平面指垫完成，这两行就是防止它回归的哨兵。
    cm = (meta.get("record_meta") or {}).get("contact") or {}
    press, gtilt = cm.get("press_mm"), cm.get("tilt_at_grasp_end_deg")
    mode = cm.get("contact_mode")
    if press is None or gtilt is None:
        print("  [⚠️] 接触真实性  meta 缺 contact 字段（老数据或探针产物），跳过")
    elif not succ:
        print(f"  [⚠️] 接触真实性  压盒 {press:+.2f}mm / 夹稳倾角 {gtilt:.2f}° —— 负样本预期不符，跳过")
    else:
        good = (press <= PRESS_MAX_MM) and (gtilt <= GRASP_TILT_MAX_DEG)
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 接触真实性  压盒 {press:+.2f} mm (≤{PRESS_MAX_MM}) / "
              f"夹稳倾角 {gtilt:.2f}° (≤{GRASP_TILT_MAX_DEG})  夹持={mode}")

    # --- 3. 非静止 ---
    cap = cv2.VideoCapture(str(out_dir / f"{CAM_ENV}.mp4"))
    frames = []
    want = {0, n_frames // 2, n_frames - 1}
    i = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if i in want:
            frames.append(fr.astype(np.int16))
        i += 1
    cap.release()
    if len(frames) >= 2:
        diff = float(np.abs(frames[0] - frames[len(frames) // 2]).mean())
        good = diff > 10.0
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 非静止      首帧 vs 中段帧平均像素差 {diff:.1f} (>10)")

    # --- 3b. 帧间亮度稳定性（2024-11 第二轮新增）---
    # 用户反馈："色彩抖动 ±8% 在一段视频内闪"。原因是当时 brightness/contrast/saturation
    # 逐帧独立重掷（等于把真机 AGC 换成随机数发生器），实测静止段帧间亮度差 6.25 DN。
    # 现在改成"每条视频基准 + 一阶低通慢漂移"，这一行就是防止它回归的哨兵：
    #   init 段机械臂完全不动，画面里唯一的帧间变化只能来自传感器模型 ->
    #   帧间平均亮度 |Δ| 必须小到肉眼不可见。
    cap = cv2.VideoCapture(str(out_dir / f"{CAM_ENV}.mp4"))
    lum_series = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        lum_series.append(float((fr.astype(np.float32)
                                 @ np.array([0.114, 0.587, 0.299], np.float32)).mean()))
    cap.release()
    lum_series = np.asarray(lum_series, float)
    ph_all = [str(f["phase"]) for f in meta.get("frames", [])][: len(lum_series)]
    init_idx = np.where(np.asarray(ph_all) == "init")[0] if len(ph_all) else np.array([], int)
    if len(init_idx) > 2 and len(lum_series) == len(ph_all):
        seg = lum_series[init_idx]
        d = np.abs(np.diff(seg))
        p95, mx = float(np.percentile(d, 95)), float(d.max())
        # 阈值 2.5 DN 的来源（实测分解）：
        #   曝光增益 ±1%/帧  -> 帧间 |Δ| mean 0.92 / p95 2.18 DN（真机 AGC 本来就逐帧微动，故意保留）
        #   色彩/暗角慢漂移   -> 只贡献 0.02 DN（肉眼不可见）
        #   第一轮逐帧重掷时 -> mean 6.25 / max 21.04 DN（就是用户看到的"闪"）
        good = (p95 < 2.5) and (mx < 5.0)
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 帧间稳定    init 静止段帧间亮度 |Δ| "
              f"p95 {p95:.2f} / max {mx:.2f} DN (需 p95<2.5, max<5)   全片极差 "
              f"{lum_series.max()-lum_series.min():.1f} DN")
    else:
        print("  [⚠️] 帧间稳定    没有 init 段可用，跳过")

    # --- 4. 物块可见性（分割渲染统计像素）---
    npz_p = Path(npz_path) if npz_path else (out_dir / "episode.npz")
    ep = load_episode(npz_p)[0] if npz_p.exists() else None
    if ep is not None:
        m = mujoco.MjModel.from_xml_path(SCENE_XML)
        d = mujoco.MjData(m)
        rig = CameraRig(m, width=meta["width"], height=meta["height"], seg=True)
        # 注意：object_physics 是纯碰撞体（rgba 全透明），分割里根本不出现。
        # 画面里看到的是同一 body 下 6 片贴图薄板，所以按 body 统计 geom 集合。
        obj_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        if obj_body < 0:
            raise RuntimeError("模型里找不到 body 'object'")
        obj_gids = set(int(g) for g in np.where(m.geom_bodyid == obj_body)[0])
        n_all = len(ep["qpos"])
        print(f"       物块 body id = {obj_body}, 相关 geom {len(obj_gids)} 个: "
              f"{sorted(obj_gids)}")
        for c in CAMS:
            counts = []
            for frac in (0.0, 0.5, 1.0):
                fi = int(round(frac * (n_all - 1)))
                d.qpos[:] = ep["qpos"][fi]
                mujoco.mj_forward(m, d)
                seg = rig.grab(d)[f"{c}_seg"]
                gid_map = seg[:, :, 0]
                counts.append(int(np.isin(gid_map, list(obj_gids)).sum()))
            good = max(counts) > 200
            # 负样本的例外：腕部相机装在指尖、朝下看夹爪，只有把药盒抬到眼前才扫得到。
            # 药盒压根没被抬起来时它当然拍到 0 像素 —— 那是"没抓到"的正常后果，
            # 不是渲染故障（正样本实测就是 0/38577/0）。外部相机不豁免：
            # 它本来就该一直看着绿垫，看不到药盒才是真的取景错了。
            forgive = (c == CAM_WRIST) and (not succ)
            ok &= bool(good or forgive)
            if good:
                print(f"  [✅] {c:14s} 物块像素数 "
                      f"首/中/末 = {counts[0]}/{counts[1]}/{counts[2]}  (需 >200)")
            elif forgive:
                print(f"  [⚠️] {c:14s} 物块像素数 首/中/末 = "
                      f"{counts[0]}/{counts[1]}/{counts[2]} —— 负样本预期不符，跳过")
            else:
                print(f"  [❌] {c:14s} 物块像素数 "
                      f"首/中/末 = {counts[0]}/{counts[1]}/{counts[2]}  (需 >200)")
        rig.close()

        # ---- 6~10：尾段（放进木盒 / 松手 / 落地 / 归位 / 没刮到木盒+密度）----
        # 全部只读 episode.npz 里的数组，不重新跑物理，所以判据与 record 完全同源。
        ph = np.asarray(ep["phase"])
        obj = np.asarray(ep["obj_pos"])
        qp = np.asarray(ep["qpos"])
        ctr = np.asarray(ep["ctrl"])
        box_n = np.asarray(ep["box_n"]) if "box_n" in ep else np.zeros(len(ph))
        spd = np.asarray(ep["obj_speed"]) if "obj_speed" in ep else np.zeros(len(ph))

        # --- 6. 药盒落在木盒内腔 ---
        bx = m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_box")]
        fin = obj[-1]
        # 木盒外 115(x) x 142(y) x 50(z)，壁 4mm -> 内腔 x±53.5 / y±67 mm；底板顶面 z=0.004
        # （2024-11 修正：长 142 是 y 向，之前模型把 x/y 写反了）
        inside = (abs(fin[0] - bx[0]) <= 0.050 and abs(fin[1] - bx[1]) <= 0.062
                  and 0.008 <= fin[2] <= 0.045)
        if inside:
            print(f"  [✅] 落在木盒里  药盒终点 "
                  f"[{fin[0]:.4f} {fin[1]:.4f} {fin[2]:.4f}]  "
                  f"相对木盒中心 ({(fin[0]-bx[0])*1000:+.1f}, {(fin[1]-bx[1])*1000:+.1f}) mm  "
                  f"z={fin[2]*1000:.1f}mm (需 ≤45 且 ≥8)")
        elif not succ:
            # 抓取失败 -> 药盒根本没被搬走，落在原处是负样本的正常结局
            print(f"  [⚠️] 落在木盒里  药盒终点 "
                  f"[{fin[0]:.4f} {fin[1]:.4f} {fin[2]:.4f}] —— 负样本预期不符，跳过")
        else:
            ok = False
            print(f"  [❌] 落在木盒里  药盒终点 "
                  f"[{fin[0]:.4f} {fin[1]:.4f} {fin[2]:.4f}]  "
                  f"相对木盒中心 ({(fin[0]-bx[0])*1000:+.1f}, {(fin[1]-bx[1])*1000:+.1f}) mm  "
                  f"z={fin[2]*1000:.1f}mm (需 ≤45 且 ≥8)")

        # --- 7. 松手后没有握持 ---
        rel = np.where(ph == "release")[0]
        if len(rel):
            after = slice(int(rel[-1]) + 1, None)
            n_hold = int((np.asarray(ep["contact_n"])[after] > 0).sum())
            good = n_hold == 0
        else:
            n_hold, good = -1, False
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 已松手      release 之后仍有握持的帧数 {n_hold} (需 0)")

        # --- 8. 落地静止 ---
        sp = float(np.abs(spd[-10:]).max())
        good = sp < 0.05
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 落地静止    末 10 帧药盒最大线速度 {sp:.4f} m/s (需 <0.05)")

        # --- 9. 回到初始姿态 ---
        arm_adr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
                            for n in ARM_JOINTS])
        gadr = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint")]
        dq_deg = np.degrees(np.abs(qp[-1, arm_adr] - qp[0, arm_adr]))
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_joint")
        grip_end = float(ctr[-1, gid])
        # 容差 0.10 rad ≈ 5.7°：末段的 ctrl 已经写回 q_home，剩下的偏差来自
        # PD 稳态误差（回程时重力/摩擦负荷与起始位不同），实测 2.4° 量级，
        # 不是"没走到位"。别把阈值收到 1° 以内，那会一直假报警。
        # 末帧爪甲必须是"收起的预设姿态"（2024-11 用户要求），不是张开状态
        good = (dq_deg.max() <= np.degrees(0.10)) and abs(grip_end - GRIP_PRESET) < 5e-3
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 回到初始位  各关节偏差 "
              f"{' '.join(f'{v:.1f}°' for v in dq_deg)} (需 ≤5.7°)  末帧夹爪 {grip_end:.3f}"
              f" (需 == 预设 {GRIP_PRESET})")

        # --- 10. 没刮到木盒 + 动作密度 ---
        # ⚠️ 这一段读的是【轨迹原始帧】（rec fps，通常 30），不是渲染后的视频帧。
        #    所以"帧数 -> 秒"必须除以轨迹帧率，不能除 meta["fps"]：
        #    stride>1 时视频帧率是 15，除错会把 0.57 s 的静止段算成 1.13 s 的假报警
        #    （负样本 demo_neg 用 stride=2，第一版就是这么假报的）。
        src_fps = float((meta.get("record_meta") or {}).get("fps") or meta["fps"])
        n_box = int((box_n > 0).sum())
        cols = np.concatenate([arm_adr, [gadr]])
        dq = np.abs(np.diff(qp[:, cols], axis=0)).max(axis=1) if len(qp) > 1 else np.zeros(0)
        static = dq < 1e-3
        keep = ~(ph[1:] == "init")
        longest, cur = 0, 0
        for s_, k_ in zip(static, keep):
            if not k_:
                cur = 0
                continue
            cur = cur + 1 if s_ else 0
            longest = max(longest, cur)
        longest_s = longest / src_fps
        good = (n_box == 0) and (longest_s <= 1.0)
        ok &= good
        print(f"  [{'✅' if good else '❌'}] 木盒未受扰  臂↔木盒接触帧 {n_box} (需 0)   "
              f"最长连续静止 {longest} 帧 = {longest_s:.2f} s (需 ≤1.0)")

    # --- 5. episode_0.npz 契约（make_video.py 直接消费这个）---
    if img_npz is not None and Path(img_npz).exists():
        d = np.load(Path(img_npz))
        has = set(d.files) >= {"env", "wrist"}
        e, w = d["env"], d["wrist"]
        shapes_ok = (e.ndim == 4 and e.shape[3] == 3 and e.shape[:3] == w.shape[:3]
                     and e.shape[0] == n_frames)
        dtype_ok = e.dtype == np.uint8 and w.dtype == np.uint8
        good = has and shapes_ok and dtype_ok
        ok &= good
        print(f"  [{'✅' if good else '❌'}] {Path(img_npz).name:14s} "
              f"env{e.shape} {e.dtype} / wrist{w.shape} {w.dtype}")
        if not shapes_ok:
            print(f"       期望 (N={n_frames}, H, W, 3) 且 env/wrist 同尺寸")

    # --- 11. 数据契约（第 10 段已严格校验；这里只把结论打出来）---
    schema_ok = meta.get("schema_ok")
    if schema_ok is None:
        # --no-schema 主动跳过时如实标 ⚠️，不假装通过
        print("  [⚠️] 数据契约    schema_ok=null（本次用 --no-schema 跳过了校验）")
    else:
        ok &= bool(schema_ok)
        print(f"  [{'✅' if schema_ok else '❌'}] 数据契约    形状/dtype 全部符合（第 10 段校验）")

    print()
    tail = "  （负样本：2/6 项为预期不符，不计入判定）" if not succ else ""
    print(f"  总判定: {'✅ 全部通过' if ok else '❌ 有项目未通过，见上方 ❌ 行'}{tail}")
    return ok


# ============================================================
# 第 10 段  数据契约校验（schema）—— 只读检查，不改任何产物
# ============================================================
#
# 为什么需要它：npz 是"没有类型系统"的数据契约。下游 make_video.py / 训练读数据器
# 只会 np.load(...)["qpos"]，形状或 dtype 错了不会抛异常，而是静默错位
# （比如 float32 被当 float64 解释、phase 变成 object 数组需要 allow_pickle）。
# 所以这里把契约写死成一张表，导出后立刻核对，不匹配就报错退出。

# 轨迹 npz 的键 -> (形状模板, dtype)。
#   模板里的字符串是占位符："N"=帧数，"nq"/"nv"/"nu"=模型自由度维度
#   （从 meta_json 里读，与 record 写进去的是同一份数字）。
#   dtype 精确到字节序：float32/float64 混用不会报错，只会让下游算错。
_TRAJ_SCHEMA = {
    "time":      (("N",), "<f8"),
    "qpos":      (("N", "nq"), "<f8"),
    "qvel":      (("N", "nv"), "<f8"),
    "ctrl":      (("N", "nu"), "<f8"),
    "phase":     (("N",), "<U16"),
    "tcp_pos":   (("N", 3), "<f8"),
    "obj_pos":   (("N", 3), "<f8"),
    "obj_quat":  (("N", 4), "<f8"),
    "contact_n": (("N",), "<i8"),
    "fn_jaw":    (("N",), "<f8"),
    "fn_grip":   (("N",), "<f8"),
    "fn_sum":    (("N",), "<f8"),
    "lift_mm":   (("N",), "<f8"),
    "hdisp_mm":  (("N",), "<f8"),
    "box_n":     (("N",), "<i8"),
    "obj_speed": (("N",), "<f8"),
}

# 图像 npz（episode_0.npz）的契约：make_video.py 只认这三个键
_IMG_KEYS = ("env", "wrist", "fps")


def _dims_ok(shape, tpl, dims):
    """形状模板比对。返回 (是否一致, 期望形状元组)。"""
    want = tuple(int(dims[t]) if isinstance(t, str) else int(t) for t in tpl)
    return tuple(int(v) for v in shape) == want, want


def validate_trajectory(path, nq=None, nv=None, nu=None):
    """
    严格校验轨迹 npz（episode.npz）。返回问题列表，空列表 = 通过。

    规则：登记的键必须全在、形状逐维相等、dtype 逐字节相等；
          额外键允许存在（向前兼容 —— 新字段不该让老数据读不了）。
    """
    p = Path(path)
    if not p.exists():
        return [f"轨迹 npz 不存在: {p}"]
    z = np.load(p)
    have = set(z.files)
    problems = []

    # 维度：优先用调用方给的（record/render 都从 meta 拿），缺失时退回 meta_json
    if nq is None or nv is None or nu is None:
        try:
            mm = json.loads(str(z["meta_json"])) if "meta_json" in have else {}
        except (KeyError, ValueError):
            mm = {}
        nq = mm.get("nq", nq)
        nv = mm.get("nv", nv)
        nu = mm.get("nu", nu)

    # 帧数：优先 time，其次任一登记键的第一维
    n = None
    if "time" in have:
        n = int(z["time"].shape[0])
    else:
        for k in _TRAJ_SCHEMA:
            if k in have:
                n = int(z[k].shape[0])
                break
    if n is None:
        return ["轨迹 npz 里没有任何登记的键（不是 record 的产物？）"]

    dims = dict(N=n, nq=nq, nv=nv, nu=nu)
    missing_dim = [k for k in ("nq", "nv", "nu") if dims[k] is None]
    if missing_dim:
        problems.append(f"无法确定维度 {missing_dim}（meta_json 里缺 nq/nv/nu）")

    for key, (tpl, dt) in _TRAJ_SCHEMA.items():
        if key not in have:
            problems.append(f"缺少键 {key}")
            continue
        # 取数组本身也可能抛：object 数组（比如 phase 被存成 Python 字符串列表）
        # 在 allow_pickle=False 下读取会 ValueError。那本身就是 dtype 违约，
        # 这里把它收敛成一条"问题"而不是让异常穿透（校验器自己不能崩）。
        try:
            a = z[key]
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{key} 无法读取（dtype 违约，如 object 数组）: {e}")
            continue
        if not missing_dim:
            good, want = _dims_ok(a.shape, tpl, dims)
            if not good:
                problems.append(f"{key} 形状 {tuple(a.shape)} != {want}")
        if a.dtype != np.dtype(dt):
            problems.append(f"{key} dtype {a.dtype} != {np.dtype(dt)}")

    if "meta_json" not in have:
        problems.append("缺少 meta_json（指标口径与标签都在里面）")
    else:
        try:
            mj = z["meta_json"]
            if mj.ndim != 0 or mj.dtype.kind != "U":
                problems.append(f"meta_json 必须是 0 维 unicode 标量，实际 {mj.shape} {mj.dtype}")
                m = {}
            else:
                m = json.loads(str(mj))
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"meta_json 无法读取: {e}")
            m = {}
        if "success" not in m:
            problems.append("meta_json 里没有 success 字段（旧版 npz，请重跑 record）")
        elif not isinstance(m["success"], bool):
            problems.append(f"success 必须是 bool，实际 {type(m['success']).__name__}")
    return problems


def validate_images(path, n_frames=None):
    """
    严格校验图像数组 npz（episode_0.npz）。返回问题列表，空列表 = 通过。

      env / wrist : (N, H, W, 3) uint8，两者同形状，N == 渲染帧数
      fps         : 0 维 float32 标量
    """
    p = Path(path)
    if not p.exists():
        return [f"图像 npz 不存在: {p}"]
    z = np.load(p)
    have = set(z.files)
    problems = []
    for key in _IMG_KEYS:
        if key not in have:
            problems.append(f"缺少键 {key}")

    shape = None
    for key in ("env", "wrist"):
        if key not in have:
            continue
        try:
            a = z[key]
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"{key} 无法读取: {e}")
            continue
        if a.ndim != 4 or a.shape[3] != 3:
            problems.append(f"{key} 形状 {tuple(a.shape)} != (N,H,W,3)")
        if a.dtype != np.uint8:
            problems.append(f"{key} dtype {a.dtype} != uint8")
        if shape is None:
            shape = a.shape
        elif shape != a.shape:
            problems.append(f"{key} 形状 {tuple(a.shape)} 与 env {tuple(shape)} 不一致")
    if shape is not None and n_frames is not None and int(shape[0]) != int(n_frames):
        problems.append(f"帧数 {int(shape[0])} != 期望 {int(n_frames)}")

    if "fps" in have:
        try:
            f = z["fps"]
        except Exception as e:                                   # noqa: BLE001
            return problems + [f"fps 无法读取: {e}"]
        if f.ndim != 0 or f.dtype != np.float32:
            problems.append(f"fps 必须是 0 维 float32，实际 {f.shape} {f.dtype}")
        elif not np.isfinite(float(f)):
            problems.append(f"fps 不是有限数: {float(f)}")
    return problems


def validate_npz_schema(traj_npz, img_npz=None, n_frames=None, rec_meta=None):
    """
    契约校验总入口：有任何一项不符就打印明细并 SystemExit（"不匹配直接报错退出"）。
    通过时返回写进 meta.json 的 schema 摘要（键/形状/dtype），便于事后审计。
    """
    rec_meta = rec_meta or {}
    problems = validate_trajectory(traj_npz, nq=rec_meta.get("nq"),
                                   nv=rec_meta.get("nv"), nu=rec_meta.get("nu"))
    summary = {"trajectory": None, "images": None}

    # 摘要只在校验通过后才有意义；如果连数组都读不出来，就让 problems 去报，
    # 这里用 try/except 兜住，别让校验器自己变成崩溃源。
    if Path(traj_npz).exists():
        try:
            z = np.load(Path(traj_npz))
            summary["trajectory"] = dict(
                path=str(traj_npz), n_frames=int(z["time"].shape[0]),
                keys={k: [list(z[k].shape), str(z[k].dtype)]
                      for k in _TRAJ_SCHEMA if k in z.files})
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"轨迹 npz 摘要无法生成: {e}")
    if img_npz is not None and Path(img_npz).exists():
        problems += validate_images(img_npz, n_frames=n_frames)
        try:
            z = np.load(Path(img_npz))
            summary["images"] = dict(
                path=str(img_npz),
                env=[list(z["env"].shape), str(z["env"].dtype)],
                wrist=[list(z["wrist"].shape), str(z["wrist"].dtype)],
                fps=[list(z["fps"].shape), str(z["fps"].dtype)])
        except Exception as e:                                   # noqa: BLE001
            problems.append(f"图像 npz 摘要无法生成: {e}")

    if problems:
        for x in problems:
            print(f"  [schema] ❌ {x}")
        raise SystemExit(f"数据契约校验失败：{len(problems)} 项不符"
                         f"（形状/dtype 与下游约定不一致），中止导出")
    n_traj = summary["trajectory"]["n_frames"] if summary["trajectory"] else "-"
    print(f"  [schema] ✅ 数据契约通过  轨迹 {len(_TRAJ_SCHEMA)} 键 + "
          f"图像 {len(_IMG_KEYS) if summary['images'] else 0} 键   N={n_traj}")
    return summary


# ============================================================
# 第 11 段  main
# ============================================================


def build_parser():
    ap = argparse.ArgumentParser(
        description="步骤三：纯物理抓取轨迹 -> 双相机 SDG 视频",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  python test_render.py record --quick
  python test_render.py render --quick
  python test_render.py                       # record + render 正式版
""")
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--out", default=str(OUT_DIR),
                       help=f"输出根目录 (默认 {OUT_DIR})")
        p.add_argument("--npz", default=str(OUT_DIR / "episode.npz"),
                       help="轨迹文件路径 (默认 <root>/demo/episode.npz)")
        p.add_argument("--width", type=int, default=None, help="宽度 (默认 640)")
        p.add_argument("--height", type=int, default=None, help="高度 (默认 480)")
        p.add_argument("--quick", action="store_true",
                       help="预览模式: 320x240 + 降采样 (出片 <20s)")
        p.add_argument("--gl", default=None, choices=["egl", "osmesa", "glfw"],
                       help="覆盖 MUJOCO_GL (需在 import 前生效, 已预扫描)")

    p_rec = sub.add_parser("record", help="跑真物理, 导出 episode.npz")
    common(p_rec)
    p_rec.add_argument("--decimate", type=int, default=None,
                       help="每 N 个 sim step 存一帧 (默认由 --fps 换算)")
    p_rec.add_argument("--fps", type=float, default=30.0, help="目标帧率 (默认 30)")
    p_rec.add_argument("--roll", type=float, default=DEF_ROLL, help="腕部滚转 (度)")
    p_rec.add_argument("--gz", type=float, default=DEF_GZ, help="抓取 TCP 高度 (m)")
    p_rec.add_argument("--grip-open", type=float, default=GRIP_APPROACH,
                       help=f"下探前的张开量 (默认 {GRIP_APPROACH}，全开=2.232)")
    p_rec.add_argument("--grip-release", type=float, default=GRIP_RELEASE,
                       help=f"释放时的张开量 (默认 {GRIP_RELEASE}，全开=2.232)")
    p_rec.add_argument("--no-transport", action="store_true", help="只抓不搬")
    p_rec.add_argument("--start-pose", default=START_POSE_DEFAULT,
                       choices=["oldros", "golden", "home"],
                       help="起止姿态 (默认 oldros = old_ros initial_pose；"
                            "golden/ home 用于 A/B 回归)")
    p_rec.add_argument("--npz-name", default="episode.npz", help="npz 文件名")

    p_ren = sub.add_parser("render", help="读 npz 出视频 (不跑物理)")
    common(p_ren)
    p_ren.add_argument("--stride", type=int, default=1, help="帧间隔>1 则降采样回放")
    p_ren.add_argument("--fps", type=float, default=None, help="覆盖帧率")
    p_ren.add_argument("--blur-sigma", type=float, default=0.8, help="高斯模糊 σ")
    p_ren.add_argument("--noise-sigma", type=float, default=3.0, help="高斯噪声 σ (DN)")
    p_ren.add_argument("--gain-jitter", type=float, default=0.01,
                       help="逐帧曝光抖动比例 (默认 0.01；真机 AGC 幅度很小)")
    p_ren.add_argument("--seed", type=int, default=0, help="随机种子")
    # ---- 视觉随机化（2024-11 第二轮）：帧间一致 + 速度驱动虚影 ----
    p_ren.add_argument("--motion-blur", action=argparse.BooleanOptionalAction,
                       default=True, help="运动模糊（默认开；--no-motion-blur 关）")
    p_ren.add_argument("--motion-prob", type=float, default=0.75,
                       help="无速度信息时每帧出拖影的概率 (默认 0.75)")
    p_ren.add_argument("--motion-px-min", type=float, default=1.0,
                       help="无速度信息时拖影下限 (px, 默认 1.0)")
    p_ren.add_argument("--motion-px-max", type=float, default=3.0,
                       help="无速度信息时拖影上限 (px, 默认 3.0)")
    p_ren.add_argument("--motion-k", type=float, default=0.6,
                       help="拖影 px / 像素速度 的系数 (默认 0.6)")
    p_ren.add_argument("--motion-max-px", type=float, default=4.0,
                       help="拖影长度上限 (px, 默认 4.0)")
    p_ren.add_argument("--vignette", type=float, default=0.30,
                       help="暗角强度：角落亮度乘 (1-v) (默认 0.30，0=关)")
    p_ren.add_argument("--vignette-pow", type=float, default=2.0,
                       help="暗角径向幂次 (默认 2.0，越大越只压边缘)")
    p_ren.add_argument("--color-jitter", type=float, default=0.03,
                       help="慢漂移幅度：亮度/对比度/饱和度的漂移比例 (默认 0.03，0=关)")
    p_ren.add_argument("--flicker-tau", type=float, default=2.0,
                       help="慢漂移时间常数 (s, 默认 2.0；越大越稳，0.2 近似逐帧重掷)")
    p_ren.add_argument("--frames", action="store_true", help="额外落盘 PNG 序列")
    p_ren.add_argument("--raw", action="store_true", help="额外落盘未劣化原图")
    p_ren.add_argument("--depth", action="store_true", help="一并渲染深度")
    p_ren.add_argument("--seg", action="store_true", help="一并渲染分割")
    p_ren.add_argument("--no-verify", action="store_true", help="跳过自检")
    p_ren.add_argument("--no-schema", action="store_true",
                       help="跳过 npz 契约校验（默认校验，不符即报错退出）")
    p_ren.add_argument("--require-success", action="store_true",
                       help="要求 success=true（负样本会被判为失败；默认允许负样本）")
    p_ren.add_argument("--export-npz", dest="export_npz", action="store_true",
                       help="额外导出图片数组 npz（env/wrist 全帧 RGB，约 0.8 GB/条；"
                            "默认不导出 —— 视频 mp4 已经是训练要的那份）")
    p_ren.add_argument("--export-name", default="episode_0.npz",
                       help="图片数组 npz 文件名 (默认 episode_0.npz)")
    p_ren.add_argument("--inspect", action="store_true",
                       help="额外出肉眼排查用的拼图/预览单帧（contact_sheet.png + "
                            "preview_*.png）。默认不出：排查请用 tools/inspect_episode.py 按需生成")
    p_ren.set_defaults(export_npz=False)

    return ap


def _apply_defaults(args):
    """
    默认值收敛到一处。--quick 只是三个默认值的简写：
      320x240  +  decimate 34 (约 15 fps)
    显式传了 --width/--height/--decimate 就不覆盖。
    """
    if getattr(args, "quick", False):
        if args.width is None:
            args.width = 320
        if args.height is None:
            args.height = 240
        if hasattr(args, "decimate") and args.decimate is None:
            args.decimate = 34
    if args.width is None:
        args.width = 640
    if args.height is None:
        args.height = 480
    if hasattr(args, "decimate") and args.decimate is None:
        dt = 0.002                       # 模型 opt.timestep，record 之后会再核对
        args.decimate = max(1, int(round(1.0 / (args.fps * dt))))
    return args


def _mk_args(ap, sub, rest):
    """给某个子命令单独解析一份 args，忽略不属于它的参数（"all" 模式要用）。"""
    args, _ = ap.parse_known_args([sub] + rest)
    return _apply_defaults(args)


def main(argv=None):
    ap = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("record", "render", "all"):
        argv = ["all"] + argv          # 允许 python test_render.py --quick
    cmd, rest = argv[0], argv[1:]

    if cmd == "record":
        return cmd_record(_mk_args(ap, "record", rest))

    if cmd == "render":
        return cmd_render(_mk_args(ap, "render", rest))

    # ---- all = record + render ----
    a_rec = _mk_args(ap, "record", rest)
    rc = cmd_record(a_rec)
    if rc:
        print("\n⚠️  record 的轨迹与步骤二基准不一致（抓取没成功，或动作序列被改动）。")
        print("    按【失败样本】继续：仍会渲染，并按 success=false 落盘（视频/npz/meta 全写）。")

    # render 必须指向 record 实际写出的那个文件（--npz-name 可能被改过）
    a_ren = _mk_args(ap, "render", rest)
    a_ren.npz = str(Path(a_rec.out) / a_rec.npz_name)
    # all 模式默认不强制成功：负样本正是这条命令要能导出的东西；
    # 显式传了 --require-success 才把"必须成功"当判据。
    rc2 = cmd_render(a_ren)
    # 退出码语义：任何一段判定未通过都返回非 0（哪怕产物已经落盘），
    # 这样 shell/CI 里 `&&` 不会把"抓取失败"当成通过。
    return rc or rc2


if __name__ == "__main__":
    sys.exit(main())
