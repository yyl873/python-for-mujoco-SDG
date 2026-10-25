================================================================================
 docs/  —— 项目文档（只有 3 份，历史都在 archive/）
================================================================================

【先看哪份】

  RUNBOOK_SDG.md      ★ 操作者看这个
        怎么生成数据、怎么导出、怎么训练、1000 条怎么跑。
        顺序：采集 → 体检 → 切分统计 →（想看图）→ 导出 lerobot →（训练）。
        每个命令的每个后缀"什么时候用"都列成表。

  PHYSICS_FACTS.md    ★ 改参数/改模型的人看这个
        现在生效的物理事实与验收数字（六项闸门、15 项自检）、
        不许动的数字（接触段步数 / 合爪 0.28 / dy=+17.5mm / roll −90° …）、
        指垫的正确贴法、已知但不算 bug 的现象（搬运歪 27~30° 等）、数据版本沿革。

  HANDOFF.md          指路牌（15 行）。原来 802 行的交接事实已归档。

【archive/ 里是什么】

  HANDOFF_full.md          802 行：交接事实、模型 4 处改动、物理事实、两条路径口径差
  MIGRATION_ACT.md         训练交付链路说明（要点已并入 RUNBOOK §7/§8）
  ROUND_video_split.md     第八轮：视频容器换 mp4 + ACT 直训数据集
  ROUND_pad_flush.md       第七轮：指垫贴合指面 + dy 偏置 + 合爪 0.28
  ROUND_contact_realism.md 第六轮：平面指垫 + 碰撞位掩码（修"戳盒/回弹"）
  ROUND_oldros_start.md    第五轮：起止姿态换成真机 initial_pose
  ROUND_golden_start.md    第四轮：起止姿态换成遥操 golden pose（已被取代）
  TOKEN_EFFICIENCY.md      会话省 Token 的操作纪律（给 AI 协作用）

  ⚠️ archive 里的数字是**当时**的实测，和现在的代码不一定对应
     （例如 sdg-v2/v3 时代的抬升 103 mm）。
     要引用数字一律用 PHYSICS_FACTS.md。

【每个目录还有一份 README.txt】

  model/ ik/ grasp/ render/ collect/ tools/ scripts/ videos/ demo/ demo_neg/ logs/
  分别讲"这个目录里每个文件是什么"。