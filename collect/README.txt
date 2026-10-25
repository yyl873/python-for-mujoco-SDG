================================================================================
 collect/  —— 批量随机生成数据（SDG）+ 数据工程的四个下游工具
================================================================================

【这个目录是干什么的】
    render/test_render.py 只产出【一条】写死的 demo（药盒固定在 (0.295, 0.090)）。
    训练要的是成百上千条"同任务、不同摆放/节奏/噪声"的样本，本目录就是那个生成器，
    以及数据生产之后**必须做**的四件事：切分、读取、体检、覆盖度统计。

【文件】
    collect.py        生成器主体（6 段：环境前置 / 随机化 / 录制 / 渲染+落盘 / 索引层 / 主流程）
    dataset.py        ★读数据的唯一入口（6 条数据契约写在它的 docstring 里）
    export_lerobot.py ★★导出成 lerobot v3.0 数据集（ACT 直接训练）—— 要用 conda lerobot312 解释器
                          ~/miniforge3/envs/lerobot312/bin/python collect/export_lerobot.py --dst dataset/x
    split.py          train/val/test 划分（random / theta-holdout / mod 三种策略）
    check_dataset.py  批量体检（schema、帧对齐、外参正交性、标签一致性、清单自洽）
    stats.py          覆盖度统计 + 数据卡（DATA_CARD.md）

【落盘分工（用户要求）】
    scripts/ep_XXXXXX.npz|json     动作脚本 + 标签 + 逐帧相机外参
    scripts/index/ep_XXXXXX.json   每条一个索引片段（原子写；进程被杀也不丢）
    scripts/index.json             聚合视图（扫片段生成，不要手改）
    scripts/splits/*.json          train/val/test + manifest.json
    scripts/report.json            体检报告（check_dataset.py）
    scripts/stats.json|stats.png|DATA_CARD.md   覆盖度（stats.py）
    videos/ep_XXXXXX_env|wrist.mp4 视频（h264；训练侧直接吃，不再转任何格式）

【★ sdg-v6 起：采集只出"训练要的那一份"】（第八轮，用户要求）
    · 视频容器 AVI(MJPG) → MP4(h264)：455 帧 640×480 从 13 MB 降到 ≈0.7 MB
    · 采集路径里**不再**生成任何观看用产物：没有 avi 转换、没有预览 png、没有拼图、不烧帧号
    · 要排查视频：tools/inspect_episode.py（按需、指定几条、也可只体检不落图）
    · paths 键 env_avi/wrist_avi → video_env/video_wrist（老数据集读取侧仍兼容）
    · 物理/轨迹与 sdg-v5 逐位相同（只换容器），老清单留档在 scripts/index.sdg-v5.json

【怎么运行】
    cd /home/yyl/deepseek-harness

    # ① 刷数据
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 20 --no-render   # 只刷物理（7 s）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 2                 # 640x480 出片
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 2 --quick         # 320x240
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --start 20 --n 5      # 续刷
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --reset --n 2         # 清索引层重刷
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --n 100 --jobs 8      # ★并行（实测≈3x）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/collect.py --render-only \
        --episodes 4,9,17 --jobs 4                                                   # 只补渲指定条目

    # ①.5 导出成 ACT 能直接训练的数据集（需要 lerobot 环境）
    env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
        collect/export_lerobot.py --check                                            # 只体检母本
    env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
        collect/export_lerobot.py --dst dataset/soarm101_sim_v1 --overwrite           # 导出+回读校验
    详见 docs/MIGRATION_ACT.md

    # ② 数据工程四步（顺序固定）
    env -u PYTHONHOME -u PYTHONPATH python3 collect/check_dataset.py      # 体检，非 0 退出=有失败
    env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py              # 切分
    env -u PYTHONHOME -u PYTHONPATH python3 collect/stats.py --card       # 覆盖度 + 数据卡
    env -u PYTHONHOME -u PYTHONPATH python3 collect/dataset.py --images   # 读取器自检

【★ v1.1 修掉的两个数据安全坑（2024-11）】

    1) 续跑会冲掉清单（已修）
       老版 main() 用 `accepted, rejected = [], []` 起步、每次整份覆盖 index.json，
       于是 `--n 20` 之后再来一次 `--start 20 --n 5`，前 20 条的清单就没了
       （npz/视频还在磁盘上，但没有清单 = 孤儿文件，训练侧根本发现不了）。
       现在：每完成一条先写 scripts/index/ep_XXXXXX.json（原子写），
             index.json 由片段聚合生成。续跑时先把片段读回来再追加。
       实测：--n 5 后 --start 5 --n 3 → index.json 里 8 条齐全。

    2) 采样 RNG 绑定到 episode_id（已修）
       老版 `rng = default_rng(base_seed)` 顺序抽，`--start 20` 抽到的是 eid 0 的随机量，
       也就是说"一次跑 25 条"与"跑 20 + 续 5"是**两套不同的样本**。
       现在每条用 SeedSequence([base_seed, episode_id])，两种跑法逐位相同（已实测）。

    另外：进程 SIGFPE 被杀时最多丢一条（正在跑的那条），已完成的条目与产物一致。

【起始 / 收尾姿态（sdg-v4 起；sdg-v3 / sdg-v2 与更早的别混训）】
    每条 episode 的**起点与收尾**都是同一个姿态 = old_ros 录制脚本的 initial_pose
    （old_ros/send_trajectory.py:369 / :552，真机"折叠待机位"）：
        shoulder_pan 0° / shoulder_lift −100° / elbow_flex 88.26° /
        wrist_flex 70.63° / wrist_roll −103.53°，夹爪收起（GRIP_PRESET=0.0）。
    实现：render/test_render.run_episode(start_pose="oldros") -> Arm.apply_start_pose()；
          姿态**不是** keyframe（scene.xml 未改，mjcf_sha256 不变），
          所以换起点只能靠 dataset_version 隔离 —— 现在是 sdg-v4。
          ★ sdg-v4 的物理与 v3 不同：加了平面指垫 + 碰撞位掩码（下探段对药盒零接触），
            mjcf_sha256 已变；v3 的 20 条（18/20 接受）留在 scripts/index.sdg-v3.json 备查。
    实测验收（2024-11 第五轮）：20/20（seed 3）、候选次数 1~7、抬升 101.5~104.5mm，
    timing_scale 0.90~1.10 十档全过（见 tools/probe_grasp_robustness.py）。
    上一轮的 golden pose 起点（sdg-v2，18/20）已被取代，可用 --start-pose golden A/B，
    但注意它必须搭配 (init_lift,retract,above)=(500,800,700) 才抓得住 ——
    起点与步数配比是耦合的，换一个就要重扫另一个。
    历史：sdg-v2 = 遥操第 0 帧 golden pose（pan −36°/lift +4°/elbow +14°/
          wrist_flex +55°/wrist_roll −104°），当年 18/20、对 timing_scale 敏感。

【索引结构（v1.1）】
    scripts/index/_policy.json  本次运行的随机化策略快照（split/stats/rebuild 不用加载模型）
    scripts/index/ep_000000.json
        {episode_id, accepted, success, reasons, dataset_version, schema_version,
         start_pose, metrics, paths{npz,json,video_env,video_wrist},
         keys{θ,dr,光照,传感器,抬升…}, run{seed,argv,pid}}
    scripts/index.json
        {dataset_version, schema_version, n_episodes, n_accepted, n_rejected, n_success,
         accept_rate, success_rate, runs[], randomization_policy, splits, accepted[], rejected[]}
      · `keys` 是随机量的**紧凑投影**：split/stats 只读 index.json 就够了，
        不用为了取一个 θ 去打开 1000 个 per-episode json。
      · `splits` 由 split.py 回填（策略/种子/各文件 sha256）。

【随机化策略（每一项都有实测依据，别凭感觉放大）】

    ★几何硬约束：药盒必须【整个 footprint 留在绿色拾取垫上】（2024-11 用户要求）
      垫半径 96.0mm、竖立药盒 footprint 半对角 37.8mm、留边 2mm
        -> 盒心离垫心最多 56.2mm；判据取"四角里最远的那个 ≤ 94.0mm"
      而抓取必须靠"绕 pan 轴的同构旋转"保住工具姿态，旋转会让盒心沿半径 271.6mm
      的圆弧走 -> 由几何反解：|θ| ≤ 10°（11° 时 footprint 最远点顶到 96.0mm 贴边）
      所有上限都是 PadZone 运行时从 model/scene.xml 反算的，不是写死的常数。

    场景旋转     ±10°（垫内上限，PadZone.theta_limit_deg 二分反算）
    径向抖动     沿"pan 轴 -> 药盒"方向 +0~6 mm（实测正向可用）
    药盒偏航     ±1.5°（再大指垫会错开窄端面，-1° 就掉）
    落定校验     每条都真落定一次，用实际 qpos 算 footprint 四角 → 越界就重新采样（≤12 次）
    节奏         init_hold/init_lift/approach_retract/approach_above/settle_a/
                 settle_b/home/tail 这几段的步数 ×0.90~1.10
                 ★descend/grasp/lift/transport/place/release/drop/retract 一步不动
    环境相机     位置 ±1~3 cm、fovy ±3°、朝向 yaw/pitch ±2°
    光照         方位角 0~360°、俯仰 5~25°、强度 0.35~0.70
    材质         药盒/木盒/桌面/垫子的 rgba 整体 ×0.90~1.10（只改明暗不改色相）
    传感器       模糊 σ0.6~1.0、噪声 σ2~4 DN、曝光抖动 0.5%~2%（逐帧，真机 AGC 量级）
    ★视觉随机化（与 render/test_render.SensorModel 同一套开关）
                 运动虚影 长度 = k×像素速度（k 0.45~0.80，上限 3.0~4.5 px，静止=0）；
                 暗角 0.22~0.38；色彩抖动"每条视频基准 + 慢漂移"±2%~±5%（τ 1.2~3.0 s）
    ✗ 不随机摩擦系数：μ=1.2 是 TPU 对纸盒的实测标定值，改了等于换了台机器人

【自动验收 / 正负样本】
    1) IK 残差 < 2 mm         4) 药盒最终落进木箱内腔
    2) 落定后药盒还立着        5) 松手后无握持接触；机械臂不碰木箱
    3) 抬升 > 95 mm           6) footprint 在绿垫上

    · 通过 → 正样本：npz + json + 两条 AVI
    · 有轨迹但没通过 → **负样本**：npz + json（success=false、has_video=false、
      相机外参照样齐，可事后补渲染），不渲染视频（--render-failed 可开）。
      负样本是稀有资源：正样本谁都能造，真实的失败模式造不出来。默认保留（--keep-failed），
      不想要用 --no-keep-failed。
    · 出生位姿反复重采样仍在垫外 → 只落 ep_XXXXXX.json（artifact="json_only"），
      证明"这个编号被试过、为什么被丢"，不留编号空洞。

【无渲染路径也有相机外参（v1.1）】
    `--no-render` 与负样本没有视频，但仍然逐帧算 cam_env_T_cw / cam_wrist_T_cw
    （`extrinsics_only()`：只做运动学，不出图，约 0.1 s/条）。
    公式与渲染路径（CameraRig）逐位一致，已用 check_dataset.py --cross-extrinsics 校验：
    外差 0.0e+00。所以这批数据以后要补渲染，外参直接可用。
    不用就加 --no-extrinsics（npz 里就没有 cam_*_T_cw，check_dataset 会标出来）。

【性能（本机实测）】
    20 条 --no-render ：5.9~7.4 s（旧基线 5.5 s，差值=外参保证，随动作候选重试次数波动）；
                        含 SIGFPE 重试的墙钟 7~9 s
    640x480 渲染      ：**有 GPU ≈ 17 s/条**（实测 8 进程 8 条 33.1 s、单条 19.1 s 含启动；
                        grab 双相机 1.7 ms/帧）；无 GPU 时退化为 software llvmpipe：174~197 ms/帧、75~90 s/条
                        ★ 本机 RTX 4060 Laptop 是好的；只有在看不到 /dev/nvidia* 的沙箱里才会掉到软件渲染
    并行（--jobs K）  ：有 GPU：8 进程 8 条 = 33.1 s；无 GPU：4 进程 4 条 = 148.7 s（≈37 s/条）
                        每进程自动设 LP_NUM_THREADS = cpu//K，避免互相抢线程
    导出 lerobot      ：实测 332 s / 20 条（≈16.6 s/条：解码 h264 + av1 crf30 重编码两路）
                        20 条 / 9102 帧 = 121 MB（≈13 KB/帧）；回读校验过（lerobot 自己读得出来）
    SIGFPE 重试套路（一次性脚本，别提交）：
        run_retry 0 20 --seed 3 --no-render       # 崩了从下一条续跑，已完成的不会重跑
    分片并行：collect.py --shard 1/4 --n 100     # 只跑 eid%4==1；分片进程不写 index.json
              （--jobs K 就是自动帮你摊成 K 个这样的子进程，跑完自动聚合清单）
              全部跑完再 collect.py --rebuild-index 统一聚合（避免多进程抢同一文件）

【实例数据（可用作回归基线）】
    --n 20 --seed 91 --no-render（垫内约束版）: 收 20 / 丢 0，接受率 100%
      footprint 最大半径 47.2~80.3mm（判据 94.0mm）→ 全部在垫上；落定偏航 -10.7°~+8.4°；
      抬升 ≥102.1 mm；重采样次数全 0
    典型一条：抬升 +103 mm、水平 180~250 mm、450 帧左右、14.1~15.7 s、试 1~2 次动作候选
    时长回归（13 条）：sim_time ≈ 6.30 + 8.68×timing_scale（r²=1.000）→ 节奏随机确实生效
