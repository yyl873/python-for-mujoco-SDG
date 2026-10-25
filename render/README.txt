================================================================================
 render/  —— 步骤三：纯净渲染采集管线（SDG 数据落地）
================================================================================

【这个目录是干什么的】
    把步骤二的真物理轨迹，变成两样东西：
      1. 带标签的双相机图像数据（meta.json 里有逐帧位姿、接触力、相机内外参）
      2. 能直接双击播放的视频（AVI / MP4）

    设计上把"动作"和"渲染"彻底拆开，靠数据契约连接：

        test_render.py record   ──>  demo/episode.npz     (物理状态轨迹 + success 标签)
                                     │
        test_render.py render   <───┘
                │
                ├──> demo/*.mp4            双相机视频（MP4/h264 —— 训练侧直接吃）
                ├──> demo/meta.json        顶层 success + 逐帧标签 + 相机内外参
                ├──> demo/contact_sheet.png 关键帧拼图        （只有加 --inspect 才出）
                └──> demo/preview_*.png     抓取瞬间单帧      （只有加 --inspect 才出）

        ★ 第八轮起：不再有 avi、不再"转一次格式才能看"，make_video.py 已删除。
          默认只出 mp4 + 轨迹；拼图/预览属于"我要看一眼"，加 --inspect 或走
          tools/inspect_episode.py（按需，指定条目）。

    demo/      = 正样本（success=true，640x480 展示件；约 800MB 的 episode_0.npz 不留）
    demo_neg/  = 负样本样例（success=false，320x240，故意夹空；详见 demo_neg/README.txt）

【15 秒动作脚本（2024-11 改版：干掉 4 秒静止，加释放 + 归位）】

    真机 151 条 / 67649 帧 / 30fps -> 平均 448 帧 ≈ 14.93 s。
    本管线 timestep=0.002 s，用 7500 步 = 15.00 s，时间栅格录制出 ~455 帧。

    #  阶段        步数   时长    内容                          备注
    1  init        800   1.60s  起始位静止（爪甲收起=预设姿态）  数据集起点
    2  approach    800   1.60s  home -> 高位撤回 z=0.24        <=900（实测上限）
    3  approach    600   1.20s  -> 药盒正上方 z=0.16（张开到位） <=700
    4  descend     400   0.80s  竖直下探                        ★接触段 锁定
    5  grasp       300   0.60s  合爪到 GRIP_SHUT=0.20            ★接触段 锁定
    6  settle_a    400   0.80s  夹稳确认
    7  lift        400   0.80s  竖直抬升                        ★接触段 锁定
    8  settle_b    300   0.60s  抬升后停
    9  transport   600   1.20s  水平搬到木盒上方 z=0.16          <=600
    10 place       300   0.60s  保持夹持下放到盒口 z=0.115       新增
    11 pre_open    100   0.20s  松手前停稳                       新增
    12 release     300   0.60s  开爪，药盒自由下落（真重力）      新增
    13 drop        300   0.60s  落进木盒并静止                   新增
    14 retract     450   0.90s  竖直提出木盒到 z=0.24（边收爪）   新增
    15 home       1300   2.60s  回初始姿态（= 首帧姿态+爪收起）   新增
    16 tail        150   0.30s  收尾静止                         新增
    ------------------------------------------------------------------
       合计       7500   15.00s

    夹爪脚本（2024-11 用户要求：起手/收尾都是"收起的预设姿态"，不必每次开最大）：

        阶段         夹爪 ctrl            说明
        ----------   ------------------   ------------------------------------
        init         0.000 GRIP_PRESET    收起，与 keyframe 一致（不再在这里张爪）
        approach     0.000 -> 1.500       边移动边张开（不额外占时间）
        descend      1.500                GRIP_APPROACH ≈86°，够跨过 74mm 盒宽
        grasp        1.500 -> 0.200        合爪贴住
        settle_a..place  0.200            夹持中
        release      0.200 -> 1.200        GRIP_RELEASE，够指尖离开药盒
        drop         1.200
        retract      1.200 -> 0.000       边上提边收爪
        home/tail    0.000                收尾爪甲同样是预设收起姿态

        扫描依据（都跑过，结论一致）：下探张开量 0.6/0.8/0.9/1.0/1.2/1.5/1.8/2.232
        全部能抓起来（抬升 103.5~103.8mm、水平 221.0mm、落点都在木盒内），
        释放张开量 0.8/1.4/2.232 都满足"松手后零接触"。所以不再开最大，
        取 1.5 / 1.2 留余量；2.232 只在真需要时用（命令 `--grip-open 2.232`）。

    改动理由：
      * 旧版 N_HOLD=2000（4.0 s 纯静止）占全片 27%，与真机分布对不上。
        新版把它换成"下放 -> 松手 -> 落进木盒 -> 提出 -> 归位"这条真动作。
      * 接触段（descend / grasp / lift）步数一个都没动 —— 实测把下探放慢到
        1150 步时结构件在盒顶压得更久，药盒被推倒（z 0.0278 -> 0.0041），
        抓取全废。所以时间只能加在自由空间段和停顿段上。
      * 静止占比：27% -> 24.7%；扣掉 init（数据集起点必需）只剩 14%，
        最长连续静止段从 4.0 s 降到 0.63 s。
      * PLACE_Z=0.115 的由来：TCP 比药盒中心高 33.2 mm、药盒半高 26 mm、
        木盒上沿 z=0.050 -> 松手时药盒底面离盒沿 6 mm，落差 61 mm，
        是"放进去"而不是"从 0.16 扔下去"（97 mm 落差会砸飞）。
      * 药盒最终躺在木盒内腔里（位置/姿态由物理决定，脚本不干预），
        末帧药盒 ≈ [0.288, -0.142, 0.030]，臂回到 q_home。

【文件】

    test_render.py   ★ 主文件（约 1900 行，12 个代码段）
        第 0 段  环境前置：MUJOCO_GL 必须在 import mujoco 之前设好（预扫描 --gl）
                 + sys.unraisablehook 静音 EGL 析构噪音
        第 1 段  常量区：所有动作常量从 test_grasp 导入，不新造；成功判据 SUCCESS_LIFT_MM
        第 2 段  EpisodeRecorder：逐帧快照 qpos/qvel/ctrl/phase/接触力
                 + arrays()（落盘与判标签共用一份构造）+ evaluate_success()（成功标签）
        第 3 段  动作序列：唯一含 ctrl 的代码，与 test_grasp 逐段同参
        第 4 段  CameraRig：一个 Renderer 轮流服务两台相机；内外参计算
        第 5 段  SensorModel：五层劣化（运动模糊 / 暗角 / 曝光 / 噪声 / 色彩抖动），seed 可复现
        第 6 段  VideoSink / PNG / 拼图
        第 7 段  record 子命令：录轨迹 + 判 success + 落盘后立刻校验契约
        第 8 段  render 子命令：纯 mj_forward 正运动学回放（负样本也照常导出）
        第 9 段  verify：13 项自动自检（负样本里 3 项降级为 ⚠️；含帧间亮度稳定性）
        第 10 段 数据契约校验：_TRAJ_SCHEMA 表 + validate_trajectory/validate_images
                 + validate_npz_schema（不符则 SystemExit）
        第 11 段 main：record / render / all（退出码语义见上）

    （make_video.py 已删除：练手时它负责 npz->AVI，现在 mp4 就是最终格式，
      要看图请用 tools/inspect_episode.py --demo）

【怎么运行】
    cd /home/yyl/deepseek-harness

    # ① 快速预览（320x240，约 17 秒）—— 只为看流程对不对
    env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py --quick

    # ② 正式出片（640x480，约 34 秒）
    env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py

    # ③ 要看拼图/预览（默认不出，按需）
    env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py render --inspect
    env -u PYTHONHOME -u PYTHONPATH python3 tools/inspect_episode.py --demo

    # ④ 只重渲染，不重跑物理（改噪声/分辨率时用）
    env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py render --noise-sigma 8

    # ⑤ 出干净图（不做传感器劣化）
    env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py render \
        --blur-sigma 0 --noise-sigma 0

【常用参数】
    record 子命令：
        --fps 30            目标帧率（换算 decimate = 1/(fps×0.002)）
        --decimate N        直接指定每 N 个 sim step 存一帧
        --roll / --gz       抓取参数，默认走步骤二验证过的 120° / 0.046
        --no-transport      只抓不搬
        --npz-name          轨迹文件名（默认 episode.npz）

    render 子命令：
        --width / --height  分辨率，默认 640 / 480
        --stride N          降采样回放（N=2 则隔帧取）
        --blur-sigma        高斯模糊 σ，默认 0.8（0 关闭；这是"失焦"，不是运动虚影）
        --noise-sigma       高斯噪声 σ(DN)，默认 3.0（0 关闭）
        --gain-jitter       逐帧曝光抖动比例，默认 0.01（真机 AGC 幅度很小）
        --seed              随机种子，默认 0（同种子同图，可复现）
        --motion-blur / --no-motion-blur   运动虚影开关，默认开
        --motion-k          拖影 px / 像素速度 的系数，默认 0.6
        --motion-max-px     拖影长度上限（px），默认 4.0
        --motion-prob       无速度信息时的出影概率，默认 0.75（有速度就不用它）
        --motion-px-min/max 无速度信息时的拖影范围（px），默认 1.0 / 3.0
        --vignette          暗角强度（角落亮度乘 1-v），默认 0.30（0=关）
        --vignette-pow      暗角径向幂次，默认 2.0
        --color-jitter      慢漂移幅度：亮度/对比度/饱和度，默认 0.03（0=关）
        --flicker-tau       慢漂移时间常数（s），默认 2.0（越大越稳）
        --frames            额外落盘 PNG 序列
        --raw               额外落盘未劣化原图（clean/noisy 配对用）
        --depth / --seg     额外渲染深度 / 分割（需配 --frames 才落盘）
        --no-export-npz     不生成 episode_0.npz
        --export-name       图片数组文件名，默认 episode_0.npz
        --gl {egl,osmesa,glfw}  覆盖 MUJOCO_GL
        --no-verify         跳过自检
        --no-schema         跳过 npz 契约校验（默认校验，不符即报错退出）
        --require-success   要求 success=true；默认允许负样本（success=false 也照常导出）

【视觉随机化（2024-11 第二轮：帧间一致 + 速度驱动虚影）】

    顺序按物理链路排（别交换，交换后就不是"从光到电"）：
        1 运动虚影 motion blur   光学  方向随机、长度 = k*(像素速度)，静止帧为 0
        2 暗角     vignette      光学  角落乘 0.70~0.78（中心 1.0，径向幂 2）
        3 曝光增益 gain          光电  整帧亮度 ±1%（真机 AGC 逐帧微动，故意保留）
        4 传感器噪声 noise       电学  σ=3 DN 高斯
        5 色彩抖动 color jitter  ISP   亮度/对比度/饱和度 慢漂移 ±3%

    ★ 第一轮踩的坑：色彩抖动是"逐帧独立重掷 ±8%"，等于把真机 AGC 换成随机数发生器。
      用户反馈"视频里在闪"。实测 demo 视频静止段（init，机械臂不动）帧间平均亮度差
      mean 6.25 / max 21.04 DN —— 肉眼就是闪。它不是噪声（噪声对整帧均值的贡献 ~0.01 DN）。
    现在的做法（帧间一致）：
      * 每条视频抽一次【基准值】，之后每帧朝"基准附近的小扰动"走一阶低通，
        时间常数 flicker_tau_s（默认 2 s），幅度 color_jitter（默认 ±3%）。
      * 实测分解（640x480 干净帧，静止 40 帧）：
            全默认                       帧间 |Δ| mean 0.92 / p95 2.18 / max 2.47 DN
            关掉曝光抖动                   mean 0.03 / p95 0.08（← 残差几乎全来自曝光 ±1%）
            再关掉色彩慢漂移                mean 0.02
            再关掉暗角慢漂移                mean 0.00
        => 闪烁的主要来源已经归零，剩下的 1 DN 是"故意保留的 AGC 微动"。

    ★ 运动虚影改成"速度驱动"（用户在真机录像里确认：静止对着不动的机械臂拍是干净的，
      只有物体/镜头相对运动时才有虚影）：
      * 第一轮是每帧独立抽 1~3 px（75% 的帧都有），静止段也在糊，像相机在抖。
      * 现在由调用方传入该帧 TCP 在【图像平面】上的像素速度 speed_px，
        拖影长度 = clip(motion_k * speed_px, 0, motion_max_px)，小于 0.2 px 视为静止 -> 不糊。
      * 实测：speed 0 -> 0.00 px（边缘能量与无模糊完全一致）；
              speed 1 -> 0.60 px；speed 5 -> 3.00 px；speed 20 -> 4.00 px（封顶）。
      * 两台相机各自算（腕部相机像素速度更大，天然更糊），这就是"跟着真实运动走"。
    这类随机化是"免费"的：不改几何、不改相机内外参、不跑物理，纯后处理。

    可复现（任务书硬要求）：所有随机量来自同一个 np.random.default_rng(seed)，
    按固定顺序抽样。同 seed + 同一批输入帧 -> 逐像素相同（慢漂移也是同一 RNG 驱动）。
    实测自检：
        同 seed 两次逐像素相同 ✅；换 seed 最大像素差 23（确实不同）✅
        暗角 角落/中心 = 0.698（= 1-0.30）✅
        静止帧拖影 = 0 px ✅

    关掉全部后处理（要"干净原图"时）：
        --blur-sigma 0 --noise-sigma 0 --gain-jitter 0 --no-motion-blur \
        --vignette 0 --color-jitter 0
    只保留视觉随机化、不要电学噪声：
        --noise-sigma 0 --blur-sigma 0 --gain-jitter 0

【自检 16 项判据（脚本打印 16 行：视频与物块可见性各按两相机分行；全绿才算成功）】
    0. 成功标签 success（抬升 ≥ 50mm 且搬运途中没掉）—— 负样本这一项是 ⚠️，不是故障
    1. 两个 mp4 能回读，帧数 / fps / 分辨率与 meta 一致
    2. 抬升 +122.22mm、水平 218.06mm 与基准一致（sdg-v5 实测 122.22 / 218.06，容差 ±5mm）
       历史：125.10 / 209.56（sdg-v4）、104.13 / 222.00（2024-10）、103.47 / 221.01（sdg-v3）—— 不要再引用
    2b. 接触真实性（第六轮新增）：压盒 ≤0.5mm 且夹稳倾角 ≤5°；实测 0.39mm / 1.06°
       （不一致说明动作序列被改了，不是渲染问题；负样本时降级为 ⚠️）
    3. 首帧 vs 中段帧平均像素差 > 10，证明视频不是静止的
   3b. 帧间亮度稳定性：init 静止段帧间 |Δ| p95 < 2.5 DN（防"色彩抖动闪"回归；
       第一轮是 6.25 DN，用户看到的就是这个）
    4. 分割渲染统计药盒像素数 > 200，证明物块真的在画面里
       （注意：object_physics 是 rgba=0 的纯碰撞体，画面里看到的是同 body 的 6 片贴图薄板）
    5. episode_0.npz 的 env / wrist 形状为 (N,H,W,3) uint8
    6. 药盒最后在木盒内腔里（|Δx|≤50mm、|Δy|≤62mm、8mm≤z≤45mm；负样本时降级 ⚠️）
    7. release 之后 contact_n 全 0（真的松手了，不是"还蹭着"）
    8. 末 10 帧药盒线速度 < 0.05 m/s（落稳了）
    9. 末帧 5 个臂关节 vs 首帧偏差 ≤ 5.7°，且末帧夹爪 ctrl == GRIP_PRESET(0.0)（归位）
   10. 臂↔木盒接触帧 == 0（提出木盒时没刮到木盒）+ 除 init 外最长连续静止段 ≤ 1.0 s
   11. 数据契约 schema_ok（形状/dtype 已由第 10 段严格校验过）

【episode_0.npz 数据契约（make_video.py 消费的就是它）】
    env   : (N, H, W, 3) uint8  RGB   外部相机 env_camera
    wrist : (N, H, W, 3) uint8  RGB   腕部相机 wrist_camera
    fps   : 0 维 float32（仅参考，make_video 按你的要求写死 30）
    注意：640x480 x 455 帧 = 约 800 MB（未压缩 npz）。所以 demo/ 默认不留这个文件，
          demo_neg/ 留了一份 320x240 的（约 100 MB）。要多 episode 就改存 JPEG 字节流。

【数据工程：success 标签 / 失败样本 / schema 校验（2024-11 新增）】
    只加"导出与校验"的辅助代码，物理控制与渲染核心一行没动。

  (1) success 标签 —— 抓取成功与否是标签，不是"有没有这条数据"
      meta.json 顶层： "success": true/false
      轨迹 npz 里 meta_json 也有同一份（只拿 npz 的下游也能读）：
          "success": false, "success_info": {"threshold_mm": 50.0, "lift_mm": 0.0,
                                             "hold_mm": 0.0, "reasons": ["lift_too_low"]}
      判据（任务书口径，阈值 50mm）：
          success = 抬升(阶段 lift 末 - grasp 末) >= 50mm  且  搬运段末没塌回去
          reasons: lift_too_low = 药盒没离开桌面（夹空/夹偏/压翻，最常见）
                   slipped      = 抬起来了，但搬运途中掉了（搬运末抬升 < 峰值一半）
      50mm 的依据：药盒半高 26mm、木盒口 z=50mm；正样本实测 125.10mm，余量 75mm。

  (2) 失败样本照样导出 —— 这是负样本，不是故障
      抓取失败时照样走完整条渲染：两个 mp4、episode_0.npz、meta.json 全部落盘，
      只是 success=false。verify 里"预期就不该成立"的三项降级为 ⚠️，不计入总判定：
          轨迹一致性（抬升必然对不上基准）、落在木盒里（药盒根本没被搬走）、
          腕部相机里的物块像素数（指尖相机只有把药盒抬到眼前才看得到）
      外部相机不豁免：它本来就该一直看着绿垫，拍不到药盒就是真的取景错了。
      其余导出健康度判据仍是硬性 ✅/❌。
      想强制要求成功：加 --require-success（负样本会被判 ❌ 并返回非 0）。

  (3) schema 校验 —— 不匹配就 SystemExit，绝不让错位数据流进训练集
      轨迹 npz 16 键 + meta_json，图像 npz 3 键，逐键核对形状与 dtype（精确到字节序）：
          time/qpos(N,nq)/qvel(N,nv)/ctrl(N,nu)/phase(<U16)/tcp_pos(N,3)/obj_pos(N,3)/
          obj_quat(N,4)/contact_n(<i8)/fn_jaw/fn_grip/fn_sum/lift_mm/hdisp_mm/
          box_n(<i8)/obj_speed   —— 数值列一律 <f8
          env/wrist(N,H,W,3) uint8 + fps 0 维 float32，且 N == 渲染帧数
      额外键允许存在（向前兼容）；meta_json 里必须有 bool 型 success。
      ★ 这条校验第一次跑就抓到真缺陷：全片无接触时 sum([]) 是 Python int 0，
        整列 fn_jaw 会被存成 int64（同一份契约的 dtype 竟然取决于数据长什么样）。
        已在 EpisodeRecorder.push 里用 float() 强制。

  (4) 造一条负样本（用来验证这条通路）
      env -u PYTHONHOME -u PYTHONPATH python3 render/test_render.py \
          --gz 0.080 --out demo_neg --quick      # 抓取高度抬到 80mm -> 夹空
      实测：success=false, reason=lift_too_low, 视频/npz/meta 全部正常落盘。

  (5) 退出码语义
      record : 抓取判定未通过 -> 1（负样本也会是 1，但 npz 已落盘）
      render : 导出/校验/自检有任何 ❌ -> 1；负样本只要导出健康 -> 0
      all    : 任一段非 0 -> 非 0（所以 shell 里 `&&` 不会把抓取失败当成通过）

【坑（都已踩过并修好，改代码时别改回去）】
    * MUJOCO_GL 必须在 import mujoco 之前设置，所以用 _prescan() 预扫命令行。
    * EGL 退出时必刷 EGLError，用 sys.unraisablehook = lambda: None 静音。
    * 抽帧会把"阶段末帧"漏掉，导致抬升显示成 109.70mm 的假象
      -> EpisodeRecorder.push(force=True) 在阶段末尾强制落帧。
    * 渲染尺寸必须全程恒定，否则 VideoWriter 静默丢帧（VideoSink 里有断言）。
    * MuJoCo 给 RGB，OpenCV 要 BGR，写之前必须 [:, :, ::-1]。
    * 【环境坑，与本项目逻辑无关】本机 `mj_step` 偶发 SIGFPE（浮点异常/核心已转储），
      概率约 7%（15 次 record 命中 1 次），报错栈顶永远是 `mujoco.mj_step`，
      每次落在不同阶段；同一份代码在单进程里连跑 60 轮一次都不崩。
      已经把录制逻辑整个关掉复测过，说明不是本项目的代码路径导致的（进程级偶发）。
      对策：跑 `all` 时套一层重试即可，物理结果是可复现的（抬升/位移每次一致）：
          for i in 1 2 3 4 5 6; do
            env -u PYTHONHOME -u PYTHONPATH python3 -u -X faulthandler \
              render/test_render.py all > logs/all_$i.log 2>&1
            grep -q "总判定" logs/all_$i.log && break
            echo "attempt $i 崩了，重试"
          done
      ⚠️ 别把输出管道到 `head`（`| head -50`）：管道被提前关闭会稳定触发同一个崩溃。
