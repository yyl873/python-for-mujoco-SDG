================================================================================
 tools/  —— 省 Token 的本地执行器（不是业务代码，别在这里放物理逻辑）
================================================================================

【为什么要这个目录】
    一轮完整验证 = 4 段脚本 ≈ 200+ 行控制台日志。这些日志 99% 的行当次读都用不上，
    但整段塞回模型上下文就是实打实的输入 Token。
    这里的工具把「执行」留在本地进程里，只把「结论字段」交出去。

【目录分层】
    tools/            入口层：流水线、审计、结果表、素材生成。命令都从这里调，路径稳定。
    tools/probes/     一次性诊断层：排完某个具体问题就归档的探针/扫描/数据修复脚本。
                      不是稳定 API，没有任何代码 import 它们，保留只为结果可复现。
                      清单与"它回答过什么问题"见 tools/probes/README.md
    判断标准：新脚本先问一句"这是一直要用的入口，还是一次性排障？"
              一次性的直接放 probes/，别再摊在 tools/ 根下。

【入口脚本清单（一句话）】

  流水线与阅读
    run_pipeline.py                ik -> grasp -> render 三段清单式执行器，只回三态回执
    inspect_episode.py           ★ 按需出单条 episode 的排查产物（拼图/预览/体检）
    digest.py                      文件/目录摘要（def/class 骨架 + 行号 + tail/grep）
    callmap.py                     一次运行实际走到的仓库内函数成本表 → "该读哪几行"
    replay_policy.py               把策略放回 MuJoCo 里开环重放 / 闭环回放

  审计与仪器（被判据引用，属于稳定面）
    probe_contact_audit.py       ★ 接触真实性六项闸门 G1~G6（0.5 s/条）
    probe_grasp_robustness.py      换起点/配比/节奏后还抓得住吗（30 s）
    probe_release_gripper.py       离线把仿真 release 帧喂给 ACT，看夹爪输出刻度
    probe_release_gripper_real.py  真机 holdout 上找"第二次张开"（释放段），喂给 ACT
    log_gripper_rollout.py         真机 rollout 记 req/sent/present + 6 个关节角 → CSV
    compare_release_ab.py          两份 roll CSV 对比 A/B 的释放时机与位姿
    compare_sim_real.py            sim156 ↔ real_train_fixed 域差（起点差 / 动作 W1）
    test_gripper_loaded.py         带载夹爪能否到 80（定时窗口，不用 input()）

  结果表与素材
    make_tables.py                 从 rl/eval/*.json 生成 docs/RESULTS.md（--check 只校验）
    make_assets.py                 演示 mp4 → README 能内嵌的 GIF（docs/assets/）
    make_compare_video.py          左右两条演示拼成上下并置对比视频 + GIF

  训练 / 评测 / 发布编排（shell）
    export_sim156_chunked.sh       分块导出 sim156，已完成块自动跳过，最后 merge + 体检
    train_A_to22500.sh             A（纯真机）17500 → 22500 步，与 B@22500 同步数对照
    train_sim156_then_real.sh      两阶段：纯仿真预训练 → 真机微调
    train_mix.py                   按"臂"训练 ACT（本地数据集、不联网、参数命令行可覆盖）
    run_armB_resume_loop.sh        armB 续训守护：崩了自动从最新 ckpt 续，最多 20 次
    run_eval_ab_cpu.sh             A/B holdout 离线评测（CPU 版，不依赖 GPU/桌面）
    run_B_release_check.sh         B 释放诊断：恢复训练模式跑一次（关 temporal ensembling）
    pre_push_check.sh              发布前自检：自检套件/密钥/大文件/README 链接/待推送
    sync_kit.sh                    把 embodied-eval-kit 同步到独立仓库 yyl873/embodied-eval-kit

【文件清单（详细用法）】

  run_pipeline.py      一条命令跑完 demo 的 ik -> grasp -> render
                       （video 段第八轮已退役：mp4 就是最终格式，不再 AVI 转码）
                       （批量刷数据不用它，直接用 collect/collect.py）
        python3 tools/run_pipeline.py                 # 全跑，回执出屏（约 10 行）
        python3 tools/run_pipeline.py ik grasp        # 只跑指定段
        python3 tools/run_pipeline.py --status        # 不跑，只看产物现状
        python3 tools/run_pipeline.py --json          # 机器可读回执（仍会真跑！）
        python3 tools/run_pipeline.py --selftest      # 只校验解析判据，不跑仿真
        段名: ik / grasp / render

        回执是三态，别把三态压成两态：
          ✅ ok    判据与约束全绿
          ⚠️  warn  脚本自判成功，但触犯了设计红线里的约束（如约束1 药盒被撞动）
          ❌ fail  脚本自判失败 / 退出码非 0 / 自检有红
        退出码: 0 = 全绿, 1 = 有 warn, 2 = 有 fail
        fail 会中断后续段（--keep-going 可继续），warn 不中断。

        全文日志按段落盘: logs/run_<段名>.log
        结构化回执落盘  : logs/run_all.json
        失败时默认只回显该段日志尾部 15 行（--tail 0 关闭，--tail N 调整）。

  inspect_episode.py   ★按需出排查产物（第八轮新增；采集阶段不再生成这些）
        python3 tools/inspect_episode.py --episodes 0,7,13   # 抽帧拼图 jpg + 夹稳预览 png
        python3 tools/inspect_episode.py --failed --limit 5  # 先看负样本
        python3 tools/inspect_episode.py --every 10          # 每 10 条抽查 1 条
        python3 tools/inspect_episode.py --all --check-only  # 只体检（帧数/尺寸/fps/可解码）
        python3 tools/inspect_episode.py --episodes 4 --render  # 没视频就补渲（纯运动学回放）
        python3 tools/inspect_episode.py --demo              # 看 demo/ 的单条 demo
        产物落 inspect/（可随时删）；体检不过 → ⚠️ + exit 1
        ★它不跑物理：补渲走 collect.render_episode（只 mj_forward），其余只是解码已有 mp4。

  digest.py            用摘要代替整读
        python3 tools/digest.py output                       # 目录统计（不列清单）
        python3 tools/digest.py render/test_render.py  # def/class 骨架 + 行号
        python3 tools/digest.py logs/run_render.log --tail 8 # 只看尾部
        python3 tools/digest.py render/test_render.py --grep "def verify"
                                                             # 只回命中数 + 首尾行号
        python3 tools/digest.py --selftest

【全局版在哪（2026-10-24 已装，跨工作区）】
    通用能力已经泛化到全局，任意目录的新会话都能用：
      dsh-digest <dir|file> [--grep PAT] [--tail N] [--json]     # ~/.dsh/toolkit/digest.py
      dsh-run [stage...] [--manifest dsh.run.json] [--json] [--status]
                                                                 # ~/.dsh/toolkit/run.py
      dsh-doctor [--json] [--quiet]                              # 核对三层是否真的生效
    全局指令 ~/.dsh/AGENTS.md（每会话注入，保持简短），
    全局技能 ~/.dsh/skills/token-efficiency/SKILL.md（目录注入、正文按需加载），
    卸载/重装：sh ~/.dsh/toolkit/install.sh [--uninstall]

    分工：本目录的 run_pipeline.py 有 4 段专用解析器与 12 项自检判据，**不能被 dsh-run 替代**，
    继续用；dsh-digest 与本目录的 digest.py 是同一套逻辑的两份拷贝（仓库版为准）。

【两条使用纪律】
  1. 所有命令都要带 `env -u PYTHONHOME -u PYTHONPATH` 前缀（HANDOFF §1.1 的坑）。
     run_pipeline.py 会替子进程清掉这两个变量，但它自己也得先能起来。
  2. 需要正文时，先用 digest 拿行号，再用 read 的 offset/limit 定向读区间。
     不要把 70 KB 的 test_render.py 或上百行日志整篇读进上下文。

【改这些工具时】
    新增一段流水线的三件套：
      a) 在 STAGES 里加一个 dict（name/desc/cmd/parse/artifacts）
      b) 写 parse_xxx(log) -> (facts: dict, status: "ok"|"warn"|"fail")
         facts 只放结论字段，绝不放日志正文
      c) 在 --selftest 里加一条断言（假日志文本，不跑仿真）
    改完必须 `--selftest` 全绿，再跑一次真流水线确认回执没变形。
