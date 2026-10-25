================================================================================
 tools/  —— 省 Token 的本地执行器（不是业务代码，别在这里放物理逻辑）
================================================================================

【为什么要这个目录】
    一轮完整验证 = 4 段脚本 ≈ 200+ 行控制台日志。这些日志 99% 的行当次读都用不上，
    但整段塞回模型上下文就是实打实的输入 Token。
    这里的工具把「执行」留在本地进程里，只把「结论字段」交出去。

【文件清单】

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
