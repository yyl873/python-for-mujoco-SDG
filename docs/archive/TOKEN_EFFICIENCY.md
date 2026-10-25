# 省 Token 工作流（Agent 侧操作规范）

> 这份文档不是"怎么用 DeepSeek"的科普，是这个仓库的**会话操作纪律**。
> 一句话规则：**执行留在本地，结论才进上下文。**
>
> 配套工具在 `tools/`（见 `tools/README.txt`）。实测数字全部来自本仓库真跑，不引用外部案例。

---

## 1. 为什么需要它

本仓库一轮完整验证 = 4 段脚本（ik / grasp / render / video），控制台日志 211 行、10517 字符。
这些日志里当次真正需要的只有一二十个数字，其余 99% 是要花输入 Token 却从不被引用的行。
更贵的是**往返**：老流程要 4 次工具调用 + 每次把整段日志塞回上下文 + 出问题还要再跑一次。

`tools/run_pipeline.py` 把 4 段塞进一个本地进程，只把结论字段带回上下文：

| | 旧路径（4 次工具调用） | 新路径（1 次调用） |
|---|---|---|
| 上下文行数 | 211 行 | 8 行 |
| 上下文字符 | 10517 | 453 |
| 约合 Token（按 1.4 字/token 估） | ~7500 | ~320 |
| 压缩比 | — | **字符 23.2× / 行数 26.4×** |
| 完整日志 | 全在上下文里 | 落盘 `logs/run_<段名>.log`，按需定向读 |
| 结构化回执 | 无 | `logs/run_all.json` |

代价说明（别只看倍数）：日志并没有被丢弃，只是从"默认进上下文"改成"落盘 + 按需取"。
真正的节省来自**默认不传**，而不是压缩失真。

---

## 2. 命令速查

```bash
# 所有命令都要带这个前缀（HANDOFF §1.1 的 PYTHONHOME 污染坑）
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py            # 全跑，回 8 行回执
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py ik grasp   # 只跑指定段
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --status   # 不跑，只看产物现状
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --json     # JSON 回执（仍会真跑！）
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --selftest # 只校验判据，不跑仿真

# 读东西之前先用摘要
env -u PYTHONHOME -u PYTHONPATH python3 tools/digest.py output                       # 目录统计
env -u PYTHONHOME -u PYTHONPATH python3 tools/digest.py render/test_render.py  # 1457 行 -> 12 行骨架
env -u PYTHONHOME -u PYTHONPATH python3 tools/digest.py render/test_render.py --grep "def verify"  # 只回命中数
env -u PYTHONHOME -u PYTHONPATH python3 tools/digest.py logs/run_render.log --tail 8 # 只看尾部
```

> ⚠️ `--json` / `--tail` / `--keep-going` **都不是只读开关**，带上它们照样会跑仿真。
> 只想看现状请用 `--status`；只想验证解析器请用 `--selftest`。

---

## 3. 五条纪律（对应通用优化清单的落地方式）

| # | 通用原则 | 本仓库的具体落法 |
|---|---|---|
| 1 | 程序化批量调用（PTC）：让模型只出"执行计划"，机械步骤交给执行器 | `tools/run_pipeline.py` 一个进程跑完 4 段；模型只在判据没绿时介入，不逐步盯每一段的交互 |
| 2 | 结构化反馈：工具直接回"成功与否 + 关键量"，不让模型猜 | 每段固定字段：`iter/residual_m`、`rise_mm/move_mm/constraint1/2`、`frames/fps/checks_pass`、产物字节数 |
| 3 | 批量操作 + 单次核验 | render 段自带 12 项自检（视频回读、非静止、物块可见、落点、松手、静止、回位、不刮木盒…），跑完即核验，**不再单独写校验脚本** |
| 4 | 精简返回：统计类只回数量 | `digest.py --grep` 只回命中数与首尾行号；`digest.py <dir>` 只回文件数/体积/分布，不列清单 |
| 5 | 长结果分页：只回前 N 行，其余按需 | 失败时只回该段日志尾部 15 行（`--tail N` 可调）；读大文件先 `digest` 拿行号，再用 `read` 的 `offset/limit` 定向读 |

另外两条通用建议在本仓库的适用性，实话实说：

- **上下文硬盘缓存（省 90%）**：能配合，但前提是**前缀稳定**。本仓库的稳定前缀顺序固定为
  `README.md → docs/HANDOFF.md → tools/README.txt`，会话内不要重写这三份文件的开头。
  收益大小取决于请求前缀是否真的逐字节一致，本仓库没有测量工具，**不要把它当成已确认的收益**。
- **max_tokens / 模型选型**：机械段（跑脚本、对数字、改一行参数）不需要旗舰模型；
  一次性给出大段设计再让模型执行，比"多轮小步试错"便宜。这是经验判断，不是本仓库的测量结论。

---

## 4. 回执契约（三态，别压成两态）

`run_pipeline.py` 的每段结论只有三种，退出码 `0 / 1 / 2` 与之对应：

| 状态 | 含义 | 流水线行为 |
|---|---|---|
| ✅ `ok` | 判据与约束全绿 | 继续 |
| ⚠️ `warn` | 脚本自判成功，但触犯了设计红线里的约束（例：约束1 药盒被撞动） | **继续**，但回执必须如实标黄 |
| ❌ `fail` | 脚本自判失败 / 退出码非 0 / 自检有红 | 中断（`--keep-going` 可继续） |

为什么坚持三态：把 `warn` 压成 `ok` 等于伪造结论（当前 grasp 段就是这种情况，见 HANDOFF §5.2）；
压成 `fail` 又会让"物理上抓成功了、只是蹭了一下"和"根本没抓住"混为一谈。

---

## 5. 扩展一段新流水线的标准做法

1. 在 `tools/run_pipeline.py` 的 `STAGES` 里加一条 `{name, desc, cmd, parse, artifacts}`；
2. 写 `parse_xxx(log) -> (facts, status)`：**facts 只放结论字段，绝不放日志正文**；
3. 在 `--selftest` 里用假日志文本加一条断言（不跑仿真，秒级）。

改完必须 `--selftest` 全绿，**再真跑一次**确认回执没变形——只跑 selftest 不算验证。

---

## 6. 反例清单（这些做法在本仓库会白烧 Token）

- 让模型"跑完再自己检查一遍"：render 的 12 项自检已是客观判据，重查 = 重复付费。
- 把 `demo/episode_0.npz`（838 MB）列进产物清单或整读：只回存在性与字节数。
- 为了定位一次失败而重跑整条流水线：先 `--status`，再 `digest --tail`。
- 不带 `env -u PYTHONHOME -u PYTHONPATH` 直接 `python3`：报 `No module named 'encodings'`，
  白跑一轮（本仓库自身踩过这个坑，见 README §2）。
- 把 `--json` 当"只读"用：它会真跑仿真并把中途产物写花（2026-10-24 实测踩过一次，
  表现为 `env_camera.mp4` 被截断成 3.1 MB，恢复方法是重跑 render 段）。

---

## 7. 全局版（跨工作区，2026-10-24 已安装）

本仓库这套东西已泛化到全局，**新会话自动生效**，与本目录无关的项目同样可用：

| 全局件 | 路径 | 生效方式 |
|---|---|---|
| 全局指令 | `~/.dsh/AGENTS.md` | 每会话自动注入（简短，≤2 KB；细节不放这里） |
| 全局技能 | `~/.dsh/skills/token-efficiency/SKILL.md` | 目录只注入 name+description，正文按需加载 |
| 全局命令 | `~/.local/bin/dsh-digest`、`~/.local/bin/dsh-run` | 已在 PATH，任意目录可用 |
| 实现体 | `~/.dsh/toolkit/{digest.py,run.py}` | 供上面两个入口调用（已清 PYTHONHOME/PYTHONPATH） |

```bash
dsh-digest <dir|file> [--grep PAT] [--tail N] [--json]   # 摘要代替整读
dsh-run [stage...] [--manifest dsh.run.json] [--json] [--status]   # 清单驱动批量执行
dsh-doctor [--json] [--quiet]                            # 核对三层是否真的生效
```

**怎么确认生效**：`dsh-doctor` 一条命令给三层判据（退出码 0/1/2）：

| 层 | 判据 | 不绿的常见原因 |
|---|---|---|
| L1 文件层 | 指令/技能/实现体/三个入口就位、指令体积 ≤4 KB、名解析到本安装 | 没装、入口不可执行、`~/.local/bin` 不在 PATH |
| L2 命令层 | `dsh-digest`、`dsh-run` 的 `--selftest` 真跑通 | 解释器或文件权限问题 |
| L3 会话层 | 本会话历史里有插件打的 typed source：`agent-instructions`、`skill-catalog` | 老会话（基线不追溯）；技能正文未加载不算失败 |

L3 是最硬的证据：它查的是插件写入的结构化标记，不是路径或名字的正则匹配。
新会话首条请求应能看到 `Instructions from: ~/.dsh/AGENTS.md` 与 `<available_skills>` 里的
`token-efficiency`（目录只含摘要，正文需 `skill` 工具加载后才出现）。

`dsh-run` 用工作区根目录的 `dsh.run.json`（`stages[]` 里每段配 `cmd/expect/forbid/warn_if/facts/artifacts`），
判定优先级与回执三态和本仓库 `run_pipeline.py` 一致（`exit!=0`→fail，`forbid`→fail，`warn_if`→warn，
`expect` 未全中→fail），退出码 0/1/2，日志落 `<工作区>/.dsh-run/logs/`。

分工：**通用能力用 `dsh-*`，本仓库仍用 `tools/run_pipeline.py`**（它有 4 段专用解析器与 12 项自检判据，
不能泛化，也不打算重写）。两份 `digest` 实现内容重复，仓库版为准、全局版是泛化拷贝，改一处记得看另一处。

卸载：`sh ~/.dsh/toolkit/install.sh --uninstall`（移除指令、技能、命令；重装同理）。
