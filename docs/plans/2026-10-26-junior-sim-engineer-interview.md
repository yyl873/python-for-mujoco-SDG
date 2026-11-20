# plan · 12 月初拿垂类实习（5 周 / 36 天）

- 对应 spec：`docs/specs/2026-10-26-junior-sim-engineer-interview.md`
- 窗口：**2026-10-26 → 2026-12-01**
- 硬节点：**11-12 简历 v1 锁死** → **11-15 投递完毕** → 11 月下旬面试 → 12 月初 offer
- 产能（2026-10-26 确认）：**每天 ≥4 h × 每周 5 天 = 20 h/周 ≈ 25 个 4 h 块 ≈ 100 h**
  → 原"必须砍项"作废，全部保留；按 4 h 块验收，一周 5 块，一个阶段 4~5 块
- 面向人读的简化版：`docs/如何学.md`（本文是"带命令的底账"，冲突时以本文为准）

> 命令前缀固定 `env -u PYTHONHOME -u PYTHONPATH`；渲染类再加 `MUJOCO_GL=egl`。
> **禁止**把 python 输出直接管道给 `head`/`tail`（本机 SIGFPE），一律 `> logs/x.log 2>&1` 再读。

---

## 0. 主线重排（2026-10-26）：实验闭环先行，学习服务实验

**原顺序（作废）**：基线 → 判据 → 闭环 → 简历 → 真机 A/B → 冲刺
**新顺序**：**连通性探针 → 真机 A/B → 闭环结论 → 简历 → 按需补学习**

用户理由（原话）：不要"学完发现实验无法闭环，再改代码再学新的"。所以先用 **1 天证明闭环可行**，
再把学习插在"解释实验结果"的位置上。下面 §"阶段 0–5"改为**实验之后的支撑段**，内容仍有效。

| 步 | 时长 | 干什么 | 通过判据 | 失败分支 |
|---|---|---|---|---|
| **E0** | 半天 | **连通性探针**：找端口 → `armA` 跑一次 autonomous rollout（10~15 s），**只验链路，不判对错** | 机械臂按策略动作 ≥10 s、双相机画面正常、单次耗时落档 | 见下方风险表 |
| **E1** | 1 天 | **真机 A/B 各 12 次**（3 起点 × 2 组 × 各 4 次，随机交替，episodic 录像） | 24 次有效试验 + 失败模式分布 | 拿不到 → 降 8 次/组并标注"样本更小" |
| **E2** | 半天 | 结论落 `docs/notes/real_ab_result.md` + 简历段落（"方向性观察" + Wilson 区间） | 结论能被追问 3 层 | — |
| **E3** | 半天（可选） | 仿真内用 `tools/replay_policy.py` 跑 A/B 对照 | 仿真/真机两侧都能闭环 | — |

**已验证的闭环前置（2026-10-26 实测，结论：不需要写新代码）**

| 项 | 状态 | 证据 |
|---|---|---|
| `armA` checkpoint | ✅ 在盘 | `/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500`（另 5000/10000/15000） |
| `armB` checkpoint | ✅ 在盘 | `/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500`（另 5000–20000） |
| 真机推理入口 | ✅ 已存在 | `lerobot_rollout.py`：`--policy.path` + `--robot.type=so_follower`，`--strategy.type=episodic` 可录 N 次 |
| 运行环境 | ✅ | `/home/yyl/lerobot/.venv` + `uv` |
| 真机标定 | ✅ | `~/.cache/huggingface/lerobot/calibration/{robots,teleoperators}`（10-18） |
| 数据规格 | ✅ | `robot_type=so_follower` / 30 fps / `action[6]`+`state[6]` / 相机键 `observation.images.env` + `.wrist`（480×640） |
| 仿真侧闭环 | ✅ | `tools/replay_policy.py --policy … --no-render --json` |
| 离线闭环 | ✅ 已跑过 | `tools/run_eval_ab_cpu.sh`（结论：不显著，已进 HANDOFF） |

**E0 的三条已知风险与对策（都是"配置级"，不是改代码）**

| 风险 | 现象 | 对策 |
|---|---|---|
| 相机键不匹配训练集 | 策略输入键名/维度报错 | 显式传 `--robot.cameras`，键名必须是 `env` / `wrist` |
| 帧率/控制频率不一致 | 动作抖、跟不上、超时 | 控制频率对齐 **30 Hz**（数据集 fps=30） |
| 动作单位（度 vs 弧度） | 机械臂乱动或超限位 | `info.json` 里 action 是**度**；若需转换，**只允许在 eval 脚本内**做 |

> E0 只要"动起来 + 画面正常"就算通过，**不要求抓到**。抓不到是 E1 要统计的事，不是 E0 的失败。

### 0.1 两日作战表（2026-10-26 用户确认：明天可测、后天去真机）

| 天 | 干什么 | 命令 / 判据 |
|---|---|---|
| **10/27 明天（4 h）** | **A1 解释器**：用 `lerobot312`；`/home/yyl/lerobot/.venv` 是空壳别用 | `env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python -c "import torch,lerobot;print(torch.__version__)"` → 2.7.1 |
| | **A2 policy 前向 smoke**（今天已替你验过，见 §0.3） | 输出 shape `(1,100,6)` |
| | **A3 真机命令拼装** | `lerobot_rollout.py --policy.path=<ckpt>/pretrained_model --robot.type=so_follower --strategy.type=episodic`，`--help` 里参数名逐个核对 |
| | **A4 记录表 + Wilson 区间** | 12 次/组能填、区间能算 |
| **10/28 后天（真机）** | **B1** `lerobot_find_port.py`；标定复用（10-18 已有，不重标） | 拿到端口号 |
| | **B2 E0 探针**：armA 跑 15 s | 臂动 ≥10 s、双相机画面正常、无维度错 |
| | **C1 E1 A/B 各 12 次**（3 起点 × 2 组 × 各 4 次，随机交替） | 24 次有效 + 失败模式分布 |
| | **C2 结论落盘** | `docs/notes/real_ab_result.md`，能被追问 3 层 |

### 0.2 今天已抓到的两个坑（后天一定会撞，先记下）

| 坑 | 现象 | 正确做法 |
|---|---|---|
| checkpoint 路径少一层 | `FileNotFoundError: config.json not found in …/017500` | 必须指到 **`…/017500/pretrained_model`** |
| preprocessor 固化了 `device: cuda` | `Requested device 'cuda' but CUDA is not available` | 本沙箱无 GPU 才报；你自己终端有 4060 应正常；真报错就加 `--device=cpu` |

### 0.3 已验证事实（2026-10-26 实跑，**非推测**）

- **解释器真身**：`envs/lerobot312`（3.12.15）= lerobot 0.6.2 + torch 2.7.1 + opencv → **真机链路够用**；
  `envs/lerobot`（3.10.22）= mujoco 3.16 + torch 2.10.0+cu126（无 lerobot）；
  `/usr/bin/python3.10` = mujoco 3.16（无 torch）；
  **`/home/yyl/lerobot/.venv` 的 site-packages 是空的，不要用**。
- `ACTPolicy` 从 `…/017500/pretrained_model` 成功加载；input_features = `observation.state` +
  `observation.images.env` + `observation.images.wrist`；`chunk_size=100 / n_action_steps=100`。
- CPU 前向通过：输入 `(1,6)` + 两张 `(1,3,480,640)` → 输出 **`(1,100,6)`**；
  未反归一化范围 `-0.72 ~ 0.30`（经 postprocessor 才是"度"）。
- 本沙箱无 `/dev/nvidia*` → 上表 cuda 报错是沙箱特有，不是你的机器问题。
- 复跑方式（明天自己再验一遍）：
  ```bash
  cd /home/yyl/deepseek-harness
  env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python - <<'EOF'
  import torch
  from lerobot.configs.policies import PreTrainedConfig
  from lerobot.policies.factory import get_policy_class
  P = "/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500/pretrained_model"
  cfg = PreTrainedConfig.from_pretrained(P); cls = get_policy_class(cfg.type)
  p = cls.from_pretrained(P, config=cfg).eval()
  out = p.predict_action_chunk({"observation.state": torch.zeros(1,6),
      "observation.images.env": torch.zeros(1,3,480,640),
      "observation.images.wrist": torch.zeros(1,3,480,640)})
  print(tuple(out.shape))   # 期望 (1, 100, 6)
  EOF
  ```

---

## 阶段 0（10/29–11/02）：护栏 + 当日基线 + 版本管理

**为什么先做**：后面所有数字都要跟"今天的机器"对齐；这仓库**没有 `.git`**，面试官看材料时会问。

**改哪些文件**：新增 `docs/notes/baseline-2026-10-26.md`；新增 `.git`（只 `add` 不 `push`）

**步骤**
1. `ls /dev/nvidia*` 记录 GPU 可见性（决定耗时口径：19.1 s/条 vs 75~90 s/条）
2. 跑流水线回执 + 接触审计，数字抄进 baseline 文档
3. 版本管理（**先写 `.gitignore` 再 add**：仓库共 2.4 GB，其中 `videos/` 2.0 GB，直接 `add -A` 会炸）

```bash
cd /home/yyl/deepseek-harness
cat > .gitignore <<'EOF'
videos/
logs/
inspect/
.cache/
.pylibs/
__pycache__/
*.npz
*.bak
*.bak.*
EOF
git init -q . && git add -A && git commit -qm "baseline: so-arm101 sdg 流水线（不含视频与日志）"
git count-objects -vH | grep size-pack   # 应该 < 100 MB，超过说明 .gitignore 没生效
```

**验证命令**
```bash
cd /home/yyl/deepseek-harness
ls /dev/nvidia* 2>&1 | head -3
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py > logs/prep_pipeline.log 2>&1
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --status
MUJOCO_GL=egl env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py > logs/prep_audit.log 2>&1; echo "audit_exit=$?"
git log --oneline | head -2; git count-objects -vH | grep size-pack
```
**通过判据**：回执 `ik ok / grasp ok / render ok`（fail=0）；`audit_exit=0` 且
G1 下探 0/400、G2 压盒 ≤0.5 mm、G3 倾角 ≤5°、G4 抬升 ≥50 mm；`git log` 有 1 条提交，
`size-pack` < 100 MB。

**2026-10-26 已实测（可直接沿用，仍需自己复跑一次确认）**
- 接触审计：G1 ok 接触 0/400 步、G2 ok 压盒 +0.39 mm、G3 夹稳倾角 1.06°、
  抬升 150.29−27.77 ≈ **+122.5 mm**；药盒静置 z=27.77 mm
- 一次抓取实际走过 **81 个仓库内函数**，主干仅 6 个（证据：`logs/map_audit.txt`）

**失败时怎么办**：`mj_step` SIGFPE（约 7%，进程级）→ 原地重跑；EGL 崩 → `JOBS=4`。
审计退出码 1 → 停下，先对照 `docs/PHYSICS_FACTS.md` 定位是哪项闸门漂了，本 plan 不允许为此改参数。

---

## 阶段 1（10/30–11/05，7 天）：把"判据"和"主干"吃下来

**改哪些文件**：新增 `docs/notes/judgement-map.md`、`docs/notes/reading-map.md`

**步骤**
1. 用 `tools/callmap.py` 拿到"这次运行真正走过的函数"，只读主干（不读全文件）
2. 读 `render/test_render.py` 的 `run_episode`(:746) / `_move`(:636) / `_move_cartesian`(:665) /
   `_hold`(:727) + `contact_audit_from_arrays`(:547) / `evaluate_success`(:585)
3. **补 IK**（2026-10-26 更正：不会，2 天）：读 `ik/test_ik.py`（**181 行**，全项目最便宜的护城河），
   能把"雅可比=关节速度→末端速度"与"PD 稳态误差→残差 0.000923 m"讲清；跑一次确认 24 次迭代收敛
4. 一页纸写清 G1–G6 定义、阈值、越界后果，每条阈值标 `文件:行号`
5. 每天 4 h 固定窗口，5 天 = 20 h，正好"判据 3 天 + IK 2 天"

**验证命令**
```bash
cd /home/yyl/deepseek-harness
MUJOCO_GL=egl env -u PYTHONHOME -u PYTHONPATH python3 tools/callmap.py \
    tools/probe_contact_audit.py > logs/map_audit.txt 2>&1
grep -n "仓库内函数\|run_episode\|_move(\|_hold(" logs/map_audit.txt | head
grep -n "0.5\|5.0\|50.0\|0.0005" render/test_render.py | head -10
```
**通过判据**：从 `judgement-map.md` 里任抽一条阈值，60 秒内 `grep` 到对应代码行；
`reading-map.md` 里每个主干函数都写了"输入/输出/失败时谁变红"。

**失败时怎么办**：读不动 → `python3 tools/digest.py <file>` 出骨架再按行读区间，
禁止把 136 KB 整文件灌进上下文。

---

## 阶段 2（11/06–11/12，7 天）：闭环 + gap 故事 + 可看作品

**改哪些文件**：新增 `docs/notes/closed-loop-map.md`、`docs/notes/interview-stories.md`、
`README` 补充一节、一条 60 秒 demo 视频（`videos/`）

**步骤**
0. **补 ACT**（2026-10-26 更正：不会，1 天）：用现成 `armA/armB` 数据跑 3 步 `lerobot-train` smoke
   （skill 记录 loss 90.2→61.7）；能讲清"动作分块为何减少累积误差与高频抖动"即可，**不实现算法**
1. 读 `tools/replay_policy.py`、`tools/train_mix.py`、`tools/run_eval_ab_cpu.sh`（关卡 6）
2. 重跑离线 A/B，把"起点差 7~17°、去掉起点后 W1 25~33°"变成自己跑出来的数字
3. 三个深挖点写成"问题→命令→数字→结论"四段式
4. 录 60 秒 demo（抓取 + 双相机画面），README 挂链接

**验证命令**
```bash
cd /home/yyl/deepseek-harness
bash tools/run_eval_ab_cpu.sh > logs/prep_eval_ab.log 2>&1; echo "exit=$?"
python3 tools/inspect_episode.py --episodes 0,7 --check-only > logs/prep_check.log 2>&1; echo "exit=$?"
```
**通过判据**：离线评测 exit=0；gap 数字与 `HANDOFF_SIM2REAL_V8.md` §1 同量级（差异必须解释，
不许静默覆盖）；三个深挖点每个都有可复跑命令。

**失败时怎么办**：评测跑不动 → 只读已有 `logs/eval_holdout_AB.log`，标明"历史证据 + 日期"，
**不许**把没跑过的数字写成自己的。

---

## 阶段 3（11/12–11/15，4 天，**硬节点**）：简历 v1 + 8 分钟自述 + 投递

**改哪些文件**：新增 `docs/resume/v1.md`（脱敏）

**步骤**
1. 主线改成：物理真实性闸门 → 域随机化 → 标准化导出 → 定量归因 sim-real gap
2. **删掉** "A/B/D/E 验证了泛化提升"（仓库事实：C/D 未训练、E 未跑、离线不显著、真机未做）
3. 必带数字：957 条成功母本 / 139 真机 eps / 62,273 帧 / 起点差 7~17° / W1 25~33° /
   下探零接触 0/400 / 压盒 0.39 mm / 19.1 s/条 → `--jobs 8` → 1000 条 1~1.5 h
4. 8 分钟自述录音一遍，删掉所有背不下来的句子
5. **11-15 前投完**（12 月初要 offer，只有一轮面试窗口）

**验证命令**
```bash
cd /home/yyl/deepseek-harness
grep -niE "消融|A/B/D/E|泛化提升|验证了.*提升" docs/resume/v1.md || echo "CLEAN_OK"
```
**通过判据**：`CLEAN_OK`；每个数字能指到 spec §3 证据表一行；自述不看稿 8 分钟讲完。

**失败时怎么办**：写不出"负结论"的价值 → 用固定句式
「我量到了 X（数字），因此瓶颈是 Y（机制），下一步是 Z（可执行）」。

---

## 阶段 4（11/16–11/22，7 天）：真机 A/B 各 12 次 + L1 演练 + ACT smoke

**产能允许后的口径（2026-10-26 更新）**：每周 20 h，真机 A/B 恢复"**各 12 次**"，分两个半天做
（3 个起点 × 2 组模型 × 各 4 次）。仍须写清"12 次样本不足以称显著"——这个诚实本身就是加分项。

**改哪些文件**：新增 `docs/notes/real_ab_result.md`；`collect/collect.py` 的 DR 区间（**用完即恢复**）

**步骤**
1. 找 so-follower 推理入口（`/home/yyl/lerobot` 侧）；没有就先写最小推理脚本（1 天）
2. 协议：3 个起点 × 2 组模型 × 各 4 次 = **12 次/组**，**随机交替顺序**，记录成功/失败模式
3. 报结果带 Wilson 区间，措辞只能是"方向性观察"
4. L1 演练：`*.bak` → 改 `noise_sigma` 2.0~4.0 → 3.0~6.0 → 先写预测 → 跑 20 条 pilot → 恢复

**验证命令**
```bash
cd /home/yyl/deepseek-harness
cp collect/collect.py collect/collect.py.bak.$(date +%Y%m%d_%H%M%S)
# ...改完跑完...
mv collect/collect.py.bak.* collect/collect.py && diff collect/collect.py collect/collect.py.bak.* \
  > /dev/null 2>&1 && echo "RESTORED_OK" || echo "STILL_DIRTY"
```
**通过判据**：两组各 ≥12 次有效试验 + 失败模式分布；`RESTORED_OK`；L1 演练有"预测 vs 实测"对照。

**失败时怎么办**：3 天内拿不到 12 次/组 → 降到 8 次/组并在材料里标"样本更小"，
仍拿不到 → **直接砍**，退回 v1 主叙事；
L1 闸门变红 → 立即恢复，把"为什么 DR 不能激进"写成反面素材（这是可讲的判断力，不算失败）。

---

## 阶段 5（11/23–12/01，9 天）：面试冲刺

**步骤**
1. 按 JD 补漏（ROS2 只到"能读能做 demo"档就够，见 `docs/如何学.md` §二）
2. 30 分钟追问式模拟 ×2，专攻："这数据有多少是真抓起来的""gap 怎么量的""你自己写的代码是哪部分"
3. **面试前 48 h** 重跑 A1/A2 两道闸门，数字以当日输出为准（不一致就改简历，不改代码）

**验证命令**
```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH python3 tools/run_pipeline.py --status
MUJOCO_GL=egl env -u PYTHONHOME -u PYTHONPATH python3 tools/probe_contact_audit.py > logs/prep_final.log 2>&1; echo "exit=$?"
```
**通过判据**：A1/A2 绿且与简历数字一致；能解释 roll 不能抖 / descend 不能放慢 / 合爪 0.28 硬绑定。

---

## 兜底延长段（2026/12/01–12/31，**仅在 11 月底仍未拿到 offer 时启用**）

**触发条件**：11-30 仍无 offer（含口头）。**未触发则整段跳过** —— 不许提前做，
因为投递窗口由招聘季决定，不由你决定；提前把 5 周排满才是主线。

| 阶段 | 日期 | 干什么 | 验证命令 / 判据 |
|---|---|---|---|
| 阶段 6 | 12/01–12/13 | **SDG 自建新场景**：换物体（新 mesh + 碰撞体）或加一个 DR 维度 → 跑 50 条 → 导出 | `collect.py --episodes 50 --jobs 6` exit=0；`probe_contact_audit.py` 退出码 0；导出后可被 `lerobot-train` 读到 |
| 阶段 7 | 12/14–12/20 | 第二批投递（≥10 家）+ 简历 v2 挂上自建场景 | `grep -niE "消融\|A/B/D/E\|验证了.*提升" docs/resume/v2.md` → `CLEAN_OK` |
| 阶段 8 | 12/21–12/31 | C++ 2 天（仅 JD 明写时）+ 补面复盘 | 能读/改一段简单 C++；复盘表落 `docs/notes/review.md` |

**兜底段的硬约束不变**：不许改 `model/*.xml` 的既有物理参数；新增物体必须 `*.bak` + 6 道闸门全绿；
`descend 400 / grasp 300 / lift 400` 一律不动。

**兜底段的验收（L3「能改」）**：新场景 50 条全部通过 G1–G6，且你能口述
"我为了换这个物体改了哪 3 处、为什么"。

---

## 回滚方案

| 改动 | 回滚 | 风险 |
|---|---|---|
| 阶段 0 的 `git init` | `rm -rf .git` | 无 |
| 阶段 4 的 `collect.py` DR 区间 | `mv *.bak.<ts> collect.py` + `diff` 自证 | 低 |
| 阶段 4 最小推理脚本 | 独立新文件，不碰既有代码 | 低 |
| 新增文档 / 视频 | 直接删 | 无 |
| 数据集 / 模型 / 物理参数 | **全程不碰** | — |

## 时间 / 算力预算

| 项 | 预算 |
|---|---|
| 墙钟 | 5 周 / 36 天，简历 11-12 锁死、11-15 投完；未拿到 offer 则启用兜底段到 12-31（+80 h） |
| GPU 训练 | **0**（不训 armC/armD/E） |
| 探针 | `probe_contact_audit` 0.5 s/条；`callmap` ≈1.1 s（实测） |
| pilot | 20 条 × `--jobs 6`，有 GPU 约 10 分钟级 |
| 真机 | 各 12 次，两个半天 |
| 磁盘 | 新增 < 10 MB（文档 + 日志 + 60 s 视频） |

## 完成定义

1. 11-12 有 `docs/resume/v1.md` 且 `CLEAN_OK`；11-15 前投递完成
2. 12-01 前 A1/A2 至少各跑绿一次，且与简历数字一致
3. 8 分钟自述不看稿，被追问三层能立刻给出**命令或文件:行号**
4. `docs/notes/` 四份齐备：judgement-map / reading-map / closed-loop-map / interview-stories
5. 真机 A/B 要么有 ≥12 次/组的诚实结论，要么明确标"未做"，**不允许含糊**
6. （若启用兜底段）12-31 前有一个"自建场景 + 50 条 + 闸门全绿"的可讲作品，且简历 v2 已挂上