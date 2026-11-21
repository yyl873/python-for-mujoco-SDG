# plan：残差 RL 算力对等复测

- 日期：2026-11-21
- 对应 spec：`docs/specs/2026-11-21-compute-parity-retest.md`（已确认方案 A）
- 当前状态：**只写了计划，未执行、未改代码**
- 一句话目标：算法/奖励/动作空间/数据池/判据全不变，只把算力提到 10×，看残差 RL 的负结论是否翻转

---

## 目标

| # | 完成什么 | 可观察结果 |
|---|---|---|
| G1 | `rl/train.py` 支持 `--n-envs N`（N>1 走 `SubprocVecEnv`） | `--n-envs 1` 行为与现在等价；`--n-envs 4/8` 能跑完 |
| G2 | 量出并行收益与内存上限 | 一张表：n_envs ↔ steps/s ↔ 峰值 RSS |
| G3 | 训出 3 个 300k 步的残差 SAC（v1 奖励、±1.43°） | `rl/ckpt/R1big/s{0,1,2}/model.zip` + 曲线 |
| G4 | 用与 §18 相同的协议评测 3 臂 | 3 个 `target_R1big_s*_clamp_90.json`，基座行 78/90 |
| G5 | 按 spec §6 的预注册规则给出判定 | 报告 §21 一张表 + 一句判定 |

## 前置条件

1. `git status` 干净（工作区当前干净：`e557e11` 之后只有文档与 probe 工具，已全部提交）。
2. 现有 R1 产物完好（不动它们）：
   ```bash
   ls rl/ckpt/R1/s{0,1,2}/model.zip rl/curves/R1/s{0,1,2}.jsonl
   ```
   期望：3 + 3 个文件都存在。
3. 环境变量固定（每个命令都要带）：
   ```bash
   env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       MPLCONFIGDIR=$PWD/.mplcache HF_HOME=$PWD/.cache/huggingface
   ```
4. 机器空闲：先看 `uptime`，1 分钟负载 < 4 再开跑（浏览器会占几核，但别和别的训练抢）。

---

## 阶段 1：给训练加并行（代码，约 20 min）

### 改哪些文件

**新增 `rl/vec_factory.py`** —— `SubprocVecEnv` 要求 env 构造函数能被 pickle，
写在 `main()` 里的闭包不行，所以必须是模块顶层函数：

```python
"""给 SubprocVecEnv 用的顶层工厂（闭包不能被 pickle）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def make_residual_env(rank: int, **kw):
    """返回一个无参 thunk（在子进程里被调用）。

    rank 用来错开训练种子：episode 是从 episode_pool 里按 env 自己的 RNG 抽的，
    各 rank 的 seed 不同 → 初始条件不同（不是同一个初始状态复制 N 份）。
    """
    kw = dict(kw)
    kw["seed"] = int(kw.get("seed", 0)) * 1000 + rank

    def _init():
        from env_residual import ResidualSO101
        return ResidualSO101(**kw)
    return _init
```

**改 `rl/train.py`**（只加分支，不动 n_envs==1 的路径）：
1. 新增参数 `--n-envs`（`type=int, default=1`）。
2. 把现在的
   ```python
   env = ResidualSO101(...)
   train_env = DummyVecEnv([lambda: env])
   ```
   改成两条路：
   ```python
   env_kw = dict(base_name=a.base, device=a.device, episode_pool=tuple(pool),
                 swap_cams=a.swap_cams, use_sensor=not a.no_sensor,
                 max_seconds=a.max_seconds, seed=seed,
                 residual_deg=(RESIDUAL_DEG if a.residual_deg is None else float(a.residual_deg)),
                 reward_version=a.reward, judge=a.judge)
   if a.n_envs <= 1:
       env = ResidualSO101(**env_kw)                 # ← 现有路径，一字不改
       train_env = DummyVecEnv([lambda: env])
       resid_deg = env.residual_deg
   else:
       from stable_baselines3.common.vec_env import SubprocVecEnv
       from vec_factory import make_residual_env
       train_env = SubprocVecEnv([make_residual_env(r, **env_kw) for r in range(a.n_envs)])
       resid_deg = env_kw["residual_deg"]
   ```
3. 收尾处 `env.close()` → `train_env.close()`（`SubprocVecEnv.close()` 会关所有子进程；
   `DummyVecEnv.close()` 也会关里面那个 env，所以两路都能用）。
4. 打印里的 `env.residual_deg` → `resid_deg`。
5. **不加** `VecMonitor`：env 自己在集末写 `ep_end_return`，`CurveCB` 读的就是它。

### 具体步骤
1. 备份 `/tmp/train.py.bak`（`cp rl/train.py /tmp/train.py.bak`），便于对比。
2. 写 `rl/vec_factory.py`。
3. 按上面 5 点改 `rl/train.py`。
4. **回归测试（关键）**：用 `--n-envs 1` 跑 600 步，确认产物结构与旧版一致。

### 验证命令
```bash
# ① n_envs=1 回归（应与旧行为完全一致）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/train.py \
    --base sim156 --seeds 0 --steps 600 --n-envs 1 --resume \
    --reward v1 --judge strict --residual-deg 1.43 --episodes 0-49 \
    --ckpt-dir .pilot/n1 --curve-dir .pilot/n1 --log-every 200
# 通过：退出码 0；.pilot/n1/s0/model.zip 存在；.pilot/n1/s0.jsonl 至少 1 行
# 失败：抛异常 / model.zip 缺失 → 停，先修再往下

# ② n_envs=4 冒烟（同一 600 步）
... 同上，把 --n-envs 改成 4，目录改成 .pilot/n4
# 通过：退出码 0；n_envs=4 的 wall time < n_envs=1 的 0.6 倍
# 失败：AttributeError: Can't pickle local object → 检查 make_residual_env 是否在模块顶层
# 失败：子进程 EGL 报错 → 在 SubprocVecEnv 里显式传 start_method="fork"（Linux 默认即是）
```

### 失败时怎么办
| 症状 | 处理 |
|---|---|
| `Can't pickle local object` | 工厂函数必须在模块顶层；`kw` 里不能塞 lambda/局部类 |
| 子进程渲染报错/全黑 | 降 `--n-envs`；在脚本里对每个子进程设 `LP_NUM_THREADS=1` |
| 训练不推进（steps 不动） | 检查 `total_timesteps` 语义：SB3 里它是**所有环境的总步数**，不是每个环境的 |
| 内存爆 | 回到阶段 2 的 4/2 环境档 |

---

## 阶段 2：Pilot 量化（约 10 min）

### 具体步骤
对 `n_envs ∈ {1, 4, 8}`，各跑 3000 步，记录 **wall / steps per second / 峰值 RSS**。
峰值 RSS 要算**进程树**（`/usr/bin/time -f %M` 只给主进程，漏掉子进程），
所以脚本里用采样：每 2 秒 `ps -o rss= --ppid <pid> -p <pid>` 求和取最大。

### 验证命令
```bash
bash rl/run_r1big.sh pilot
```
通过判据（写死在脚本里）：
- 8 环境的 **峰值 RSS < 10 GB**（可用内存 12 GB）；
- 8 环境的 **steps/s ≥ 4 × 单环境**；
- 三个档位都**没有**子进程异常退出。

### 预期结果
| n_envs | 预期 steps/s | 预期峰值 RSS |
|---|---|---|
| 1 | ~72（现有实测） | ~2 GB |
| 4 | 200–280 | ~5 GB |
| 8 | 300–500 | ~9 GB（有 OOM 风险） |

### 失败时怎么办
- 8 环境 RSS ≥ 10 GB 或 OOM → **降 4**（训练时间 ×2，仍可接受）；
- 4 环境也过不了 → 降 2，并在报告里写明"算力只提到 ~2× 并行，是内存限制"；
- 若并行几乎不提速（瓶颈在 EGL 软件渲染）→ 记录这个事实，考虑 `--no-sensor`
  **不行**（会改变观测分布，破坏对照）→ 那就接受较低倍数，并在 §21 里如实说明"算力提升倍数受渲染限制"。

---

## 阶段 3：正式训练（30–120 min）

### 具体步骤
3 个种子串行（每种子用 pilot 选出的 `N`），每种子一个外层重试循环（最多 10 次，`--resume` 保进度）。
命令行参数与 §5 训练 9 个模型时**逐项一致**，只有 `--steps` 和 `--n-envs` 变：

```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-rl/bin/python rl/train.py \
    --base sim156 --seeds "$S" --steps 300000 --n-envs "$N" \
    --reward v1 --judge strict --residual-deg 1.43 --episodes 0-49 \
    --ckpt-dir rl/ckpt/R1big --curve-dir rl/curves/R1big --resume
```

### 验证命令
```bash
bash rl/run_r1big.sh train
# 通过：3 个 rl/ckpt/R1big/s*/model.zip 存在
#      每个 rl/curves/R1big/s*.jsonl 最后一行 episodes >= 600（期望 ~1300）
# 失败：episodes < 600 → 并行没生效（还在 72 步/秒）；查 --n-envs 是否被吞
```

### 预期结果
| | 旧（30k 步） | 新（300k 步，8 环境） |
|---|---|---|
| wall / 种子 | 415 s | 600–900 s |
| 见过 episode | 131–143 | 1300–1500 |
| 训练内成功率 | 0.00–0.22，无趋势 | 待测（这是本实验的核心看点之一） |

### 已知取舍（要写进报告）
`buffer_size` 仍是 **200 000**（与旧配置一致）。300k 步 → 前 100k 样本会被挤出回放池。
**保持 200k 是为了"只改算力"的严格对照**；如果结果很接近，可以再做一次 `buffer_size=1M` 的补充（不在本次范围）。

---

## 阶段 4：评测（约 40 min）

### 具体步骤
每臂 90 次 rollout，协议与 §18 相同（`judge=target --jaw-min-deg 16.05`），
带 `--resume` 与外层重试（3 路并发、`OMP_NUM_THREADS=1`，每臂约 13 min）。

```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl .venv-rl/bin/python rl/eval_ab.py \
    --base sim156 --policy rl/ckpt/R1big/s0/model.zip --reward v1 \
    --episodes 100-129 --seeds 3 --judge target --jaw-min-deg 16.05 --resume \
    --out rl/eval/target_R1big_s0_clamp_90.json
```

### 验证命令
```bash
bash rl/run_r1big.sh eval
python3 rl/clamp_analysis.py --pat "target_R1big_s*_clamp_90.json"
# 通过：3 个 json 各 180 行（base 90 + base+R 90）
#      每份的 base 行都是 78/90、clean 50/90（§18.4 的确定性校验）
# 失败：base 行不是 78/90 → 评测环境被改过，先查再说，不要继续下结论
```

---

## 阶段 5：结论与文档（约 30 min）

1. 报告新增 `§21`，一张表：
   `臂 | 训练步数 | 见过 episode | wall | success | clean | 配对 McNemar vs 基座+钳爪(b/c/p) | 判定`
2. 判定只能取 spec §6 的三档之一（翻转 / 不翻转 / 中间态）。
3. 若"不翻转"：在 §18.3 那段加一句限定语的解除——原文"方差比收益大一个量级"是低预算下的观察，
   现在补上"10× 预算下仍然如此"，并说明 §19/§20 的探针结论**不依赖训练预算**（这是它们更硬的原因）。
4. `docs/INTERVIEW_PACK.md` 的数字表加一行"10× 算力对照"。
5. 提交（一个 commit，只含新产物 + 文档）。

---

## 回滚方案

| 层级 | 操作 |
|---|---|
| 只回滚代码 | `git checkout -- rl/train.py && rm rl/vec_factory.py` |
| 清训练产物 | `rm -rf rl/ckpt/R1big rl/curves/R1big .pilot` |
| 清评测产物 | `rm -f rl/eval/target_R1big_s*_clamp_90.json` |
| 全清 | 上面三条一起 + `git checkout -- docs/REPORT_RESIDUAL_RL.md docs/INTERVIEW_PACK.md` |

**永不触碰**：`rl/ckpt/R1/`、`rl/curves/R1/`、`rl/eval/target_R1_s*_90.json`、§17–§20 的既有产物。

## 时间 / 算力预算

| 阶段 | 顺利 | 不顺 |
|---|---|---|
| 1 代码 + 回归 | 20 min | 40 min（pickle/EGL 排查） |
| 2 pilot | 10 min | 20 min（退到 4 环境） |
| 3 训练 | 30–40 min（8 env） | 90–120 min（4 env / SIGFPE 重试） |
| 4 评测 | 40 min | 60 min |
| 5 文档 | 30 min | — |
| **合计** | **~2.2 h** | **~4 h** |
| 磁盘 | ckpt 3 × ~1 MB（SAC 小网络）+ 曲线 ×3 + eval json ×3，**< 50 MB** | — |

## 完成定义（DoD）

- [ ] `rl/train.py --n-envs 1` 与改动前行为一致（600 步回归通过）
- [ ] pilot 表落地：`n_envs ↔ steps/s ↔ 峰值 RSS`
- [ ] 3 个 300k 步 ckpt + 曲线，`episodes ≥ 600`
- [ ] 3 份 90-rollout 评测，基座行复现 78/90
- [ ] 报告 §21 表格 + 按预注册规则给出的判定
- [ ] 面试包数字表加一行
- [ ] 一个 commit；工作区干净

## 不在本计划内（避免范围蔓延）

- chunk 级残差 / 任务空间残差（§20.4 的另一条线）
- `buffer_size` 调整、奖励函数修改、幅度修改
- VLA：等本计划出结论后**另起 spec**（开放问题见 spec §10，含"数据集里 instruction 字段是否多任务"）