# 上下文压缩 · 2026-10-28

## 1. 项目目标

SO-ARM101 纯物理 MuJoCo 数据合成 + LeRobot ACT Sim2Real。
当前阶段：A/B 真机 pilot 已完成 12 次/臂，但 B 出现系统性释放失败，正在诊断夹爪。

---

## 2. 数据

| 数据集 | episodes | frames | 用途 |
|---|---:|---:|---|
| `real_train_fixed` | 139 | 62,273 | P1 真机训练池 |
| `holdout_fixed` | 12 | 5,376 | 测试，禁止训练 |
| `sim156` | 156 | 70,824 | 仿真抽样 |
| `sim39` | 39 | 17,753 | B 的仿真子集 |
| `armA` | 139 | 62,273 | 纯真机 |
| `armB` | 178 | 80,026 | 139 真机 + 39 sim |
| `armC` | 295 | 133,097 | 139 真机 + 156 sim，未训练 |
| `armD` | 156 | 70,824 | 纯仿真，未训练 |

路径：`/home/yyl/lerobot_datasets/`

---

## 3. 模型

| 臂 | checkpoint | 状态 |
|---|---|---|
| armA | `/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500` | ✅ 完成 |
| armB | `/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500` | ✅ 完成（中途崩过，已 resume） |
| armC/D | — | 未训练 |

旧基线：`/home/yyl/lerobot/outputs/train/act_yyl151_lance/checkpoints/020000`

---

## 4. 离线 A/B 结果

`holdout_fixed`，每 10 帧取点，538 点：

| 指标 | A | B |
|---|---:|---:|
| L1@1 | 2.490 | 2.408 |
| L1@10 | 2.618 | 2.627 |
| L1@50 | 3.524 | 3.679 |
| L1@100 | 4.227 | 4.688 |
| RMSE | 8.302 | 9.198 |
| 夹爪 L1@1 | 2.133 | 1.435 |

结论：B 短时和夹爪略好；长时明显更差；12 条测试集上不显著（B 更好 7/12，bootstrap CI 跨 0，p=0.387）。

---

## 5. 真机 A/B 结果（12 次/臂）

九宫格：9 次竖抓 + 3 次横抓。

### A

```text
成功 4，半成功 0，失败 8
2 次够不到 + 6 次夹空
任务成功 4/12 = 33.3%
抓取成功 4/12 = 33.3%
抓取后释放成功 4/4 = 100%
```

成功位置：左下竖、中下竖、中中竖、左中横。

### B

```text
成功 2，半成功 7，失败 3
2 次够不到 + 1 次夹空
任务成功 2/12 = 16.7%
抓取成功（成功+半成功）9/12 = 75.0%
抓取后释放成功 2/9 = 22.2%
```

成功位置：中下竖、右中横。  
半成功：左上、中上、右上、左中、中中、左下（竖）+ 中中横。  
半成功定义：抓到了、搬到了，但释放时夹爪没有正常张开。

统计（探索性）：

```text
任务成功 A 4/12 vs B 2/12：p = 0.64
抓取成功 A 4/12 vs B 9/12：p = 0.0995
抓取后释放 A 4/4 vs B 2/9：p = 0.021
```

结论：B 抓取更强，但释放系统性失败；不能宣布 B 更好。

---

## 6. 夹爪诊断现状

手动测试：

```text
cmd=0    present=0.756    → 闭合
cmd=20   present=19.38
cmd=40   present=39.45
cmd=60   present=59.52
cmd=80   present=79.73
cmd=100  present=99.38    → 张开
```

结论：

```text
0 = 闭合，100 = 张开：方向正常
空载命令→实际：正常
```

B 策略日志：

```text
gripper_req 从 ~38 升到 99.38
gripper_sent 与 req 一致，也到 99.38
```

结论：

```text
B 确实发了开爪命令
动作没有被裁剪
不是“策略没发开爪”
```

待确认：

```text
夹着盒子时，夹爪能不能张到 80 以上？
```

---

## 7. 下一步（当前最优先）

### 7.1 带载夹爪测试

```bash
cd /home/yyl/deepseek-harness

env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
  tools/test_gripper_loaded.py
```

它会张开 → 给 10 秒放盒子 → 闭合 → 再张开，打印 `loaded open`。

判断：

```text
>=80   带载能开，问题在释放时机/位置
50~80  部分开，看盒子是否掉落
<50    带载卡住，查机械/扭矩
```

### 7.2 如果带载能开，记录 B 运行时的 present

使用 `docs/GRIPPER_RELEASE_DIAGNOSTIC.md` 里的 `log_gripper_rollout2.py`，在 B 的一个半成功位置跑 60 秒，查看：

```text
gripper_req
gripper_sent
present_before
present_after
```

判断：

```text
req低 sent低 present低    策略输出问题
req高 sent低 present低    动作映射/裁剪问题
req高 sent高 present低    带载机械问题
req高 sent高 present高    释放时机/位置问题
```

### 7.3 单变量测试

如果策略输出正常但释放仍失败：

```bash
--policy.temporal_ensemble_coeff=0.0
# 或
--policy.n_action_steps=100
```

一次只改一个，同位置、同时长。

---

## 8. 环境与脚本

### Python 环境

```bash
env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python ...
```

`~/lerobot` 是源码（editable install），`lerobot312` 是环境；不要删 `~/lerobot`。

### 常用文件

```text
docs/HANDOFF_SIM2REAL_V8.md               # 总交接
docs/REAL_ROBOT_AB_PROTOCOL.md            # 12 次流程
docs/REAL_AB_RESULTS_2026-10-28.md        # A/B 真机结果
docs/GRIPPER_RELEASE_DIAGNOSTIC.md        # 夹爪诊断
tools/test_gripper_loaded.py              # 带载测试
tools/eval_act_holdout.py                 # 离线评测
tools/run_eval_ab_cpu.sh                  # A/B 离线评测
tools/compare_sim_real.py                 # sim/real 分布 W1
```

### Git

```text
本地 main 已 commit: ac201bc
领先 origin/main 5 个 commit
未 push：ghproxy 无凭据，直连 GitHub TLS 被阻断
```

个人文档 `docs/如何学.md`、`docs/plans/`、`docs/specs/` 未提交。

### 机器稳定性

出现过：

```text
kernel BUG at fs/inode.c:649
clear_inode -> shmem_evict_inode
Comm: deepseek-harnes
```

长任务用 `systemd-run --user` 或 `nohup`；Python 已设：

```bash
PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX=...
```

GPU 掉驱动时先重启；`torch.cuda.is_available()` 必须为 True。

---

## 9. 现在不要做

- 不要跑 C/D/E
- 不要因为 B 抓取 9/12 就宣布 B 更好
- 不要把“半成功”算成功
- 不要同时改多个释放相关参数
