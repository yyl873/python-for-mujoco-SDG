# 新窗口交接 · 2026-10-28

> 项目：SO-ARM101 纯物理 MuJoCo SDG + LeRobot ACT Sim2Real  
> 当前卡点：B 组真机“抓取成功但释放失败”，正在定位是夹爪幅度、释放时机还是位置问题。

---

## 1. 当前结论（先看这里）

```text
A 组：抓得少，但抓到后释放可靠。
B 组：抓得多，但释放系统性失败，任务成功率反而低。
```

真机 12 次/臂：

| 指标 | A | B |
|---|---:|---:|
| 任务成功 | 4/12 = 33.3% | 2/12 = 16.7% |
| 半成功 | 0 | 7/12 = 58.3% |
| 失败 | 8/12 = 66.7% | 3/12 = 25.0% |
| 抓取成功（成功+半成功） | 4/12 = 33.3% | 9/12 = 75.0% |
| 抓取后释放成功 | 4/4 = 100% | 2/9 = 22.2% |

A 失败：2 次够不到 + 6 次夹空。  
B 失败：2 次够不到 + 1 次夹空。  
B 半成功：抓到了、搬到了，但夹爪没有正常张开释放。

---

## 2. 夹爪诊断：已经排除了什么

### 2.1 方向正常

```text
cmd=0    → 闭合
cmd=100  → 张开
空载命令/实际跟随正常
```

### 2.2 B 不是“没发开爪命令”

B 真机 rollout 日志：

```text
release 尝试期间 gripper_req ≈ 25~30
gripper_sent 与 req 基本一致
present 也基本跟随（24~30）
```

之前看到的 `99.38` 是 **return-to-initial-position**，不是释放命令。

### 2.3 “B 被真机刻度拉低”不成立

离线探针：

```text
仿真观测：
  B release_mid  pred 30.78 / gt 25.01
  B drop_start   pred 42.35 / gt 39.26

真机 holdout：
  A release_mid  pred 21.80 / gt 21.78
  B release_mid  pred 21.24 / gt 21.78
```

结论：

```text
B 在仿真观测上能输出 42.35 的大张开量。
在真机观测上，A/B 的释放输出几乎一样（~21）。
A 用 ~21 能释放 4/4，所以“B 幅度被拉低”不能解释 B 的失败。
```

---

## 3. 现在最可能的原因

按优先级：

1. **释放位置/时机不对**  
   B 在错误的位置或错误的时间打开夹爪，盒子没进木盒。
2. **盒子在指间卡住**  
   虽然 present 到了 25~30，但当前抓取姿态下不足以让盒子脱出。
3. **B 的 closed-loop 轨迹与 A 不同**  
   B 抓得更深/更偏，导致释放时卡住。

**不是**：夹爪方向反了、策略不发开爪命令、动作被裁剪。

---

## 4. 下一步只做两个最小实验

### 实验 1：带载夹爪测试

```bash
cd /home/yyl/deepseek-harness

env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
  tools/test_gripper_loaded.py
```

看 `loaded open`：

```text
>=80   带载能开 → 不是机械问题
50~80  部分开 → 看盒子是否掉落
<50    带载卡住 → 机械/扭矩问题
```

### 实验 2：同位置 A/B 各跑 60 秒，记录 release 窗口

用已固化脚本：

```text
tools/log_gripper_rollout.py
```

跑 B：

```bash
cd /home/yyl/lerobot

GRIPPER_LOG=/home/yyl/deepseek-harness/logs/gripper_B_samepos.csv \
env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
/home/yyl/deepseek-harness/tools/log_gripper_rollout.py \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00 \
  --robot.id=yyl_follower_arm \
  --robot.cameras='{
    env: {
      type: opencv,
      index_or_path: /dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0,
      width: 640, height: 480, fps: 30, fourcc: "MJPG"
    },
    wrist: {
      type: opencv,
      index_or_path: /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0,
      width: 640, height: 480, fps: 60, fourcc: "MJPG"
    }
  }' \
  --policy.path=/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  --task="Grab the black cube" \
  --duration=60 \
  --fps=30 \
  --display_data=false
```

A 用同一个位置、同一条命令，只改：

```text
--policy.path=/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500/pretrained_model
```

日志写到 `logs/gripper_A_samepos.csv`。

对比：

```text
A/B 的 gripper_req 峰值
A/B 开始张开的时间点
A/B 张开时的末端位置
```

判断：

```text
A 明显更高       → B 幅度不足
A/B 差不多       → 查释放位置/时机/盒子卡住
```

---

## 5. 暂时不要做

- 不要重跑 12 次 A/B
- 不要跑 C/D/E
- 不要同时改 temporal ensemble 和 n_action_steps
- 不要把“半成功”算作成功

---

## 6. 数据和模型

| 数据集 | episodes | frames |
|---|---:|---:|
| `real_train_fixed` | 139 | 62,273 |
| `holdout_fixed` | 12 | 5,376 |
| `sim156` | 156 | 70,824 |
| `sim39` | 39 | 17,753 |
| `armA` | 139 | 62,273 |
| `armB` | 178 | 80,026 |
| `armC` | 295 | 133,097（未训练） |
| `armD` | 156 | 70,824（未训练） |

模型：

```text
A: /home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500/pretrained_model
B: /home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model
旧基线: act_yyl151_lance/checkpoints/020000
```

---

## 7. 环境和稳定性

Python：

```bash
env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python ...
```

`~/lerobot` 是源码（editable install），`lerobot312` 是环境；不要删 `~/lerobot`。

机器出现过：

```text
kernel BUG at fs/inode.c:649
clear_inode -> shmem_evict_inode
Comm: deepseek-harnes
```

长任务用 `systemd-run --user` 或 `nohup`。  
GPU 掉驱动时先重启，`torch.cuda.is_available()` 必须为 True。

---

## 8. 文件索引

```text
docs/HANDOFF_NEXT_WINDOW_2026-10-28.md      # 本文件
docs/GRIPPER_RELEASE_DIAGNOSTIC.md          # 夹爪诊断全文
docs/REAL_AB_RESULTS_2026-10-28.md          # 真机 A/B 结果
docs/REAL_ROBOT_AB_PROTOCOL.md              # 12 次流程
docs/HANDOFF_SIM2REAL_V8.md                 # 总交接

tools/test_gripper_loaded.py                # 带载夹爪测试
tools/log_gripper_rollout.py                # rollout 夹爪日志
tools/probe_release_gripper.py              # 仿真释放探针
tools/probe_release_gripper_real.py         # 真机 holdout 释放探针
tools/eval_act_holdout.py                   # 离线 A/B 评测
tools/compare_sim_real.py                   # sim/real 分布 W1
```

Git：

```text
本地 main 已 commit: ac201bc
领先 origin/main 5 个 commit
未 push：ghproxy 无凭据，直连 GitHub TLS 被阻断
```
