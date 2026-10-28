# 明天（2026-10-29）执行计划

> 目标：消掉 A/B 对比里唯一的混淆项（训练步数不同），把"仿真在 sim2real 上有帮助"从
> 「方向性观察」变成「同步数单变量对照」。**不需要训练。**

---

## 1. 为什么做这件事

现在的对比有一个被人一问就塌的漏洞：

| | A | B |
|---|---|---|
| 数据 | 真机 139 集 | 真机 139 集 + 仿真 39 集 |
| 步数 | **17,500** | **22,500** |

抓取率 4/12 → 9/12 的差异，无法完全排除"多训 5000 步"的贡献。

**解法不需要重训**：`act_armB/checkpoints/017500` 已经存在（131 MB，与 A 的 017500 同步数）。
用它跑同样的 12 次，就得到「同数据不同训练量」和「同步数不同数据」两组对照。

```
A@17500  真机数据        17500 步
B@17500  真机+仿真数据    17500 步   ← 明天要跑的
B@22500  真机+仿真数据    22500 步   ← 已跑过（9/12 抓取）
```

判读关键：**B@17500 vs A@17500** 是干净的单变量（只差数据）。

---

## 2. 准备工作（约 10 分钟）

1. 确认 GPU 无关（本次不训练），但确认真机链路：
   ```bash
   ls -l /dev/serial/by-id/
   ls -l /dev/v4l/by-id/
   ```
   串口应看到 `usb-1a86_USB_Single_Serial_5B91029226-if00`；相机应看到
   `usb-Generic_Web_Camera...`（env）和 `usb-XHH-260202-H_Integrated_Camera...`（wrist）。

2. 摆九宫格圆垫，药盒放到起始位，木盒放到位（与上次 12 次完全一致的位置）。

3. 确认要跑的权重存在：
   ```bash
   ls -l /home/yyl/lerobot/outputs/train/act_armB/checkpoints/017500/pretrained_model/model.safetensors
   ```

---

## 3. 执行：12 次（9 竖 + 3 横，与上次同协议）

**权重换成 B@017500**，其余参数一律不动（保持与上次可比）：

```bash
cd /home/yyl/lerobot

env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
  -m lerobot.scripts.lerobot_rollout \
  --strategy.type=base \
  --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00 \
  --robot.id=yyl_follower_arm \
  --robot.cameras='{env: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}, wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0, width: 640, height: 480, fps: 60, fourcc: MJPG}}' \
  --policy.path=/home/yyl/lerobot/outputs/train/act_armB/checkpoints/017500/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  --task="Pick up the medicine box and place it into the wooden box." \
  --duration=60 \
  --fps=30 \
  --display_data=false
```

> 注意：`--policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1` 与上次 12 次
> 保持一致，**不要改成 None/100**，否则又多一个变量。

---

## 4. 记录表（照这个填，跑完直接贴给我）

每格填：`成功` / `半` / `够不到` / `夹空`

| 位置 | 竖抓 | 横抓 |
|---|---|---|
| 左上 | | |
| 中上 | | |
| 右上 | | |
| 左中 | |  |
| 中中 | | |
| 右中 | |  |
| 左下 | | — |
| 中下 | | — |
| 右下 | | — |

汇总三行：

```text
成功：__ / 12
半成功：__ / 12
失败：__ / 12（够不到 __ + 夹空 __）
```

---

## 5. 判读规则（跑完对照这张表）

| B@17500 的结果 | 说明什么 |
|---|---|
| 抓取率仍明显高于 A@17500（比如 ≥8/12） | **仿真数据的增益成立**，与训练步数无关 → 结论可以定稿 |
| 抓取率掉到 A 水平（≈4/12） | 增益主要来自多训的 5000 步，不是仿真 → 需要修正结论 |
| 释放仍然大面积失败（半成功多） | 释放问题与训练步数无关，是数据/指垫问题 → 与今天的判断一致 |

---

## 6. 明天不要做的事

- ❌ 不要重训（时间不够，且本次不需要）
- ❌ 不要改 `temporal_ensemble_coeff` / `n_action_steps`（会引入新变量）
- ❌ 不要跑 22500 的 B（已有数据，重复无意义）
- ❌ 不要动 C/D/E

---

## 7. 时间预算

| 步骤 | 预计 |
|---|---|
| 设备检查 + 摆位 | 10 分钟 |
| 12 次 rollout | 40~60 分钟 |
| 填表 + 汇总 | 10 分钟 |
| **合计** | **约 1~1.5 小时** |