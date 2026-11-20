# 真机 A/B 评测流程 · 每臂 12 次抓取

> 目的：在同一套摆放、同一套相机、同一套参数下比较 `armA` 与 `armB`。
> 这是 pilot，不是统计定论；12 次只能看方向，不能证明显著。

---

## 0. 前置检查

- follower 已连接：`/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00`
- env 相机：`/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0`
- wrist 相机：`/dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0`
- 两路相机都是 640x480 @ 30fps，MJPG
- `torch.cuda.is_available() == True`
- 桌面、灯光、药盒、木盒位置固定
- A/B 两组的 `duration`、`temporal_ensemble_coeff`、`n_action_steps` 完全一致

---

## 1. 摆放 12 个起点

在拾取垫上标 12 个位置（例如 3×4 网格），每个位置用同一朝向。

要求：

- A/B 两组使用**同一组 12 个位置**
- 每个位置至少跑 1 次 A、1 次 B
- 不要中途改变灯光、相机、木盒位置

推荐顺序（减少时间漂移）：

```text
A: pos1~pos6
B: pos1~pos6
A: pos7~pos12
B: pos7~pos12
```

---

## 2. 每次 trial 的固定步骤

1. 把药盒放到当前 `position_id`，朝向固定。
2. 机械臂回到同一个 start pose。
3. 确认 env/wrist 画面正常。
4. 开始录像（手机或采集卡均可）。
5. 启动对应策略 rollout。
6. 手放在急停附近，但**非危险不干预**。
7. 等 rollout 结束或超时。
8. 判定并记录。
9. 复位药盒和机械臂，进入下一条。

---

## 3. 成功判据

一次 trial 记为成功，必须同时满足：

- 药盒被夹起，离开桌面/垫面 ≥ 50 mm
- 被移动到木盒上方并放入
- 松爪后药盒留在木盒内
- 全程没有人工干预
- 没有撞飞、掉落、卡死

否则记失败，并写失败模式。

---

## 4. 失败模式

```text
no_contact          没碰到盒
grasp_slip          夹住但滑落
collision           撞盒/撞桌/撞木盒
wrong_place         放偏/没放进木盒
timeout             超时未完成
human_intervention  人工干预
camera_error        相机掉线/画面异常
robot_error         串口/机械臂异常
```

---

## 5. 记录表

使用：

```text
logs/real_ab_trials.csv
```

字段：

```text
trial_id, arm, position_id, start_time, success,
failure_mode, time_to_grasp_s, human_intervention,
video_file, notes
```

---

## 6. 跑完后看什么

每臂：

```text
success / 12
失败模式分布
平均完成时间（只统计成功 trial）
是否出现明显抖动/抽搐
夹爪是否提前闭合或闭合不足
```

A/B 对比：

```text
A 成功数 vs B 成功数
同一 position_id 上 A/B 是否一个成一个败
B 是否只在某些位置更好
```

12 次样本很小，报告时写清楚：

```text
这是方向性 pilot，不是显著性结论。
```

---

## 7. 策略命令

### A

```bash
cd /home/yyl/lerobot

lerobot-rollout \
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
      width: 640, height: 480, fps: 30, fourcc: "MJPG"
    }
  }' \
  --policy.path=/home/yyl/lerobot/outputs/train/act_armA/checkpoints/017500/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 \
  --policy.n_action_steps=1 \
  --task="Grab the black cube" \
  --duration=180 \
  --display_data=true
```

### B

只把 `--policy.path` 换成：

```text
/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model
```
