# 数据体检报告 · 181 条真机 + 64 条仿真

> 生成时间 2026-10-25。全部数字为**实测**，命令可复跑。
> 体检工具：`tools/check_datasets_health.py`（本轮新增，只读，不写数据集）。

## 0. 一句话结论

**数据本身是干净的（无 NaN、无坏帧、视频帧数与 meta 全部对齐、时间戳精确 1/30），
可以直接进训练。** 但有 **1 个必须先修的量纲问题（夹爪）** 和 **2 个不阻塞的 warn**；
另外"仿真能不能提升真机"这件事，从数据多样性看**期望值不高**（见 §4）。

## 1. 复跑命令

```bash
cd /home/yyl/deepseek-harness
PY=/home/yyl/miniforge3/envs/lerobot312/bin/python

# 三个集子一起查（约 2 min，含全量视频解码）
env -u PYTHONHOME -u PYTHONPATH $PY tools/check_datasets_health.py \
  --root /home/yyl/lerobot_datasets/yyl100_dataset_fixed \
  --root /home/yyl/lerobot_datasets/yyl100_corners \
  --root /home/yyl/deepseek-harness/dataset/soarm101_sim_v1 \
  --out logs/dataset_health.json

# 只看数据不看视频（10 s）
... --no-video

# 退出码：0=ok / 1=warn / 2=fail
```

仿真母本侧另有仓库自带体检（1000 条 npz+mp4 与 index.json 对账）：

```bash
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 collect/check_dataset.py --quiet
```

## 2. 体检结果总表

| 项 | 真机 `yyl100_dataset_fixed` | 真机 `yyl100_corners` | 仿真 `soarm101_sim_v1` |
|---|---|---|---|
| 状态 | **warn** | **ok** | **ok** |
| episodes / frames | 151 / 67,649 | 30 / 11,870 | 64 / 29,176（母本可用 **1000**） |
| fps / dt | 30 / 精确 0.033333 | 30 / 精确 0.033333 | 30 / 精确 0.033333 |
| episode 长度 | 448 或 449（**只有 2 种**） | 308~578（28 种） | 429~481（39 种） |
| action/state | 6 维 float32，关节名一致 | 同 | 同 |
| robot_type | so_follower | so_follower | so_follower |
| task 文本 | `Grab the black cube` | `Grab the box` | `Pick up the medicine box and place it into the wooden box.` |
| NaN / Inf | 0 / 0 | 0 / 0 | 0 / 0 |
| 全局 index 连续 | **✗ 缺 448 个（max 68096, n 67649）** | ✓ | ✓ |
| 视频 | env+wrist 640×480 **av1**，16 文件/相机 | 同，1 文件 | 同，env 2 文件 |
| 视频解码错 / 帧数不符 | 0 / 0（151 条全对） | 0 / 0（30 条全对） | 0 / 0（64 条全对） |
| 重复 episode（动作指纹） | 0 组 | 0 组 | 0 组 |
| 夹爪 action 区间 | 0.0 ~ 49.68 | 0.46 ~ 42.58 | 0.0 ~ 85.94 |
| 夹爪 起始/结束均值 | 0.43 / 0.37 | 3.40 / 3.18 | 0.00 / 0.00 |
| `wrist_roll` 跨度 | 127.4°（-112.3~+15.1） | 123.3°（-128.7~-5.3） | **13.7°**（-103.6~-89.9） |
| 图像亮度 env/wrist | 140.5 / 133.4 | 145.0 / 143.4 | 137.3 / 140.1 |
| 图像锐度 env/wrist（Laplacian var） | **421 / 1381** | 991 / 460 | **57.5 / 35.8** |
| `\|action-state\|` 均值 (deg) | 0.94/2.62/2.43/0.84/1.24/1.65 | 1.12/3.05/3.16/0.83/1.54/1.31 | 0.48/1.10/0.65/0.76/0.14/1.30 |
| 单步跳变 >20° | 1 处 | 0 | 0 |

关节行程（max−min，度）：

| joint | fixed | corners | sim |
|---|---|---|---|
| shoulder_pan | 76.0 | 72.8 | 60.0 |
| shoulder_lift | 158.2 | 161.2 | 110.4 |
| elbow_flex | 148.4 | 150.3 | **53.9** |
| wrist_flex | 86.9 | 63.8 | 77.0 |
| wrist_roll | 127.4 | 123.3 | **13.7** |
| gripper | 49.7 | 42.1 | 85.9 |

## 3. 发现（按严重程度）

### B1（阻塞）夹爪量纲不一致：仿真给的是**度**，真机给的是**0~100 归一化**

- 真机侧：`so_follower.py` 里 5 个臂关节用 `DEGREES`，**夹爪硬编码 `RANGE_0_100`**
  → `gripper.pos` 是标定行程的百分比，不是度。
- 真机标定跨度：raw 1580~3035 = **1455 counts × 0.0879°/count = 127.9°**，
  正好等于 MJCF 里 `gripper_joint` 的行程 `range="0.0 2.232"` rad = 127.9°。
- 仿真侧：`gripper.pos` 就是 `gripper_joint` 的角度（0~127.9°，本批用到 0~85.94°）。
- **两者 0 点含义一致（0 = 闭合）**，差的只是比例：`真机值 = 度 × 100/127.9 = 度 × 0.782`。
- 不修直接混：模型看到两套夹爪语义（仿真的"张开 86" vs 真机的"张开 67"），
  归一化统计被拉偏，夹爪通道学成两套目标 → 真机夹取大概率退化。
- **20 秒上机确认**（只需确认 0/100 端点，方向已由数据判定为 0=闭合）：

```bash
cd /home/yyl/lerobot
env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python - <<'EOF'
from lerobot.robots.so_follower import SOFollower, SOFollowerConfig
r = SOFollower(SOFollowerConfig(port="/dev/ttyACM0"))
r.connect()
with r.bus.torque_disabled():
    input("把夹爪【完全闭合】后回车…"); print("闭合 gripper.pos =", r.get_observation()["gripper.pos"])
    input("把夹爪【完全张开】后回车…"); print("张开 gripper.pos =", r.get_observation()["gripper.pos"])
r.disconnect()
EOF
```
期望：闭合 ≈ 0，张开 ≈ 100。若确实如此，仿真夹爪按 `×0.782` 换算后再导出即对齐
（不要用 50 —— 真机数据里最大只到 49.68 是"没开到最大"，不是"开到头"）。

#### ✅ 已定案（2026-10-25，三条独立证据一致）：`--gripper-space real-closed`，0 = 闭合

| 证据 | 内容 | 结论 |
|---|---|---|
| ① 标定文件 | `~/.cache/huggingface/lerobot/calibration/robots/so_follower/yyl_follower_arm.json`：gripper `range_min=1580 / range_max=3035` = 1455 counts × 0.0879°/count = **127.9°**，与 MJCF `gripper_joint` 行程 `0~2.232 rad` 完全一致 | 尺子确认：`真机值 = 度 × 0.782` |
| ② 正运动学 + 数据 | ep0 抓取途中停住值 25.2 → 关节 32.2° → 指垫间距 **42 mm**（张开接近，合理）；若 0=张开则为 **95 mm**（全程几乎不闭合，抓不住 25~35 mm 的物体） | 方向：**0 = 闭合** |
| ③ 人手目视 | 用户确认 `inspect/frames/real_ep0_env/file-000_t0.000s.png` 起始姿态夹爪**是闭合的**，对应首帧 `gripper ≈ 0.43` | 方向二次确认 |

另：首帧 home 姿态批次一致性检查（151 fixed vs 30 corners）—— 5 个关节差 ≤3.5°，
仅 `wrist_roll` 差 8.5°（两批 std 6.3/7.1，≈1.3σ，属人摆位波动），**不像换过标定**，
两批可直接混训。（本机确有 3 份 follower 标定：`my_awesome` 1517~2990 / `my_awesome` 1458~2919 /
`yyl` 1580~3035，当前在用最后一份。）

### B2（阻塞/设计前提）仿真与真机**不是同一个任务**

- 真机：`Grab the black cube`（151 条）+ `Grab the box`（30 条），无语言输入，ACT 单策略。
- 仿真：药盒（15 mm 薄盒）→ 放进木箱，15 段 plan（含 transport/place/release）。
- 因此仿真只能提供"视觉/动力学正则"，**不能替代真机监督**。混合比例要按这个前提定（见计划 §2）。

### B3（阻塞/影响比例）仿真视觉域差很大

- 锐度（Laplacian 方差）：真机 env 421 / wrist 1381，仿真 env 57.5 / wrist 35.8 → **差 7~24 倍**。
- 说明仿真出图明显更"糊/更平"（AA + 运动虚影 + 传感器模糊 aug + 低纹理）。
- 同一套 resnet18 主干要同时吃两种质感，混合比例越高，真机那侧的视觉特征越容易被稀释。
- 若要提高仿真的可用率，优先降 blur（`--render-only` 无法改 aug，需改 collect 的 sensor 随机化）。

### W1（不阻塞）`yyl100_dataset_fixed` 的全局 index 有 448 个空洞

- `n=67649` 但 `max index=68096` → 恰好少一个 448 帧的 episode，说明这个集**被删过 1 条**。
- 读取不受影响（行数、episode 边界、`dataset_from/to_index` 全部自洽，回读正常），
  但 `lerobot_edit_dataset` 某些操作依赖 index 连续性，建议先 `recompute_stats` 或按原样用、不要原地编辑。

### W2（不阻塞但影响泛化）fixed 集 151 条全是 448/449 帧

- 只有 2 种长度 → 录制是按固定时长截的，**节奏多样性≈0**；
  corners 集（308~578，28 种）才是正常人工演示分布。
- 混合时建议把 corners 视为"节奏多样性来源"，不要按帧数比例边缘化它（见计划 §2）。

## 4. 关键量化：仿真的"有效多样性"只有真机的 1/10

按"重采样到 100 点后两两动作轨迹平均 |Δ|（度）"度量：

| 集合 | N | 批内两两 |Δ| 均值 | p10 | p50 | p90 |
|---|---|---|---|---|
| 仿真 64 条 | 64 | **1.1°** | 0.4 | 1.1 | 1.8 |
| 真机 fixed | 151 | 10.6° | 6.1 | 10.1 | 15.3 |
| 真机 corners | 30 | 16.8° | 9.4 | 16.7 | 24.3 |
| 跨域（仿真 vs 真机） | 10×10 | **25.8°** | — | — | — |

读法：
1. 仿真 64 条互相只差 1.1° → 本质是**同一条轨迹 + 随机外观**（θ±10°、光照、相机抖动、模糊）。
   加 1000 条仿真 ≈ 加 1 条轨迹的 1000 张外观图，**动作信息量不随之增长**。
2. 仿真与真机的轨迹差 25.8°（lift/elbow/flex 各差 30~44°）→ 强行高比例混合是
   "用一条不属于本任务的轨迹去稀释 10 倍多样性的真机数据"，冲突风险高、收益存疑。
3. 因此：**仿真占比必须低（按样本 ≤20~25%）**，或用"预训练→真机微调"两阶段把冲突关在门外。

## 5. 与已有 ACT 的关系

- 已训练模型 `outputs/train/act_yyl151_lance`：20000 步 / batch 8 / act(resnet18, chunk 100,
  use_vae=false, lr 1e-5)，数据 = `yyl100_lance_fixed`（**只有 151 条，没吃 corners 的 30 条**）。
- 也就是说：**光把 corners 30 条并进来，就相当于白捡 17% 的新数据** —— 这比"加仿真"更可能提升真机。
- 训练速度实测：5000 步 ≈ 27 min（batch 8, num_workers=2）→ **0.32 s/步**，单臂 20000 步 ≈ 1.8 h。