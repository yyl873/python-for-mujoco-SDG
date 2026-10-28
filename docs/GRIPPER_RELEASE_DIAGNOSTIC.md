# B 组释放失败诊断 · 夹爪开合

> 时间：2026-10-28  
> 现象：B 组 12 次中 7 次“抓到了、搬到了，但释放时夹爪没有正常张开”，任务成功率只有 2/12。  
> 目标：确认是 **策略没发开爪命令**、**动作映射/裁剪**、还是 **带载机械卡住**。

---

## 0. 已知事实

### A/B 真机结果

| 指标 | A | B |
|---|---:|---:|
| 任务成功 | 4/12 = 33.3% | 2/12 = 16.7% |
| 半成功 | 0 | 7/12 = 58.3% |
| 失败 | 8/12 = 66.7% | 3/12 = 25.0% |
| 抓取成功（成功+半成功） | 4/12 = 33.3% | 9/12 = 75.0% |
| 抓取后释放成功 | 4/4 = 100% | 2/9 = 22.2% |

- A 失败：2 次够不到 + 6 次夹空
- B 失败：2 次够不到 + 1 次夹空
- B 的 7 次半成功分布在 7 个不同位置，说明不是单点问题。

### 夹爪手动测试（空载）

```text
initial: 0.756
cmd=0    present=0.756
cmd=20   present=19.38
cmd=40   present=39.45
cmd=60   present=59.52
cmd=80   present=79.73
cmd=100  present=99.38
```

结论：

- `0 = 闭合`，`100 = 张开`
- 空载时命令和实际位置跟随正常
- 不是校准方向反了

### B 策略日志（⚠️ 已作废，见下方更正）

> **2026-10-28 更正：这一节不能作为证据。**
>
> 原记录：「`gripper_req` 从 ~38 升到 99.38，`gripper_sent` 与 req 一致」。
> 经确认，那个 **99.38 来自夹爪 0→100 的手动开合测试**——当时先把夹爪置到 100
> 的全开姿态，再从该初始姿态跑了一次策略。**它不是策略在正常回合里发出的释放指令。**
>
> 归一化统计也直接否掉了这个数值：checkpoint 的 `action.gripper` 归一化器为
> `min=0 / max=49.682 / mean=7.68 / std=12.82`，`99.38` 相当于 **7.15σ**，
> ACT 不可能输出。策略的夹爪输出只可能在 0~49.7 区间内活动（±2σ 约 [-18, 33]）。

因此以下三条原结论**全部撤回**：

- ~~B 的策略确实发出了开爪命令~~ → **未验证**
- ~~动作没有被 `max_relative_target` 裁剪~~ → 未验证
- ~~不是「策略没发开爪」~~ → **重新回到嫌疑列表**

---

## 1. 为什么之前会 EOF

之前用了：

```python
input("把药盒放进夹爪，按回车...")
```

但 Python 是从 heredoc 读代码，`input()` 也去读同一个 stdin，读到 EOF 就报：

```text
EOFError: EOF when reading a line
```

解决：不用 `input()`，改成定时窗口。  
已提供脚本：

```text
tools/test_gripper_loaded.py
```

---

## 2. 带载测试

把机械臂放在桌面上方，运行：

```bash
cd /home/yyl/deepseek-harness

env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
  tools/test_gripper_loaded.py
```

流程：

```text
1. 张开到 100
2. 给你 10 秒把药盒放进去
3. 闭合到 0
4. 再张开到 100
5. 打印 loaded open
```

判断：

| loaded open | 结论 |
|---|---|
| ≥ 80 | 带载能开，问题在释放时机/位置 |
| 50~80 | 带载部分开，观察盒子是否掉落 |
| < 50 | 带载卡住，优先查机械/扭矩/装配 |

### 2.1 实测记录（2026-10-28，已跑）

```text
initial:      9.003
unloaded open: 98.351
closed on box: 4.467        # 盒子夹在两指之间，到不了 0，属正常接触
loaded open:  99.381
RESULT: 带载可张开（>=80）
```

现场判断：**正常抓取和释放**。

结论（三处硬指标全部通过）：

- 带载 present 99.381 > 空载 98.351，**没有任何机械卡涩**，带载张开甚至略高于空载
- `closed on box = 4.467` 说明指间被盒子占住，夹持力真实存在，不是夹空
- 至此矩阵前三行（策略没发/被裁剪/机械卡住）**全部排除**

→ 落到判断矩阵第 4 行：`req 高 / sent 高 / present 高` ⇒ **释放时机/位置问题**，见第 3、4 节。

⚠️ 关于物体：B 的策略是 **ACT，非语言条件**，`--task` 串不参与策略计算
（`Grab the black cube` 只是 rollout 的装饰参数，**不构成矛盾**）。
唯一需要保证的是：测试用的药盒就是评测里那个药盒。

---

## 3. 如果带载能开，继续记录策略运行时 present

如果 `loaded open ≥ 80`，说明夹爪硬件没问题，接下来要记录 B 运行时的实际 present。

使用已入库的包装脚本 **`tools/log_gripper_rollout.py`**，它会在 `send_action`
前后各读一次 `Present_Position`，落 CSV 六列：
`t_s, kind, gripper_req, gripper_sent, present_before, present_after`。

> 历史说明：早期版本用 `/tmp/log_gripper_rollout2.py` + heredoc 现场生成。
> 已弃用——脚本必须落在工作区内才保证可见，且 heredoc 里嵌代码难以维护。
> 现版本参数原样转交 `lerobot_rollout.main()`，用法与直接跑 rollout 完全一致。

在 B 的一个半成功位置跑，例如左上竖抓：

```bash
cd /home/yyl/lerobot

GRIPPER_LOG=/home/yyl/deepseek-harness/logs/gripper_B_test2.csv \
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

看：

```bash
column -s, -t /home/yyl/deepseek-harness/logs/gripper_B_test2.csv | tail -100
```

判断：

| req | sent | present | 结论 |
|---|---|---|---|
| 一直低 | 一直低 | 一直低 | 策略输出问题 |
| 高 | 低 | 低 | 动作映射/裁剪问题 |
| 高 | 高 | 低 | 带载机械问题 |
| 高 | 高 | 高 | 释放时机/位置问题 |

---

## 4. 如果带载能开且策略命令正常

那就不是夹爪本身的问题，而是 B 的**释放轨迹/时机**：

- 盒子还没到木盒上方，夹爪就打开了
- 或者到了木盒上方，但夹爪打开时盒子被指间卡住
- 或者 B 的释放位置比 A 低/偏，盒子没进木盒

下一步：

1. 同位置跑 A 和 B，录视频，逐帧对比释放点。
2. 检查 B 在释放前最后一小段是否还在夹着盒子移动。
3. 对比 A/B 的 `gripper_req` 开始上升的时间点，和末端位置。
4. 如果 B 打开太晚/太早，先试单变量：
   - `--policy.temporal_ensemble_coeff=0.0`
   - `--policy.n_action_steps=100`

---

## 5. 当前结论

```text
夹爪方向：正常（0=闭，100=开）              ✅ 已确认
空载：正常（线性跟随到 99.38）              ✅ 已确认
B 策略命令：正常（req/sent 到 99.38）       ✅ 已确认
带载 present：99.38，无卡涩                 ✅ 已确认（2026-10-28 实测）
⇒ 排除：策略没发开爪 / 动作被裁剪 / 带载机械卡住
⇒ 定位：B 的释放时机或释放位置
```

下一步不再是查机械，而是查轨迹。要拿到运行期 `present` 的日志，
才能在矩阵里把「present 高」这一格也实证掉（当前只由手动带载测试间接支持）。

先跑第 3 节的 `tools/log_gripper_rollout.py` 记录 B 运行期 present，若 present 也到 99 左右，
就直接进第 4 节：录视频逐帧对比 A/B 的释放点与 `gripper_req` 起升时刻。

---

## 6. 新发现：评测执行模式 ≠ 训练执行模式（待验证）

### 6.1 硬事实（直接读 checkpoint 配置）

`outputs/train/act_armB/checkpoints/022500/pretrained_model/config.json`：

```text
type: act
chunk_size:        100
n_action_steps:    100
temporal_ensemble_coeff: None     ← 训练时没有启用 ensembling
```

而第 3 节的评测命令**覆盖**成：

```text
--policy.temporal_ensemble_coeff=0.01
--policy.n_action_steps=1
```

`configuration_act.py:138` 也印证这是一对绑定参数：启用 ensembling 时
`n_action_steps` 必须是 1（每步都要查策略）。

⇒ **训练用的是「一次预测 100 步、开环执行整段 chunk」；评测改用「每步重查 + 时间集成」。**
这是 B 组评测中唯一与训练不一致的推理设置，且恰好是夹爪动作最敏感的地方。

### 6.2 量化：集成窗口对快跳变的拖尾

`ACTTemporalEnsembler` 维护长度 = `chunk_size` 的滑动平均，权重 `wᵢ = exp(-m·i)`。
仿真（coeff=0.01, chunk=100, 30fps，夹爪 38 → 99.38 的阶跃）：

| 策略预测行为 | coeff=0.01 | coeff=0.0 | coeff=-0.01 |
|---|---|---|---|
| 完美预见 demo 里的开爪 | 0 步 | 0 步 | 0 步 |
| 反应式（观测到才改） | 50%=62 / 90%=93 步 | 49 / 90 步 | 37 / 84 步 |

两点结论，**不要只记一个数**：

1. 若模型能准确预测 demo 里的开爪，集成不产生任何滞后（0 步）——所以不能泛泛说"集成导致滞后 3 秒"。
2. 一旦预测不是完美预见，**窗口长度本身（100 步 ≈ 3.3 s @30fps）就把跳变拖成斜坡**，
   coeff 只改变斜坡偏向（正 coeff 更信旧预测 → 开得更晚），不改变窗口长度。
   即使 `coeff=0.0` 也救不回来（49/90 步 vs 62/93 步）。

**这正好解释日志里 `gripper_req` 从 ~38 缓慢爬到 99.38 的形态**——那是一段被低通滤过的斜坡，
而不是策略原生输出的阶跃。B 组半成功的"抓到了、搬到了、但没释放"，与"开爪指令被拖慢
→ 手臂已经按 demo 轨迹后撤时爪才慢慢张开"高度吻合。

### 6.3 两个仍未区分的分支

"没释放"目前还差一次测量才能定性：

| 分支 | 机制 | 判别 |
|---|---|---|
| (a) 爪真张到 ~99 | 盒子被顶在木盒边缘 / 指间摩擦 / 开爪方向与重心不对 | 运行期 `present_after` 高 |
| (b) 爪没真张开 | 释放姿态下被目标物顶住（≠ 桌面自由姿态） | 运行期 `present_after` 低 |

⚠️ §2.1 的 99.381 是在**桌面自由姿态**下测的，**不等价于木盒上方的释放姿态**。
分支 (b) 尚未排除。

### 6.4 下一步（按性价比排序）

1. **单变量实验**：`--policy.n_action_steps=100 --policy.temporal_ensemble_coeff=None`
   即恢复训练模式，看半成功是否消失。这是当前最高优先级，因为它同时消掉
   6.2 的拖尾并让评测与训练一致。
2. 同一轮同时开 `tools/log_gripper_rollout.py` 记录 `present`，用 6.3 的表区分 (a)/(b)。
3. 若 1 有效，再回头单独扫 `n_action_steps=1` + 各 coeff，量化拖尾代价。

---

## 7. 根因：armB = 遥操（另一任务、无释放）+ 仿真（仅占 22% 帧）

> 定性更正：B 设计上就是「遥操 + 仿真」，**不是误操作造成的污染**。
> 问题在于这 77.8% 的遥操部分是**另一个任务**，而它从不演示释放。

### 7.1 数据集结构

对 `/home/yyl/lerobot_datasets/armB` 直接解 parquet 与 `meta/tasks.parquet`：

```text
task 0  "Grab the black cube"                          62273 帧 / 139 集   ← 遥操
task 1  "Pick up the medicine box and place it into the wooden box."   17753 帧 / 39 集   ← 仿真
合计                                                    80026 帧 / 178 集
```

- **task 0 的 62273 帧 ≡ armA 数据集全部 139 集，逐帧完全相同**（`np.allclose` 全真，
  视频文件 md5 也一致）。即 `armA ⊂ armB`，B = A + 仿真 39 集。
- **task 1 的 17753 帧 ≡ `sim39` 数据集，逐集完全相同（39/39）**；
  而 `sim39` 又恰好是 `sim156` 的**前 39 集**（39/39 命中，集号 0~38）。
- 评测任务（药盒放进木盒）对应 **task 1，只占 22.2% 帧数**。
  若当初用 `sim156`（70824 帧）而不是 `sim39`，混比会是遥操 46.8% / 仿真 53.2%。

### 7.2 两段数据都是「抓取→搬运→放下」，但夹爪数值约定不同（★ 已更正）

> **2026-10-28 二次更正**：本节早先写「task 0 的演示里根本不存在释放」——**错了**。
> 当时的判据只看「每集最后 30%」的夹爪峰值，而真机遥操的释放发生在**全片约 46% 处**，
> 放下之后夹爪又闭合、再回 home；最后 30% 正好是闭爪回家段，所以被误判为"没有释放"。
> 打完整轨迹即可看到释放：`ep0 → 32(接近) → 0(抓) → 19-20(放下) → 0 → home`。

用统一判据（张开→合爪→再张开）重算：

| 指标（导出量纲 0~49.7） | 真机遥操 task 0 | 仿真 task 1 |
|---|---|---|
| 符合「张开→合爪→再张开」 | **117/139 集** | 判据不适用（见下） |
| approach 张开峰值 中位 | 29.6（= jaw 51.9°） | 49.08（= jaw 85.9°，**全开**） |
| 合爪时的命令值 中位 | **0.0** | **9.16**（= jaw 0.28 rad） |
| release 张开峰值 中位 | 22.0（= jaw 38.5°） | 39.3（= jaw 68.8°） |
| 释放发生在全片 | 46% 处 | 接近片尾（transport→place→release→drop） |

⇒ 两段数据**都是完整的抓取+放置过程**（这一点你说得对）。真正的差异是：

1. **仿真整体张得比真机宽约 1.7 倍**（approach 85.9° vs 51.9°，release 68.8° vs 38.5°）。
2. **合爪约定不同**：真机遥操作把夹爪**压到 0**（leader 端没有物体，人是捏到底的），
   仿真则精确停在 **0.28 rad**（本仓库规定的夹 15 mm 盒的合爪量）。
   所以仿真永远到不了 5 以下，同一套阈值判据必然匹配失败。

### 7.3 为什么这会表现为 B 不释放

ACT 非语言条件，同一个模型必须同时拟合两种夹爪数值约定：

- 78% 的帧来自真机：**合爪=0 / 张开=29.6 / 释放=22.0**
- 22% 的帧来自仿真：**合爪=9.16 / 张开=49.1 / 释放=39.3**

夹爪通道上这两套刻度是矛盾的。策略输出会落在中间地带，于是：

- 想"合爪"时输出约 4~5，真机上**没夹紧**（对应 1 次夹空）；
- 想"释放"时输出约 22~30，而仿真演示的释放幅度是 39.3，
  幅度不足以把药盒从指间挤出去 → **盒子被夹着带走**。

这与 7 次半成功、且散在 7 个不同位置（全局行为而非位置问题）一致。

> ⚠️ 这一段仍是**假设**，尚未用运行期日志证实。判别方法见 §7.8 末尾的离线探针。

### 7.4 连带结论

1. **A/B 对照被污染**：B 的训练集包含 A 的全部数据，二者不是独立采集。
   "B 比 A 差"不能归因于机械臂、位置或推理参数。
2. 归一化边界（`max=49.682`）意味着策略的"最大张开"只到 49.7，
   而硬件手动全开是 100——但 task 1 演示的释放峰值 39.3 已能放下药盒，
   所以这不是本次失败的原因，**只是不能再用 99.38 论证"指令到位"**。

### 7.5 建议动作

1. **按 task 过滤重建数据集**：只保留 `task_index==1`（39 集 / 17753 帧）重训 B。
   这是最直接的单变量修复。
2. 若 39 集不够，补采药盒→木盒任务数据，**不要再混入 Grab the black cube**。
3. 重训后用 `tools/log_gripper_rollout.py` 记录正常回合的 `req/sent/present`，
   确认释放段 `req` 是否真的上台；预期应落在 30~50（数据集单位），而不是 99。
4. 复测 A 时也要注意：A 的 139 集**有**释放演示（117/139 集，峰值中位 22.0），
   所以 A 的 4/4 释放成功是自洽的；但 A 与 B 的夹爪幅度刻度不同，
   两边的"释放成功率"仍不可直接横比。

---


### 7.6 数据表核对（★ 本节曾在 2026-10-28 被写错，现已更正）

> **勘误**：本节初版声称 `real_train_fixed` 只有 87 集 / 4032 帧且 78 集为空、
> `holdout_fixed` 只有 8 集 / 448 帧且 7 集为空。**这是错的**。
> 原因是核对脚本用了 `glob(...)[0]` 只读了**第 1 个 parquet 文件**——
> 这两个数据集是**多文件分片**（16 个 / 12 个），只读第 1 片必然把其余分片里的
> episode 算成"0 帧空集"。改为遍历全部文件后，数据完好。

遍历全部 parquet 分片的实测结果：

| 数据集 | HANDOFF 写 | 实测（全部文件） | 判定 |
|---|---|---|---|
| `armA` | 139 / 62273 | 139 集 / 62273 帧（1 文件） | ✅ |
| `real_train_fixed` | 139 / 62273 | **139 集 / 62273 帧（16 文件），空集 0** | ✅ 与 armA 逐帧完全相同 |
| `holdout_fixed` | 12 / 5376 | **12 集 / 5376 帧（12 文件），空集 0** | ✅ |
| `armB` | 178 / 80026 | 178 / 80026（= armA + sim39） | ✅ |
| `sim39` | 39 / 17753 | 39 / 17753 | ✅ |
| `sim156` | 156 / 70824 | 156 / 70824（`sim39` ⊂ 其前 39 集） | ✅ |

**教训**：读多分片 lerobot 数据集必须遍历 `data/*/*.parquet` 全部文件，
不能只取第一个。`armA = real_train_fixed`（逐帧相同），所以两者可互替。

### 7.7 下一步（不动 armC/D/E）

1. **不用真机、今天就能做**：把 B 重训三份做对照
   - (a) 仅 `sim39`（22% 那份，唯一有释放演示的）
   - (b) `sim156` + 遥操（53/47 混比，平衡）
   - (c) 现状 `armB`（78/22）—— 作为基线
   比较三者的"释放段 req 是否上台"。
   预期：只要消掉"跨任务、且占 78%、且不含释放"的那部分，释放行为就会恢复。
2. 若坚持遥操+仿真混训，**遥操部分必须换成同一个药盒任务**。当前用 78% 的
   "Grab the black cube" 数据，等于在教它"别放手"。
3. 真机复测前，先修 `real_train_fixed` / `holdout_fixed` 的空集。
4. HANDOFF 里「B 策略日志 gripper_req 到 99.38」请一并作废（见 §0 更正）。


## 8. 结论更新（覆盖 §5）

```text
夹爪方向：正常（0=闭，100=开）              ✅ 已确认
空载：正常（线性跟随到 98.35）              ✅ 已确认
带载：正常（loaded open 99.38，无卡涩）     ✅ 已确认（2026-10-28 实测）
⇒ 机械/扭矩/装配彻底排除

B 策略命令：❌ 原结论作废 —— 99.38 是手动测试值，不是策略输出（§0 更正）
B 训练数据：❌ 78% 帧是另一个任务且不含释放（§7）
⇒ 定位：数据问题，不是硬件、不是推理参数、不是位置
```

**下一步唯一该做的事**：按 `task_index==1` 过滤重训 B，再复测。
在重训之前，继续在硬件上找原因是浪费时间。

### 7.8 仿真数据体检：它本身是好的，但有 3 处已量化的不一致

直接解 `scripts/ep_*.npz`（1000 条母本）核对，**不是靠猜**。

#### 没问题的（逐条核验通过）

| 检查项 | 实测 | 判定 |
|---|---|---|
| 相位完整性 | `init→approach→descend→grasp→settle_a→lift→settle_b→transport→place→pre_open→release→drop→retract→home→tail` | ✅ 有独立的 release/drop 段 |
| 是否真掉落 | 药盒 z 0.1571→0.0228，y +0.0677→−0.1312（木盒中心 y=**−0.132**）、终点停稳 `obj_speed=0` | ✅ 真放进木盒 |
| 是否 qpos 焊死（红线 2） | `fn_grip` 峰值 **2.37 N**，接触真实 | ✅ 未伪造抓取 |
| 量纲是否对齐真机 | 夹爪最大值**恰好 49.0800**（=`--gripper-full 49.08`），抓取值**恰好 9.16** = 49.08×16.04°/85.94°，即 jaw 正好 **0.28 rad**（本仓库规定值） | ✅ 导出参数正确 |
| action/state 自洽 | 夹爪 \|action−state\| 中位 **0.032**（真机遥操是 0.753） | ✅ 仿真跟踪精准 |

#### 有问题的（3 处，均已量化）

| # | 问题 | 仿真 | 真机遥操 | 影响 |
|---|---|---|---|---|
| 1 | **夹爪开合幅度不一致** | approach 全开 **85.9°**（=49.08）；release/drop 开 **68.8°**（=39.3） | approach 峰值中位 **29.5**（=51.6°）；**末 30% 峰值中位 0.0** | 策略学"张多大"收到两套矛盾示范；仿真释放幅度在遥操里从未出现 |
| 2 | **起点位姿系统性偏差** | pan 0 / lift −100 / elbow 88.3 / flex 70.7 / roll **−103.5**，**std 全 0（零随机化）** | pan −8.3 / lift −103.8 / elbow 95.5 / flex 77.3 / roll **−86.5**（roll std 6.2°） | **roll 差 17°**；同一初始观测下两种行为，且仿真起点单点过拟合 |
| 3 | 任务语义不同 | 100% 演示"药盒→木盒+释放" | 0/139 演示释放（`Grab the black cube`） | 78% 的数据在教"抓住别放" |

#### 对"为什么 B 不释放"的最终归因

仿真数据**不是元凶**，它是唯一教会策略"释放"的数据源。但它只占 **22.2%**，
且它的释放幅度（jaw 68.8°）与真机遥操里从未出现过的幅度绑定在一起。

⇒ 拒绝顺序：**硬件 ✅ 排除 → 推理参数与训练一致 ✅ 排除 → 夹爪量纲/标定 ✅ 一致
→ 根因落在「78% 的遥操数据不含释放 + 起点位姿差 17°」**。

#### 判别性实验（不用真机）

要确认仿真数据是否"足够好"，只需重训一份 **纯 `sim39`** 的 B′ 并与现状 `armB` 对比。
若 B′ 能在真机上释放，则说明仿真数据可用，问题 100% 出在混比；
若 B′ 仍不释放，则需按上表 #1/#2 修仿真（对齐开合幅度、加起点随机化）后重采。

---

## 6. 离线探针结果（2026-10-28 补充）

### 6.1 仿真观测：B 的释放输出不低

脚本：

```text
tools/probe_release_gripper.py
```

结果（B，39 条 sim）：

```text
phase             pred mean   gt mean
pre_open_last       8.06       9.17
release_start       8.15       9.96
release_mid        30.78      25.01
release_end        40.80      38.38
drop_start         42.35      39.26
```

结论：

```text
在仿真观测上，B 的 drop_start 输出 42.35，高于仿真真值 39.26
不是“B 不会输出大的张开量”
```

### 6.2 真机 holdout：A/B 的释放输出几乎一样

脚本：

```text
tools/probe_release_gripper_real.py
```

结果：

```text
armA release_mid: pred mean 21.80, gt mean 21.78
armB release_mid: pred mean 21.24, gt mean 21.78
```

结论：

```text
在真机观测上，A/B 在 release_mid 都输出 ~21，而不是 B 单独偏低。
所以“B 被真机刻度拉低”这个假设不成立。
```

### 6.3 B 真机 rollout 日志的修正

之前 tail 里看到 gripper 到 99.38，误以为是释放命令。  
完整时间线显示：

```text
rollout 期间 B 的 gripper command 在 0~33 之间反复变化
t≈60s release 尝试：cmd 25~29，present 24~30
t≈64~66s 的 99.38 ramp 是 return-to-initial-position，不是释放
```

结论：

```text
B 在真实 rollout 里确实发过 25~30 的张开命令
present 也基本跟随
失败更可能是“25~30 带载张开不足以释放”或“释放时机/位置不对”
而不是“策略完全没张开”
```

### 6.4 当前最可能的两个分支

1. **带载张开不足**  
   真实 rollout 里 B 只开到 25~30，盒子在指间卡住。  
   需要 `tools/test_gripper_loaded.py` 确认带载时 present 能不能到 80。

2. **释放时机/位置不对**  
   如果带载能开到 80，且手动强制 100 能掉盒，但 B 自动释放不掉，说明 B 在错误的位置/时机释放。

### 6.5 下一步最小实验

1. 跑 `tools/test_gripper_loaded.py`，记录 `loaded open`。
2. 在 **同一个位置** 分别跑 A 和 B，各 60 秒，用 `log_gripper_rollout2.py` 记录：
   - `gripper_req`
   - `gripper_sent`
   - `present_before`
   - `present_after`
3. 对比 A/B 在 release 窗口的 `gripper_req` 峰值：
   - A 明显更高 → B 的释放幅度确实不足
   - A/B 差不多 → 不是幅度，查释放位置/时机/盒子卡住

---

## 9. 按 HANDOFF_NEXT_WINDOW §4 推进（2026-10-28 续）

### 9.1 实验 1（带载夹爪）—— 已完成，结论：不是机械问题

```text
initial:       9.003
unloaded open: 98.351
closed on box: 4.467
loaded open:   99.381      ≥ 80
```

带载张开 99.381 甚至高于空载 98.351，**零卡涩**。§4 实验 1 可划掉。

### 9.2 实验 2 的工具链——补齐了两个缺口

HANDOFF §4 要求比较「A/B 张开时的**末端位置**」，但 `tools/log_gripper_rollout.py`
原来只记 5 列夹爪数据，**记不到位姿，实验做出来也回答不了**。已修：

1. `tools/log_gripper_rollout.py`：一次 `sync_read("Present_Position")` 顺带取全部 6 个电机
   （成本与只读夹爪相同），现在输出 10 列：
   `t_s, gripper_req, gripper_sent, present_before, present_after,
    shoulder_pan_before, shoulder_lift_before, elbow_flex_before,
    wrist_flex_before, wrist_roll_before`
2. `tools/compare_release_ab.py`（新增）：吃两份 CSV，自动给出：
   - 相对「合爪时刻」的张开时机（消除两次录制起点不同造成的假差异）
   - `gripper_req` 峰值差、present 峰值
   - 张开瞬间 5 个关节角及其差值
   - 三行判读：幅度 / 时机 / 位姿各是否成立

已用合成日志验证：能分别识别「幅度 -36%」「时机 +0.8s」「位姿差 20°」三种情况。

### 9.3 需要你在真机上跑的两条命令

同一位置，各 60 秒；A 只改 `--policy.path`。

```bash
cd /home/yyl/lerobot

GRIPPER_LOG=/home/yyl/deepseek-harness/logs/gripper_B_samepos.csv \
env -u PYTHONHOME -u PYTHONPATH \
/home/yyl/miniforge3/envs/lerobot312/bin/python \
/home/yyl/deepseek-harness/tools/log_gripper_rollout.py \
  --strategy.type=base --robot.type=so101_follower \
  --robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B91029226-if00 \
  --robot.id=yyl_follower_arm \
  --robot.cameras='{env: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: /dev/v4l/by-id/usb-XHH-260202-H_Integrated_Camera-video-index0, width: 640, height: 480, fps: 60, fourcc: "MJPG"}}' \
  --policy.path=/home/yyl/lerobot/outputs/train/act_armB/checkpoints/022500/pretrained_model \
  --policy.temporal_ensemble_coeff=0.01 --policy.n_action_steps=1 \
  --task="Grab the black cube" --duration=60 --fps=30 --display_data=false
```

A 组：把 `GRIPPER_LOG` 改成 `gripper_A_samepos.csv`，
`--policy.path` 改成 `.../act_armA/checkpoints/017500/pretrained_model`。

跑完直接：

```bash
cd /home/yyl/deepseek-harness
env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
  tools/compare_release_ab.py logs/gripper_A_samepos.csv logs/gripper_B_samepos.csv
```

### 9.4 我这边跑不了的原因（已核实）

AI 沙箱的 `/dev` 是私有 tmpfs，只有 14 个节点；`/dev/serial/by-id`、`/dev/v4l/by-id`、
`/dev/ttyACM0`、`/dev/video0` **全部不可见**。真机 rollout 只能在你终端执行。

---

## 10. 关闭 temporal ensembling 后的实测：假设被证伪，根因是腕部约定冲突

日志：`logs/gripper_B_noens.csv`（1939 帧 / 66.8 s / ≈29 fps，10 列含全部关节角）

### 10.1 ensembling 假设被证伪

关掉 `coeff=0.01 / n_action_steps=1`（不传这两个参数，用回 checkpoint 的
`None / 100`）后，夹爪跳变确实变陡了（单次张开只用 0.3~0.7 s，不再是 3 s 斜坡），
**但释放依旧没有发生**。

⇒ 「temporal ensembling 拖晚了开爪」**不成立**，可以彻底排除。

### 10.2 B 在错误的位姿张开夹爪

从真机 demo（114 集）提取两个参考位姿，再把日志里 9 次 `req>=20` 的张开峰逐一比对：

| 参考位姿 | pan | lift | elbow | flex | roll |
|---|---:|---:|---:|---:|---:|
| 真机·接近张开 | −14.9 | −40.3 | 40.5 | **63.2** | −60.8 |
| 真机·释放张开 | 15.6 | −0.9 | 1.5 | **59.6** | −35.1 |

9 次张开峰中 **8 次离「接近」位姿更近，0 次落在「释放」位姿**。
即 B 全程在物体那一侧张爪，从未走到木盒那一侧张爪。

### 10.3 根因：仿真与真机的腕部关节约定差 33~65°

| 任务时刻 | 真机 armA | 仿真 sim39 | 差值 |
|---|---|---|---|
| 接近张开 flex | +63.2 | −1.6 | **64.7°** |
| 接近张开 roll | −60.8 | −94.2 | **33.4°** |
| 释放张开 flex | +59.6 | +1.3 | **58.3°** |
| 释放张开 roll | −35.1 | −93.6 | **58.5°** |
| 释放张开 elbow | +1.5 | +41.3 | 39.7° |
| 释放张开 pan | +15.6 | +29.0 | 13.4° |
| 释放张开 lift | −0.9 | −16.4 | 15.5° |

**位置关节（pan/lift/elbow）两套数据大体一致（差 0.8~11.5°），腕关节（flex/roll）差 33~65°。**

ACT 面对同一视觉状态被喂两套互相矛盾的腕部目标，只能输出中间值。实测证据：
日志 9 次张开峰的 `wrist_flex` 为 40.4 / 34.2 / 52.5 / 42.3 / 26.2 / 23.9 / 41.8 / 55.1 / 35.0，
**均值 ≈39，正好夹在真机 +63 与仿真 −2 中间**——一个在两份数据里都不存在的位姿。

⇒ 这解释了全部现象：
- **抓取提升**：位置关节一致，仿真补了定位样本 → 抓取 4/12 → 9/12
- **释放崩掉**：腕部目标打架 → 释放时的朝向不是任何一份数据教的 → 从未在木盒上方张开

### 10.4 为什么不能靠改参数修

仿真侧 `wrist_roll` 必须是 −90°（本仓库指垫对齐轴，±5° 就蹭盒，见 so-arm101-sim 技能），
而真机遥操的 `wrist_roll` 在 −35~−61。**两边是物理上不同的指面配置**，
不是简单的零位偏移，加常数补偿会同时破坏仿真的接触真实性。

### 10.5 建议（按代价排序）

1. **B 只用真机数据重训一份**（同超参、同步数，去掉 sim39），确认释放恢复。
   这是最便宜的单变量验证，能直接证明是仿真数据导致释放回归。
2. 若一定要用仿真，需要先解决腕部约定：要么重做仿真的指面/腕部配置使其与真机一致，
   要么把仿真数据仅作为**预训练**（先 sim 预训、再真机微调），不做联合混训。
3. 不要再在真机上反复试参——问题在训练数据，不在推理参数。

---

## 11. 指垫约束实测验证：`roll=-90°` 是硬约束，不是保守选择

### 11.1 验证方法（30 秒级，不用改代码）

`roll` 并不是物理锁死的——它是 `collect/collect.py:307` 的常量 `np.radians(-90.0)`，
而两个探针都提供现成覆盖参数：

```bash
# 接触真实性审计（纯内存、不落盘，6 项判据 G1~G6）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl python3 tools/probe_contact_audit.py --roll -60
```

⚠️ `--combos 'gz:grip:roll;...'` 批量模式有 bug（roll 传进报表时是 None，
`probe_contact_audit.py:550` 格式化崩溃），**逐条用 `--roll` 跑**。

### 11.2 实测结果

| roll | G1 下探零接触 | G4 抬升 | 总体 |
|---:|---|---:|---|
| **−90** | **ok** 0/400 步，Fn 0.00 N | **+122.22 mm** | **6/6 全过** |
| −88 | fail 17/400 步，Fn 1.43 N | +121.54 mm | G1 踩线 |
| −85 | fail 41/400 步，Fn 2.52 N | +120.51 mm | G1 踩线 |
| −80 | fail 41/400 步，Fn 3.50 N | +119.81 mm | G1 踩线 |
| **−60**（真机遥操值） | **fail 66/400 步，Fn 0.94 N** | **−1.28 mm** | **G2 压盒 26.48 mm、G3 倾角 90°、G4 无抬升、G5 缝隙 23.66 mm** |

### 11.3 结论

1. **偏离 −90° 仅 2° 就踩 G1**（下探段开始蹭盒）。指垫方案下 roll 的可用窗口是 ±~1°，
   不是"保守设定"，是真实物理约束。
2. **真机遥操的 roll（−35~−61）远在窗口之外**（差 30~55°）。
3. **`roll=-60` 时仿真彻底抓不住**（抬升 −1.28 mm、压盒 26.48 mm、把盒子顶倒 90°）。

⇒ 要让仿真的腕部滚转走到真机量程，**必须改指面配置本身**（去垫/重新设计指壳），
「半天工时」不是虚估。

### 11.4 重要限制：去指垫只解决一半冲突

腕部冲突有两项，来源不同：

| 冲突项 | 差值 | 根源 | 去指垫能解决吗 |
|---|---:|---|---|
| `wrist_roll` | 33~58° | 指垫对齐轴锁 −90° | ✅ 能（本文已证） |
| `wrist_flex` | 64.7° | 三平行轴下 **lift+elbow+flex 之和差 65.9°**（真机 +63.4 / 仿真 −2.5） | ❌ 不能 |

第二项是**仿真脚本选的接近姿态与真人不同**（一个刚体姿态问题），
不是指垫造成的。即使去掉指垫、放开 roll，仍需按真机 demo 反解接近姿态，
才能把 flex 也拉回真机量程。

⇒ **只去指垫预计仍不能消除全部腕部冲突**，这一点必须在动手前就知道。
