# SO-ARM101 纯物理 MuJoCo 数据合成（零 ROS）

用 MuJoCo 里的 SO-ARM101 机械臂，**真物理**抓一个 15 mm 药盒放进木盒，
批量生成"同任务、不同摆放 / 不同节奏 / 不同噪声"的训练数据，并导出成 **lerobot 数据集**给 ACT 训练。

**没有伪造**：机械臂不写 `qpos`，握持来自 `mj_step` 的接触力与摩擦（法向力 2.6~4 N，药盒重力只有 0.113 N）。

---

## 我要生成数据 → 只看这一份

👉 **[`docs/RUNBOOK_SDG.md`](docs/RUNBOOK_SDG.md)**（操作手册：顺序、命令、每个参数什么时候用、1000 条怎么跑）

三分钟版：

```bash
cd /home/yyl/deepseek-harness

# ① 采集 20 条（带视频；有 GPU 时 20 条约 1 分钟）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
    collect/collect.py --n 20 --jobs 8 --seed 3

# ② 体检（秒级）
/usr/bin/python3 tools/inspect_episode.py --all --check-only

# ③ 导出成 ACT 可训练的数据集（换解释器！）
env -u PYTHONHOME -u PYTHONPATH ~/miniforge3/envs/lerobot312/bin/python \
    collect/export_lerobot.py --dst dataset/soarm101_sim_v1 --overwrite
```

三条命令各自的判据、失败怎么办、参数怎么选，都在手册里。

---

## 这个仓库是什么

| 阶段 | 文件 | 作用 |
|---|---|---|
| 一 IK | `ik/test_ik.py` | 逆运动学（24 次迭代，残差 0.9 mm） |
| 二 抓取审计 | `tools/probe_contact_audit.py` | ★ 六项闸门：下探是否戳盒 / 压盒 / 撞倒 / 真抬起来 / 指缝 / 穿模。**0.9 s/条，改参数前先跑它** |
| 三 录制+渲染 | `render/test_render.py` | 录一条 demo → 渲染 mp4 → 15 项自检 → `success` 标签 |
| 四 批量合成 | `collect/collect.py` | 随机摆放/节奏/噪声，批量刷数据（`--jobs` 并行） |
| 五 训练交付 | `collect/export_lerobot.py` | 母本 → **lerobot v3.0 数据集**（ACT 直接训） |

配套工具：`tools/run_pipeline.py`（一条命令跑前三段，只回三态回执）、
`tools/inspect_episode.py`（按需出拼图/预览/体检）、`tools/replay_policy.py`（把策略放回 MuJoCo 跑）、
`tools/probe_grasp_robustness.py`（十档节奏鲁棒性）。

## 目录地图

```
model/        MJCF 与网格（改模型只动这里）
ik/  grasp/  render/  collect/     四步代码 + 自检
tools/        执行器与探针（流水线、审计、排查、策略回放）
scripts/      母本·轨迹与标签 ep_XXXXXX.npz|json + index.json + splits/
videos/       母本·视频 ep_XXXXXX_env|wrist.mp4（h264）
dataset/      导出的 lerobot 数据集（训练指这里）
inspect/      按需排查产物（可删）
demo/ demo_neg/   单条 demo 正/负样本
logs/         运行日志
old_ros/      旧 ROS 代码，仅参考，禁止运行
```

## 文档（只有三份，别的都在 `docs/archive/`）

| 文档 | 给谁看 |
|---|---|
| [`docs/RUNBOOK_SDG.md`](docs/RUNBOOK_SDG.md) | **操作者**：怎么生成数据、怎么导出、怎么训练、1000 条怎么跑 |
| [`docs/PHYSICS_FACTS.md`](docs/PHYSICS_FACTS.md) | 改代码/调参数的人：现在生效的物理事实、验收数字、不许动的东西 |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | 指路（历史长文已归档到 `docs/archive/`） |

## 三条硬约束（违反即返工）

1. **零 ROS**：不引入 `rclpy`/`moveit`；`old_ros/` 只作参考，禁止运行。
2. **不伪造抓取**：不许写 `data.qpos` 把药盒"焊"在夹爪上，也不许在采集路径里覆写药盒位姿
   （唯一例外是出生那一刻落地）。
3. **渲染与控制解耦**：`render/test_render.py` 的 render 阶段不写 `ctrl`、不调 `mj_step`，
   只对已录轨迹做 `mj_forward` 正运动学回放。