# ft_probe —— 第 3 步的前置实验（噪声地板标定，见 `docs/REPORT_RESIDUAL_RL.md` §14）

全部是 **CPU 上的最小验证**：用 `--steps=3` 的"微调"（权重几乎没动）去评测，
目的是标定 held-out 指标本身的噪声地板，不是微调效果。

| 文件 | 是什么 | 关键数字 |
|---|---|---|
| `ab_base_recheck.json` | 基座复现（seeds=1，n=30） | 8/30，与 `ab_base_heldout_strict.json` 逐条一致 |
| `ab_smoke.json` | 微调 3 步（seed 1000，自带归一化），n=30 | 17/30（clean 9） |
| `ab_ftw_normbase.json` | 同一权重 + **基座**归一化，n=30 | 15/30 |
| `ab_basew_ftnorm.json` | **基座**权重 + 微调归一化，n=30 | 6/30 |
| `ab_base_90.json` | 基座，seeds=3（n=90） | 12/90（clean 10） |
| `ab_ft3_90.json` | 微调 3 步（seed 1000），n=90 | 35/90（clean 12） |
| `ab_ft_s1001_90.json` | 微调 3 步，**训练 seed=1001**，n=90 | 6/90（clean 2） |

结论：权重变化 ≤3e-5 就能让 held-out 成功在 6/90 ~ 35/90 之间摆动 → **噪声地板 ±15pp**；
且 success 涨、clean 几乎不涨 → 主指标必须是 `clean_success`。

复现命令（CPU，全部离线）：

```bash
# 3 步微调（沙箱可跑，约 10 s）
env -u PYTHONHOME -u PYTHONPATH HF_HOME=$PWD/.cache/huggingface .venv-rl/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/sim156_plus_dagger3 --dataset.root=$PWD/rl/dagger_ds/sim156_plus_dagger3 \
  --policy.type=act --policy.repo_id=local/act_dagger_ft \
  --policy.pretrained_path=/home/yyl/lerobot/outputs/train/act_sim156_pre/checkpoints/last/pretrained_model \
  --policy.push_to_hub=false --save_checkpoint_to_hub=false \
  --batch_size=2 --steps=3 --save_freq=3 --log_freq=1 \
  --output_dir=$PWD/.ftsmoke/act_dagger_ft --seed=1000 --num_workers=0

# 评测（90 次 rollout，约 6 min）
env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl LP_NUM_THREADS=8 MPLCONFIGDIR=$PWD/.mplcache \
  .venv-rl/bin/python rl/eval_ab.py \
  --base .ftsmoke/act_dagger_ft/checkpoints/000003/pretrained_model \
  --episodes 100-129 --seeds 3 --judge strict --out rl/eval/ft_probe/ab_ft3_90.json
```
