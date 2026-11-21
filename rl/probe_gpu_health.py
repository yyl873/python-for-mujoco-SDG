#!/usr/bin/env python3
"""rl/probe_gpu_health.py —— 训崩了先跑这个：到底是「显卡/驱动面」还是「torch 轮子面」。

为什么要它：GPU 上两次崩溃是两种症状——
  ① rc=139 SIGSEGV（无 traceback，worker 侧，20 个泄漏信号量）
  ② `RuntimeError: d.is_cuda() INTERNAL ASSERT FAILED at c10/cuda/impl/CUDAGuardImpl.h:34`，
     且崩前进度条卡在同一 step 约 20 s（2.93 → 1.19 step/s）
「卡住再抛 CUDA 断言」既可能是驱动/供电面（Xid），也可能是 cu126 轮子缺 sm_120 内核。
本脚本只碰显卡、不碰数据集和 dataloader，能在 1~2 分钟内把两者分开。

判读
----
- A 段（纯矩阵乘 + backward）就炸        → 显卡/驱动面（看 dmesg 里的 Xid）
- A 段过、B 段（真 ACT 前向 + backward）炸 → torch/lerobot 算子面（换 torch 轮子或降档）
- 两段都过（哪怕慢）                      → 计算没问题，嫌疑在 dataloader worker / 视频解码

用法（★ 必须在能看到 /dev/nvidia* 的终端里跑）
    env -u PYTHONHOME -u PYTHONPATH /home/yyl/miniforge3/envs/lerobot312/bin/python \
        rl/probe_gpu_health.py --steps 200
    # 想定位到具体算子（慢很多）：
    CUDA_LAUNCH_BLOCKING=1 env -u PYTHONHOME -u PYTHONPATH .../python rl/probe_gpu_health.py --steps 20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CKPT = "/home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model"


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception as e:                                       # noqa: BLE001
        return f"<失败: {e}>"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200, help="每段跑多少步 forward+backward")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--force-cpu", action="store_true",
                    help="只验证本脚本自身逻辑（无显卡时用，结论作废）")
    a = ap.parse_args()

    import torch

    print("=" * 72)
    print(" 环境")
    print("=" * 72)
    print(f"  torch          : {torch.__version__}")
    print(f"  cuda 可用      : {torch.cuda.is_available()}  设备数 {torch.cuda.device_count()}")
    if a.force_cpu:
        print("  （--force-cpu：只用 CPU 验证脚本自身的 A/B 两段能否跑通，结论不代表显卡）")
    if not torch.cuda.is_available() and not a.force_cpu:
        print("  ★ 看不到显卡：这段必须在**你自己的终端**里跑（沙箱内 NVML 初始化失败）")
        return 2
    if torch.cuda.is_available():
        print(f"  GPU            : {torch.cuda.get_device_name(0)}  "
              f"capability {torch.cuda.get_device_capability(0)}")
        arch = torch.cuda.get_arch_list()
        print(f"  编译架构(arch) : {arch[:8]}{'...' if len(arch) > 8 else ''}")
    print(f"  驱动(nvidia-smi): {sh('nvidia-smi --query-gpu=driver_version --format=csv,noheader')}")
    print(f"  CUDA_LAUNCH_BLOCKING = {sh('printenv CUDA_LAUNCH_BLOCKING') or '(未设)'}")

    dev = torch.device("cpu" if a.force_cpu else "cuda:0")

    # ---------------- A 段：纯计算 ----------------
    print()
    print("=" * 72)
    print(f" A 段：纯 matmul + backward × {a.steps}")
    print("=" * 72)
    t0 = time.time()
    try:
        x = torch.randn(2048, 2048, device=dev, dtype=torch.float32)
        w = torch.randn(2048, 2048, device=dev, dtype=torch.float32, requires_grad=True)
        for i in range(a.steps):
            y = (x @ w).relu().sum()
            y.backward()
            w.grad = None
            if not torch.isfinite(y):
                print(f"  ★ 第 {i} 步出现非有限值 —— 数值面问题")
                return 3
        if dev.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:                                       # noqa: BLE001
        print(f"  ✗ A 段失败（{type(e).__name__}）：{e}")
        print("  → 纯计算都过不去 ⇒ 显卡/驱动面。请贴这两条的尾部：")
        print("      dmesg | tail -30 | grep -iE 'xid|nvrm|reset'")
        print("      nvidia-smi -q | grep -iE 'xid|throttle|temp' | head")
        return 1
    print(f"  ✓ A 段通过：{a.steps} 步 / {time.time() - t0:.1f} s"
          f"（{a.steps / max(1e-9, time.time() - t0):.1f} step/s）")

    # ---------------- B 段：真 ACT 前向 + backward ----------------
    print()
    print("=" * 72)
    print(f" B 段：{Path(a.ckpt).name} 前向 + backward × {a.steps}")
    print("=" * 72)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    try:
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: PLC0415
        from replay_policy import load_policy_config                # noqa: PLC0415

        cfg = load_policy_config(Path(a.ckpt))
        cfg.device = str(dev)
        cls = get_policy_class(cfg.type)
        policy = cls.from_pretrained(str(a.ckpt), config=cfg).to(dev).train()
        imgs = [k for k in cfg.input_features if "images" in k]
        n_act = cfg.output_features["action"].shape[0]
        chunk = int(getattr(cfg, "chunk_size", 100))      # ACT 的 action 是 (B, chunk, dim)
        n_state = cfg.input_features["observation.state"].shape[0]
        batch = {"observation.state": torch.randn(8, n_state, device=dev),
                 "action": torch.randn(8, chunk, n_act, device=dev),
                 "action_is_pad": torch.zeros(8, chunk, dtype=torch.bool, device=dev)}
        for k in imgs:
            shape = cfg.input_features[k].shape
            batch[k] = torch.rand(8, *shape, device=dev)
        t0 = time.time()
        for i in range(a.steps):
            loss, _ = policy.forward(batch)
            loss.backward()
            for p in policy.parameters():
                p.grad = None
            if i % 25 == 0:
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                print(f"    step {i:4d}  loss={float(loss):8.4f}  "
                      f"{(i + 1) / max(1e-9, time.time() - t0):5.2f} step/s", flush=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:                                       # noqa: BLE001
        import traceback                                       # noqa: PLC0415
        print(f"  ✗ B 段失败（{type(e).__name__}）：{e}")
        print("  --- 完整 traceback（贴给我用）---")
        traceback.print_exc()
        print("  → 纯计算过了、模型算子过不去 ⇒ torch/lerobot 算子面。")
        print("      试 CUDA_LAUNCH_BLOCKING=1 复跑定位具体算子；仍崩就换 torch 轮子（cu128 起对 sm_120 支持更全）")
        return 4
    dt = time.time() - t0
    print(f"  ✓ B 段通过：{a.steps} 步 / {dt:.1f} s（{a.steps / max(1e-9, dt):.2f} step/s，batch 8）")

    print()
    print("=" * 72)
    print(" 结论：两段都过 ⇒ 计算面没问题，嫌疑落在 dataloader worker / 视频解码。")
    print(" 下一步：WORKERS=0 跑训练（run_lang_train.sh 会自动降档）。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())