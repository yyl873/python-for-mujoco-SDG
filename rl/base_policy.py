#!/usr/bin/env python3
"""rl/base_policy.py —— 冻结的 ACT 基座 + 观测口径补偿（spec §6.1 / 障碍 O3）

为什么需要这个文件
------------------
纯真机遥操数据训练的 checkpoint 放进 MuJoCo 闭环时，观测有两处**已知**口径差：

  ① 相机通道标签：`yyl100_lance_fixed`（`act_yyl151_lance` 的训练集）里
     `observation.images.env` 通道实际装的是**腕部画面**，两路互换且未修
     —— 证据见 `docs/REAL_DATASET_CAMERA_AUDIT.md` §5（元数据血缘相关 +1.0000，
     平均绝对差 0.0000）。→ 开关 `swap_cams`
  ② 传感器劣化：SDG 采集时每帧都过 `test_render.SensorModel`（blur/noise/gain/
     vignette/color/motion blur），而 `tools/replay_policy.run_closed_loop` 直接吃
     原始锐利渲染图。→ 开关 `use_sensor`

两处都做成显式开关，交给 `rl/pilot.py` 做 2×2 对照，用数据选口径，不靠猜。

不改动 `tools/replay_policy.py` / `render/test_render.py` / `collect/collect.py`
（spec 非目标 N5），只 import 复用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# 与 scripts/ep_000000.json 的 randomization.sensor 同口径（sdg-v6 实测值，别手改）
SENSOR_DEFAULT = dict(blur_sigma=0.67, noise_sigma=3.536, gain_jitter=0.0191,
                      motion_px_max=3.01, motion_k=0.638, vignette=0.23,
                      color_jitter=0.032, flicker_tau_s=2.73)

OBS_ENV = "observation.images.env"
OBS_WRIST = "observation.images.wrist"

# 可用基座（spec §5 的 4 级 fallback，L1→L4；L4 是脚本专家，不在这里）
CKPTS = {
    "lance": "/home/yyl/lerobot/outputs/train/act_yyl151_lance/checkpoints/last/pretrained_model",
    "armA": "/home/yyl/lerobot/outputs/train/act_armA/checkpoints/last/pretrained_model",
    "armB": "/home/yyl/lerobot/outputs/train/act_armB/checkpoints/last/pretrained_model",
    "sim156": "/home/yyl/lerobot/outputs/train/act_sim156_pre/checkpoints/last/pretrained_model",
}


def patch_lerobot_cpu_device():
    """把 lerobot 预处理管线里写死的 device=cuda 覆盖成 cpu（仅无 GPU 时，仅本进程）。

    背景：checkpoint 侧 `policy_preprocessor.json` 把 `{"device": "cuda"}` 存进了管线配置，
    没有 CUDA 的机器上 `make_pre_post_processors()` 会在 from_pretrained 里直接抛
    `ValueError: Requested device 'cuda' but CUDA is not available`。
    这里只在 `ProcessorPipeline._instantiate_step` 上加一层参数覆盖，**不改 lerobot 源码、
    不改 checkpoint 文件、也不改 tools/replay_policy.py**（spec 非目标 N5）。
    """
    import torch
    if torch.cuda.is_available():
        return False
    from lerobot.processor.pipeline import DataProcessorPipeline as PP
    if getattr(PP, "_dsh_cpu_patched", False):
        return True
    orig = PP._instantiate_step.__func__

    def _instantiate_step(cls, step_entry, step_class, step_key, overrides):
        if isinstance(step_entry, dict):
            cfg = step_entry.get("config")
            if isinstance(cfg, dict) and cfg.get("device") not in (None, "cpu"):
                step_entry = {**step_entry, "config": {**cfg, "device": "cpu"}}
        return orig(cls, step_entry, step_class, step_key, overrides)

    PP._instantiate_step = classmethod(_instantiate_step)
    PP._dsh_cpu_patched = True
    return True


class ObsAdapter:
    """渲染原图 -> checkpoint 训练时见到的观测（含两处口径补偿）。"""

    def __init__(self, *, swap_cams: bool = False, use_sensor: bool = True,
                 sensor_cfg: dict | None = None, seed: int = 0):
        self.swap_cams = bool(swap_cams)
        self.use_sensor = bool(use_sensor)
        self.sensor_cfg = dict(SENSOR_DEFAULT if sensor_cfg is None else sensor_cfg)
        self.seed = int(seed)
        self._sensor = None

    def reset(self, seed: int | None = None):
        from test_render import SensorModel            # noqa: PLC0415
        if seed is not None:
            self.seed = int(seed)
        if not self.use_sensor:
            self._sensor = None
            return
        c = self.sensor_cfg
        self._sensor = SensorModel(
            blur_sigma=c["blur_sigma"], noise_sigma=c["noise_sigma"],
            gain_jitter=c["gain_jitter"], seed=self.seed,
            motion_px=(1.0, c["motion_px_max"]), motion_k=c["motion_k"],
            vignette=c["vignette"], color_jitter=c["color_jitter"],
            flicker_tau_s=c["flicker_tau_s"], fps=30.0)

    def __call__(self, raw: dict, speeds_px: dict | None = None) -> dict:
        from test_render import CAM_ENV, CAM_WRIST  # noqa: PLC0415
        speeds_px = speeds_px or {}
        out = {}
        for cam, key in ((CAM_ENV, OBS_ENV), (CAM_WRIST, OBS_WRIST)):
            img = raw[cam]
            if self._sensor is not None:
                img = self._sensor.apply(img, speed_px=float(speeds_px.get(cam, 0.0)))
            out[key] = img
        if self.swap_cams:                       # ★ 口径补偿①
            out[OBS_ENV], out[OBS_WRIST] = out[OBS_WRIST], out[OBS_ENV]
        return out


class ACTBase:
    """冻结的 ACT 基座：act(state_deg[6], raw_images) -> action_deg[6]。"""

    def __init__(self, ckpt: str | Path, *, device: str = "cpu",
                 swap_cams: bool = False, use_sensor: bool = True, seed: int = 0):
        from replay_policy import LerobotPolicy      # noqa: PLC0415
        patch_lerobot_cpu_device()                   # ★ 无 GPU 时把管线里的 cuda 覆盖成 cpu
        self.ckpt = str(ckpt)
        self.policy = LerobotPolicy(Path(ckpt), device=device)
        self.adapter = ObsAdapter(swap_cams=swap_cams, use_sensor=use_sensor, seed=seed)
        self.calls = 0
        self._pre_cached = None
        # temporal ensembling 时每步都要真观测，缓存会改变语义 → 该模式下禁用懒渲染
        self._lazy_ok = getattr(self.policy.policy.config, "temporal_ensemble_coeff", None) is None

    def reset(self, task: str = "", seed: int | None = None):
        # ★ ACT 内部有 action chunk 队列，跨 episode 不 reset 会接着上一条走
        self.policy.prepare(task or "")
        self.adapter.reset(seed=seed)
        self.calls = 0
        self._pre_cached = None

    # ---- chunk 队列感知（性能关键）----
    # ACT 的 n_action_steps=100：`select_action` 只在队列空时用 batch 算整块，
    # 其余 99 步只是 popleft（batch 参数根本没被读）。所以"队列非空就不必渲染"，
    # 与真机部署一致（真机每步都读相机，但图像只在块边界被用）。
    # 收益：一条 15 s episode 的渲染次数从 450 次降到 ~5 次。
    def queue_len(self) -> int:
        q = getattr(self.policy.policy, "_action_queue", None)
        return len(q) if q is not None else 0

    def needs_images(self) -> bool:
        if not self._lazy_ok:
            return True
        return self._pre_cached is None or self.queue_len() == 0

    def act(self, state_deg, raw: dict | None = None,
            speeds_px: dict | None = None) -> np.ndarray:
        t = self.policy.torch
        if self.needs_images():
            if raw is None:
                raise RuntimeError("ACT action chunk 队列已空，必须提供图像")
            obs = self.adapter(raw, speeds_px)
            batch = {"observation.state": t.from_numpy(np.asarray(state_deg, np.float32))}
            for k, v in obs.items():          # HWC uint8 RGB -> CHW float[0,1]（同 lerobot 口径）
                batch[k] = t.from_numpy(np.asarray(v, np.float32) / 255.0).permute(2, 0, 1)
            with t.inference_mode():
                self._pre_cached = self.policy.pre(batch)
        with t.inference_mode():
            a = self.policy.post(self.policy.policy.select_action(self._pre_cached))
        self.calls += 1
        return np.asarray(a.detach().cpu().numpy().reshape(-1)[:6], float)


def make_base(name: str, *, device="cpu", swap_cams=False, use_sensor=True, seed=0) -> ACTBase:
    path = CKPTS.get(name, name)
    if not Path(path).exists():
        raise SystemExit(f"checkpoint 不存在: {path}")
    return ACTBase(path, device=device, swap_cams=swap_cams,
                   use_sensor=use_sensor, seed=seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="基座自检：加载一个 checkpoint，出一维动作")
    ap.add_argument("--ckpt", default="armA")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--swap-cams", action="store_true")
    ap.add_argument("--no-sensor", action="store_true")
    a = ap.parse_args()
    b = make_base(a.ckpt, device=a.device, swap_cams=a.swap_cams,
                  use_sensor=not a.no_sensor)
    rng = np.random.default_rng(0)
    raw = {"env_camera": (rng.random((480, 640, 3)) * 255).astype(np.uint8),
           "wrist_camera": (rng.random((480, 640, 3)) * 255).astype(np.uint8)}
    b.reset("Pick up the medicine box and place it into the wooden box.")
    act = b.act(np.zeros(6), raw)
    print(f"ok ckpt={a.ckpt} action_dim={act.shape[0]} action={np.round(act, 2).tolist()}")