#!/usr/bin/env python3
"""rl/env_residual.py —— 残差 RL 的 Gymnasium 环境（spec §6）

    obs ──▶ 基座 ACT（冻结，含 chunk 队列） ──▶ action_il(6 维, 度)
                                                │
                                        + 残差 a(6 维, 度, |a| ≤ 5.73°≈0.1 rad)
                                                ▼
                                   ctrl = _ctrl_from_deg(action_il + a)   （单尺度，只缩放一次）
                                                ▼
                                          mj_step × N

设计要点（都是为了"两天能跑出趋势"）
------------------------------------
1. **残差只吃低维观测（14 维）**，视觉留在基座里。渲染 1.7 ms/帧（GPU）或 174 ms/帧（CPU），
   把 RL 的观测做成图等于把训练成本乘以百倍，而 6 维残差的活是纠偏、不是看世界。
2. **动作空间就是最终幅度**：`Box(±5.73°, 6)`。原计划的 `Box(±0.1)` 再乘 0.1 是双重缩放，
   实际只剩 ±0.01 rad，残差等于没开，而曲线会"看起来在跑"。
3. **闸门与奖励分离**：G1-G6 只做验收（`probe_contact_audit` / `eval_ab`），**不进 reward**，
   否则 RL 会学着擦边过闸门。
4. **不伪造抓取**：残差只能经 `ctrl` + `mj_step` 生效，绝不写 `qpos`（红线 2）。
5. `terminated`（成功/撞倒）与 `truncated`（超时/盒被推出工作区）严格分开。

注意：`mj_step` 有约 7% 进程级 SIGFPE，进程会被直接杀掉 —— 进程内 try/except 兜不住，
所以训练入口（`rl/train.py`）必须做**进程级**重试 + 断点续跑。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gymnasium as gym                                  # noqa: E402
from gymnasium import spaces                             # noqa: E402

OBS_DIM = 14
RESIDUAL_DEG = float(np.degrees(0.1))                    # ±0.1 rad = ±5.7296°
DEG = 1.0


def load_episode(scripts: Path, eid: int):
    import json                                          # noqa: PLC0415
    p = Path(scripts) / f"ep_{eid:06d}.npz"
    j = Path(scripts) / f"ep_{eid:06d}.json"
    if not p.exists():
        raise SystemExit(f"没有母本轨迹: {p}")
    with np.load(p, allow_pickle=True) as z:
        ep = {k: np.array(z[k]) for k in z.files if k != "meta_json"}
    meta = json.loads(j.read_text(encoding="utf-8")) if j.exists() else {}
    return ep, meta


class ResidualSO101(gym.Env):
    """ACT 基座 + 6 维残差。整个环境只在 `step` 里通过 ctrl 影响物理。"""

    metadata = {"render_modes": []}

    def __init__(self, *, base_name: str = "sim156", device: str = "cpu",
                 swap_cams: bool = False, use_sensor: bool = True,
                 control_hz: float = 30.0, max_seconds: float = 15.0,
                 episode_pool=(0,), scripts=None, randomize: bool = True,
                 residual_deg: float = RESIDUAL_DEG, seed: int = 0,
                 width: int = 640, height: int = 480):
        super().__init__()
        import mujoco                                    # noqa: PLC0415
        import test_render as R                          # noqa: PLC0415
        from base_policy import make_base                # noqa: PLC0415
        from replay_policy import _ctrl_from_deg, _deg_from_ctrl   # noqa: PLC0415

        self._mujoco = mujoco
        self._R = R
        self._ctrl_from_deg = _ctrl_from_deg
        self._deg_from_ctrl = _deg_from_ctrl

        self.base = make_base(base_name, device=device, swap_cams=swap_cams,
                              use_sensor=use_sensor, seed=seed)
        self.base_name = base_name
        self.control_hz = float(control_hz)
        self.max_seconds = float(max_seconds)
        self.randomize = bool(randomize)
        self.scripts = Path(scripts or (ROOT / "scripts"))
        self.episode_pool = list(episode_pool)

        self.residual_deg = float(residual_deg)
        self.action_space = spaces.Box(low=-self.residual_deg, high=self.residual_deg,
                                       shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(OBS_DIM,), dtype=np.float32)

        self.arm = R.Arm(R.SCENE_XML)
        self.rig = R.CameraRig(self.arm.m, width=width, height=height)
        self.m, self.d = self.arm.m, self.arm.d

        # 随机化要能复位：先存模型原值
        self._mass0 = self.m.body_mass.copy()
        self._fric0 = self.m.geom_friction.copy()
        self._light0 = self.m.light_diffuse.copy()

        self._episodes = {eid: load_episode(self.scripts, eid) for eid in self.episode_pool}
        self._rng = np.random.default_rng(seed)
        self._prev_uv = {c: None for c in R.CAMS}
        self._speeds = {c: 0.0 for c in R.CAMS}
        self._obj0 = None
        self._t0 = 0.0
        self._max_lift = -1e9
        self._tcp_box_min = 1e9
        self._episode_id = None
        self._task = ""

    # ---------------- 观测 / 奖励 ----------------
    def _obs(self) -> np.ndarray:
        d = self.d
        state = self._deg_from_ctrl(self.arm, d.ctrl)           # 6 维（度）
        box = self.arm.obj_pos()
        tcp = self.arm.tcp_pos()
        rel = box - tcp
        jid = self.m.body_jntadr[self.arm.obj]                   # 药盒 free joint
        dadr = int(self.m.jnt_dofadr[jid])                       # ★ qvel 用 dof 地址，不是 qpos 地址
        vel = d.qvel[dadr:dadr + 3]                              # 线速度（世界系）
        n_con = len(self.arm.grip_contacts())
        t_norm = min(1.0, (d.time - self._t0) / max(1e-9, self.max_seconds))
        obs = np.concatenate([state / 100.0, rel, vel * 10.0,
                              [n_con / 4.0, t_norm]]).astype(np.float32)
        assert obs.shape[0] == OBS_DIM, obs.shape
        return obs

    def _penetration_mm(self) -> float:
        """药盒与手指/桌面的最大穿透深度（mm，>=0）。用于压盒惩罚。"""
        d, m = self.d, self.m
        worst = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            if m.geom_bodyid[c.geom1] != self.arm.obj and m.geom_bodyid[c.geom2] != self.arm.obj:
                continue
            worst = max(worst, -float(c.dist))
        return worst * 1000.0

    def _box_tilt_deg(self) -> float:
        z = self.d.xmat[self.arm.obj].reshape(3, 3)[2, 2]
        return float(np.degrees(np.arccos(np.clip(z, -1.0, 1.0))))

    def _reward(self, lift_mm: float, n_con: int, pen_mm: float):
        tcp = self.arm.tcp_pos()
        box = self.arm.obj_pos()
        dist = float(np.linalg.norm(tcp - box))
        r = -10.0 * dist
        if n_con >= 2 and lift_mm > 5.0:
            r += 0.5
        rise = max(0.0, lift_mm - self._max_lift)
        r += 2.0 * float(np.tanh(rise / 50.0))
        if pen_mm > 0.5:
            r -= 1.0
        return r

    # ---------------- Gym 接口 ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        eid = int(self._rng.choice(self.episode_pool))
        ep, meta = self._episodes[eid]
        self._episode_id = eid
        self._task = (meta.get("instruction")
                      or "Pick up the medicine box and place it into the wooden box.")

        d, m = self.d, self.m
        d.qpos[:] = ep["qpos"][0]
        d.qvel[:] = 0.0
        d.ctrl[:] = ep["ctrl"][0]

        # 随机化（可复现：同一个 self._rng）
        m.body_mass[:] = self._mass0
        m.geom_friction[:] = self._fric0
        m.light_diffuse[:] = self._light0
        if self.randomize:
            adr = self.arm.oadr
            d.qpos[adr + 0] += self._rng.uniform(-0.030, 0.030)
            d.qpos[adr + 1] += self._rng.uniform(-0.030, 0.030)
            yaw = np.radians(self._rng.uniform(-15.0, 15.0))
            cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
            q = d.qpos[adr + 3:adr + 7].copy()
            # 绕世界 z 预乘
            d.qpos[adr + 3] = cy * q[0] - sy * q[3]
            d.qpos[adr + 4] = cy * q[1] - sy * q[2]
            d.qpos[adr + 5] = cy * q[2] + sy * q[1]
            d.qpos[adr + 6] = cy * q[3]
            m.body_mass[self.arm.obj] *= float(self._rng.uniform(0.9, 1.1))
            m.geom_friction[:] *= float(self._rng.uniform(0.9, 1.1))
            m.light_diffuse[:] *= float(self._rng.uniform(0.9, 1.1))

        self._mujoco.mj_forward(m, d)

        self.base.reset(self._task, seed=int(self._rng.integers(0, 2 ** 31 - 1)))
        self._prev_uv = {c: None for c in self._R.CAMS}
        self._speeds = {c: 0.0 for c in self._R.CAMS}
        self._obj0 = self.arm.obj_pos().copy()
        self._t0 = float(d.time)
        self._max_lift = -1e9
        self._tcp_box_min = 1e9
        return self._obs(), dict(episode_id=eid, task=self._task)

    def step(self, action):
        d, m = self.d, self.m
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[:6],
                    -self.residual_deg, self.residual_deg)

        state = self._deg_from_ctrl(self.arm, d.ctrl)
        # ★ 只有 ACT 的 chunk 队列空了才需要新图（n_action_steps=100 → 渲染次数 ÷100）
        if self.base.needs_images():
            raw = self.rig.grab(d)
            for c in self._R.CAMS:                    # 运动虚影要跟着真实运动
                uv = self._R._project_px(self.rig, d, c, self.arm.tcp_pos())
                if uv is not None and self._prev_uv[c] is not None:
                    self._speeds[c] = float(np.linalg.norm(uv - self._prev_uv[c]))
                self._prev_uv[c] = uv
            action_il = self.base.act(state, raw, self._speeds)
        else:
            action_il = self.base.act(state)
        target_deg = action_il + a                      # ★ 单尺度：残差直接就是最终幅度
        d.ctrl[:] = self._ctrl_from_deg(self.arm, target_deg)

        step_dt = 1.0 / self.control_hz
        t_begin = d.time - self._t0
        while d.time - self._t0 < t_begin + step_dt:
            self._mujoco.mj_step(m, d)

        obj = self.arm.obj_pos()
        lift_mm = float((obj[2] - self._obj0[2]) * 1000.0)
        hdisp_mm = float(np.linalg.norm(obj[:2] - self._obj0[:2]) * 1000.0)
        n_con = len(self.arm.grip_contacts())
        pen_mm = self._penetration_mm()
        tilt = self._box_tilt_deg()
        self._tcp_box_min = min(self._tcp_box_min, float(np.linalg.norm(self.arm.tcp_pos() - obj)))

        reward = self._reward(lift_mm, n_con, pen_mm)   # ★ 必须在更新 _max_lift 之前算（rise 依赖旧值）
        self._max_lift = max(self._max_lift, lift_mm)
        success = bool(self._max_lift >= 50.0 and n_con >= 2)
        fell = bool(tilt > 45.0)
        if success:
            reward += 50.0
        if fell:
            reward -= 10.0
        terminated = bool(success or fell)
        timeout = (d.time - self._t0) >= self.max_seconds
        out_of_area = hdisp_mm > 120.0
        truncated = bool(timeout or out_of_area)

        info = dict(lift_mm=lift_mm, max_lift_mm=float(self._max_lift),
                    hdisp_mm=hdisp_mm, contact_n=n_con, penetration_mm=pen_mm,
                    box_tilt_deg=tilt, success=success, episode_id=self._episode_id,
                    act_il_deg=action_il.tolist(), act_res_deg=a.tolist())
        return self._obs(), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.rig.close()
        except Exception:                              # noqa: BLE001
            pass
        super().close() if hasattr(super(), "close") else None


def _selftest(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="S1 环境自检：200 步随机残差，验证维度/终止/奖励")
    # spec §9 的验收命令写的是 `env_residual.py --selftest`；本文件直接跑就是 selftest，
    # 所以这个旗标只作为别名接受（不改变行为），免得照 spec 抄命令的人被 argparse 拒掉。
    ap.add_argument("--selftest", action="store_true", help="别名：本文件默认即 selftest")
    ap.add_argument("--base", default="sim156",
                    help="基座 ckpt（S0 pilot 结论：sim156 唯一闭环成功，故为默认）")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--episodes", default="0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-randomize", action="store_true")
    a = ap.parse_args(argv)
    pool = tuple(int(x) for x in a.episodes.split(",") if x.strip())
    env = ResidualSO101(base_name=a.base, device=a.device, episode_pool=pool,
                        randomize=not a.no_randomize)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (OBS_DIM,), obs.shape
        n_term = n_trunc = 0
        r_sum = 0.0
        for _ in range(a.steps):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            r_sum += r
            n_term += int(term)
            n_trunc += int(trunc)
            if term or trunc:
                obs, info = env.reset()
        # 200 步 @30 Hz 只有 6.7 s，够不到 15 s 的超时截断，所以单开一段把 truncated
        # 分支也走通：同一环境实例把 max_seconds 缩到 2 s（不重载 ACT，省 20 s）。
        env.max_seconds = 2.0
        obs, info = env.reset(seed=1)
        for _ in range(4 * int(round(2.0 * env.control_hz)) + 40):
            obs, r, term, trunc, info = env.step(env.action_space.sample())
            n_term += int(term)
            n_trunc += int(trunc)
            if trunc:
                break
            if term:
                obs, info = env.reset()
        dims_ok = bool(obs.shape == (OBS_DIM,) and env.action_space.shape == (6,))
        covered = bool(n_term >= 1 and n_trunc >= 1)          # spec §9 S1 硬判据
        print(f"{'ok' if (dims_ok and covered) else 'FAIL'} steps={a.steps} "
              f"obs_dim={obs.shape[0]} act_dim={env.action_space.shape[0]} "
              f"reward_sum={r_sum:.2f} n_terminated={n_term} n_truncated={n_trunc} "
              f"last_lift_mm={info.get('max_lift_mm', 0):.1f}")
        return 0 if (dims_ok and covered) else 1
    finally:
        env.close()


if __name__ == "__main__":
    sys.exit(_selftest())