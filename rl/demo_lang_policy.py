#!/usr/bin/env python3
"""rl/demo_lang_policy.py —— 语言条件 ACT 的"看得见"演示：一句话 → 仿真里挑箱子 → 出 mp4

为什么要它：`rl/run_lang_eval.sh` 只出 JSON（落点/成功率），面试或汇报时没法看。
本脚本把同一套策略跑成视频：策略看到的画面（env + wrist 双视角并排）+ 烧上去的指令横幅。

指令怎么进来（三选一）
----------------------
  --instr left|right   直接给（最稳，推荐）
  --text "把药盒放进左边的木箱"   中英文关键词判断（左右/left/right 命中即用）
  --mic                **用耳机麦克风说话**：录 N 秒 → 优先 vosk 离线识别 → 失败则退化成
                       能量+过零率的关键词兜底；两者都不行会明确报错，不会瞎猜
  ★ MuJoCo 没有音频子系统：语音只能在这里被转成 2 维 one-hot 指令再喂给策略，
    这就是"语言条件"的全部含义（模型不认识自由文本，只认 task_order 里那两条）。

用法（必须在能看到显示/显卡的终端；沙箱里 MUJOCO_GL=egl 也能出视频，只是慢）
    P=.venv-rl/bin/python
    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl $P rl/demo_lang_policy.py \
        --ckpt outputs/act_lang_pair/checkpoints/005000/pretrained_model \
        --instr left --episode 2400 --seed 0 --out rl/eval/demo_left.mp4

    # 耳机说话版（需先 pip install sounddevice；想用离线识别再 pip install vosk）
    ... --mic --out rl/eval/demo_mic.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "rl", ROOT / "tools", ROOT / "render", ROOT / "grasp"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

BOX_LABEL = {None: "未落箱", "target_box": "箱 A（左）", "target_box_b": "箱 B（右）"}
# 指令下标 → 期望落点（与 rl/eval_lang.py 的口径一致：lift 指令 #0 对应左箱 target_box）
EXPECT_BOX = {0: "target_box", 1: "target_box_b"}


# ---------------------------------------------------------------- 指令解析
def task_order_from_ds(ds: Path | None) -> list[str]:
    if ds is None:
        return ["left", "right"]
    prov = Path(ds) / "meta" / "sim_provenance.json"
    if prov.exists():
        return list(json.loads(prov.read_text(encoding="utf-8")).get("task_order") or ["left", "right"])
    return ["left", "right"]


def pick_index(text: str, order: list[str]) -> int:
    """按关键词把自由文本映射到 task_order 的下标（left/右/right/左）。"""
    t = text.lower()
    for i, name in enumerate(order):
        key = name.lower().split(".")[0]
        if "left" in key and ("left" in t or "左" in t):
            return i
        if "right" in key and ("right" in t or "右" in t):
            return i
    raise SystemExit(f"认不出左右指令: {text!r}（task_order={order}）")


def listen_mic(seconds: float, order: list[str]) -> tuple[int, str]:
    """录音 → 转指令。优先 vosk 离线识别；没有就退化到能量/过零率兜底。"""
    try:
        import sounddevice as sd                                   # noqa: PLC0415
    except Exception as e:                                          # noqa: BLE001
        raise SystemExit(
            "麦克风需要 sounddevice：先 `pip install sounddevice`（可选 `pip install vosk` 用离线识别）。\n"
            f"  当前环境缺依赖：{e}"
        )
    rate = 16000
    print(f"🎤 请说指令（{seconds}s，例如 \"put it in the left box\" 或 \"放左边\"）...")
    audio = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
    sd.wait()
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt((audio ** 2).mean()))
    print(f"   录到 {len(audio)/rate:.1f}s，RMS={rms:.4f}")
    if rms < 1e-3:
        raise SystemExit("几乎没收到声音（RMS<0.001）：检查麦克风设备/静音键。")

    try:                                                            # ① vosk 离线识别
        import vosk                                                 # noqa: PLC0415

        model_dir = ROOT / "rl" / "vosk-model-small-en-us"
        if not model_dir.exists():
            print("   ⚠ 没找到 rl/vosk-model-small-en-us，跳过 vosk，用兜底判定")
        else:
            rec = vosk.KaldiRecognizer(vosk.Model(str(model_dir)), rate)
            rec.AcceptWaveform((audio * 32767).astype(np.int16).tobytes())
            text = json.loads(rec.FinalResult()).get("text", "")
            print(f"   vosk 识别: {text!r}")
            if "left" in text or "right" in text:
                return pick_index(text, order), text
            print("   识别文本里没有 left/right，转兜底")
    except ImportError:
        print("   （没装 vosk，用兜底判定）")

    # ② 兜底：说话里的左右通常落在不同的共振峰/能量包络，这里只用最粗的两个判别特征，
    #    准确率不高，但绝不会静默瞎猜 —— 会打印得分让你确认。
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    freqs = np.fft.rfftfreq(len(audio), 1 / rate)
    centroid = float((spec * freqs).sum() / (spec.sum() + 1e-9))
    print(f"   兜底特征：谱质心 {centroid:.0f} Hz（左/右各录一次做标定更靠谱）")
    raise SystemExit(
        "兜底判定不可靠，已停止。请用 --instr left|right，或装 vosk 做离线识别：\n"
        "  pip install vosk sounddevice && 下载 small 模型解压到 rl/vosk-model-small-en-us"
    )


# ---------------------------------------------------------------- 写视频
class MP4Sink:
    """pyav 直写 h264；尺寸恒定（沿用本仓库"尺寸变了就静默丢帧"的教训，这里直接断言）。"""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        import av                                                  # noqa: PLC0415

        self.size = size
        path.parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), mode="w")
        self.s = self.c.add_stream("libx264", rate=int(fps))
        self.s.width, self.s.height = size
        self.s.pix_fmt = "yuv420p"
        self.n = 0

    def write(self, rgb: np.ndarray):
        import av                                                  # noqa: PLC0415

        h, w = rgb.shape[:2]
        assert (w, h) == self.size, f"帧尺寸 {w}x{h} ≠ {self.size}（会静默丢帧）"
        for pkt in self.s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")):
            self.c.mux(pkt)
        self.n += 1

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def banner(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    import cv2                                                     # noqa: PLC0415

    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, color, 2, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="明确 ckpt 目录（checkpoints/005000/pretrained_model）")
    ap.add_argument("--ds", default=str(ROOT / "rl" / "lang_ds" / "v2_pair"),
                    help="数据集（读 task_order；必须与训练一致）")
    ap.add_argument("--instr", choices=("left", "right"), default=None)
    ap.add_argument("--text", default=None, help="自由文本指令（关键词判定左右）")
    ap.add_argument("--mic", action="store_true", help="用麦克风说话给指令")
    ap.add_argument("--mic-seconds", type=float, default=3.0)
    ap.add_argument("--episode", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "demo_lang.mp4"))
    ap.add_argument("--fps", type=float, default=15.0, help="出片帧率（每 1/fps 秒抓一帧）")
    ap.add_argument("--views", choices=("side", "env", "wrist"), default="side")
    a = ap.parse_args()

    order = task_order_from_ds(Path(a.ds) if a.ds else None)
    if a.instr:
        ix = order.index(next(t for t in order if a.instr in t.lower()))
        said = f"--instr {a.instr}"
    elif a.text:
        ix = pick_index(a.text, order)
        said = f'"{a.text}"'
    elif a.mic:
        ix, said = listen_mic(a.mic_seconds, order)
        said = f'🎤 "{said}"'
    else:
        raise SystemExit("三选一：--instr left|right / --text \"...\" / --mic")

    onehot = np.zeros(len(order), np.float32)
    onehot[ix] = 1.0
    target = order[ix]
    print(f"指令 → #{ix} {target!r}（one-hot {onehot.tolist()}，来自 {said}）")

    from env_residual import ResidualSO101                          # noqa: PLC0415

    env = ResidualSO101(base_name=a.ckpt, episode_pool=(a.episode,),
                        max_seconds=a.max_seconds, judge="target",
                        instr_onehot=onehot, device="cpu", seed=a.seed)
    n_frames_target = int(round(a.max_seconds * env.control_hz))
    every = max(1, int(round(env.control_hz / a.fps)))
    W, H = 640, 480
    size = (W * 2, H) if a.views == "side" else (W, H)
    sink = MP4Sink(Path(a.out), a.fps, size)

    t0 = time.time()
    try:
        obs, info = env.reset(seed=a.seed)
        step = 0
        landed = None
        for k in range(n_frames_target):
            _, _, term, trunc, info = env.step(np.zeros(6, np.float64))   # 残差=0 → 纯基座策略
            step = k + 1
            if k % every == 0 or term or trunc:
                raw = env.rig.grab(env.d)
                env_img, wrist_img = raw["env_camera"], raw["wrist_camera"]
                if a.views == "side":
                    frame = np.concatenate([env_img, wrist_img], axis=1)
                else:
                    frame = env_img if a.views == "env" else wrist_img
                tag = f"instr=#{ix} {target[:28]}  step {step}"
                sink.write(banner(frame, tag, (120, 255, 120)))
            if term or trunc:
                break
        landed = info.get("landed_box") if isinstance(info, dict) else None
        if landed is None:
            from eval_lang import classify_box                          # noqa: PLC0415

            pos = np.asarray(env.d.qpos[env.arm.oadr:env.arm.oadr + 3], float)
            landed = classify_box(pos)
    finally:
        env.close()
        sink.close()

    expect = EXPECT_BOX.get(ix)
    ok = landed == expect
    print(f"\n结果: 指令 #{ix}={target!r} → 期望 {BOX_LABEL.get(expect, expect)}，"
          f"实际落在 {BOX_LABEL.get(landed, landed)}"
          f" → {'✅ 跟随指令' if ok else '❌ 未跟随/未落箱'}")
    print(f"帧数 {sink.n}，用时 {time.time()-t0:.1f}s，视频 {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())