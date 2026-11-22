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
import os
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


def _read_wav(path: Path, rate: int = 16000) -> "np.ndarray":
    import wave                                                     # noqa: PLC0415

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == rate, f"采样率 {w.getframerate()} != {rate}"
        buf = w.readframes(w.getnframes())
    return (np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0)


def _record_cli(seconds: float, rate: int = 16000) -> "np.ndarray | None":
    """没有 PortAudio 时的零依赖录音：arecord / pw-record 二选一（都是系统自带常见命令）。

    背景：`pip install sounddevice` 只装 Python 绑定，运行期还要系统里有 libportaudio2；
    本机实测 `import sounddevice` 会抛 PortAudio library not found → 直接退到命令行录音，
    免得为了 3 秒录音去装系统包。
    """
    import shutil                                                   # noqa: PLC0415
    import subprocess                                               # noqa: PLC0415
    import tempfile                                                 # noqa: PLC0415

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "mic.wav"
        if shutil.which("arecord"):
            print("   （用 arecord 录音）")
            r = subprocess.run(["arecord", "-q", "-f", "S16_LE", "-r", str(rate),
                                "-c", "1", "-d", str(int(round(seconds))), "-t", "wav", str(wav)],
                               capture_output=True, text=True)
            if r.returncode == 0 and wav.exists() and wav.stat().st_size > 1024:
                return _read_wav(wav, rate)
            print(f"   arecord 失败(rc={r.returncode}): {r.stderr.strip()[:160]}")
        if shutil.which("pw-record"):
            print("   （用 pw-record 录音）")
            pr = subprocess.Popen(["pw-record", "--format", "s16", "--rate", str(rate),
                                   "--channels", "1", str(wav)],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(seconds)
            pr.terminate()
            try:
                pr.wait(timeout=5)
            except Exception:                                       # noqa: BLE001
                pr.kill()
            if wav.exists() and wav.stat().st_size > 1024:
                return _read_wav(wav, rate)
            print("   pw-record 失败")
    return None


def record(seconds: float, rate: int = 16000) -> "np.ndarray":
    """录音：sounddevice → arecord → pw-record，三路兜底；带电平自检。"""
    print(f"🎤 请说指令（{seconds}s，例如 \"放左边\" / \"left box\"）...", flush=True)
    audio = None
    try:
        import sounddevice as sd                                   # noqa: PLC0415

        buf = sd.rec(int(seconds * rate), samplerate=rate, channels=1, dtype="float32")
        sd.wait()
        audio = np.asarray(buf, dtype=np.float32).reshape(-1)
    except Exception as e:                                          # noqa: BLE001
        msg = str(e).splitlines()[0] if str(e) else type(e).__name__
        print(f"   sounddevice 不可用（{msg}）→ 退到命令行录音")
        audio = _record_cli(seconds, rate)
    if audio is None:
        raise SystemExit(
            "录音失败：三条后端都不通。任选其一修好即可：\n"
            "  A) sudo apt install libportaudio2        ← 让 sounddevice 能用（推荐）\n"
            "  B) sudo apt install alsa-utils           ← 提供 arecord\n"
            "  C) sudo apt install pipewire-utils       ← 提供 pw-record"
        )
    rms = float(np.sqrt((audio ** 2).mean()))
    peak = float(np.abs(audio).max())
    print(f"   录到 {len(audio)/rate:.1f}s，RMS={rms:.4f} peak={peak:.3f}")
    if rms < 1e-3:
        raise SystemExit(
            "几乎没收到声音（RMS<0.001）：检查麦克风设备、系统输入源（pavucontrol）、静音键。" )
    return audio


def mel_feat(audio: "np.ndarray", rate: int = 16000, n_mel: int = 24, n_ceps: int = 13) -> "np.ndarray":
    """MFCC 风格特征（numpy 手搓，不依赖 librosa）：log-mel → 每帧去均值 → DCT → 均值+标准差。

    为什么不是直接用 log-mel：log-mel 全为正、受音量主导，实测两个完全不同的音
    （220 Hz 与 1500 Hz）余弦相似度还有 0.958，做最近邻会分不开。
    每帧去均值（抹掉音量/增益）+ DCT 取低阶倒谱（只保留谱包络形状）后，
    同类相似度高、异类明显分开，才有资格做"声纹标定"。
    """
    win, hop = int(0.025 * rate), int(0.010 * rate)
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    n = 1 + (len(audio) - win) // hop
    frames = np.stack([audio[i * hop:i * hop + win] * np.hanning(win) for i in range(n)])
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / rate)

    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel2hz(np.linspace(hz2mel(80), hz2mel(rate / 2), n_mel + 2))
    fb = np.zeros((n_mel, len(freqs)))
    for m in range(n_mel):
        lo, ce, hi = edges[m], edges[m + 1], edges[m + 2]
        up = (freqs - lo) / max(1e-9, ce - lo)
        dn = (hi - freqs) / max(1e-9, hi - ce)
        fb[m] = np.clip(np.minimum(up, dn), 0, None)
    logmel = np.log(spec @ fb.T + 1e-8)                  # (frames, n_mel)
    logmel -= logmel.mean(1, keepdims=True)              # ★ 抹掉音量/增益
    m, k = np.arange(n_mel), np.arange(n_ceps)[:, None]
    dct = np.cos(np.pi / n_mel * (m + 0.5) * k)          # (n_ceps, n_mel)
    ceps = logmel @ dct.T                                 # (frames, n_ceps)
    feat = np.concatenate([ceps.mean(0), ceps.std(0)])    # 48 维
    return feat / (np.linalg.norm(feat) + 1e-9)


CALIB = ROOT / "rl" / "voice_calib.json"


def calibrate(order: list[str], seconds: float, reps: int = 3) -> dict:
    """录几次标定样本，存成每类一个质心（只要 sounddevice，不需要任何 ASR）。"""
    cents: dict[str, list[list[float]]] = {}
    for word in ("left", "right"):
        cents[word] = []
        for r in range(reps):
            print(f"\n标定 {word}  第 {r+1}/{reps} 次")
            cents[word].append(mel_feat(record(seconds)).tolist())
    data = {"order": order, "feat": "log-mel mean+std (24 mel)", "centroids": {
        w: np.mean(np.array(v), 0).tolist() for w, v in cents.items()}}
    CALIB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {CALIB}（同一个麦克风/同一个距离才有效）")
    return data


def listen_mic(seconds: float, order: list[str], save_wav: bool = False) -> tuple[int, str]:
    """录音 → 指令。后端顺序：faster-whisper → vosk → 已标定的声纹最近邻 → 报错。"""
    audio = record(seconds)
    if save_wav:
        out = ROOT / "rl" / "eval" / "last_mic.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        import wave                                                # noqa: PLC0415

        with wave.open(str(out), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes((audio * 32767).astype(np.int16).tobytes())
        print(f"   波形已存 {out}（识别不准时把它发我）")

    # ① faster-whisper：中英都认，离线，装一次就能用
    #    ★ 首次运行会从 HuggingFace 下模型（tiny≈75MB / small≈484MB）。国内建议走镜像：
    #      HF_ENDPOINT=https://hf-mirror.com（没设的话这里默认给上，不影响已下好的缓存）
    try:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from faster_whisper import WhisperModel                     # noqa: PLC0415

        size = os.environ.get("WHISPER_MODEL", "tiny")
        print(f"   用 faster-whisper({size}) 识别中...", flush=True)
        model = WhisperModel(size, device="cpu", compute_type="int8")
        segs, _ = model.transcribe(audio, language=None, beam_size=1)
        text = "".join(s.text for s in segs).strip()
        print(f"   识别: {text!r}")
        if "left" in text or "right" in text or "左" in text or "右" in text:
            return pick_index(text, order), text
        print("   文本里没有左右关键词，继续下一个后端")
    except ImportError:
        print("   （没装 faster-whisper；pip install faster-whisper 可换来真正的语音识别）")
    except Exception as e:                                          # noqa: BLE001
        print(f"   faster-whisper 出错: {e}")

    # ② vosk（模型目录用 --vosk-model 指定）
    try:
        import vosk                                                 # noqa: PLC0415

        md = Path(os.environ.get("VOSK_MODEL", str(ROOT / "rl" / "vosk-model-small-en-us")))
        if md.exists():
            rec = vosk.KaldiRecognizer(vosk.Model(str(md)), 16000)
            rec.AcceptWaveform((audio * 32767).astype(np.int16).tobytes())
            text = json.loads(rec.FinalResult()).get("text", "")
            print(f"   vosk 识别: {text!r}")
            if "left" in text or "right" in text:
                return pick_index(text, order), text
        else:
            print(f"   （vosk 模型目录不存在：{md}；VOSK_MODEL=<dir> 可指定）")
    except ImportError:
        print("   （没装 vosk）")

    # ③ 声纹标定最近邻：只要 sounddevice + 一次 --calibrate
    if CALIB.exists():
        data = json.loads(CALIB.read_text(encoding="utf-8"))
        f = mel_feat(audio)
        scores = {w: float(np.dot(f, np.asarray(c, float))) for w, c in data["centroids"].items()}
        best = max(scores, key=scores.get)
        margin = scores[best] - scores[min(scores, key=scores.get)]
        print(f"   声纹匹配: {scores} → {best}（margin {margin:.3f}）")
        if margin < 0.08:
            raise SystemExit(
                f"两个词得分太接近（margin {margin:.3f} < 0.08），不敢替你选。\n"
                "  请重跑 `--calibrate`（同一位置、同一音量各说 3 次），或直接用 --instr left|right。"
            )
        return order.index(next(t for t in order if best in t.lower())), f"voice-calib:{best}"

    raise SystemExit(
        "三条语音后端都不可用。二选一：\n"
        "  A) pip install faster-whisper sounddevice     ← 真语音识别（中英都认）\n"
        "  B) pip install sounddevice && --calibrate     ← 不用 ASR，录 6 次做声纹标定\n"
        "  或直接用 --instr left|right / --text \"放左边\""
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
    ap.add_argument("--calibrate", action="store_true",
                    help="录 6 次做声纹标定（不需要任何 ASR；之后 --mic 用最近邻判定）")
    ap.add_argument("--save-wav", action="store_true", help="把这次麦克风波形存 rl/eval/last_mic.wav")
    ap.add_argument("--live", action="store_true", help="开一个实时窗口，边说边看他动")
    ap.add_argument("--no-play", dest="play", action="store_false", help="跑完不要自动播放视频")
    ap.add_argument("--realtime", action="store_true", help="按真实时间步进（默认全速计算）")
    ap.add_argument("--episode", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--out", default=str(ROOT / "rl" / "eval" / "demo_lang.mp4"))
    ap.add_argument("--fps", type=float, default=15.0, help="出片帧率（每 1/fps 秒抓一帧）")
    ap.add_argument("--views", choices=("side", "env", "wrist"), default="side")
    a = ap.parse_args()

    order = task_order_from_ds(Path(a.ds) if a.ds else None)
    if a.calibrate:
        calibrate(order, a.mic_seconds)
        return 0
    if a.instr:
        ix = order.index(next(t for t in order if a.instr in t.lower()))
        said = f"--instr {a.instr}"
    elif a.text:
        ix = pick_index(a.text, order)
        said = f'"{a.text}"'
    elif a.mic:
        ix, said = listen_mic(a.mic_seconds, order, a.save_wav)
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
    live = a.live
    if live:
        import cv2                                                  # noqa: PLC0415

        cv2.namedWindow("lang-conditioned ACT", cv2.WINDOW_NORMAL)
        print("（实时窗口已开：按 q 或 ESC 可提前结束，视频照样会存）")
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
                shown = banner(frame, tag, (120, 255, 120))
                sink.write(shown)
                if live:
                    cv2.imshow("lang-conditioned ACT", shown[:, :, ::-1])
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        print("   手动结束")
                        break
                if a.realtime:
                    target_t = (k + 1) / env.control_hz
                    slack = target_t - (time.time() - t0)
                    if slack > 0:
                        time.sleep(slack)
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
        if live:
            import cv2                                              # noqa: PLC0415

            cv2.destroyAllWindows()

    expect = EXPECT_BOX.get(ix)
    ok = landed == expect
    print(f"\n结果: 指令 #{ix}={target!r} → 期望 {BOX_LABEL.get(expect, expect)}，"
          f"实际落在 {BOX_LABEL.get(landed, landed)}"
          f" → {'✅ 跟随指令' if ok else '❌ 未跟随/未落箱'}")
    print(f"帧数 {sink.n}，用时 {time.time()-t0:.1f}s，视频 {a.out}")
    if a.play and not live:
        import subprocess                                           # noqa: PLC0415

        try:
            subprocess.Popen(["xdg-open", str(a.out)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            print("（已调用 xdg-open 播放；没反应就手动打开上面的路径）")
        except Exception as e:                                      # noqa: BLE001
            print(f"（自动播放失败：{e}，手动打开 {a.out}）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())