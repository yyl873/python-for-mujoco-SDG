#!/usr/bin/env python3
"""rl/mix_dataset.py —— 把 DAgger 纠正集与母本数据集混合（lerobot v3 aggregate）

为什么需要单独一步
------------------
`rl/collect_dagger.py` 产出的是**纠正集**：帧数少（几千帧）但每条都落在基座真正会漂移的
状态上；`sim156`（`/home/yyl/lerobot_datasets/sim156`，70824 帧）是**母本集**：帧数多但只有
正轨分布。微调 ACT 要两者同时在场，否则：

  * 只用纠正集 → 模型忘掉正常轨迹（灾难性遗忘，ACT 的 chunk 预测直接崩）
  * 只用母本集 → 就是我们现在的基座，什么也没变

lerobot 的 `aggregate_datasets` 按 episode 拼接，训练采样是**按帧均匀**的，所以混比 ≈ 帧数比。
本脚本把它包一层，顺便：

  * `--dagger-repeat K`：把纠正集重复 K 份（等价于给纠正帧加权，不改 adam/steps）
  * 写出 `meta/mix_provenance.json`：记清混比、来源、帧数 —— 事后对账不用猜

★ 这一步不需要 GPU，纯文件搬运（视频会被 concatenate 复用，不重新编码时很快）。

用法
----
```bash
env -u PYTHONHOME -u PYTHONPATH .venv-rl/bin/python rl/mix_dataset.py \
    --base-ds /home/yyl/lerobot_datasets/sim156 --dagger rl/dagger_ds/dagger_v1 \
    --repeat 3 --out /home/yyl/lerobot_datasets/sim156_plus_dagger3
```
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE / "datasets"))


def _ffmpeg_bin() -> str:
    """找 ffmpeg（PATH 优先，其次 conda 环境里的那个）。"""
    f = shutil.which("ffmpeg")
    if f:
        return f
    for c in ("/home/yyl/miniforge3/envs/lerobot312/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if Path(c).exists():
            return c
    raise SystemExit("找不到 ffmpeg（重编码需要它）")


# lerobot 的 metadata 里 video.codec 是容器级名字，映射到 ffmpeg 编码器
_CODEC_MAP = {"av1": "libsvtav1", "h264": "libx264", "hevc": "libx265"}


def reencode_video(path: Path, vinfo: dict) -> None:
    """把一个 mp4 重编码成 vinfo 声明的编码参数（让 aggregate 的 remux 合法）。

    ★为什么必须重编码而不是改元数据：lerobot 的 `aggregate_datasets` 走
      `video_utils.concatenate_video_files`，那是 **demux + packet 直通 remux**，不做重编码；
      两个来源 codec 不同时要么兼容性检查直接报错，要么拼出来的流是坏帧。
    """
    import subprocess                                            # noqa: PLC0415

    codec = _CODEC_MAP.get(str(vinfo.get("video.codec") or "av1"), "libsvtav1")
    cmd = [_ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(path),
           "-c:v", codec, "-pix_fmt", str(vinfo.get("video.pix_fmt") or "yuv420p")]
    if vinfo.get("video.crf") is not None:
        cmd += ["-crf", str(vinfo["video.crf"])]
    if vinfo.get("video.preset") is not None:
        cmd += ["-preset", str(vinfo["video.preset"])]
    if vinfo.get("video.g") is not None:
        cmd += ["-g", str(vinfo["video.g"])]
    tmp = path.with_name(path.stem + ".rc.mp4")
    cmd += ["-an", str(tmp)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        raise SystemExit(f"ffmpeg 重编码失败 {path}: {r.stderr[-400:]}")
    tmp.replace(path)


def info_of(root: Path) -> dict:
    p = root / "meta" / "info.json"
    if not p.exists():
        raise SystemExit(f"不是 lerobot 数据集（缺 meta/info.json）: {root}")
    return json.loads(p.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DAgger 纠正集 + 母本集 -> 混合数据集",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--base-ds", required=True, help="母本数据集根（如 sim156）")
    ap.add_argument("--dagger", required=True, help="纠正集根（rl/collect_dagger.py 的 --out）")
    ap.add_argument("--repeat", type=int, default=1, help="纠正集重复份数（默认 1；加权用 2~3）")
    ap.add_argument("--out", required=True, help="混合数据集输出根（必须不存在或 --overwrite）")
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-reencode", action="store_true",
                    help="不把纠正集视频重编码到母本编码参数（默认会重编码）。"
                         "★lerobot 的 aggregate 是 **remux 不重编码**（video_utils."
                         "concatenate_video_files 的 packet 直通 + codec 兼容性检查），"
                         "所以两个来源的视频必须是同一 codec/crf/preset，否则拼出来的流是坏的。")
    a = ap.parse_args(argv)

    base = Path(a.base_ds).expanduser()
    dag = Path(a.dagger).expanduser()
    out = Path(a.out).expanduser()
    ib, id_ = info_of(base), info_of(dag)
    if ib["codebase_version"] != id_["codebase_version"]:
        raise SystemExit(f"codebase_version 不一致: {ib['codebase_version']} vs {id_['codebase_version']}")
    if list(ib["features"]) != list(id_["features"]):
        raise SystemExit("特征不一致，无法 aggregate（检查相机键/维度）")
    # 视频编码器参数口径：只比 shape/dtype/names，不计 info 里的 codec/crf/preset
    def _sans_codec(feats):
        out = {}
        for k, v in feats.items():
            v = dict(v)
            if isinstance(v.get("info"), dict):
                v["info"] = {kk: vv for kk, vv in v["info"].items()
                             if kk not in ("video.codec", "video.crf", "video.preset",
                                           "video.g", "video.fast_decode")}
            out[k] = v
        return out
    if _sans_codec(ib["features"]) != _sans_codec(id_["features"]):
        raise SystemExit("特征不一致（除编码器参数外），无法 aggregate")
    real_codec = (id_["features"].get("observation.images.env", {}).get("info") or {}).get("video.codec")
    if a.repeat < 1:
        raise SystemExit("--repeat 必须 >= 1")

    fb, fd = int(ib["total_frames"]), int(id_["total_frames"])
    total = fb + fd * a.repeat
    print("=" * 78)
    print(" mix_dataset  纠正集 + 母本集")
    print("=" * 78)
    print(f"  母本    : {base}  {ib['total_episodes']} 条 / {fb} 帧")
    print(f"  纠正集  : {dag}  {id_['total_episodes']} 条 / {fd} 帧  ×{a.repeat}")
    print(f"  混合后  : {total} 帧，纠正帧占比 {100.0 * fd * a.repeat / max(1, total):.1f}%")
    if fd * a.repeat < 0.02 * total:
        print("  ⚠️  纠正帧 < 2%：可能被母本淹没，考虑 --repeat 2~3")
    print(f"  输出    : {out}")
    print()

    if out.exists():
        if not a.overwrite:
            raise SystemExit(f"{out} 已存在（换路径或加 --overwrite）")
        shutil.rmtree(out)

    from lerobot.datasets.dataset_tools import aggregate_datasets        # noqa: PLC0415

    # ---- 编码器口径垫片：拷一份纠正集，只把 meta/info.json 的视频编码声明换成母本的 ----
    dag_used = dag
    base_vinfo = (ib["features"].get("observation.images.env") or {}).get("info") or {}
    if not a.no_reencode and real_codec != base_vinfo.get("video.codec"):
        shim = out.parent / f".{out.name}_shim"
        if shim.exists():
            shutil.rmtree(shim)
        shutil.copytree(dag, shim)
        vids = sorted(shim.rglob("*.mp4"))
        print(f"  重编码  : {len(vids)} 个视频 {real_codec} → {base_vinfo.get('video.codec')} "
              f"(crf {base_vinfo.get('video.crf')}, preset {base_vinfo.get('video.preset')})")
        t_rc = time.time()
        for i, v in enumerate(vids, 1):
            reencode_video(v, base_vinfo)
            if i % 20 == 0 or i == len(vids):
                print(f"    {i}/{len(vids)}  {time.time() - t_rc:.0f}s", flush=True)
        fi = json.loads((shim / "meta" / "info.json").read_text(encoding="utf-8"))
        for k in ("observation.images.env", "observation.images.wrist"):
            if k in fi["features"] and k in ib["features"]:
                fi["features"][k]["info"] = json.loads(json.dumps(ib["features"][k]["info"]))
        (shim / "meta" / "info.json").write_text(
            json.dumps(fi, ensure_ascii=False, indent=1), encoding="utf-8")
        dag_used = shim

    # aggregate 用 (repo_id, root) 定位来源；同一来源重复就是重复计入
    srcs = [x for x in ([str(dag_used)] * a.repeat) + [str(base)]]
    roots = [Path(x) for x in srcs]
    names = [f"local/{p.name}" for p in roots]
    aggr_repo = a.repo_id or f"local/{out.name}"
    t0 = time.time()
    aggregate_datasets(repo_ids=names, roots=roots, aggr_repo_id=aggr_repo,
                       aggr_root=out, concatenate_videos=True)
    io = info_of(out)

    prov = dict(
        generator="rl/mix_dataset.py",
        generated_at=str(np.datetime64("now")),
        base_dataset=str(base), base_episodes=int(ib["total_episodes"]), base_frames=fb,
        dagger_dataset=str(dag), dagger_episodes=int(id_["total_episodes"]),
        dagger_frames=fd, dagger_repeat=int(a.repeat),
        dagger_video_codec_real=real_codec,
        dagger_video_codec_used=str(dag_used != dag and base_vinfo.get("video.codec") or real_codec),
        total_frames=int(io["total_frames"]), total_episodes=int(io["total_episodes"]),
        correction_frac=round(fd * a.repeat / max(1, total), 4),
        note=("纠正帧来自 rl/collect_dagger.py（审计后的专家组合）；训练采样按帧均匀，"
              "所以纠正帧占比 ≈ 纠正数据的影响力。混比是超参，本文件记录实际值便于复现。"),
    )
    (out / "meta" / "mix_provenance.json").write_text(
        json.dumps(prov, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  ✅ 完成: {io['total_episodes']} 条 / {io['total_frames']} 帧  "
          f"纠正帧占比 {prov['correction_frac']:.1%}  耗时 {time.time() - t0:.0f}s")
    print(f"     溯源: {out / 'meta' / 'mix_provenance.json'}")
    if dag_used != dag:
        shutil.rmtree(dag_used, ignore_errors=True)
    print()
    print("  下一步（微调，需要 GPU）:")
    print(f"    /home/yyl/miniforge3/envs/lerobot312/bin/lerobot-train \\")
    print(f"      --dataset.repo_id={aggr_repo} --dataset.root={out} \\")
    print(f"      --policy.type=act \\")
    print(f"      --policy.pretrained_path=/home/yyl/lerobot/outputs/train/act_sim156_pre/checkpoints/last/pretrained_model \\")
    print(f"      --batch_size=8 --steps=10000 --save_freq=2500 \\")
    print(f"      --output_dir=/home/yyl/lerobot/outputs/train/act_dagger_ft --seed=1000 --num_workers=4")
    print()
    print("  然后评测（沙箱 CPU 也跑得动）:")
    print("    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl LP_NUM_THREADS=8 \\")
    print("        .venv-rl/bin/python rl/eval_ab.py \\")
    print("        --base /home/yyl/lerobot/outputs/train/act_dagger_ft/checkpoints/last/pretrained_model \\")
    print("        --episodes 100-129 --seeds 1 --judge strict \\")
    print("        --skip-base --base-from rl/eval/ab_base_heldout_strict.json \\")
    print("        --out rl/eval/ab_dagger_ft.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
