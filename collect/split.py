#!/usr/bin/env python3
"""
split.py —— 给 SDG 数据集切 train / val / test（可复现、按场景旋转留出）
========================================================================

为什么必须单独一步、而且必须现在做
----------------------------------
`index.json` 里只有"哪条是哪条"，没有任何划分。等到训练时再临时切，就会出现
"张三切一版、李四切一版"，指标互相不可比 —— 而这正是"同一份数据两套结论"的根源。
所以划分是**数据生产的产物**，不是训练的临时动作：

    scripts/splits/train.json  val.json  test.json   ← episode_id 列表（训练侧只读它）
    scripts/splits/manifest.json                     ← 策略/种子/各文件 sha256（防手改）
    scripts/index.json 的 "splits" 段                 ← 指向上面这份 manifest 的摘要

三种策略（--policy）
-------------------
  1) random       默认。按 success 分层（正/负样本在三份里的比例一致），
                  这样 val/test 的失败率不会因为切分抖动而漂移。
  2) theta-holdout 按【场景旋转 θ】留出：train/val 只含 |θ| ≤ T，test 只含 |θ| > T。
                  为什么值得单独支持：θ 是这里最强的泛化轴（同构解的极限就在 ±10°），
                  用中间段训练、边界段测试，测的是"外推到几何极限"的能力。
                  默认 T = 7°，即 test = 7° < |θ| ≤ 10°。
  3) mod          完全确定：val = (eid % round(1/val) == 0)，test = (eid % round(1/test) == 1)。
                  与随机种子无关，任何人任何机器算出来都一样 —— 适合做对照实验的固定划分。

可复现性
--------
* 同一份数据 + 同一 `--seed` + 同一策略 → **逐字节相同的三份文件**。
  （`created` 时间戳不进 sha256，只出现在 manifest 里，所以校验哈希是稳定的。）
* 划分只依赖 `episode_id` 与 `keys.theta_deg`（没有就退回读 per-episode json），
  不依赖文件遍历顺序。

用法
----
    cd /home/yyl/deepseek-harness
    env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py                      # 80/10/10 随机
    env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py --policy theta-holdout --theta 7
    env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py --policy mod
    env -u PYTHONHOME -u PYTHONPATH python3 collect/split.py --dry-run            # 只看数量不落盘
"""

import json
import argparse
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
SPLITS_DIR = SCRIPTS_DIR / "splits"


class SplitError(RuntimeError):
    pass


def _atomic_write(path, obj, indent=1):
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _sha(obj):
    """对"去掉时间戳"的规范 JSON 求 sha256 —— 重跑一次如果划分没变，哈希就不该变。"""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _theta_of(entry):
    """取场景旋转角：优先索引里的 keys.theta_deg，老数据退回读 per-episode json。"""
    k = entry.get("keys") or {}
    if k.get("theta_deg") is not None:
        return float(k["theta_deg"])
    jp = (entry.get("paths") or {}).get("json")
    if jp and (ROOT / jp).exists():
        try:
            m = json.loads((ROOT / jp).read_text(encoding="utf-8"))
            return float(m["randomization"]["spawn"]["theta_deg"])
        except Exception:                                    # noqa: BLE001
            return None
    return None


def load_entries(scripts=None):
    scripts = Path(scripts) if scripts else SCRIPTS_DIR
    idx_path = scripts / "index.json"
    if not idx_path.exists():
        raise SplitError(f"找不到 {idx_path}：先跑 collect/collect.py（或 --rebuild-index）")
    index = json.loads(idx_path.read_text(encoding="utf-8"))
    rows = list(index.get("accepted") or []) + list(index.get("rejected") or [])
    if not rows:
        raise SplitError("清单里一条样本都没有")
    items = []
    for e in rows:
        items.append(dict(episode_id=int(e["episode_id"]),
                          success=bool(e.get("success", e.get("accepted"))),
                          theta=_theta_of(e),
                          dataset_version=e.get("dataset_version", "legacy"),
                          paths=e.get("paths") or {}))
    items.sort(key=lambda x: x["episode_id"])
    return index, items


def check_consistency(items, allow_mixed=False):
    vers = sorted({str(x["dataset_version"]) for x in items})
    if len(vers) > 1 and not allow_mixed:
        raise SplitError(
            f"清单里混了多个 dataset_version: {vers}\n"
            "  混版本的数据不能切在一起（随机化/元数据结构都不同，指标不可比）。\n"
            "  要么 --reset 后重刷，要么确认后加 --allow-mixed。")
    return vers


def _stratified(items, val, test, seed):
    """按 success 分层随机划分。返回 {train,val,test} 的 episode_id 列表。"""
    rng = np.random.default_rng(int(seed))
    out = {"train": [], "val": [], "test": []}
    for cls in (True, False):
        group = [x["episode_id"] for x in items if x["success"] is cls]
        rng.shuffle(group)
        n = len(group)
        n_test = int(round(n * test))
        n_val = int(round(n * val))
        # 小样本兜底：只要还有剩余，就让 test / val 至少各有一条（否则 val 为空没法用）
        if n and test > 0 and n_test == 0 and n >= 3:
            n_test = 1
        if n and val > 0 and n_val == 0 and n - n_test >= 2:
            n_val = 1
        out["test"] += group[:n_test]
        out["val"] += group[n_test:n_test + n_val]
        out["train"] += group[n_test + n_val:]
    for k in out:
        out[k].sort()
    return out


def _mod_split(items, val, test):
    """
    完全确定的划分（与种子无关）：val 取 eid % k_val == 0，test 取 eid % k_test == 1。
    两类余数不同（余 0 vs 余 1），所以 val/test 比例相同也不会撞车。
    """
    k_val = max(2, int(round(1.0 / val))) if val > 0 else 0
    k_test = max(2, int(round(1.0 / test))) if test > 0 else 0
    out = {"train": [], "val": [], "test": []}
    for x in items:
        eid = x["episode_id"]
        if k_test and eid % k_test == 1:
            out["test"].append(eid)
        elif k_val and eid % k_val == 0:
            out["val"].append(eid)
        else:
            out["train"].append(eid)
    return out


def _theta_split(items, theta, val, seed):
    """按 θ 留出：test = |θ| > T；剩下的按原比例（val/(1-test)）随机分层再切。"""
    hi = [x for x in items if x["theta"] is not None and abs(x["theta"]) > theta]
    rest = [x for x in items if not (x["theta"] is not None and abs(x["theta"]) > theta)]
    if not hi:
        raise SplitError(f"没有 |θ| > {theta}° 的样本，theta-holdout 切不出 test（检查 --theta）")
    out = {"train": [], "val": [], "test": sorted(x["episode_id"] for x in hi)}
    if rest:
        # 在剩余的里按 val 比例切（test 已经留出，所以这里 test=0）
        part = _stratified(rest, val, 0.0, seed)
        out["train"] = part["train"]
        out["val"] = part["val"]
    return out


def build_split(index, items, policy, seed, val, test, theta, allow_mixed=False,
                scripts=None, verbose=True, write=True):
    scripts = Path(scripts) if scripts else SCRIPTS_DIR
    splits_dir = scripts / "splits"
    if index.get("mjcf_mixed") and not allow_mixed:
        raise SplitError(
            "这批样本来自不同版本的 MJCF（模型被改过）：mjcf_mixed=true。\n"
            "  物理/画面分布已经不是同一件事，不能切在同一份数据里（--allow-mixed 可强制放行）。")
    vers = check_consistency(items, allow_mixed)
    if policy == "random":
        parts = _stratified(items, val, test, seed)
    elif policy == "theta-holdout":
        parts = _theta_split(items, theta, val, seed)
    elif policy == "mod":
        parts = _mod_split(items, val, test)
    else:
        raise SplitError(f"未知策略 {policy}")
    if not parts["train"]:
        raise SplitError("train 切出来是空的（样本太少或 ratio 太大）")

    by_id = {x["episode_id"]: x for x in items}
    files, shas, counts = {}, {}, {}
    for name in ("train", "val", "test"):
        rows = [by_id[i] for i in parts[name]]
        payload = dict(
            split=name, policy=policy, seed=int(seed), theta_deg=theta,
            ratios=dict(val=val, test=test),
            dataset_version=(vers[0] if len(vers) == 1 else "mixed"),
            n=len(rows),
            n_success=int(sum(1 for r in rows if r["success"])),
            n_failed=int(sum(1 for r in rows if not r["success"])),
            episode_ids=[r["episode_id"] for r in rows],
            entries=[dict(episode_id=r["episode_id"], success=r["success"],
                          theta_deg=r["theta"], **{k: r["paths"].get(k) for k in
                                                  ("npz", "json", "video_env", "video_wrist")})
                     for r in rows],
        )
        shas[name] = _sha(payload)
        counts[name] = len(rows)
        payload["created"] = str(np.datetime64("now"))
        payload["content_sha256"] = shas[name]
        files[name] = (str((splits_dir / f"{name}.json").relative_to(ROOT)) if write
                       else None)
        if write:
            _atomic_write(splits_dir / f"{name}.json", payload)

    manifest = dict(
        policy=policy, seed=int(seed), theta_deg=theta, ratios=dict(val=val, test=test),
        dataset_version=(vers[0] if len(vers) == 1 else "mixed"),
        n=counts, files_sha256=shas,
        index_sha256=_sha({"ids": [x["episode_id"] for x in items]}),
        created=str(np.datetime64("now")),
        note="sha256 不含 created 字段：同一份数据 + 同一 seed + 同一策略必须哈希一致",
    )
    mpath = _atomic_write(splits_dir / "manifest.json", manifest) if write else None

    # 把摘要挂回 index.json（index.json 是生成物，只改这一处派生字段）
    idx_path = scripts / "index.json"
    index = dict(index)
    index["splits"] = {k: manifest[k] for k in
                       ("policy", "seed", "theta_deg", "ratios", "n", "created",
                        "files_sha256", "index_sha256")}
    if write:
        _atomic_write(idx_path, index)

    if verbose:
        tot = len(items)
        print(f"  策略      : {policy}" + (f"（θ 留出 ±>{theta}°）" if policy == "theta-holdout"
                                          else f"  seed={seed}"))
        for name in ("train", "val", "test"):
            n = counts[name]
            ns = sum(1 for r in [by_id[i] for i in parts[name]] if r["success"])
            th = [r["theta"] for r in [by_id[i] for i in parts[name]] if r["theta"] is not None]
            ths = f"  θ∈[{min(th):+.1f}°,{max(th):+.1f}°]" if th else ""
            print(f"  {name:<9} : {n:>4} 条（{100*n/max(1,tot):>5.1f}%）  成功 {ns}"
                  f" / 失败 {n-ns}{ths}")
        if counts["val"] == 0 or counts["test"] == 0:
            print("  ⚠️ 有一份是空的：样本量太小，训练侧请自行处理（或用 --policy mod）")
        print(f"  产物      : {splits_dir.relative_to(ROOT)}/{{train,val,test}}.json + manifest.json")
        print(f"  sha256    : train={shas['train']} val={shas['val']} test={shas['test']}")
    return manifest, files


def main(argv=None):
    ap = argparse.ArgumentParser(description="切分 SDG 数据集（train/val/test）")
    ap.add_argument("--policy", default="random",
                    choices=("random", "theta-holdout", "mod"))
    ap.add_argument("--seed", type=int, default=0, help="随机划分的种子（默认 0，固定）")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--theta", type=float, default=7.0,
                    help="theta-holdout 的阈值（度）：test 只含 |θ|>该值")
    ap.add_argument("--allow-mixed", action="store_true",
                    help="允许跨 dataset_version 切（默认禁止）")
    ap.add_argument("--dry-run", action="store_true", help="只打印数量，不落盘")
    args = ap.parse_args(argv)

    print("=" * 78)
    print(" split  数据集切分")
    print("=" * 78)
    index, items = load_entries()
    print(f"  数据集    : {len(items)} 条（收 {sum(1 for x in items if x['success'])} / 丢 "
          f"{sum(1 for x in items if not x['success'])}）"
          f"  version={index.get('dataset_version')}")
    if args.dry_run:
        build_split(index, items, args.policy, args.seed, args.val,
                    args.test, args.theta, args.allow_mixed, verbose=True, write=False)
        print("  --dry-run : 未落盘")
        return 0
    build_split(index, items, args.policy, args.seed, args.val, args.test,
                args.theta, args.allow_mixed, verbose=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
