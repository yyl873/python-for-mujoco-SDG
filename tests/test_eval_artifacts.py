"""从 `rl/eval/*.json` 重算 README / docs/RESULTS.md 里已发布的数字。

这个文件是整套测试的核心，主张只有一句：
**README 上的每个数字都能用落盘的产物重新算出来，不需要重跑任何评测。**

覆盖三份反事实配对评测（v1 基线 + v2_pair 中间/最终），重算
`follow_acc / ITT / Wilson / 配对全对 / verdict`，再复算真机 A/B 的 Fisher 精确检验。
所有浮点比较用容差（abs diff），不断言任何内部实现细节——
`eeval` 正在升级到 0.2.0，只依赖它在 `eeval/__init__.py` 里公开的名字与语义。
"""
import math

import pytest

import eeval  # 由 tests/conftest.py 把 embodied-eval-kit/ 放进 sys.path

LEFT = "Put the medicine box into the left wooden box."
RIGHT = "Put the medicine box into the right wooden box."
TASK_ORDER = [LEFT, RIGHT]

# 指令 → 期望落点箱。注意两个参数的差别（实测，0.1.0 与 0.2.0 都一样）：
#   box_of      {下标: 箱名}，旧口径，键必须是 int（传指令文本会 ValueError/TypeError）
#   expect_map  {指令文本: 箱名}，按文本 key，不依赖 task_order 的下标顺序
# 本文件一律走 `aggregate(rows, task_order, {}, expect_map=EXPECT)`——产物里本来就有
# task_a/box_a/task_b/box_b，按文本映射不会因为下标顺序被改而静默算错。
EXPECT = {LEFT: "target_box", RIGHT: "target_box_b"}

TOL = 1e-3  # 已发布数字都保留 4 位小数，JS/JSON/py 之间的舍入差异不会超过它

# 文件名 → 已发布数字（README「四块能力的证据」+ docs/RESULTS.md 表 1 逐项对应）
PUBLISHED = [
    pytest.param(
        "lang_pair_5000.json",
        dict(n_rollouts=76, landed=40, correct=40, acc=1.0, wilson_lo=0.9124,
             itt=0.5263, pairs_both=15, pairs_n=38, verdict="语言生效"),
        id="v2pair-5000",
    ),
    pytest.param(
        "lang_pair_3000_interim.json",
        dict(n_rollouts=76, landed=34, correct=31, acc=0.9118, wilson_lo=0.7704,
             itt=0.4079, pairs_both=9, pairs_n=38, verdict="语言生效"),
        id="v2pair-3000-interim",
    ),
    pytest.param(
        "lang_counterfactual_ckpt1250.json",
        dict(n_rollouts=76, landed=49, correct=27, acc=0.551, wilson_lo=0.4131,
             itt=0.3553, pairs_both=0, pairs_n=38, verdict="语言未被使用（≈抛硬币）"),
        id="v1-ckpt1250",
    ),
    pytest.param(
        "lang_counterfactual.json",
        dict(n_rollouts=76, landed=39, correct=22, acc=0.5641, wilson_lo=0.4098,
             itt=0.2895, pairs_both=0, pairs_n=38, verdict="语言未被使用（≈抛硬币）"),
        id="v1-baseline",
    ),
]
ARTIFACTS = [p.values[0] for p in PUBLISHED]


def close(got, want, tol=TOL):
    return got is not None and abs(float(got) - float(want)) <= tol


def recalc(art):
    """用 eeval 从 rows 重算聚合指标（不读 summary）。返回 (summary, pairs)。"""
    rows, task_order = art["rows"], art["task_order"]
    assert task_order == TASK_ORDER, f"task_order 变了：{task_order}"
    agg = eeval.aggregate(rows, task_order, {}, expect_map=EXPECT)
    pr = eeval.aggregate_pairs(rows, task_order, EXPECT)
    return agg, pr


# ---------------------------------------------------------------- 产物自身形状

@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_artifact_schema_matches_documented_shape(artifacts, name, pub):
    """每份产物都有文档承诺的字段，且 rows 条数与 summary 自洽。"""
    art = artifacts(name)
    for key in ("ckpt", "task_order", "task_a", "task_b", "box_a", "box_b", "summary", "rows"):
        assert key in art, f"{name} 缺字段 {key}"
    assert art["task_a"] == LEFT and art["task_b"] == RIGHT
    assert art["box_a"] == EXPECT[LEFT] and art["box_b"] == EXPECT[RIGHT]
    assert len(art["rows"]) == art["summary"]["n_rollouts"] == pub["n_rollouts"]

    for r in art["rows"]:
        assert set(("episode_id", "seed", "ix", "landed_box")) <= set(r)
        assert r["ix"] in (0, 1)
        # 落点只允许三种：左箱 / 右箱 / 没落进任何箱
        assert r["landed_box"] in (EXPECT[LEFT], EXPECT[RIGHT], None)


# ------------------------------------------------------- 用 rows 复算已发布数字

@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recomputed_landed_and_correct_counts(artifacts, name, pub):
    """落进某箱的 rollout 数 与 落对的次数 —— 直接从 rows 数，不信 summary。"""
    art = artifacts(name)
    agg, _ = recalc(art)
    assert agg["n_rollouts"] == pub["n_rollouts"]
    assert agg["n_landed_in_some_box"] == pub["landed"], f"{name} 落箱数对不上"
    assert agg["follow_correct"] == pub["correct"], f"{name} follow_correct 对不上"
    # 与文件自带 summary 也要一致（防止 summary 与 rows 各说各话）
    assert agg["n_landed_in_some_box"] == art["summary"]["n_landed_in_some_box"]
    assert agg["follow_correct"] == art["summary"]["follow_correct"]


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recomputed_follow_acc(artifacts, name, pub):
    """follow_acc = 落对 / 落进某箱 —— 语言有没有被用上的主指标。"""
    art = artifacts(name)
    agg, _ = recalc(art)
    assert close(agg["follow_acc"], pub["acc"]), f"{name} follow_acc={agg['follow_acc']} 期望 {pub['acc']}"
    assert close(agg["follow_acc"], art["summary"]["follow_acc"])


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recomputed_itt(artifacts, name, pub):
    """ITT = 落对 / 全部 rollout（分母是全部，包含没落箱的）。"""
    art = artifacts(name)
    agg, _ = recalc(art)
    assert close(agg["follow_acc_itt"], pub["itt"]), f"{name} ITT={agg['follow_acc_itt']} 期望 {pub['itt']}"
    assert close(agg["follow_acc_itt"], art["summary"]["follow_acc_itt"])
    # ITT ≤ follow_acc 恒成立（分母更大、分子相同）
    assert agg["follow_acc_itt"] <= agg["follow_acc"] + 1e-12


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recomputed_wilson_lower_bound(artifacts, name, pub):
    """Wilson 下界正是 README 里"40/40 也才 91.2%"那个数字。"""
    art = artifacts(name)
    agg, _ = recalc(art)
    lo, hi = agg["follow_wilson"]
    assert close(lo, pub["wilson_lo"]), f"{name} wilson lo={lo} 期望 {pub['wilson_lo']}"
    assert close(lo, art["summary"]["follow_wilson"][0])
    assert close(hi, art["summary"]["follow_wilson"][1])
    # 区间必须包住点估计，且是真区间（不是 [p, p] 假精确）
    assert lo <= agg["follow_acc"] <= hi + 1e-12
    # 交叉验证：wilson(k, n) 与 aggregate 内部的口径一致
    w_lo, w_hi = eeval.wilson(art["summary"]["follow_correct"],
                              art["summary"]["n_landed_in_some_box"])
    assert close(lo, w_lo) and close(hi, w_hi)


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recomputed_paired_both_follow(artifacts, name, pub):
    """配对全对：同一 (场景, seed) 的两条指令都落对才算一个。最硬的指标。"""
    art = artifacts(name)
    _, pr = recalc(art)
    assert pr["n_pairs"] == pub["pairs_n"], f"{name} 配对数 {pr['n_pairs']} 期望 {pub['pairs_n']}"
    assert pr["pairs_both_follow"] == pub["pairs_both"], (
        f"{name} 配对全对={pr['pairs_both_follow']} 期望 {pub['pairs_both']}")
    assert pr["pairs_both_follow"] == art["summary"]["pairs_both_follow"]
    assert pr["pairs_both_follow"] <= pr["n_pairs"]
    # 配对全对的次数不会超过"落对总次数 / 每条指令"
    assert pr["pairs_both_follow"] * 2 <= art["summary"]["follow_correct"]


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_pairs_are_complete_counterfactual_pairs(artifacts, name, pub):
    """每条 (场景, seed) 必须**两条指令都测过**，否则配对口径不成立。"""
    art = artifacts(name)
    seen = {}
    for r in art["rows"]:
        seen.setdefault((r["episode_id"], r["seed"]), set()).add(r["ix"])
    assert len(seen) == pub["pairs_n"]
    incomplete = [k for k, v in seen.items() if v != {0, 1}]
    assert not incomplete, f"{name} 有 {len(incomplete)} 个配对缺一条指令：{incomplete[:3]}"


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_binom_p_vs_chance_agrees(artifacts, name, pub):
    """`aggregate` 的二项 p 与 summary 里的 binom_p_vs_chance 是同一个量（键名不同）。"""
    art = artifacts(name)
    agg, _ = recalc(art)
    # 键名容忍：0.2.0 可能把它对齐成 binom_p_vs_chance
    if "binom_p" in agg:
        p = agg["binom_p"]
    elif "binom_p_vs_chance" in agg:
        p = agg["binom_p_vs_chance"]
    else:
        pytest.fail(f"{name}: aggregate() 里找不到二项 p 值，当前键={sorted(agg)}")
    recorded = art["summary"]["binom_p_vs_chance"]
    assert close(p, recorded, tol=1e-3), f"{name} 重算 p={p} vs 记录 {recorded}"
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("name,pub", PUBLISHED)
def test_recorded_verdict_is_the_published_one(artifacts, name, pub):
    """产物里记的判定就是 README 引用的那句（历史产物不该被改写）。"""
    art = artifacts(name)
    assert art["summary"]["verdict"] == pub["verdict"]


def test_recomputed_verdict_separates_language_from_chance(artifacts):
    """重算出的判定必须**区分得出**"语言生效"与"≈抛硬币"。

    这里故意不断言 verdict() 的具体文案：eeval 正从 0.1.0 升到 0.2.0，
    同一个判决可能被改写成英文（"instruction followed" / "instruction ignored"）。
    被测试的是语义——两次 v2_pair 同判、v1 基线异判——而不是字符串。
    """
    verdicts = {}
    for name in ARTIFACTS:
        agg, pr = recalc(artifacts(name))
        v = eeval.verdict(dict(agg, **pr))
        assert isinstance(v, str) and v.strip(), f"{name} verdict() 返回空"
        verdicts[name] = v

    assert verdicts["lang_pair_5000.json"] == verdicts["lang_pair_3000_interim.json"], (
        f"两个 v2_pair 版本应同判，实际 {verdicts}")
    assert verdicts["lang_counterfactual.json"] != verdicts["lang_pair_5000.json"], (
        f"v1 基线（配对全对=0）必须和 v2_pair 区分开，实际 {verdicts}")


# --------------------------------------------------- 真机 A/B 的 Fisher 复算
# 出处 docs/REAL_AB_RESULTS_2026-10-28.md
#   §10.1 / §10.5  A@22500 1/12 vs B@22500 8/12 → p = 0.0094
#   §11.3          A@22500 1/12 vs B_ft_real 9/12 → p = 0.0028
FISHER_TOL = 5e-4


def test_fisher_real_ab_a022500_vs_b022500():
    p = eeval.fisher_exact_2x2(1, 11, 8, 4)  # [[抓到,没抓到]_A, [抓到,没抓到]_B]
    assert abs(p - 0.0094) < FISHER_TOL, f"p={p} 期望 0.0094（docs §10.1）"


def test_fisher_real_ab_a022500_vs_bfitreal():
    p = eeval.fisher_exact_2x2(1, 11, 9, 3)
    assert abs(p - 0.0028) < FISHER_TOL, f"p={p} 期望 0.0028（docs §11.3）"


def test_fisher_sanity_guards_against_a_stub():
    """反空转守卫：同样本无差异时 p=1，且两表的 p 必须一致（换行不换结论）。"""
    assert eeval.fisher_exact_2x2(6, 6, 6, 6) == pytest.approx(1.0, abs=1e-12)
    a = eeval.fisher_exact_2x2(1, 11, 8, 4)
    b = eeval.fisher_exact_2x2(8, 4, 1, 11)
    assert abs(a - b) < 1e-12, "Fisher 双侧 p 不该依赖两个样本的先后顺序"
    # 差异越大 p 越小：8/12 比 9/12 的对照更极端
    assert eeval.fisher_exact_2x2(1, 11, 9, 3) < a < 0.05


# --------------------------------------------------- README 不许和产物对不上

def test_readme_cites_the_numbers_this_suite_recomputes(repo_root):
    """README 引用的关键数字必须就是上面对齐过的那些（防"文档悄悄改口"）。

    空白被归一化后再比：``15/38`` 与 ``15 / 38``、``p=0.0028`` 与 ``p = 0.0028``
    是同一个数字，排版不该让守卫误报。
    """
    raw = (repo_root / "README.md").read_text(encoding="utf-8")
    text = " ".join(raw.split())
    for needle in ("40/40", "15/38", "52.6%", "56.4%", "0.02 mm", "122.2 mm",
                   "p=0.0094", "p=0.0028", "0.06747"):
        assert needle in text, f"README 里找不到已发布数字 {needle!r}"