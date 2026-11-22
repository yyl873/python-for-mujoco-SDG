"""纯函数单测：`eeval` 的判据/回执 + 仓库里能安全 import 的纯标准库模块。

能进来测的模块必须**顶层不 import mujoco / torch / lerobot / cv2 / numpy**：

- `eeval.*`                     纯标准库（统计 / 判据 / 反事实聚合 / 回执）
- `rl/clamp_analysis.py`        纯标准库（自己又写了一份 wilson/mcnemar → 正好交叉验证）
- `rl/audit_hack.py`            纯标准库（奖励作弊审计的 summarize）
- `rl/widen_state.py`           torch 只在函数体内 import，模块本身可安全 import（只测 patch_config）
- `tools/digest.py`             纯标准库（human / grep_file）

明确**不测**（顶层 import 了重依赖，见回执）：`ik/ik.py`（mujoco）、
`collect/split.py`、`collect/stats.py`（numpy）。
"""
import importlib
import importlib.util

import pytest

import eeval


def load_tool_module(path):
    """按文件路径加载没有 __init__.py 的脚本（tools/ 下都是这种）。"""
    spec = importlib.util.spec_from_file_location(path.stem + "_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ eeval 判据

def test_judge_episode_accepts_a_clean_lift():
    ep = eeval.Episode(lift_mm=[0, 10, 55, 60, 62, 61, 60, 58],
                       tilt_deg=[0, 1, 2, 2, 3, 2, 2, 1],
                       penetration_mm=[0, 0, 0.1, 0.2, 0.1, 0, 0, 0.1],
                       landed_box="target_box")
    out = eeval.judge_episode(ep, eeval.CleanCfg(require_release=True))
    assert out["ok"] is True
    assert out["reasons"] == []
    assert out["consecutive_ok"] is True
    # max 字段是"取整后的最大值"，量纲与输入一致
    assert abs(out["max_lift_mm"] - 62.0) < 1e-6
    assert abs(out["max_tilt_deg"] - 3.0) < 1e-6


def test_judge_episode_lists_every_failing_reason():
    """失败要能按类型计数：一次列出所有不满足的条件，而不是只报第一条。

    只数条数与结构，不锁死 reasons 的具体措辞（eeval 0.1.0→0.2.0 正在改文案）。
    """
    ep = eeval.Episode(lift_mm=[0, 10, 20, 10, 5],        # 从没到 50mm，也凑不出连续 5 步
                       tilt_deg=[0, 2, 9.5, 2, 1],        # 超过默认 5°
                       penetration_mm=[0, 0.1, 0.6, 0, 0],  # 超过默认 0.5mm
                       landed_box=None)
    out = eeval.judge_episode(ep)  # 默认 CleanCfg
    assert out["ok"] is False
    assert out["consecutive_ok"] is False
    # 4 条都该报：抬升不够 / 连续不足 / 太歪 / 穿透
    assert len(out["reasons"]) == 4, out["reasons"]
    assert all(isinstance(r, str) and r.strip() for r in out["reasons"])
    # 每个失败维度都要有自己的措辞，不能四条一模一样（否则没法按类型计数）
    assert len(set(out["reasons"])) == 4, out["reasons"]


def test_judge_episode_require_release_toggle():
    """require_release 一开，同样的轨迹就因为"没落箱"多一条失败原因。"""
    ep = eeval.Episode(lift_mm=[60] * 6, tilt_deg=[0] * 6, penetration_mm=[0] * 6,
                       landed_box=None)
    lenient = eeval.judge_episode(ep, eeval.CleanCfg(require_release=False))
    strict = eeval.judge_episode(ep, eeval.CleanCfg(require_release=True))
    assert lenient["ok"] is True
    assert strict["ok"] is False
    assert len(strict["reasons"]) == len(lenient["reasons"]) + 1

    ep.landed_box = "target_box_b"
    assert eeval.judge_episode(ep, eeval.CleanCfg(require_release=True))["ok"] is True


def test_judge_episode_consecutive_lift_is_actually_consecutive():
    """抬升够了但中间断掉（"瞬间弹起"）不算洁净成功。"""
    ep = eeval.Episode(lift_mm=[60, 60, 10, 60, 60], tilt_deg=[0] * 5, penetration_mm=[0] * 5)
    out = eeval.judge_episode(ep)
    assert out["consecutive_ok"] is False        # 最长连续只有 2 步 < 默认 5
    assert out["ok"] is False


# ------------------------------------------------------------- eeval 三态回执

def test_receipt_exit_code_and_worst():
    r = eeval.Receipt().add("数据", "ok", n=76).add("仿真", "warn", clamps=3)
    assert r.worst() == "warn"
    assert eeval.exit_code(r) == 1               # warn 不许被压成 ok
    assert eeval.exit_code(eeval.Receipt().add("x", "ok")) == 0
    assert eeval.exit_code(eeval.Receipt().add("x", "fail")) == 2
    assert eeval.exit_code(eeval.Receipt()) == 0  # 空回执按 ok 处理


def test_receipt_rejects_unknown_status():
    with pytest.raises(ValueError):
        eeval.Stage(name="x", status="green")


def test_receipt_renders_facts_into_json_and_markdown():
    r = eeval.Receipt().add("统计", "warn", p=0.0094, n=12)
    js = r.to_json()
    md = r.to_markdown()
    assert '"overall": "warn"' in js
    assert "统计" in md and "**warn**" in md
    assert "p=0.0094" in md and "n=12" in md


# ------------------- 交叉验证：仓库里手写的 wilson/mcnemar 与 eeval 必须一致
# 注意 n=0 时两边口径**不同**（rl/clamp_analysis.wilson 返回 (0.0, 0.0)，
# eeval.wilson 返回 (0.0, 1.0)），所以这里只比 n>0 的取值，离散口径不去强求。

@pytest.mark.parametrize("k,n", [(40, 40), (31, 34), (22, 39), (1, 12), (8, 12), (0, 40)])
def test_clamp_analysis_wilson_matches_eeval(repo_root, k, n):
    ca = importlib.import_module("rl.clamp_analysis")
    a = ca.wilson(k, n)
    b = eeval.wilson(k, n)
    assert abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9, f"wilson({k},{n}) 不一致：{a} vs {b}"


@pytest.mark.parametrize("b,c", [(3, 0), (5, 2), (0, 0), (9, 1)])
def test_clamp_analysis_mcnemar_matches_eeval(repo_root, b, c):
    ca = importlib.import_module("rl.clamp_analysis")
    assert abs(ca.mcnemar_exact(b, c) - eeval.mcnemar_exact(b, c)) < 1e-12


def test_clamp_analysis_pct_uses_wilson():
    ca = importlib.import_module("rl.clamp_analysis")
    # 40/40 的 Wilson 下界是 91% —— 这条把"别把 100% 当真"钉在测试里
    assert ca.pct(40, 40) == "40/40 = 100.0% [91,100]"
    assert ca.pct(22, 39).startswith("22/39 = 56.4%")


# ------------------------------------------------------- rl/audit_hack.summarize

def test_audit_hack_summarize_counts_dirty_success(repo_root):
    ah = importlib.import_module("rl.audit_hack")
    rows = [
        # 成功且洁净
        {"success": True, "clean_success": True, "G6c_pen_max_mm": 1.0,
         "end_reason": "ok", "steps": 300, "tilt_max_deg": 2},
        # 成功但脏（穿透越过 gate）→ 记成 dirty success
        {"success": True, "clean_success": True, "G6c_pen_max_mm": 30.0,
         "end_reason": "ok", "steps": 400, "tilt_max_deg": 4},
        # 失败且早退
        {"success": False, "clean_success": False, "G6c_pen_max_mm": 0.0,
         "end_reason": "fell", "steps": 100, "tilt_max_deg": 9},
        {"success": False, "clean_success": False, "G6c_pen_max_mm": 0.0,
         "end_reason": "timeout", "steps": 600, "tilt_max_deg": 1},
    ]
    out = ah.summarize(rows, pen_gate=20.8)
    assert out["n"] == 4
    assert out["n_success"] == 2
    assert abs(out["success_rate"] - 0.5) < 1e-9
    assert out["n_clean_success"] == 1
    assert abs(out["clean_success_rate"] - 0.25) < 1e-9
    assert out["clean_recorded"] is True
    assert abs(out["dirty_success_frac"] - 0.5) < 1e-9      # 2 个成功里 1 个脏
    assert abs(out["early_exit_frac"] - 0.25) < 1e-9        # "fell" 属于早退
    assert out["end_reasons"] == {"ok": 2, "fell": 1, "timeout": 1}
    assert abs(out["steps_mean"] - 350.0) < 1e-9
    assert abs(out["pen_max_mm"] - 30.0) < 1e-9
    assert abs(out["tilt_max_deg"] - 9.0) < 1e-9


def test_audit_hack_summarize_empty_is_zero_not_crash(repo_root):
    ah = importlib.import_module("rl.audit_hack")
    assert ah.summarize([]) == {"n": 0}


# ------------------------------------------------- rl/widen_state.patch_config

def test_widen_state_patch_config_widens_state_and_writes_names(repo_root, tmp_path):
    """语言 one-hot 加宽：config.json 里 observation.state 的形状改成 new_dim。"""
    ws = importlib.import_module("rl.widen_state")
    import json

    src = tmp_path / "config.json"
    src.write_text(json.dumps({
        "input_features": {"observation.state": {"dtype": "float32", "shape": [6]}},
        "output_features": {},
    }), encoding="utf-8")
    dst = tmp_path / "out.json"

    msg = ws.patch_config(src, dst, 8, ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"])
    assert "[6] -> [8]" in msg

    cfg = json.loads(dst.read_text(encoding="utf-8"))
    st = cfg["input_features"]["observation.state"]
    assert st["shape"] == [8]
    assert st["names"][-1] == "s8"
    # 其它字段不许被顺手改掉
    assert cfg["output_features"] == {}
    # 源文件不动（patch_config 只读 src）
    assert json.loads(src.read_text(encoding="utf-8"))["input_features"]["observation.state"]["shape"] == [6]


def test_widen_state_patch_config_without_names_keeps_old_names(repo_root, tmp_path):
    ws = importlib.import_module("rl.widen_state")
    import json

    src = tmp_path / "config.json"
    src.write_text(json.dumps({
        "input_features": {"observation.state": {"shape": [6], "names": ["a", "b", "c", "d", "e", "f"]}},
    }), encoding="utf-8")
    dst = tmp_path / "out.json"
    ws.patch_config(src, dst, 6, None)
    st = json.loads(dst.read_text(encoding="utf-8"))["input_features"]["observation.state"]
    assert st["shape"] == [6]
    assert st["names"] == ["a", "b", "c", "d", "e", "f"]


# --------------------------------------------------------- tools/digest 纯函数

def test_digest_human_formats_sizes(repo_root, find_script):
    d = load_tool_module(find_script("digest.py"))
    assert d.human(512) == "512 B"
    assert d.human(2048) == "2.0 KB"
    assert d.human(5 * 1024 * 1024) == "5.0 MB"


def test_digest_grep_file_reports_counts_not_text(find_script, tmp_path):
    d = load_tool_module(find_script("digest.py"))
    f = tmp_path / "log.txt"
    f.write_text("alpha\nbeta\nmujoco crash\ngamma\nmujoco again\n", encoding="utf-8")
    out = d.grep_file(f, "mujoco")
    assert out["count"] == 2
    assert out["first"] == [3, 5]
    assert out["last"] == [5]
    assert d.grep_file(f, "not-there-at-all")["count"] == 0

# ------------------------------------- tools/compare_release_ab.find_release
# 抓-放时序判定：先张开(>thr) → 合拢(<=5) = 合爪 → 再张开(>=thr) = 释放。
# 真机 A/B 里"释放失败"就是靠它定位的（docs/REAL_AB_RESULTS §10.4）。

def _grip_rows(reqs, t0=0.0):
    return [{"t": t0 + i, "req": r, "sent": r, "present": r,
             "shoulder_pan": 1.0, "shoulder_lift": 2.0, "elbow_flex": 3.0,
             "wrist_flex": 4.0, "wrist_roll": 5.0}
            for i, r in enumerate(reqs)]


def test_find_release_detects_grasp_and_release(repo_root, find_script):
    m = load_tool_module(find_script("compare_release_ab.py"))
    rows = _grip_rows([60, 60, 2, 3, 40, 55, 59])
    res = m.find_release(rows, thr=30)
    assert res["ok"] is True
    assert res["t_grasp"] == 2.0 and res["t_open"] == 4.0
    assert res["dt_open"] == 2.0
    assert res["peak"] == 59                      # 从释放点起的峰值
    assert res["pose"]["wrist_roll"] == 5.0       # 姿态取释放那一帧


def test_find_release_reports_why_when_release_fails(repo_root, find_script):
    m = load_tool_module(find_script("compare_release_ab.py"))
    # 合爪后再没张开到 thr —— 就是"放不开"那种失败
    stuck = m.find_release(_grip_rows([60, 2, 3, 4, 5]), thr=30)
    assert stuck["ok"] is False and "放" in stuck["why"]
    # 整段都没张开过
    never = m.find_release(_grip_rows([0, 1, 2]), thr=30)
    assert never["ok"] is False and "30" in never["why"]


# ------------------------------------------- rl/failure_analysis.load_rows

def test_failure_analysis_load_rows_indexes_by_episode(repo_root, tmp_path):
    fa = importlib.import_module("rl.failure_analysis")
    import json

    payload = {"detail": {"base+R": {"rows": [
        {"episode_id": 7, "seed": 0, "success": False, "end_reason": "fell"},
        {"episode_id": 8, "seed": 1, "success": True, "end_reason": "ok"},
    ]}, "base": {"rows": [{"episode_id": 99}]}}}
    f = tmp_path / "ab.json"
    f.write_text(json.dumps(payload), encoding="utf-8")

    rows = fa.load_rows(str(f))                    # 默认 arm="base+R"
    assert sorted(rows) == [7, 8]                  # 键是 int 的 episode_id，不是字符串
    assert rows[7]["end_reason"] == "fell"
    assert fa.load_rows(str(f), arm="base").keys() == {99}
    assert fa.load_rows(str(f), arm="不存在") == {}


def test_failure_analysis_mcnemar_matches_eeval(repo_root):
    """又一份手抄的 mcnemar —— 必须和 eeval 的结果一致，否则同仓两套结论会打架。"""
    fa = importlib.import_module("rl.failure_analysis")
    for b, c in [(3, 0), (5, 2), (0, 0), (9, 1), (12, 4)]:
        assert abs(fa.mcnemar_exact(b, c) - eeval.mcnemar_exact(b, c)) < 1e-12, (b, c)
