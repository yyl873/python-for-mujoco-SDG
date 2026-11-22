"""`docs/RESULTS.md` 必须能从 `rl/eval/*.json` 重新生成，且表不能落后于数据。

机制：`tools/make_tables.py` 负责生成，`--check` 负责校验（文件齐不齐、字段全不全、
`docs/RESULTS.md` 是否与当前 json 一致）。这里用 subprocess 真的跑一次 `--check`，
因为"表是最新的"这件事只有在子进程里跑真脚本才算证明。
"""
import importlib.util
import os
import subprocess
import sys

import pytest

PUBLISHED_PCT = {
    # 三份已发布评测在 docs/RESULTS.md 表 1 里的 follow_acc 写法（生成器格式）
    "lang_pair_5000.json": "100.0% (40/40)",
    "lang_pair_3000_interim.json": "91.2% (31/34)",
    "lang_counterfactual.json": "56.4% (22/39)",
}


def clean_env():
    """本机 PYTHONHOME/PYTHONPATH 被 AppImage 污染，子进程里必须摘掉。"""
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def test_make_tables_check_passes(repo_root, find_script):
    """`--check` rc=0：文件齐全 + 字段齐全 + 结果表是最新的。"""
    tool = find_script("make_tables.py")
    r = subprocess.run([sys.executable, str(tool), "--check"],
                       cwd=str(repo_root), env=clean_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=60)
    out = r.stdout.decode("utf-8", "replace")
    assert r.returncode == 0, f"--check 失败 rc={r.returncode}：\n{out}"
    assert "✓" in out, f"--check 没打印通过标记：\n{out}"


def test_make_tables_check_is_read_only(repo_root, find_script):
    """`--check` 只校验不写入：跑完 docs/RESULTS.md 的字节必须一模一样。"""
    doc = repo_root / "docs" / "RESULTS.md"
    before = doc.read_bytes()
    r = subprocess.run([sys.executable, str(find_script("make_tables.py")), "--check"],
                       cwd=str(repo_root), env=clean_env(),
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       timeout=60)
    assert r.returncode == 0
    assert doc.read_bytes() == before, "--check 改了 docs/RESULTS.md（应只读）"


def load_make_tables(find_script, name="make_tables_under_test"):
    """按文件路径加载生成器，并**注册进 sys.modules**。

    只做 `spec_from_file_location` 不注册时，`from __future__ import annotations` 下的
    dataclass 会在解析字符串注解时找不到自己的模块而炸掉——这是 importlib 的用法要求，
    不是被测代码的问题。
    """
    tool = find_script("make_tables.py")
    spec = importlib.util.spec_from_file_location(name, tool)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_make_tables_source_runs_all_exist(repo_root, find_script):
    """生成器声明的数据源在仓库里都真实存在，否则表是空中楼阁。"""
    mod = load_make_tables(find_script, "make_tables_sources")

    assert mod.LANGUAGE_RUNS, "LANGUAGE_RUNS 为空"
    for run in mod.LANGUAGE_RUNS:
        assert (repo_root / "rl" / "eval" / run.filename).exists(), \
            f"{run.label} 的来源文件缺 {run.filename}"

    # 三份在本套件里被复算的产物必须都在生成器的清单里
    listed = {run.filename for run in mod.LANGUAGE_RUNS}
    for fn in PUBLISHED_PCT:
        assert fn in listed, f"{fn} 没被 make_tables 收录，结果表不会体现它"


def test_build_is_deterministic(repo_root, find_script):
    """重复生成必须逐字节一致（防止有人往生成器里塞时间戳/随机数）。"""
    mod = load_make_tables(find_script, "make_tables_determinism")
    assert mod.build_report() == mod.build_report()


def test_results_md_carries_the_published_numbers(repo_root):
    """表 1 里的三行必须是上面测试复算过的那些数字（同一套 json 生成的）。"""
    text = (repo_root / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    for fn, needle in PUBLISHED_PCT.items():
        assert needle in text, f"docs/RESULTS.md 里找不到 {fn} 的 {needle!r}（表可能落后于 json）"