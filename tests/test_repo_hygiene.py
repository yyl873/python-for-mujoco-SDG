"""守卫性测试：仓库卫生 + "测试本身不许碰重依赖"。

这一组不是功能测试，是**防止仓库悄悄腐化**的哨兵：
- 有人提交一个 200 MB 的 ckpt → 这里红
- README 里的链接指向被删掉的文件 → 这里红
- docs/RESULTS.md 丢了表头 → 这里红
- 有人往测试里塞 `import mujoco`（本机会 SIGFPE）→ 这里红
"""
import ast
import os
import re
import subprocess

import pytest

MAX_TRACKED_BYTES = 5 * 1024 * 1024  # 5 MB：超过这个就不该进 git

# 测试里禁止在模块层 import 的重依赖（本机 mujoco SIGFPE、torch 无 GPU、numpy 被禁用）
HEAVY_DEPS = {"mujoco", "torch", "numpy", "cv2", "lerobot", "safetensors"}


def git_ls_files(repo_root):
    """`git ls-files -z`，返回相对路径列表。git 不可用时 skip。"""
    r = subprocess.run(["git", "ls-files", "-z"], cwd=str(repo_root),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        pytest.skip(f"git ls-files 失败：{r.stderr.decode('utf-8', 'replace')[:200]}")
    return [os.fsdecode(x) for x in r.stdout.split(b"\0") if x]


def test_no_tracked_file_over_5mb(repo_root):
    """被 git 跟踪的文件里不该有 > 5MB 的（大产物重生成即可，别塞仓库）。"""
    offenders = []
    for rel in git_ls_files(repo_root):
        if rel.split("/")[0] == ".git":
            continue
        p = repo_root / rel
        try:
            size = p.stat().st_size
        except OSError:
            continue  # 已删除但仍在 index 里（脏工作区），不算体积问题
        if size > MAX_TRACKED_BYTES:
            offenders.append((rel, size / 1024 / 1024))
    assert not offenders, "以下被跟踪文件 > 5MB：" + ", ".join(
        f"{rel} ({mb:.1f}MB)" for rel, mb in sorted(offenders))


def _is_tracked(path: str, tracked: set[str]) -> bool:
    """文件必须在索引里；目录只要"里面至少有一个被跟踪的文件"就算入库。"""
    if path in tracked:
        return True
    prefix = path.rstrip("/") + "/"
    return any(entry.startswith(prefix) for entry in tracked)


def test_readme_relative_links_point_to_existing_files(repo_root):
    """README 里所有相对路径的 markdown 链接/图片都必须真实存在**且已入库**。

    "本地存在但没 commit" 是真实踩过的坑：``.gitignore`` 里有一条 ``*.mp4`` 兜底规则，
    生成 ``docs/assets/real_robot_ab.mp4`` 后 ``git status`` 里根本看不到它，
    README 的链接在 GitHub 上就是死的——本地测试却全绿。所以这里同时查 git 索引。
    """
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(\s*<?([^)\s>]+)>?", readme)

    tracked = set(git_ls_files(repo_root))
    checked, missing, untracked = 0, [], []
    for t in targets:
        if t.startswith(("http://", "https://", "mailto:", "#")):
            continue  # 外链 / 邮箱 / 页内锚点不查
        path = t.split("#", 1)[0]
        if not path:
            continue
        checked += 1
        if not (repo_root / path).exists():
            missing.append(t)
        elif tracked and not _is_tracked(path, tracked):
            untracked.append(t)
    assert checked >= 5, f"README 里只解析到 {checked} 个相对链接，解析可能失效了"
    assert not missing, f"README 里有死链：{missing}"
    assert not untracked, (
        f"README 链接指向的文件没入库（本地能开、GitHub 上是死的）：{untracked}；"
        "先 git add 再跑测试，或检查 .gitignore 是否把它们挡住了"
    )


def test_results_md_exists_with_three_tables(repo_root):
    """docs/RESULTS.md 必须在，且三张表的表头都在（它是数字的唯一出处）。"""
    doc = repo_root / "docs" / "RESULTS.md"
    assert doc.exists(), "缺 docs/RESULTS.md（跑 tools/make_tables.py 生成）"
    text = doc.read_text(encoding="utf-8")
    for heading in ("## 表 1", "## 表 2", "## 表 3"):
        assert heading in text, f"docs/RESULTS.md 缺 {heading}"
    for header in ("| 模型 | 落点", "| 步数 |", "| 配置 |"):
        assert header in text, f"docs/RESULTS.md 缺表头 {header!r}"


def test_test_files_have_no_module_level_heavy_imports(repo_root):
    """测试自己在模块层不许 import mujoco/torch/numpy 等（跑了就崩/没 GPU）。

    只查 `Module.body` 里的 import：函数体内的延迟 import（见 sim 守卫用例）是允许的，
    因为它只有在显式开启时才执行。
    """
    violations = []
    for path in sorted((repo_root / "tests").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:  # 只看模块层，不进函数体
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            for n in names:
                if n.split(".")[0] in HEAVY_DEPS:
                    violations.append(f"{path.name}:{node.lineno} import {n}")
    assert not violations, "测试模块层出现了重依赖 import：" + "; ".join(violations)


def test_conftest_puts_eeval_on_sys_path(repo_root):
    """conftest 的 sys.path 设置真的生效（否则 eeval 在 CI 上找不到）。"""
    import sys
    from pathlib import Path

    assert str(repo_root / "embodied-eval-kit") in sys.path
    assert str(repo_root) in sys.path

    import eeval
    assert "embodied-eval-kit" in Path(eeval.__file__).parts