"""pytest 全局设置。

两件事：
1. 把**仓库根目录**（为了 `import rl.xxx`）和 **embodied-eval-kit/**（零依赖统计包 `eeval`）放进 sys.path，
   这样测试不依赖任何 pip 安装、也不依赖当前工作目录。
2. 提供只读 fixture 读 `rl/eval/*.json` 里的真实评测产物。

设计约束（本机环境决定的）：
- 测试里**不** import mujoco / torch / lerobot / cv2 / numpy：本机 import mujoco 会 SIGFPE，
  torch 没有 GPU。需要仿真的用例走 `@pytest.mark.sim`（见 pytest.ini）。
- 测试只读仓库，不写任何被 git 跟踪的文件（tmp_path 除外）。
"""
import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "rl" / "eval"

# 顺序有意为之：embodied-eval-kit 在最前，避免同名的 tests/ 目录互相遮蔽。
for _p in (REPO_ROOT / "embodied-eval-kit", REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


@pytest.fixture(scope="session")
def repo_root():
    """仓库根目录（绝对路径）。"""
    return REPO_ROOT


@pytest.fixture(scope="session")
def eval_dir():
    """`rl/eval/`：真实评测产物所在目录。"""
    return EVAL_DIR


# 解析结果缓存在 session 里（每个文件 ~19 KB，解析一次足够）。
_CACHE = {}


@pytest.fixture(scope="session")
def find_script(repo_root):
    """按文件名定位仓库里的脚本，返回绝对路径。

    `tools/` 正在被重组（probe 类脚本陆续移进 `tools/probes/`），所以不把测试
    钉死在 `tools/xxx.py` 上：先看惯用位置，再全仓找，找不到就 fail 并说清原因。
    """
    def _find(filename):
        preferred = repo_root / "tools" / filename
        hits = [preferred] if preferred.is_file() else [
            p for p in repo_root.rglob(filename)
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts
        ]
        if not hits:
            pytest.fail(f"仓库里找不到 {filename}（tools/ 目录结构被改过？）")
        return sorted(hits, key=lambda p: len(p.parts))[0]

    return _find


@pytest.fixture(scope="session")
def artifacts():
    """只读 fixture：`artifacts("lang_pair_5000.json") -> dict`。

    返回深拷贝，测试怎么改都不会污染别的测试；文件缺失时 skip 而不是 fail
    （产物是本机重新生成出来的，clone 下来可能没有）。
    """
    def _get(name):
        path = EVAL_DIR / name
        if not path.exists():
            pytest.skip(f"评测产物不存在：{path.relative_to(REPO_ROOT)}（本机重跑评测后才有）")
        if name not in _CACHE:
            _CACHE[name] = json.loads(path.read_text(encoding="utf-8"))
        return copy.deepcopy(_CACHE[name])

    return _get