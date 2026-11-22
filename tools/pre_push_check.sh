#!/usr/bin/env bash
# 发布前自检：一条命令确认"这个仓库可以推到 GitHub 了"
#   bash tools/pre_push_check.sh
# 检查项：① 自检套件 ② 密钥泄漏 ③ 大文件 ④ README 链接 ⑤ 待推送提交与体积
set -u
cd "$(dirname "$0")/.." || exit 1
fail=0
hr() { printf '%s\n' "────────────────────────────────────────"; }

hr; echo "① 自检套件（结果表 / json 字段 / 语法 / 评测工具包测试）"
make check >/tmp/_ppc_check.log 2>&1 && tail -1 /tmp/_ppc_check.log || { fail=1; tail -8 /tmp/_ppc_check.log; }

hr; echo "② 密钥泄漏扫描（tracked 文件内容）"
hits=$(git grep -lIE "hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN [A-Z ]*PRIVATE KEY" -- . 2>/dev/null | head -5)
if [ -n "$hits" ]; then echo "  ✗ 可疑文件："; echo "$hits" | sed 's/^/    /'; fail=1; else echo "  ✓ 未发现"; fi

hr; echo "③ 入库大文件（>5 MB 会拖慢克隆）"
big=$(git ls-files -s | awk '{print $4}' | xargs -r du -m 2>/dev/null | awk '$1>5{print "  ⚠ "$1" MB "$2}')
if [ -n "$big" ]; then echo "$big"; else echo "  ✓ 无（最大 2.6 MB，网格 STL）"; fi

hr; echo "④ README 内部链接"
env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 - <<'PY'
import pathlib, re, sys
s = pathlib.Path('README.md').read_text(encoding='utf-8')
links = sorted(set(re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', s)))
miss = [l for l in links if not l.startswith('http') and not pathlib.Path(l).exists()]
print(f"  {'✓' if not miss else '✗'} {len(links)} 个链接，缺失 {len(miss)}" + (f": {miss}" if miss else ""))
sys.exit(1 if miss else 0)
PY
[ $? -ne 0 ] && fail=1

hr; echo "⑤ 待推送状态"
echo "  分支     : $(git branch --show-current)"
echo "  远端     : $(git remote get-url origin 2>/dev/null || echo '未配置')"
echo "  本地提交 : $(git rev-list --count HEAD)"
echo "  未提交改动: $(git status --porcelain | wc -l) 项"
echo "  仓库体积 : $(du -sh .git | cut -f1)"
if [ -n "$(git status --porcelain)" ]; then echo "  ⚠ 有未提交改动，推送前先 git add -A && git commit"; fail=1; fi

hr
if [ "$fail" -eq 0 ]; then
  echo "✓ 可以推送。在你的终端执行（沙箱里没有你的 GitHub 凭据）："
  echo "    git push origin main"
  echo "  若代理 502：重试 2~3 次，或换代理/直连后再推"
else
  echo "✗ 有检查项未通过，先修上面的问题"
fi
rm -f /tmp/_ppc_check.log
exit "$fail"
