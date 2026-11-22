#!/usr/bin/env bash
# 把主仓 embodied-eval-kit/ 的最新改动同步到独立仓库 yyl873/embodied-eval-kit
#
# 背景：kit 平时在主仓里改（跟主仓一起测试），需要发布时再拆出去。
# 环境坑：全局配置里有 url.<ghproxy>.insteadOf 重写规则，且 gh 的凭据助手只认 github.com，
#         所以这里用 GIT_CONFIG_GLOBAL=/dev/null + gh 凭据助手直连。
set -eu
cd "$(dirname "$0")/.." || exit 1
BR=eeval-only
URL=https://github.com/yyl873/embodied-eval-kit.git

git subtree split -P embodied-eval-kit -b "$BR" >/dev/null
env GIT_CONFIG_GLOBAL=/dev/null GIT_TERMINAL_PROMPT=0 \
  git -c credential.helper='!gh auth git-credential' \
  push "$URL" "$BR":main
echo "✓ 已同步到 $URL （分支 $BR → main）"
