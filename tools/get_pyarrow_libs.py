#!/usr/bin/env python3
"""给系统 Python 配一个可用的 pyarrow（不改系统环境）：下载官方 cp310 manylinux wheel 解压到 .cache/pylibs。

为什么需要：本机 /usr/bin/python3 没有 pyarrow；两个 conda 环境（lerobot / lerobot312）一启动就崩；
pip 又因为 /usr/lib/python3.10 下有损坏的 .pyc 报 `bad marshal data`。所以直接抓 wheel 解压，
用 PYTHONPATH 挂上去即可（pyarrow wheel 自带 libarrow，不需要编译）。

    env -u PYTHONHOME -u PYTHONPATH /usr/bin/python3 tools/get_pyarrow_libs.py
    PYTHONPATH=$PWD/.cache/pylibs env -u PYTHONHOME /usr/bin/python3 tools/fix_cam_labels.py

只有 tools/fix_cam_labels.py 改 episodes parquet 那一步需要它；扫描与导出不需要。
"""
import json, ssl, sys, urllib.request, zipfile
from pathlib import Path
ctx = ssl._create_unverified_context()
j = json.load(urllib.request.urlopen("https://pypi.org/pypi/pyarrow/json", timeout=30, context=ctx))
pick = None
for ver in sorted(j["releases"], key=lambda s: [int(x) for x in s.split(".") if x.isdigit()], reverse=True):
    for f in j["releases"][ver]:
        n = f["filename"]
        if "cp310" in n and "x86_64" in n and "linux" in n and n.endswith(".whl"):
            pick = (ver, f); break
    if pick: break
if not pick:
    sys.exit("没有 cp310 x86_64 wheel")
ver, f = pick
print("选中", ver, f["filename"], "%.1f MB" % (f["size"] / 1e6))
dest = Path(".cache/wheels"); dest.mkdir(parents=True, exist_ok=True)
wp = dest / f["filename"]
if not wp.exists():
    urllib.request.urlretrieve(f["url"], wp)
libs = Path(".cache/pylibs"); libs.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(wp) as z:
    z.extractall(libs)
print("解压到", libs.resolve())
