#!/bin/bash
cd /home/yyl/deepseek-harness

# 👇 加上这一行，自动创建 logs 文件夹（如果不存在的话）
mkdir -p logs

for start in 0 200 400 600 800
do
    echo "=================================================="
    echo "开始生成：--start $start --n 200"
    echo "=================================================="
    
    env -u PYTHONHOME -u PYTHONPATH MUJOCO_GL=egl /usr/bin/python3 \
        collect/collect.py --start $start --n 200 --jobs 4 --seed 3 \
        > logs/collect_${start}.log 2>&1
        
    echo "批次 $start 完成，日志见 logs/collect_${start}.log"
    sleep 5
done
