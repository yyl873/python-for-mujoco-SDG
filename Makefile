# SO-ARM101 · 语言条件 ACT —— 一条命令复现每个结论
#
# 设计原则：make 只做"读已有产物"和"一次性可跑完"的事，
# 不在这里偷偷跑 8 小时训练（训练命令写在 README 里，需要显式执行）。

SHELL := /bin/bash
RUN   := env -u PYTHONHOME -u PYTHONPATH
PY    := /usr/bin/python3            # 纯 numpy/标准库工具用它（dsh 脚本约定）
RL_PY := .venv-rl/bin/python         # 有 mujoco / torch / av / cv2 的环境
LP    := /home/yyl/miniforge3/envs/lerobot312/bin/python   # lerobot 环境（有 safetensors）

CKPT  ?= outputs/act_lang_pair/checkpoints/005000/pretrained_model
DS    ?= rl/lang_ds/v2_pair
EP    ?= 2400
SEED  ?= 1

.DEFAULT_GOAL := help

## help        显示所有目标
help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /' | column -t -s'|' 2>/dev/null || \
	 grep -E '^## ' $(MAKEFILE_LIST)

## tables      从 rl/eval/*.json 重新生成 docs/RESULTS.md（不重跑任何实验）
tables:
	$(RUN) $(PY) tools/make_tables.py

## check       仓库自检：结果表最新 / json 字段齐 / 脚本可编译 / 评测工具包测试通过
check:
	$(RUN) $(PY) tools/make_tables.py --check
	$(RUN) $(PY) -m py_compile tools/make_tables.py rl/probe_lang_weight.py rl/demo_lang_policy.py
	$(RUN) $(PY) embodied-eval-kit/tests/test_eeval.py
	@echo "✓ 语法、结果表、评测工具包自检全部通过"

## kit-demo    跑评测工具包的合成数据演示（零依赖，不需要 mujoco）
kit-demo:
	$(RUN) $(PY) embodied-eval-kit/examples/demo_synthetic.py

## weights     重算语言通道权重曲线 → rl/eval/lang_weight_curve.json（需要 outputs 里的 ckpt）
weights:
	$(RUN) $(RL_PY) - <<'EOF'
	import importlib.util as iu, json, glob
	from pathlib import Path
	spec = iu.spec_from_file_location('p', 'rl/probe_lang_weight.py')
	m = iu.module_from_spec(spec); spec.loader.exec_module(m)
	rows = []
	for d in sorted(glob.glob('outputs/act_lang_pair/checkpoints/[0-9]*'), key=lambda x: int(Path(x).name)):
	    r = m.probe(d); t = r.get('tensors') or {}
	    enc = t.get('model.encoder_robot_state_input_proj.weight')
	    vae = t.get('model.vae_encoder_robot_state_input_proj.weight')
	    if enc:
	        rows.append({"step": int(Path(d).name), "encoder": round(enc['per_col'], 5),
	                     "vae_encoder": round(vae['per_col'], 5) if vae else None})
	base = m.probe('/home/yyl/lerobot/outputs/train/act_sim156_lang8/pretrained_model').get('tensors') or {}
	out = {"tool": "rl/probe_lang_weight.py", "rows": rows,
	       "metric": "per_col = (lang_norm/sqrt2)/(other_norm/sqrt6)",
	       "baseline": {"encoder": round(base['model.encoder_robot_state_input_proj.weight']['per_col'], 5)}}
	Path('rl/eval/lang_weight_curve.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
	print(f"✓ {len(rows)} 个 ckpt 已写入 rl/eval/lang_weight_curve.json")
	EOF

## demo        跑一条演示（左指令，出 mp4）：make demo EP=2400 SEED=1
demo:
	$(RUN) MUJOCO_GL=egl $(RL_PY) rl/demo_lang_policy.py --ckpt $(CKPT) \
	  --instr left --episode $(EP) --seed $(SEED) --out rl/eval/demo_left.mp4 --no-play

## demo-right  同场景只说"右"（与 demo 配对，用来看语言是否真的改变落点）
demo-right:
	$(RUN) MUJOCO_GL=egl $(RL_PY) rl/demo_lang_policy.py --ckpt $(CKPT) \
	  --instr right --episode $(EP) --seed $(SEED) --out rl/eval/demo_right.mp4 --no-play

## demo-loop   三轮循环演示（同场景、左右自动交替；换成 --mic 就是说话版）
demo-loop:
	$(RUN) MUJOCO_GL=egl $(RL_PY) rl/demo_lang_policy.py --ckpt $(CKPT) \
	  --alternate --loop --rounds 3 --episode $(EP) --seed $(SEED) \
	  --out rl/eval/demo_loop.mp4 --no-play

## eval        重跑最终反事实评测（76 rollout，CPU 约 8~15 分钟）
eval:
	$(RUN) CKPT=$(PWD)/$(CKPT) DS=$(PWD)/$(DS) OUT=rl/eval/lang_pair_5000.json \
	  bash rl/run_lang_eval.sh

## train       训练（5000 步；CPU ≈7.7h / GPU ≈30min）——不会自动跑，需显式 make train
train:
	$(RUN) DS_ROOT=$(PWD)/$(DS) STEPS=5000 SAVE_FREQ=250 \
	  REPO_ID=local/v2_pair POLICY_ID=local/act_v2_pair \
	  OUT=$(PWD)/outputs/act_lang_pair bash rl/run_lang_train.sh

## audit       接触真实性审计（0.5 s/条，纯内存）
audit:
	$(RUN) MUJOCO_GL=egl $(RL_PY) tools/probe_contact_audit.py

## clean       只清缓存与演示中间物，不动真实结果
clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -f rl/eval/demo_loop_r*.mp4
	@echo "✓ 已清理（结果 json 与 ckpt 未动）"

.PHONY: help tables check kit-demo weights demo demo-right demo-loop eval train audit clean