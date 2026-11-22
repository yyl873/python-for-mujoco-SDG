# tools/probes/

One-off diagnostic scripts kept for reproducibility — they are **not** a stable API and nothing in the repo imports them.

| script | what question it answered | note |
| --- | --- | --- |
| `check_datasets_health.py` | Is a lerobot dataset (real or sim) self-consistent: structure, dims, NaN/Inf, index continuity, per-episode timestamp monotonicity, teleports, video decodability, gripper range, duplicate action fingerprints, sim-vs-real image statistics? | read-only, writes `logs/<name>.json` + terminal summary (exit 0=ok / 1=warn / 2=fail); needs `pyarrow` (lerobot312 env) |
| `clean_archived_sessions.py` | Which archived sessions and dangling cache entries in `~/.dsh` can be moved to trash or purged, without breaking `workspace.json` / `session_projcache.json`? | must be run by a human outside the agent sandbox; aborts while the harness is running unless `--force` |
| `dump_frames.py` | What do frames at given timestamps of an mp4 actually look like (gripper open/close, domain gap, visual anomalies)? | decode-only, writes PNGs to `inspect/frames/`; needs `av` + `PIL` |
| `eval_act_holdout.py` | Offline action L1@1/10/50/100 + RMSE per joint for an ACT checkpoint on the holdout set? | screening only, does not replace real success rate; invoked by `tools/run_eval_ab_cpu.sh`; needs torch |
| `export_real_env_frames.py` | Which camera stream is really the fixed environment camera, and what does it frame (20 real + 3 sim frames, montage, pad statistics)? | judges by content (`static%`), not folder name; writes `inspect/env_cam_20/` |
| `fix_cam_labels.py` | Can the swapped env/wrist camera labels be fixed consistently across `videos/`, `meta/info.json`, `meta/stats.json` and episodes parquet? | dry-run by default (`--apply` / `--revert`); the parquet step needs `pyarrow` |
| `get_pyarrow_libs.py` | How to give the system Python a usable `pyarrow` without pip and without touching the system environment? | downloads a cp310 manylinux wheel into `.cache/pylibs`; needs network |
| `probe_env_cam_jitter.py` | After `collect`'s env_camera jitter, how many frames still keep the green pad clipped by the frame border? | needs mujoco + `MUJOCO_GL=egl`; pure memory, writes nothing |
| `probe_motor_bus.py` | Which Feetech motors answer on the bus, and where is the daisy chain broken (`There is no status packet!`)? | needs the real arm: serial port + calibration file |
| `scan_env_framing.py` | Is the real env-camera framing the same in every episode (fixed camera aiming off) or only in some (repositioned mid-recording), and is the pad cropped? | `--unit file\|episode`; episode mode needs `pyarrow`; writes `inspect/env_framing/` |
| `scan_real_cams.py` | Which mp4 is the fixed environment camera and which is the wrist camera (static% and pad_cv criteria)? | writes `inspect/cam_scan/`; its CSV is the input of `fix_cam_labels.py` |
| `solve_env_camera.py` | Which `env_camera` position makes the sim reproduce the real `_fixed` framing (pad 270 px wide, right edge at x=639, top at 286 px)? | `--write` rewrites `model/so_arm101.xml`; needs `MUJOCO_GL=egl` |
