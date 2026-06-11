# Notes for Claude (AI Assistant)

This directory vendors the **diffusion_policy** training workspace (formerly
named "cupid" here): `train.py`, `eval.py` and the `diffusion_policy` package,
matching the original upstream repo's layout.

Keep it that way: policy_doctor-specific code (attribution, eval-episode
saving, analysis, UI) lives in the main `policy_doctor` package — do not add
new top-level scripts or packages here. Attribution entry points are in
`policy_doctor/attribution/`; the influence/clustering data layer is in
`policy_doctor/behavior_graphs/`.

`data/` is an untracked symlink into bulk storage; relative
`data/...` paths embedded in old checkpoints resolve from this directory
(subprocess steps run with `cwd=DIFFUSION_POLICY_ROOT`).
