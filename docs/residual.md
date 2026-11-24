# Active study and reproducible commands

## What is trained

`lerobot/smolvla_robocasa` is already pretrained. Inference episodes do not train it.
Its weights remain frozen; TD3 learns six bounded corrections to action indices
5:11. Base motion, control mode, and gripper stay under SmolVLA control. The
default correction scale is 0.1. This limited authority constrains recoverable failures.

The default 59-value residual input contains 16 proprioceptive values, 12 pending
VLA actions, chunk position, and 30 ground-truth fridge geometry values. Legacy
29-input models require `--observation-mode proprio`; they cannot be loaded as
59-input models. Object state is privileged simulation data, not estimated vision.

Reward is 5 times signed normalized door-closing progress plus 10 for native
success, minus 0.05 times squared executed correction (the runner's existing
factory default). It offers little reaching guidance before door motion. Success
remains the upstream predicate requiring all relevant fridge doors closed.
Reopening is penalized; timeout is failure. The 900-step CloseFridge horizon
matches the local RoboCasa registry. Short smoke horizons are not task evaluations.

## Checkpoint and task evidence

Cached checkpoint revision: `f7af588d11e30b0069f200792fb285904b74c80b`.
Its train_config names `lerobot/robocasa_target_human_unified`. Dataset task
metadata at revision `8e9ca9d` was inspected directly on 2026-11-23:
https://huggingface.co/datasets/lerobot/robocasa_target_human_unified/tree/8e9ca9d/meta

The task strings are class-name labels, so `CloseFridge` is the correct default
language contract, not the environment's longer natural-language description.
`--instruction` is an explicit experimental override. Coverage includes
CloseFridge, SlideDishwasherRack, LoadDishwasher, and RecycleBottlesByType, but
does not include CloseDishwasher. Coverage does not establish competence.
Local source contains CloseDishwasher, but no trash-lid task was found.
Rack sliding and bottle sorting are approved candidate extensions, still requiring
task-specific observations, rewards, live validation, and native success checks.
Rack direction varies by episode; sorting requires correct clusters, table contact,
and gripper withdrawal. A fridge-angle reward must not be reused for these tasks.

## Run from the active PyCharm repository

```powershell
Set-Location 'C:\Users\Joshua\PycharmProjects\VLA-RL'
$python = 'C:\Users\Joshua\miniconda3\envs\vla-rl\python.exe'
& $python -m unittest discover -s tests -v
& $python scripts/train_residual.py --mode reward-check --output runs/reward-new
& $python scripts/train_residual.py --mode zero-check --steps 60 --horizon 900 --output runs/zero-live-new
& $python scripts/train_residual.py --mode zero-check --zero-policy-images recorded --steps 60 --horizon 900 --output runs/zero-recorded-new
```

The live zero-check requires exact images, actions, and observed dynamics. A
recorded-image check feeds the original trajectory images to the second policy
pass while still comparing live dynamics and recording live camera differences.
It isolates wrapper equivalence and MUST NOT be called end-to-end equivalence.
Both checks retain strict action, qpos, proprioception, sparse reward, and success
assertions. Camera mismatches save NPZ evidence. No pixel tolerance is assumed.

Reproducible repeated frozen baseline (same initialization, crossed policy seeds,
and identical-seed reruns):

```powershell
& $python scripts/eval_study.py --method base_vla --seeds 100 101 102 --policy-seeds 0 1 --repeats 2 --horizon 900 --save-video --output runs/base-new
```

Use a fresh output directory. JSONL is flushed after every episode. Reports record
initial-state and model hashes, instruction, policy seed, simulator seed, physics,
timeout, success, completion steps for successes only, final openness, and timing.
Source hashes, Git revisions, package versions, and checkpoint path are stored in
run_config.json. Repeated trials are not independent training runs.

## Training and shared evaluation

These are candidate training budgets, not validated learning-quality settings:

```powershell
& $python scripts/train_residual.py --mode train --perturbation nominal --seed 0 --policy-seed 10000 --train-seeds 200 201 202 203 --steps 9000 --horizon 900 --output runs/nominal-s0-new
& $python scripts/train_residual.py --mode train --perturbation phys_random_train --seed 0 --policy-seed 10000 --train-seeds 200 201 202 203 --steps 9000 --horizon 900 --output runs/random-s0-new
& $python scripts/eval_study.py --method residual_td3 --model-path runs/nominal-s0-new/td3.zip --seeds 100 101 102 --policy-seeds 0 1 --repeats 2 --horizon 900 --output runs/nominal-eval-new
& $python scripts/eval_study.py --method residual_td3 --model-path runs/random-s0-new/td3.zip --seeds 100 101 102 --policy-seeds 0 1 --repeats 2 --horizon 900 --output runs/random-eval-new
```

Repeat both training conditions with independent learner seeds 1 and 2 at equal
budgets. Do not fine-tune the randomized condition from a nominal model for the
main comparison. Reserve initialization seeds 100-102 for evaluation. Do not tune
against their outcomes. Report uncertainty across independent training runs.
SmolVLA now owns a private Torch RNG stream, so policy resets and inference do
not reset or consume TD3's Torch RNG. Training uses a separate policy seed sequence.
training_resets.jsonl records each actual reset and sampled physical parameters.

Physics randomization currently samples damping in [0.8, 1.8], added joint
friction loss in [0, 0.35], and sliding contact friction scale in [0.7, 1.4].
Contact changes now target the task fixture and gripper, excluding floor/wheels
and unrelated scene geometry. Contact pair combination can limit the effect of
lowering one surface's coefficient; physical feasibility and realized effects
still require live tests. Fixed held-out presets are exploratory, not a validated
severity study. Lighting/texture presets remain legacy code outside this study.

## Observed audit checks, 2026-11-23

- 31 unit/integration tests pass, including private RNG isolation and contact target selection.
- An initial eight-step zero-check passed. It was insufficient to establish long-run equivalence.
- Strict live-image 60-step check failed at step 13: three left-camera channel
  values differed by one intensity level; qpos, proprioception, and actions matched
  through that step. No relaxed tolerance was substituted.
- The recorded-image 60-step check passed exact actions, qpos, proprioception,
  rewards, and success across chunk boundaries. Live rendering differences remain.
- A separate native-instruction diagnostic also failed action equality and is
  not a checkpoint-matched baseline; do not pool it with class-label runs.

See runs/audit-zero-strict-label-20261123 and runs/audit-zero-recorded-20261123.
The full nominal-versus-randomized study, successful residual learning, validated
task extensions, severity feasibility sweeps, and causal rescue claims remain
unestablished. Identical initial hashes are necessary but do not remove rendering
variability. Paired outcome changes must be compared against repeated baselines.


## Reset fix and bounded training validation

The simulator-only audit found 69 geometry-color values changed between the
first and second seed-100 resets. Lazy construction consumed the global RNG
before the first episode. `seeded_reset` now initializes the lazy environment
inside a saved RNG context, then reseeds before the actual episode reset.
After the fix, qpos, qvel, and every inspected numeric model array matched:
`runs/audit-reset-fixed-20261123/report.json`.

The XML hash in episode records is a serialization checksum, not a complete
semantic equivalence test. The XML exporter can omit `content_type` on later
exports; use the numeric-model reset audit to diagnose such hash differences.

The pre-fix four-episode baseline had 0/4 successes (900 steps each), approximately
16 simulator steps/second including reset. Keep it separate from post-fix runs.
Within the matched later pre-fix pair (policy seed 1), final openness differed
by 0.0396 despite identical initial state and XML hashes. Rendering variability
can therefore matter for outcome-pair interpretation even after matching resets.

A randomized-physics integration run completed 128 steps and 96 optimizer
updates with the 59-value observation, finite replay rewards, changed actor
parameters, and frozen SmolVLA verified. It completed zero episodes, so it is
NOT evidence of learning quality. Its actual physics and independent seeds are
in `runs/audit-training-integration-20261123/training_resets.jsonl`.

Reproduce that bounded integration check with a fresh directory:

```powershell
& $python scripts/train_residual.py --mode train --steps 128 --learning-starts 32 --batch-size 16 --horizon 900 --perturbation phys_random_train --output runs/integration-new
& $python scripts/audit_resets.py --seed 100 --output runs/reset-audit-new
& $python scripts/summarize_audit.py runs/audit-base-fixed-20261123/eval_report.json --output runs/audit-base-fixed-20261123/summary
```

Post-fix frozen baseline (one simulator initialization, policy seed 0):

| Repeat | Success | Steps | Final mean openness |
| --- | --- | --- | --- |
| 0 | False | 900 | 0.690975 |
| 1 | False | 900 | 0.620590 |

Both videos and full episode records are under `runs/audit-base-fixed-20261123`.
This sample cannot estimate general task competence. Successful residual
learning, task extensions, multiple independent training runs, held-out severity
sweeps, feasibility validation, and rescue/harm analysis remain future work.
