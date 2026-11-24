# CloseFridge residual milestone

The active study compares frozen SmolVLA, nominal residual TD3, and physically
randomized residual TD3 under held-out lighting, texture, friction, and hinge
resistance changes. OpenFridge, OpenCabinet, and dish placement into a sink are
all required. CloseFridge is the first implementation task. This change does not
implement randomization or the remaining tasks, and does not revise the study
plan. The older tracking-fault documents/configuration are historical scaffolding.

## Implemented contract

- SmolVLA runs in eval/inference mode with gradients disabled on every parameter.
- The learner has six normalized actions, scaled by 0.1 by default, affecting
  only indices 5:11. Base motion, control mode, and gripper remain exact copies.
- Arm corrections are bounded and respect action limits when the base is within
  them. An already out-of-range VLA action remains unchanged at zero residual;
  corrections cannot extend its excursion or exceed the scale. There is no
  global clipping that could silently change the baseline.
- Learner observation: 16 robot proprioceptive values, 12 current pending VLA
  actions, and one normalized actual chunk position (29 float32 values).
- Simulator door state is read only by the reward provider. Images stay on the
  VLA path; neither images nor privileged state enter SB3 replay observations.
- The wrapper enforces its horizon and returns the next pending VLA action even
  on timeouts, without auto-resetting. SB3's normal VecEnv/replay path handles
  terminal observations and timeout bootstrapping. No standalone replay buffer.

Training reward is `5 * (previous_mean_openness - current_mean_openness)` plus
`10 * is_success`, minus `0.001 * squared_norm(executed_arm_correction)`.
Normalized fridge-only door state comes from the fixture's `get_joint_state`,
which handles different ranges and reversed hinge directions. Reopening gives
negative progress; no motion gives zero progress. Success remains the upstream
all-doors-closed predicate. Freezer doors are excluded. Reward components and the
original sparse reward are logged in step info. This shaping is not asserted to
preserve the optimal discounted policy; it also provides no reaching gradient
before door movement. Baseline competence remains unestablished (two saved
single-episode failures).

## Commands

Run in PowerShell from the project root using the existing GPU environment:

```powershell
$python = 'C:\Users\Joshua\miniconda3\envs\vla-rl\python.exe'
& $python -m pip install 'stable-baselines3==2.7.0'
& $python -m unittest discover -s tests -v
& $python scripts/train_residual.py --mode reward-check --output runs/reward-check
& $python scripts/train_residual.py --mode zero-check --steps 8 --horizon 8 --output runs/zero-check
& $python scripts/train_residual.py --mode train --steps 64 --horizon 32 --output runs/td3-smoke
```

The smoke runner uses CUDA for SmolVLA and TD3, a 10,000-transition CPU replay
buffer, batch size 8, and a small MLP. Its actor starts at zero correction; warmup
actions and exploration noise still produce bounded nonzero corrections. It
checks that optimizer updates occurred, actor parameters changed, SmolVLA stayed
frozen, and rewards were finite. It saves a TD3 model, monitor CSV, and report.
These short-run hyperparameters are integration settings, not study settings.
Each invocation should have its own output directory. No build step is required.
The validated environment uses SB3 2.7.0, Gymnasium 1.2.3, and Matplotlib 3.11.2.
Installing SB3 replaced Gymnasium 1.4.0 with the compatible 1.2.3 release.
The pre-existing RoboCasa 1.0.1 pin to LeRobot 0.3.3 conflicts with the installed
editable LeRobot 0.7.1. The local integration passes the checks below; neither
downloaded source checkout was replaced to resolve that metadata conflict.

Checkpoint loading is offline by default and uses the existing Hugging Face
cache. Supply `--checkpoint <snapshot-directory>` if needed. Download checkpoint
dependencies with the existing baseline launcher first if the cache is incomplete.
The camera map matches that launcher. `--project-root` supports validating a staged
copy against existing downloaded libraries without modifying their source files.

`zero-check` compares seeded direct frozen-policy and zero-residual trajectories:
pending and executed actions, qpos, proprioception, camera hashes, sparse reward,
and success. It fails on reset nondeterminism rather than silently relaxing the
equivalence assertion. `reward-check` temporarily sets actual fridge door joints
to open, half-open, and closed positions, checks normalization and the upstream
success predicate, stationary progress and reopening, and restores qpos. Run
these before longer training. Explicitly seeded resets seed Python and legacy
NumPy RNGs as well as RoboCasa's kitchen RNG; fixture construction also uses
global NumPy randomness. This fix is local to the runner, preserving the
downloaded library sources. Camera failures save expected/actual frames as NPZ
files and report the step, changed-value count, and maximum intensity difference.

## Validation results and limits

Validation on 2026-11-15 used the specified `vla-rl` Python environment:

- All 21 unit/integration tests passed, including SB3 replay timeout handling.
- Live reward validation passed on actual CloseFridge joints: normalized openness
  `[1, 0.5, 0]`, closure rewards `[2.5, 12.5]`, stationary progress reward `0`, and
  reopening reward `-2.5`. The upstream success predicate accepted closed doors
  and rejected half-open doors. See `runs/reward-check-seeded/report.json`.
- Zero-residual equivalence passed for 8 steps, including exact pending/executed
  actions, qpos, proprioception, all three camera hashes (reset plus every step),
  sparse reward, and success. See `runs/zero-check-seeded/report.json`. Initial
  attempts with only the kitchen RNG seeded failed on one-level camera intensity
  differences; exact comparisons were retained when fixing the reset seeding.
- CUDA TD3 smoke training passed: 64 transitions over two 32-step episodes and
  56 optimizer updates. The actor changed, rewards were finite, and every SmolVLA
  parameter remained frozen and unchanged, with no gradients. Artifacts are
  `runs/td3-smoke-seeded/report.json`, `monitor.csv`, and `td3.zip`.

These are integration checks, not evidence of task competence or training
convergence. No residual task-performance result is claimed. Numba compilation
caches remain under `runs/`. Existing downloaded libraries
and their pre-existing changes were preserved.

The SB3 API used here follows its [TD3 2.7.0 documentation](https://stable-baselines3.readthedocs.io/en/v2.7.0/modules/td3.html).
