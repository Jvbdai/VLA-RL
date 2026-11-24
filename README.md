# VLA-RL

Simulation-only research on frozen vision-language-action policies with learned
residual control in RoboCasa, using the PandaOmron robot.

**Research question:** Does training a residual RL policy across varied physical
conditions improve unseen household interactions, and what failures remain when
the residual receives accurate object state?

The experiment compares frozen SmolVLA, SmolVLA plus nominally trained residual
TD3, and SmolVLA plus physics-randomized residual TD3. The contribution must be
a measured experimental finding. Combining these existing methods or measuring
harmful corrections is not itself a novelty claim.

## What runs today

- **Pretrained policy:** `lerobot/smolvla_robocasa`, kept frozen during TD3 training.
  Running inference does not fine-tune SmolVLA.
- **Residual:** six bounded arm corrections at action indices `5:11`, with default
  scale 0.1. SmolVLA retains base movement, control mode, and gripper control.
- **Observation:** 59 values by default: proprioception, pending VLA action,
  chunk position, and simulator-ground-truth fridge geometry. Legacy 29-input
  checkpoints require explicit `--observation-mode proprio`.
- **Current task:** CloseFridge, using native success and a 900-step horizon.
- **Reproducibility:** separate simulator, policy, and learner RNG streams;
  repeated baseline evaluation; checkpoint contracts; source/version provenance;
  sampled physics logs; per-episode results and optional three-camera videos.

Ground-truth geometry is an intentional experimental condition, not evidence of
robustness to tracking errors. See [implementation and commands](docs/residual.md)
for the reward, observations, physics ranges, and validation limitations.

## Continuation status — 2026-11-24

- A 9,000-step nominal CloseFridge pilot completed 8,100 optimizer updates;
  SmolVLA remained frozen. It succeeded in 3 of 11 completed training episodes,
  all on initialization seed 201. This is not evidence of held-out improvement.
  See [training summary](docs/results/2026-11-24/learning-summary.json).
- Numeric model fingerprints and strict matched-outcome reports supplement XML
  hashes. A simulator-only replay found exact qpos/qvel and matching models,
  but repeated rendering of the same state can differ by one intensity level.
- Rack sliding and bottle sorting now have task-specific observations, rewards,
  diagnostics, and registered horizons (450 and 2,850 steps). Their physics
  randomization remains disabled until validated; live integration results are
  recorded separately from competence evaluations.
- Training saves intermediate checkpoints and learning curves. The latest suite
  contains 39 passing tests. The prior audit below is retained as history.

For a new checkout, initialize the pinned simulator dependencies with
`git submodule update --init --recursive`. Install the simulator assets separately.
The validated external LeRobot checkout is https://github.com/huggingface/lerobot.git
at revision `45a0756dfda4010eef6c0bfb2a95c5dd23c03fc6`; it remains outside Git under
`third_party/lerobot`. Full installation of the CUDA environment and model cache
is still required; submodule initialization alone is not a complete installation.

## Validation status

The 2026-11-23 audit established integration checks, not successful residual learning:

| Check | Observed result |
| --- | --- |
| Unit/integration suite | 31 tests passed |
| Same-seed reset audit after RNG fix | Exact qpos, qvel, and inspected numeric model arrays |
| Zero residual with recorded policy images | Exact actions and observed dynamics for 60 steps |
| Strict live-image zero residual | Failed on small rendering differences; end-to-end equivalence remains unresolved |
| Randomized TD3 smoke run | 128 steps, 96 optimizer updates, actor changed, SmolVLA frozen; zero complete episodes |
| Post-fix frozen baseline | 0/2 successes at 900 steps on simulator seed 100 and policy seed 0 |
| Final mean door openness in those repeats | 0.691 and 0.621 (lower is better) |

The small sample cannot establish general task competence or residual efficacy.
The recorded-image check isolates wrapper behavior; it is not an end-to-end
equivalence pass. Identical seeds alone do not guarantee identical rendered
trajectories. XML hash differences can also reflect serialization changes, so
inspect numeric model arrays before attributing a mismatch to physical state.

Compact [audit reports](docs/results/2026-11-23) are versioned for this handoff.
Full `runs/` outputs, videos, model weights, and downloaded assets remain local
and ignored by Git. Their original locations appear in the reports.

## Quick start

Use the configured `vla-rl` environment with CUDA, RoboCasa/robosuite assets,
editable LeRobot, and cached SmolVLA dependencies. This is not a standalone
standard-library project. On the validated Windows machine:

```powershell
Set-Location 'C:\Users\Joshua\PycharmProjects\VLA-RL'
$python = 'C:\Users\Joshua\miniconda3\envs\vla-rl\python.exe'
& $python -m unittest discover -s tests -v
& $python scripts/audit_resets.py --seed 100 --output runs/reset-audit-new
& $python scripts/eval_study.py --method base_vla --seeds 100 --policy-seeds 0 1 --repeats 2 --horizon 900 --save-video --output runs/base-new
```

Each invocation should use a fresh output directory. On another machine, adapt
the paths and use its configured Python interpreter. Checkpoint loading is
offline by default. Supply `--checkpoint <snapshot-directory>` when necessary.
The validated snapshot is `f7af588d11e30b0069f200792fb285904b74c80b`.
Dependency versions and source revisions from the baseline audit are included in
the versioned provenance report; assets and dependencies must be installed separately.

For a bounded training integration check:

```powershell
& $python scripts/train_residual.py --mode train --steps 128 --learning-starts 32 --batch-size 16 --horizon 900 --perturbation phys_random_train --output runs/integration-new
```

This deliberately short run tests optimizer updates and frozen weights. Use the
[full training and shared evaluation commands](docs/residual.md#training-and-shared-evaluation)
for candidate nominal/randomized experiments; their budgets are not validated
learning-quality settings. The same evaluation entry point supports `base_vla`
and `residual_td3 --model-path <td3.zip>`.

## Next research steps

1. Resolve or quantify live-rendering variability and strengthen matching checks
   before interpreting paired rescue/harm outcomes.
2. Establish full-task CloseFridge learning curves and held-out performance, with
   equal budgets and multiple independent learner seeds. Keep debugging curricula
   separate from full-task results.
3. Add task-specific dishwasher and bottle-sorting observations, rewards, and
   diagnostics. Coverage is verified for SlideDishwasherRack, LoadDishwasher,
   and RecycleBottlesByType in the checkpoint's training metadata; competence is
   not verified. These tasks are not equivalent to trash-lid manipulation.
4. Validate task-relevant physical perturbations and feasibility, then run held-out
   severity sweeps and report uncertainty across training runs.

See [checkpoint task coverage](docs/checkpoint_task_coverage.json) for source and
metadata identifiers. CloseDishwasher exists locally but is absent from that
checkpoint's task list. No environment success definition should be changed.

## Repository guide and handoff notes

| Location | Purpose |
| --- | --- |
| `src/vla_rl/` | Frozen policy, residual wrapper, rewards, physics, observations, seeding, provenance |
| `scripts/train_residual.py` | Reward/zero checks and TD3 training |
| `scripts/eval_study.py` | Frozen/residual evaluation, repeats, reports, videos |
| `scripts/audit_resets.py` | Numeric simulator reset and XML comparison |
| `scripts/summarize_audit.py` | Baseline CSV table and repeat-variability plot |
| `tests/` | Behavioral and integration regression checks |
| `docs/residual.md` | Detailed current protocol, commands, and limitations |
| `docs/results/2026-11-23/` | Small handoff evidence snapshots |

Older tracking-fault, temporal-gating, camera-corruption, and alternative-task
plans are historical, including the older configs and documents not linked as
the current protocol above. Previous README and residual instructions are
preserved in `docs/history/`.

The local RoboCasa checkout has pre-existing asset working-tree changes. They
are not part of this parent-repository commit; the pinned dependency revision
is unchanged. Downloaded LeRobot and simulator assets are also not bundled.
