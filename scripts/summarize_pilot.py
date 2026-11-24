"""Archive compact pilot evidence and write a dated research handoff."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    runs = ROOT/'runs'
    train = json.loads((runs/'20261124-nominal-pilot-s0/learning_summary.json').read_text())
    base = json.loads((runs/'20261124-base-heldout/eval_report.json').read_text())
    residual = json.loads((runs/'20261124-nominal-heldout/eval_report.json').read_text())
    pairs = json.loads((runs/'20261124-paired/paired_outcomes.json').read_text())
    output = ROOT/'docs/results/2026-11-24'
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        'training-report.json': '20261124-nominal-pilot-s0/report.json',
        'learning-summary.json': '20261124-nominal-pilot-s0/learning_summary.json',
        'training-episodes.jsonl': '20261124-nominal-pilot-s0/training_episodes.jsonl',
        'paired-outcomes.json': '20261124-paired/paired_outcomes.json',
        'paired-outcomes.csv': '20261124-paired/paired_outcomes.csv',
    }
    for target, source in sources.items():
        shutil.copyfile(runs/source, output/target)
    # Keep compact episode evidence; full per-field model fingerprints stay in runs.
    for label, report in [('baseline', base), ('residual', residual)]:
        compact = dict(report, episodes=[{k: v for k, v in e.items() if k != 'initial_numeric_model'}
                                        | {'initial_numeric_model': {'schema': e['initial_numeric_model']['schema'],
                                                                     'sha256': e['initial_numeric_model']['sha256']}}
                                        for e in report['episodes']])
        (output/f'{label}-evaluation.json').write_text(json.dumps(compact, indent=2), encoding='utf-8')
    render = json.loads((runs/'20261124-render-replay/report.json').read_text())
    rendering = {'steps': render['steps'], 'exact_qpos_qvel': render['exact_qpos_qvel'],
                 'numeric_models_equal': render['numeric_models_equal'],
                 'max_same_context_pixel_difference': max(e['max'] for e in render['same_state_same_context_rerenders']),
                 'max_between_reset_pixel_difference': max(d['max'] for e in render['between_reset_camera_differences'] for d in e['cameras'].values())}
    (output/'rendering-summary.json').write_text(json.dumps(rendering, indent=2), encoding='utf-8')
    content = '''# CloseFridge pilot and task interfaces — 2026-11-24

This is a nominal-training pilot with one learner seed, not the completed
nominal-versus-randomized study. No claim of algorithmic novelty or causal rescue
is established. Native simulator success predicates remain unchanged.

## Rendering and initial-condition matching

The simulator-only replay audit used 60 identical recorded actions across two
seeded resets. Numeric model identities and qpos/qvel matched exactly. Even two
renders of the same state in the same context differed by up to one intensity
level. The strict live zero-residual check remains strict and failed; it was not
converted into a pass by increasing tolerance. Recorded-image checks isolate
wrapper dynamics, not end-to-end rendered equivalence.

Evaluation now records per-field numeric model fingerprints (including numeric
solver options) alongside the existing XML checksum, initial qpos/qvel hash,
initial camera hashes, instruction, and actual physical parameters. XML
serialization differences alone do not establish a physical mismatch. Numeric
fingerprints do not claim to capture every controller or renderer state.

## Full-horizon nominal pilot

Configuration: 9,000 environment steps, 900-step horizon, learner seed 0,
policy seed sequence starting at 10,000, initialization pool 200–203,
59-dimensional privileged fridge observation, residual scale 0.1, and 900
warm-up steps. Existing reward and native success were retained. Checkpoints
were saved at steps 0, 3,000, 6,000, and 9,000. The learner writes losses,
episode returns, contact counts/loss events, final errors, and correction
saturation. Contact is gripper-to-fixture, not necessarily to the fridge door.
Positive-progress counts use a small numerical threshold and can include drift.

'''
    content += f"Observed training: **{train['training_successes']}/{train['episodes']} completed episodes succeeded**, with **{train['optimizer_updates']} optimizer updates** in {train['elapsed_s']:.1f} seconds. SmolVLA's frozen weights were checked after training. Training episodes include exploration and are not held-out evidence.\n\n"
    content += '''## Held-out nominal comparison

The final checkpoint and frozen baseline use simulator seeds 100 and 101,
policy seed 0, two repeats each, and a 900-step horizon. The small sample and
single training seed do not support generalization or confidence claims across
training runs. Repeats are not independent initialization samples.

| Method | Successes / episodes | Timeouts/failures | Mean final openness |
| --- | --- | --- | --- |
'''
    for label, report in [('Frozen SmolVLA', base), ('Nominal TD3 pilot', residual)]:
        e = report['episodes']
        content += f"| {label} | {sum(x['success'] for x in e)}/{len(e)} | {sum(not x['success'] for x in e)} | {sum(x['final_openness'] for x in e)/len(e):.4f} |\n"
    content += '\nDescriptive paired outcomes: `' + json.dumps(pairs['counts']) + '`. Mismatched initial conditions are excluded from rescue/harm counts. These outcomes are associations, not causal effects.\n\n'
    content += '''## Approved task extensions

`SlideDishwasherRack` uses 16 privileged features plus the 29-value common
input (45 total), a 450-step default horizon, and signed reduction in remaining
normalized rack travel. The target direction is explicit in the residual input.

`RecycleBottlesByType` uses 88 privileged features plus the common input (117
total), a 2,850-step horizon, and signed reduction in mean excess XY distance
beyond the native 0.30 m cluster threshold. The features include seven bottle
poses, three cluster distances, and the mystery bottle's true category. This is
privileged semantic state as well as geometry. Zero placement error is not
success: native table-contact and gripper-withdrawal requirements still apply.

Both adapters retain class-name VLA instructions matching checkpoint metadata.
Physics randomization is intentionally rejected for these new tasks until their
task-specific physical parameters and feasibility are validated. They are rack
sliding and bottle sorting, not dishwasher-door closing or trash-lid manipulation.

Live adapter checks use four zero-action simulator steps and are integration
checks, not policy competence or completed-task evaluations. See accompanying
task audit reports when present.

## Commands

Run from the active VLA-RL repository with the configured vla-rl interpreter:

```powershell
$python = 'C:\\Users\\Joshua\\miniconda3\\envs\\vla-rl\\python.exe'
& $python scripts/train_residual.py --steps 9000 --horizon 900 --seed 0 --policy-seed 10000 --train-seeds 200 201 202 203 --checkpoint-every 3000 --output runs/nominal-new
& $python scripts/plot_learning.py runs/nominal-new
& $python scripts/eval_study.py --method base_vla --seeds 100 101 --policy-seeds 0 --repeats 2 --output runs/base-new
& $python scripts/eval_study.py --method residual_td3 --model-path runs/nominal-new/td3.zip --seeds 100 101 --policy-seeds 0 --repeats 2 --output runs/residual-new
& $python scripts/compare_study.py --baseline runs/base-new/eval_report.json --residual runs/residual-new/eval_report.json --output runs/paired-new
& $python scripts/audit_task_interfaces.py --task SlideDishwasherRack --output runs/rack-audit-new
& $python scripts/audit_task_interfaces.py --task RecycleBottlesByType --output runs/sorting-audit-new
```

Task evaluation uses `--task SlideDishwasherRack` or `--task RecycleBottlesByType`
with default `task_ground_truth` observations and the registered task horizon.
Use fresh output directories. Short horizon overrides are integration only.

For a controlled CloseFridge severity sweep, evaluation accepts
`--hinge-damping-scale`, `--hinge-frictionloss-add`, and
`--contact-friction-scale` with `--perturbation nominal`. Set only the intended
variable and run every method at the identical level. Actual numeric parameters
are saved. These CLI controls do not establish feasibility, and no severity
performance result has yet been measured in this pilot.

Next: validate competence on the new tasks, assess the pilot's held-out failures
and saturation, then freeze training choices and execute matched nominal and
randomized runs with multiple learner seeds and physically feasible held-out
severity levels. Preserve the repeated-baseline controls.
'''
    (ROOT/'docs/pilot-2026-11-24.md').write_text(content, encoding='utf-8')
    for task, run in [('rack','20261124-rack-interface'),('sorting','20261124-sorting-interface')]:
        path=runs/run/'report.json'
        if path.exists(): shutil.copyfile(path,output/f'{task}-interface.json')

if __name__ == '__main__': main()
