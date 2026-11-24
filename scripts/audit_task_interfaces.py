"""Live integration checks for approved tasks; not policy-quality evaluation."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["SlideDishwasherRack", "RecycleBottlesByType"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[200, 201])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT/'robocasa')
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output/'numba-cache'))
    import numpy as np
    from lerobot.sim.integrations.robocasa import RoboCasaEnv
    from vla_rl.seed_schedule import seeded_reset
    from vla_rl.task_interfaces import TASK_HORIZONS, task_observation, task_diagnostics
    from vla_rl.rewards import make_reward
    observation = task_observation(args.task, "task_ground_truth")
    diagnostics = task_diagnostics(args.task)
    reward = make_reward(args.task)
    env = RoboCasaEnv(task=args.task, episode_length=TASK_HORIZONS[args.task])
    episodes = []
    try:
        for seed in args.seeds:
            raw, _ = seeded_reset(env.reset, seed)
            reward.reset(env)
            features = observation(env)
            assert features.shape == (observation.size,) and np.isfinite(features).all()
            k = env._env.env
            initial = diagnostics(env)
            initial_native_success = bool(k._check_success())
            checks = []
            for _ in range(4):
                raw, sparse, done, truncated, info = env.step(np.zeros(12, dtype=np.float32))
                native = bool(k._check_success())
                assert info["is_success"] == native
                shaped, terms = reward.step(env, native, np.zeros(6))
                assert np.isfinite(shaped) and np.isfinite(observation(env)).all()
                assert terms["success_bonus"] == 10 * native
                checks.append({"native_success": native, "task_error": terms["task_error"], "reward": shaped})
                if done or truncated:
                    break
            episodes.append({"seed": seed, "initial_native_success": initial_native_success,
                             "initial_diagnostics": initial, "final_diagnostics": diagnostics(env), "steps": checks})
        report = {"task": args.task, "kind": "integration_only_zero_actions", "horizon": TASK_HORIZONS[args.task],
                  "object_schema": observation.schema, "residual_observation_dim": 29 + observation.size,
                  "episodes": episodes}
        (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        env.close()

if __name__ == '__main__':
    main()
