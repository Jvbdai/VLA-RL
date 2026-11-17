"""CloseFridge nominal TD3 smoke run or paired zero-residual integration check."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "zero-check", "reward-check"), default="train")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "residual-smoke")
    parser.add_argument("--checkpoint", help="Cached snapshot path; Hub lookup is offline")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.1)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.project_root = args.project_root.resolve()
    if args.steps < 1 or args.horizon < 1 or not 0 <= args.scale <= 1:
        parser.error("steps/horizon must be positive; scale must lie in [0, 1]")
    if args.mode == "train" and importlib.util.find_spec("stable_baselines3") is None:
        parser.error("Install stable-baselines3==2.7.0 in the vla-rl environment first")

    # Avoid outer checkout directory shadowing the editable RoboCasa package.
    os.chdir(args.project_root / "robocasa")
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != args.project_root.resolve()]
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Keep generated compilation caches out of downloaded library checkouts.
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output / "numba-cache"))
    import numpy as np
    import torch
    from lerobot.sim.integrations.robocasa import RoboCasaEnv
    from vla_rl.residual import ResidualEnv
    from vla_rl.rewards import CloseFridgeReward

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    class SeededRoboCasaEnv(RoboCasaEnv):
        def reset(self, *, seed=None, **kwargs):
            # RoboCasa's Gym wrapper seeds kitchen.rng, but fixture construction
            # also uses the Python and legacy NumPy global random generators.
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
            return super().reset(seed=seed, **kwargs)

    env = SeededRoboCasaEnv(task="CloseFridge", episode_length=args.horizon)
    try:
        if args.mode == "reward-check":
            env.reset(seed=args.seed)
            kitchen = env._env.env
            saved = kitchen.sim.data.qpos.copy()
            reward = CloseFridgeReward()
            try:
                values = []
                shaped = []
                for openness in (1.0, 0.5, 0.0):
                    kitchen.fxtr.set_joint_state(openness, openness, kitchen,
                                                kitchen.fxtr._fridge_door_joint_names)
                    kitchen.sim.forward()
                    values.append(reward.openness(env))
                    if openness == 1.0:
                        reward.reset(env)
                    else:
                        shaped.append(reward.step(env, kitchen._check_success(), np.zeros(6))[0])
                np.testing.assert_allclose(values, [1, .5, 0], atol=1e-6)
                np.testing.assert_allclose(shaped, [2.5, 12.5], atol=1e-6)
                assert kitchen._check_success()
                stationary = reward.step(env, False, np.zeros(6))[0]
                kitchen.fxtr.set_joint_state(0.5, 0.5, kitchen,
                                            kitchen.fxtr._fridge_door_joint_names)
                kitchen.sim.forward()
                reopening = reward.step(env, kitchen._check_success(), np.zeros(6))[0]
                np.testing.assert_allclose([stationary, reopening], [0, -2.5], atol=1e-6)
                assert not kitchen._check_success()
                report = {"mode": args.mode, "normalized_openness": values,
                          "closed_success": True, "closure_rewards": shaped,
                          "stationary_reward": stationary, "reopening_reward": reopening}
            finally:
                kitchen.sim.data.qpos[:] = saved
                kitchen.sim.forward()
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA required for the local GPU SmolVLA run")
            from huggingface_hub import snapshot_download
            from vla_rl.smolvla import FrozenSmolVLA

            checkpoint = args.checkpoint or snapshot_download(
                "lerobot/smolvla_robocasa", local_files_only=True,
                allow_patterns=["*.json", "*.safetensors"],
            )
            base = FrozenSmolVLA(checkpoint, env)
            wrapped = ResidualEnv(env, base, CloseFridgeReward(), scale=args.scale, horizon=args.horizon)
            if args.mode == "zero-check":
                # Same simulator and seed, two full reset paths; compare actual
                # dynamics and pixels, not just action composition arithmetic.
                def digest(obs):
                    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in obs["pixels"].items()}

                raw, _ = env.reset(seed=args.seed)
                base.reset(args.seed)
                print("Recording direct frozen-policy trajectory", flush=True)
                qpos = [env._env.env.sim.data.qpos.copy()]
                pixels = [digest(raw)]
                frames = [{k: v.copy() for k, v in raw["pixels"].items()}]
                actions, states, successes, sparse_rewards = [], [raw["agent_pos"].copy()], [], []
                for _ in range(min(args.steps, args.horizon)):
                    action, _ = base.action(raw)
                    raw, sparse, done, trunc, info = env.step(action)
                    actions.append(action.copy())
                    qpos.append(env._env.env.sim.data.qpos.copy())
                    states.append(raw["agent_pos"].copy())
                    pixels.append(digest(raw))
                    frames.append({k: v.copy() for k, v in raw["pixels"].items()})
                    successes.append(bool(info["is_success"]))
                    sparse_rewards.append(sparse)
                    if done or trunc:
                        break
                # Raw pixels are retained by a process-local recording hook.
                original = wrapped._observation
                recorded_pixels = []
                def capture(raw):
                    recorded_pixels.append(digest(raw))
                    capture.frames = {k: v.copy() for k, v in raw["pixels"].items()}
                    return original(raw)
                wrapped._observation = capture
                obs, _ = wrapped.reset(seed=args.seed)
                print("Comparing zero-residual trajectory", flush=True)
                np.save(args.output / "direct-actions.npy", np.asarray(actions))
                np.testing.assert_array_equal(env._env.env.sim.data.qpos, qpos[0])
                np.testing.assert_array_equal(obs[:16], states[0])
                assert recorded_pixels[-1] == pixels[0]
                for i, action in enumerate(actions):
                    np.testing.assert_array_equal(wrapped.pending_action, action)
                    obs, _, done, trunc, info = wrapped.step(np.zeros(6, np.float32))
                    np.testing.assert_array_equal(info["executed_action"], action)
                    np.testing.assert_array_equal(env._env.env.sim.data.qpos, qpos[i + 1])
                    np.testing.assert_array_equal(obs[:16], states[i + 1])
                    if recorded_pixels[-1] != pixels[i + 1]:
                        differences = {}
                        max_diff_overall = 0
                        for camera, expected in frames[i + 1].items():
                            actual = capture.frames[camera]
                            delta = np.abs(actual.astype(int) - expected.astype(int))
                            max_cam_diff = int(delta.max())
                            max_diff_overall = max(max_diff_overall, max_cam_diff)
                            differences[camera] = {"changed_values": int(np.count_nonzero(delta)),
                                                   "max_difference": max_cam_diff}
                            if max_cam_diff > 1:
                                np.savez_compressed(args.output / f"mismatch-step-{i + 1}-{camera}.npz",
                                                    expected=expected, actual=actual)
                        if max_diff_overall > 1:
                            raise AssertionError(f"camera mismatch at step {i + 1}: {differences}")
                    assert info["is_success"] == successes[i]
                    assert info["sparse_reward"] == sparse_rewards[i]
                report = {"mode": args.mode, "compared_steps": len(actions), "exact_equivalence": True,
                          "seed": args.seed, "global_reset_rngs_seeded": True}
            else:
                from stable_baselines3 import TD3
                from stable_baselines3.common.buffers import ReplayBuffer
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.noise import NormalActionNoise

                frozen_before = {k: v.detach().cpu().clone() for k, v in base.policy.named_parameters()}
                model = TD3("MlpPolicy", Monitor(wrapped, str(args.output / "monitor.csv")),
                            buffer_size=10_000, learning_starts=8, batch_size=8,
                            train_freq=1, gradient_steps=1, policy_delay=2,
                            policy_kwargs={"net_arch": [128, 128]},
                            action_noise=NormalActionNoise(np.zeros(6), .1 * np.ones(6)),
                            seed=args.seed, device="cuda", verbose=1)
                # Start the learned correction at zero, preserving the initial baseline.
                for actor in (model.actor, model.actor_target):
                    torch.nn.init.zeros_(actor.mu[-2].weight)
                    torch.nn.init.zeros_(actor.mu[-2].bias)
                actor_before = [p.detach().clone() for p in model.actor.parameters()]
                model.learn(total_timesteps=args.steps)
                assert type(model.replay_buffer) is ReplayBuffer
                assert model._n_updates > 0, "smoke run did not perform optimizer updates"
                assert any(not torch.equal(a, b) for a, b in zip(actor_before, model.actor.parameters()))
                assert all(torch.equal(frozen_before[k], v.detach().cpu()) and v.grad is None
                           and not v.requires_grad for k, v in base.policy.named_parameters())
                assert np.isfinite(model.replay_buffer.rewards[:model.replay_buffer.size()]).all()
                model.save(str(args.output / "td3"))
                report = {"mode": args.mode, "steps": model.num_timesteps, "updates": model._n_updates,
                          "replay_class": type(model.replay_buffer).__name__, "frozen_verified": True,
                          "checkpoint": str(checkpoint), "seed": args.seed,
                          "scale": args.scale, "horizon": args.horizon}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
