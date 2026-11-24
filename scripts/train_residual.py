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
    parser.add_argument("--zero-policy-images", choices=["live", "recorded"], default="live", help="Recorded images isolate wrapper dynamics; not end-to-end equivalence")
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "residual-smoke")
    parser.add_argument("--instruction", default=None, help="Override checkpoint dataset task label; changes the policy contract")
    parser.add_argument("--checkpoint", help="Cached snapshot path; Hub lookup is offline")
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--horizon", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0, help="TD3 learner seed (zero-check simulator seed)")
    parser.add_argument("--policy-seed", type=int, default=10000)
    parser.add_argument("--scale", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-starts", type=int, default=900)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[200, 201, 202, 203])
    parser.add_argument("--observation-mode", choices=["proprio", "fridge_ground_truth"], default="fridge_ground_truth")
    parser.add_argument("--init-model", type=Path, help="Optional pretrained td3.zip to fine-tune from")

    parser.add_argument(
        "--task",
        choices=("CloseFridge", "OpenFridge", "OpenCabinet", "PlaceDishInSink"),
        default="CloseFridge",
    )
    parser.add_argument(
        "--perturbation",
        choices=(
            "nominal",
            "phys_random_train",
            "heldout_high_resistance",
            "heldout_low_friction",
            "heldout_dim_lighting",
            "heldout_texture_tint",
            "heldout_combined",
        ),
        default="nominal",
    )

    args = parser.parse_args()
    args.output = args.output.resolve()
    args.project_root = args.project_root.resolve()
    if args.init_model is not None:
        args.init_model = args.init_model.resolve()
        if not args.init_model.exists():
            parser.error(f"--init-model not found: {args.init_model}")
    if args.steps < 1 or args.horizon < 1 or not 0 <= args.scale <= 1 or args.learning_starts < 0:
        parser.error("steps/horizon must be positive; learning-starts nonnegative; scale in [0, 1]")
    if args.observation_mode == "fridge_ground_truth" and args.task not in ("CloseFridge", "OpenFridge"):
        parser.error("Fridge observations require a fridge task; use --observation-mode proprio otherwise")
    if args.mode == "reward-check" and args.task != "CloseFridge":
        parser.error("reward-check currently validates CloseFridge only")
    if not 0 < args.gamma < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("Invalid gamma, batch size, or learning rate")
    if args.mode == "train" and args.steps <= args.learning_starts + 1:
        parser.error("Training must allow at least two optimizer updates after learning-starts")
    if min([args.seed, args.policy_seed] + args.train_seeds) < 0:
        parser.error("Seeds must be nonnegative")
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
    from vla_rl.perturbations import PRESETS, PerturbationWrapper
    from vla_rl.rewards import CloseFridgeReward, make_reward
    from vla_rl.door_observation import FridgeDoorObservation
    from vla_rl.seed_schedule import EpisodeSeedSchedule, seeded_reset

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    schedule = EpisodeSeedSchedule(args.train_seeds)
    class SeededRoboCasaEnv(RoboCasaEnv):
        def reset(self, *, seed=None, **kwargs):
            effective_seed = schedule.next() if args.mode == "train" else (args.seed if seed is None else seed)
            return seeded_reset(super().reset, effective_seed, **kwargs)

    env = SeededRoboCasaEnv(task=args.task, episode_length=args.horizon)
    if args.perturbation != "nominal":
        env = PerturbationWrapper(env, PRESETS[args.perturbation])
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
            from vla_rl.provenance import provenance
            (args.output / "provenance.json").write_text(json.dumps(provenance(ROOT, checkpoint), indent=2), encoding="utf-8")
            base = FrozenSmolVLA(checkpoint, env, instruction=args.instruction or args.task)
            object_obs = FridgeDoorObservation() if args.observation_mode == "fridge_ground_truth" else None
            wrapped = ResidualEnv(env, base, make_reward(args.task), scale=args.scale,
                                  horizon=args.horizon, object_observation=object_obs)
            (args.output / "run_config.json").write_text(json.dumps(dict(vars(args),
                observation_dim=wrapped.observation_space.shape[0], checkpoint=str(checkpoint)),
                indent=2, default=str), encoding="utf-8")
            if args.mode == "zero-check":
                # Same simulator and seed, two full reset paths; compare actual
                # dynamics and pixels, not just action composition arithmetic.
                def digest(obs):
                    return {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in obs["pixels"].items()}

                raw, _ = env.reset(seed=args.seed)
                base.reset(args.seed)
                print("Recording direct frozen-policy trajectory", flush=True)
                qpos = [env.unwrapped._env.env.sim.data.qpos.copy()]
                qvel = [env.unwrapped._env.env.sim.data.qvel.copy()]
                pixels = [digest(raw)]
                frames = [{k: v.copy() for k, v in raw["pixels"].items()}]
                actions, states, successes, sparse_rewards = [], [raw["agent_pos"].copy()], [], []
                for _ in range(min(args.steps, args.horizon)):
                    action, _ = base.action(raw)
                    raw, sparse, done, trunc, info = env.step(action)
                    actions.append(action.copy())
                    qpos.append(env.unwrapped._env.env.sim.data.qpos.copy())
                    qvel.append(env.unwrapped._env.env.sim.data.qvel.copy())
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
                camera_differences = []
                def capture(raw):
                    recorded_pixels.append(digest(raw))
                    capture.frames = {k: v.copy() for k, v in raw["pixels"].items()}
                    if args.zero_policy_images == "recorded":
                        raw = dict(raw, pixels=frames[min(len(recorded_pixels) - 1, len(frames) - 1)])
                    return original(raw)
                wrapped._observation = capture
                obs, _ = wrapped.reset(seed=args.seed)
                print("Comparing zero-residual trajectory", flush=True)
                np.save(args.output / "direct-actions.npy", np.asarray(actions))
                np.testing.assert_array_equal(env.unwrapped._env.env.sim.data.qpos, qpos[0])
                np.testing.assert_array_equal(env.unwrapped._env.env.sim.data.qvel, qvel[0])
                np.testing.assert_array_equal(obs[:16], states[0])
                if recorded_pixels[-1] != pixels[0]:
                    differences = {}
                    for camera, expected in frames[0].items():
                        actual = capture.frames[camera]
                        delta = np.abs(actual.astype(int) - expected.astype(int))
                        differences[camera] = {
                            "changed_values": int(np.count_nonzero(delta)),
                            "max_difference": int(delta.max()),
                            "mean_difference": float(delta.mean()),
                        }
                        np.savez_compressed(args.output / f"mismatch-reset-{camera}.npz",
                                            expected=expected, actual=actual)
                    (args.output / "reset-camera-differences.json").write_text(
                        json.dumps(differences, indent=2), encoding="utf-8")
                    camera_differences.append({"step": 0, "cameras": differences})
                    if args.zero_policy_images == "live":
                        raise AssertionError(f"reset camera mismatch: {differences}")
                for i, action in enumerate(actions):
                    np.testing.assert_array_equal(wrapped.pending_action, action, err_msg=f"pending action mismatch at step {i}")
                    obs, _, done, trunc, info = wrapped.step(np.zeros(6, np.float32))
                    np.testing.assert_array_equal(info["executed_action"], action)
                    np.testing.assert_array_equal(env.unwrapped._env.env.sim.data.qpos, qpos[i + 1])
                    np.testing.assert_array_equal(env.unwrapped._env.env.sim.data.qvel, qvel[i + 1])
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
                            if max_cam_diff > 0:
                                np.savez_compressed(args.output / f"mismatch-step-{i + 1}-{camera}.npz",
                                                    expected=expected, actual=actual)
                        camera_differences.append({"step": i + 1, "cameras": differences})
                        (args.output / "camera-differences.json").write_text(json.dumps(camera_differences, indent=2), encoding="utf-8")
                        if max_diff_overall > 0 and args.zero_policy_images == "live":
                            raise AssertionError(f"camera mismatch at step {i + 1}: {differences}")
                    assert info["is_success"] == successes[i]
                    assert info["sparse_reward"] == sparse_rewards[i]
                report = {"mode": args.mode, "compared_steps": len(actions), "exact_actions_and_state": True,
                          "camera_max_allowed_intensity_difference": 0,
                          "policy_images": args.zero_policy_images,
                          "end_to_end_equivalence": args.zero_policy_images == "live",
                          "camera_differences": camera_differences,
                          "seed": args.seed, "global_reset_rngs_seeded": True}
            else:
                from stable_baselines3 import TD3
                from stable_baselines3.common.buffers import ReplayBuffer
                from stable_baselines3.common.monitor import Monitor
                from stable_baselines3.common.noise import NormalActionNoise

                frozen_before = {k: v.detach().cpu().clone() for k, v in base.policy.named_parameters()}
                class PolicySeedWrapper(__import__("gymnasium").Wrapper):
                    episode = 0
                    def reset(self, *, seed=None, options=None):
                        policy_seed = args.policy_seed + self.episode
                        self.episode += 1
                        obs, info = self.env.reset(seed=seed, options=dict(options or {}, policy_seed=policy_seed))
                        with (args.output / "training_resets.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(info, default=str) + "\n")
                        return obs, info
                class EpisodeLogWrapper(__import__("gymnasium").Wrapper):
                    def step(self, action):
                        obs, reward, done, trunc, info = self.env.step(action)
                        return obs, reward, done, trunc, dict(info, door_openness=info["reward_terms"].get("door_openness", float("nan")))
                monitored = Monitor(EpisodeLogWrapper(PolicySeedWrapper(wrapped)), str(args.output / "monitor.csv"),
                                    info_keywords=("is_success", "door_openness"))
                model = TD3("MlpPolicy", monitored,
                            learning_rate=args.lr,
                            buffer_size=100_000, learning_starts=args.learning_starts, batch_size=args.batch_size,
                            gamma=args.gamma,
                            train_freq=1, gradient_steps=1, policy_delay=2,
                            policy_kwargs={"net_arch": [128, 128]},
                            action_noise=NormalActionNoise(np.zeros(6), 0.03 * np.ones(6)),
                            seed=args.seed, device="cuda", verbose=1)
                # Start the learned correction at zero, preserving the initial baseline.
                for actor in (model.actor, model.actor_target):
                    torch.nn.init.zeros_(actor.mu[-2].weight)
                    torch.nn.init.zeros_(actor.mu[-2].bias)
                if args.init_model is not None:
                    pretrained = TD3.load(str(args.init_model), device="cuda")
                    if pretrained.observation_space.shape != wrapped.observation_space.shape:
                        raise ValueError("Old model observation shape differs; train the new observation model from scratch")
                    model.policy.load_state_dict(pretrained.policy.state_dict())
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
                          "scale": args.scale, "horizon": args.horizon,
                          "observation_mode": args.observation_mode, "train_seeds": args.train_seeds,
                          "episodes": len(model.ep_info_buffer), "gamma": args.gamma}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
