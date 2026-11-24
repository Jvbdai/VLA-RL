"""Evaluate frozen SmolVLA or trained Residual TD3 across tasks and perturbations."""

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _save_rollout_video(frames, video_path: Path, fps: int = 20):
    """Write side-by-side 3-camera RGB frames to an MP4 (or GIF fallback)."""
    if frames is None or len(frames) == 0:
        return None
    video_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(video_path), fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)
        return str(video_path)
    except Exception:
        try:
            import torchvision.io as tvio
            import torch
            tensor = torch.from_numpy(frames)
            tvio.write_video(str(video_path), tensor, fps=fps)
            return str(video_path)
        except Exception:
            from PIL import Image
            gif_path = video_path.with_suffix(".gif")
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(
                str(gif_path),
                save_all=True,
                append_images=imgs[1:],
                duration=int(1000 / fps),
                loop=0,
            )
            return str(gif_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("CloseFridge", "OpenFridge", "OpenCabinet", "PlaceDishInSink", "SlideDishwasherRack", "RecycleBottlesByType"),
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
    parser.add_argument(
        "--method",
        choices=("base_vla", "residual_td3"),
        default="base_vla",
    )
    parser.add_argument("--hinge-damping-scale", type=float, default=None)
    parser.add_argument("--hinge-frictionloss-add", type=float, default=None)
    parser.add_argument("--contact-friction-scale", type=float, default=None)
    parser.add_argument("--model-path", type=Path, help="Path to trained td3.zip checkpoint")
    parser.add_argument("--instruction", default=None, help="Override checkpoint dataset task label; changes the policy contract")
    parser.add_argument("--checkpoint", help="Cached SmolVLA snapshot path")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[0], help="Independent policy RNG seeds, crossed with simulator seeds")
    parser.add_argument("--repeats", type=int, default=1, help="Identical-seed reruns to measure nondeterminism")
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=None, help="Defaults to task-specific registered horizon")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--observation-mode", choices=["proprio", "fridge_ground_truth", "task_ground_truth"], default=None)
    parser.add_argument("--save-video", action="store_true", help="Save 3-camera rollout videos")
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "eval-study")
    args = parser.parse_args()
    from vla_rl.task_interfaces import TASK_HORIZONS, task_observation, task_diagnostics
    args.horizon = args.horizon if args.horizon is not None else TASK_HORIZONS.get(args.task, 900)
    from vla_rl.perturbations import PerturbationSpec
    physical_overrides = {k: getattr(args, k) for k in ("hinge_damping_scale", "hinge_frictionloss_add", "contact_friction_scale") if getattr(args, k) is not None}
    custom_spec = None
    if physical_overrides:
        if args.task != "CloseFridge" or args.perturbation != "nominal":
            parser.error("Explicit physics levels currently require CloseFridge and --perturbation nominal")
        try:
            custom_spec = PerturbationSpec(name="custom_physics", **physical_overrides)
        except ValueError as error:
            parser.error(str(error))
        args.perturbation = "custom_physics"
    if args.task in ("SlideDishwasherRack", "RecycleBottlesByType") and args.perturbation != "nominal":
        parser.error("New task physics must be validated separately; only nominal is currently supported")

    if args.model_path is not None:
        args.model_path = args.model_path.resolve()
    if args.method == "residual_td3" and (args.model_path is None or not args.model_path.exists()):
        parser.error("--model-path pointing to a trained td3.zip is required for residual_td3")
    contract = {}
    if args.method == "residual_td3":
        config_path = args.model_path.parent / "run_config.json"
        if config_path.exists():
            contract = json.loads(config_path.read_text(encoding="utf-8"))
            if contract["task"] != args.task:
                parser.error("Checkpoint was trained on a different task")
            trained_instruction = contract.get("instruction") or contract["task"]
            if (args.instruction or args.task) != trained_instruction:
                parser.error("Instruction differs from the trained checkpoint contract")
            if args.checkpoint and str(Path(args.checkpoint).resolve()) != str(Path(contract["checkpoint"]).resolve()):
                parser.error("VLA checkpoint differs from training")
            args.checkpoint = contract["checkpoint"]
            for arg in ("scale", "observation_mode"):
                value = getattr(args, arg)
                if value is not None and value != contract[arg]:
                    parser.error(f"{arg} differs from the trained checkpoint contract")
                setattr(args, arg, contract[arg])
        elif args.scale is None or args.observation_mode is None:
            parser.error("Legacy checkpoint: explicitly supply --scale and --observation-mode proprio")
    args.scale = 0.1 if args.scale is None else args.scale
    args.observation_mode = args.observation_mode or ("task_ground_truth" if args.task in ("SlideDishwasherRack", "RecycleBottlesByType") else "fridge_ground_truth")
    if args.repeats < 1 or min(args.seeds + args.policy_seeds) < 0:
        parser.error("repeats must be positive and seeds nonnegative")
    if args.horizon < 1 or args.episodes_per_seed < 1 or not 0 <= args.scale <= 1:
        parser.error("Invalid horizon, episodes, or scale")
    if args.observation_mode == "fridge_ground_truth" and args.task not in ("CloseFridge", "OpenFridge"):
        parser.error("Fridge observations require a fridge task")

    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "episodes.jsonl").exists():
        parser.error("Output already contains episodes; use a fresh output directory")

    os.chdir(ROOT / "robocasa")
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT.resolve()]
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output / "numba-cache"))

    import numpy as np
    import torch
    from huggingface_hub import snapshot_download
    from lerobot.sim.integrations.robocasa import RoboCasaEnv
    from vla_rl.perturbations import PRESETS, PerturbationWrapper
    from vla_rl.residual import ResidualEnv
    from vla_rl.rewards import make_reward
    from vla_rl.smolvla import FrozenSmolVLA
    from vla_rl.door_observation import FridgeDoorObservation
    from vla_rl.seed_schedule import seeded_reset

    class SeededRoboCasaEnv(RoboCasaEnv):
        def reset(self, *, seed=None, **kwargs):
            return seeded_reset(super().reset, 0 if seed is None else seed, **kwargs)

    base_env = SeededRoboCasaEnv(task=args.task, episode_length=args.horizon)
    env = (
        PerturbationWrapper(base_env, custom_spec or PRESETS[args.perturbation])
        if args.perturbation != "nominal"
        else base_env
    )

    try:
        checkpoint = args.checkpoint or snapshot_download(
            "lerobot/smolvla_robocasa",
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors"],
        )
        from vla_rl.provenance import provenance
        (args.output / "run_config.json").write_text(json.dumps({
            "arguments": vars(args), "provenance": provenance(ROOT, checkpoint)}, indent=2, default=str), encoding="utf-8")
        vla = FrozenSmolVLA(checkpoint, env, instruction=args.instruction or args.task)
        from vla_rl.diagnostics import fridge_diagnostics, EpisodeDiagnostics
        wrapped = ResidualEnv(
            env, vla, make_reward(args.task), scale=args.scale, horizon=args.horizon,
            object_observation=task_observation(args.task, args.observation_mode),
            diagnostics=task_diagnostics(args.task),
        )

        # Hook to capture the 3 camera views side-by-side when --save-video is enabled
        ep_frames = []
        if args.save_video:
            orig_obs_fn = wrapped._observation
            def _capture_obs(raw):
                if "pixels" in raw and isinstance(raw["pixels"], dict):
                    cams = [
                        np.asarray(raw["pixels"][k], dtype=np.uint8)
                        for k in sorted(raw["pixels"].keys())
                    ]
                    if cams:
                        ep_frames.append(np.concatenate(cams, axis=1))
                return orig_obs_fn(raw)
            wrapped._observation = _capture_obs

        td3_model = None
        if args.method == "residual_td3":
            from stable_baselines3 import TD3
            td3_model = TD3.load(str(args.model_path), device="cuda")
            if td3_model.observation_space.shape != wrapped.observation_space.shape:
                raise ValueError("Residual checkpoint observation shape does not match evaluation")

        episodes = []
        t0 = time.time()
        import itertools
        for seed, policy_seed, repeat in itertools.product(args.seeds, args.policy_seeds, range(args.repeats)):
            for ep_idx in range(args.episodes_per_seed):
                ep_seed = seed * 1000 + ep_idx if args.episodes_per_seed > 1 else seed
                ep_frames.clear()
                episode_start = time.perf_counter()
                obs, reset_info = wrapped.reset(seed=ep_seed, options={"policy_seed": policy_seed})
                import hashlib
                initial_qpos = env.unwrapped._env.env.sim.data.qpos.copy()
                initial_qvel = env.unwrapped._env.env.sim.data.qvel.copy()
                initial_model_hash = hashlib.sha256(env.unwrapped._env.env.sim.model.get_xml().encode()).hexdigest()
                from vla_rl.model_identity import numeric_model_identity
                initial_numeric_model = numeric_model_identity(env.unwrapped._env.env.sim.model)
                initial_state_hash = hashlib.sha256(initial_qpos.tobytes() + initial_qvel.tobytes()).hexdigest()
                initial_openness = wrapped.reward_provider.openness(env) if hasattr(wrapped.reward_provider, "openness") else None
                diagnostic_summary = EpisodeDiagnostics()
                total_reward = 0.0
                total_progress = 0.0
                steps = 0
                success = False
                done = False

                while not done:
                    if args.method == "base_vla":
                        residual_action = np.zeros(6, dtype=np.float32)
                    else:
                        residual_action, _ = td3_model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = wrapped.step(residual_action)
                    diagnostic_summary.update(info)
                    total_reward += float(reward)
                    total_progress += float(info["reward_terms"]["progress"])
                    steps += 1
                    success = bool(info.get("is_success", False))
                    done = bool(terminated or truncated)

                saved_video = None
                if args.save_video and ep_frames:
                    vid_file = (
                        args.output
                        / "videos"
                        / f"{args.task}_{args.method}_{args.perturbation}_seed{ep_seed}_policy{policy_seed}_repeat{repeat}.mp4"
                    )
                    saved_video = _save_rollout_video(np.stack(ep_frames, axis=0), vid_file)
                    print(f"Saved video to: {saved_video}", flush=True)

                ep_record = {
                    "seed": ep_seed, "policy_seed": policy_seed, "repeat": repeat,
                    "initial_state_sha256": initial_state_hash, "initial_model_sha256": initial_model_hash,
                    "initial_numeric_model": initial_numeric_model,
                    "initial_camera_sha256": reset_info.get("initial_camera_sha256"),
                    "instruction": args.instruction or args.task,
                    "wall_time_s": time.perf_counter() - episode_start,
                    "completion_steps": steps if success else None,
                    "success": success,
                    "steps": steps,
                    "total_reward": round(total_reward, 4),
                    "total_progress": round(total_progress, 4),
                    "initial_openness": initial_openness,
                    "final_openness": info["reward_terms"].get("door_openness"),
                    "final_task_error": info["reward_terms"].get("task_error", info["reward_terms"].get("door_openness", info["reward_terms"].get("dish_to_sink_distance"))),
                    "terminated": bool(terminated), "truncated": bool(truncated),
                    "perturbation_parameters": reset_info.get("perturbation"),
                    "video": saved_video,
                    "diagnostics": diagnostic_summary.summary(),
                }
                episodes.append(ep_record)
                with (args.output / "episodes.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(ep_record) + "\n")
                print(f"Episode seed={ep_seed}: success={success}, steps={steps}, "
                      f"reward={total_reward:.3f}, progress={total_progress:.3f}", flush=True)

        success_rate = float(np.mean([e["success"] for e in episodes]))
        mean_reward = float(np.mean([e["total_reward"] for e in episodes]))
        mean_progress = float(np.mean([e["total_progress"] for e in episodes]))
        mean_steps = float(np.mean([e["steps"] for e in episodes]))

        summary = {
            "task": args.task,
            "perturbation": args.perturbation,
            "method": args.method,
            "horizon": args.horizon, "scale": args.scale,
            "observation_mode": args.observation_mode,
            "checkpoint": str(checkpoint), "instruction_override": args.instruction,
            "policy_seeds": args.policy_seeds, "repeats": args.repeats,
            "failures": sum(not e["success"] for e in episodes),
            "training_seeds": contract.get("train_seeds"),
            "learner_seed": contract.get("seed"),
            "model_path": str(args.model_path) if args.model_path else None,
            "num_episodes": len(episodes),
            "success_rate": round(success_rate, 4),
            "mean_reward": round(mean_reward, 4),
            "mean_progress": round(mean_progress, 4),
            "mean_steps": round(mean_steps, 2),
            "wall_time_s": round(time.time() - t0, 2),
            "episodes": episodes,
        }
        (args.output / "eval_report.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
