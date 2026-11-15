"""Run the upstream SmolVLA RoboCasa evaluator with the checkpoint camera mapping."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, help="Override episode horizon for a smoke test")
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="Save three-camera video and per-step door/action data")
    parser.add_argument("--instruction", default="CloseFridge", help="Language input; defaults to the checkpoint dataset task label")
    args = parser.parse_args()
    if args.episodes < 1 or (args.steps is not None and args.steps < 1):
        parser.error("episodes and steps must be positive")
    root = Path(__file__).resolve().parents[1]
    # Running from the clone avoids the outer clone folder shadowing its package.
    simulator = root / "robocasa"
    if not (simulator / "robocasa" / "__init__.py").exists():
        parser.error(f"Expected RoboCasa checkout at {simulator}")
    output = root / "runs" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S-smolvla")
    camera_map = {
        "observation.images.robot0_agentview_left": "observation.images.camera1",
        "observation.images.robot0_agentview_right": "observation.images.camera2",
        "observation.images.robot0_eye_in_hand": "observation.images.camera3",
    }
    command = [
        sys.executable, "-m", "lerobot.scripts.lerobot_eval",
        "--policy.path=lerobot/smolvla_robocasa",
        "--env.type=robocasa", "--env.task=CloseFridge",
        "--eval.batch_size=1", f"--eval.n_episodes={args.episodes}",
        "--eval.use_async_envs=false", f"--policy.device={args.device}",
        f"--seed={args.seed}", f"--output_dir={output}",
        "--rename_map=" + json.dumps(camera_map),
    ]
    if args.steps is not None:
        command.append(f"--env.episode_length={args.steps}")
    print("Interpreter:", sys.executable, flush=True)
    print("Output directory:", output, flush=True)
    print("This runs a frozen pretrained policy and saves video; it does not train RL.", flush=True)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Activate vla-rl with GPU PyTorch, or pass --device cpu.")
    # LeRobot converts policy.path to pathlib.Path. On Windows a Hub ID then
    # becomes 'lerobot\\smolvla_robocasa', which the Hub rejects. Resolve the
    # repository to a complete local snapshot before handing it to LeRobot.
    from huggingface_hub import snapshot_download
    print("Resolving pretrained checkpoint (cached after first download)...", flush=True)
    checkpoint = snapshot_download(
        repo_id="lerobot/smolvla_robocasa",
        allow_patterns=["*.json", "*.safetensors"],
    )
    command[3] = f"--policy.path={Path(checkpoint).resolve()}"
    print("Checkpoint:", checkpoint, flush=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["MUJOCO_GL"] = "glfw" if os.name == "nt" else env.get("MUJOCO_GL", "egl")
    if args.diagnostic or args.instruction is not None:
        env["VLA_DIAGNOSTIC_DIR"] = str(output / "diagnostics")
        if args.instruction is not None:
            env["VLA_INSTRUCTION"] = args.instruction
        else:
            env.pop("VLA_INSTRUCTION", None)
        command = [sys.executable, str(root / "scripts" / "diagnostic_eval.py"), *command[3:]]
    result = subprocess.run(command, cwd=simulator, env=env)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
