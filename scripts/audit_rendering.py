"""Replay identical recorded actions and measure pixels without a VLA in the loop."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.actions, args.output = args.actions.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT / "robocasa")
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output / "numba-cache"))
    import numpy as np
    from lerobot.sim.integrations.robocasa import RoboCasaEnv
    from vla_rl.seed_schedule import seeded_reset
    from vla_rl.model_identity import numeric_model_identity
    actions = np.load(args.actions)
    if actions.ndim != 2 or actions.shape[1] != 12 or not np.isfinite(actions).all():
        raise ValueError("Expected finite N x 12 actions")
    env = RoboCasaEnv(task="CloseFridge", episode_length=len(actions))
    reference, identities, camera_diffs, rerender_diffs = [], [], [], []
    def differences(a, b):
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
        return {"max": int(delta.max()), "changed_values": int(np.count_nonzero(delta)),
                "mean": float(delta.mean())}
    try:
        for repeat in range(2):
            raw, _ = seeded_reset(env.reset, args.seed)
            sim = env._env.env.sim
            identities.append(numeric_model_identity(sim.model))
            for step in range(len(actions) + 1):
                if repeat == 0:
                    reference.append((sim.data.qpos.copy(), sim.data.qvel.copy(),
                                      {k: v.copy() for k, v in raw["pixels"].items()}))
                else:
                    qpos, qvel, pixels = reference[step]
                    np.testing.assert_array_equal(qpos, sim.data.qpos, err_msg=f"qpos at {step}")
                    np.testing.assert_array_equal(qvel, sim.data.qvel, err_msg=f"qvel at {step}")
                    camera_diffs.append({"step": step, "cameras": {
                        k: differences(pixels[k], raw["pixels"][k]) for k in pixels}})
                if step % 10 == 0:
                    for camera in raw["pixels"]:
                        a = sim.render(256, 256, camera_name=camera).copy()
                        b = sim.render(256, 256, camera_name=camera).copy()
                        rerender_diffs.append({"repeat": repeat, "step": step, "camera": camera,
                                               **differences(a, b)})
                if step < len(actions):
                    raw, *_ = env.step(actions[step])
        report = {"seed": args.seed, "steps": len(actions), "exact_qpos_qvel": True,
                  "numeric_models_equal": identities[0] == identities[1],
                  "numeric_models": identities, "between_reset_camera_differences": camera_diffs,
                  "same_state_same_context_rerenders": rerender_diffs}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"steps": len(actions), "exact_qpos_qvel": True,
            "numeric_models_equal": report["numeric_models_equal"],
            "max_between_reset_pixel_difference": max(d["max"] for s in camera_diffs for d in s["cameras"].values()),
            "max_same_context_rerender_difference": max(d["max"] for d in rerender_diffs)}, indent=2))
    finally:
        env.close()

if __name__ == "__main__":
    main()
