"""Inspect repeated seeded simulator resets without loading a policy."""
import argparse
import difflib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT / "robocasa")
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() != ROOT]
    os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(args.output / "numba-cache"))
    import numpy as np
    from lerobot.sim.integrations.robocasa import RoboCasaEnv
    from vla_rl.seed_schedule import seeded_reset
    env = RoboCasaEnv(task="CloseFridge", episode_length=900)
    try:
        from vla_rl.model_identity import numeric_model_identity
        snapshots, xmls, states, identities = [], [], [], []
        for repeat in range(2):
            obs, _ = seeded_reset(env.reset, args.seed)
            sim = env._env.env.sim
            model = sim.model._model
            identities.append(numeric_model_identity(model))
            arrays = {}
            for name in dir(model):
                value = getattr(model, name)
                if isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
                    arrays[name] = value.copy()
            snapshots.append(arrays)
            states.append((sim.data.qpos.copy(), sim.data.qvel.copy()))
            xmls.append(sim.model.get_xml())
            (args.output / f"reset-{repeat}.xml").write_text(xmls[-1], encoding="utf-8")
        differences = {}
        for name, before in snapshots[0].items():
            after = snapshots[1][name]
            if before.shape != after.shape:
                differences[name] = {"shapes": [before.shape, after.shape]}
            elif not np.array_equal(before, after, equal_nan=True):
                finite = np.isfinite(before) & np.isfinite(after)
                delta = np.abs(before[finite].astype(float) - after[finite].astype(float))
                differences[name] = {"changed_finite": int(np.count_nonzero(delta)),
                                     "max_finite_difference": float(delta.max()) if delta.size else None,
                                     "nonfinite_mask_changed": bool(np.any(np.isfinite(before) != np.isfinite(after)))}
        report = {"seed": args.seed, "qpos_exact": bool(np.array_equal(states[0][0], states[1][0])),
                  "qvel_exact": bool(np.array_equal(states[0][1], states[1][1])),
                  "model_array_differences": differences,
                  "numeric_identity_equal": identities[0] == identities[1],
                  "numeric_identities": identities}
        (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (args.output / "model.diff").write_text("".join(difflib.unified_diff(xmls[0].splitlines(True), xmls[1].splitlines(True))), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        env.close()

if __name__ == "__main__":
    main()
