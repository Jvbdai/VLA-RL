"""Physical and visual domain perturbations for RoboCasa MuJoCo environments.

Supports training-time physical randomization and held-out evaluation shifts across:
- Hinge resistance (joint damping and frictionloss on fixture doors)
- Contact friction (surface tangential/torsional/rolling friction on fixtures/grippers)
- Lighting (ambient and diffuse intensity scaling + position shifts)
- Surface appearance / texture (material/geom RGBA color and reflectance shifts)

Baseline values are snapshotted on the first reset of each underlying model and
restored before applying new multipliers so perturbations never compound across episodes.
"""

from dataclasses import dataclass
import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class PerturbationSpec:
    name: str = "nominal"
    hinge_damping_scale: float = 1.0
    hinge_frictionloss_add: float = 0.0
    contact_friction_scale: float = 1.0
    light_intensity_scale: float = 1.0
    light_pos_jitter: float = 0.0
    color_shift_strength: float = 0.0
    randomize_physical: bool = False

    def __post_init__(self):
        if self.hinge_damping_scale <= 0 or not np.isfinite(self.hinge_damping_scale):
            raise ValueError("hinge_damping_scale must be finite and positive")
        if self.hinge_frictionloss_add < 0 or not np.isfinite(self.hinge_frictionloss_add):
            raise ValueError("hinge_frictionloss_add must be finite and nonnegative")
        if self.contact_friction_scale <= 0 or not np.isfinite(self.contact_friction_scale):
            raise ValueError("contact_friction_scale must be finite and positive")
        if self.light_intensity_scale < 0 or not np.isfinite(self.light_intensity_scale):
            raise ValueError("light_intensity_scale must be finite and nonnegative")
        if self.light_pos_jitter < 0 or not np.isfinite(self.light_pos_jitter):
            raise ValueError("light_pos_jitter must be finite and nonnegative")
        if not 0.0 <= self.color_shift_strength <= 1.0:
            raise ValueError("color_shift_strength must lie in [0, 1]")


PRESETS = {
    "nominal": PerturbationSpec(name="nominal"),
    "phys_random_train": PerturbationSpec(
        name="phys_random_train",
        randomize_physical=True,
    ),
    "heldout_high_resistance": PerturbationSpec(
        name="heldout_high_resistance",
        hinge_damping_scale=3.0,
        hinge_frictionloss_add=1.5,
    ),
    "heldout_low_friction": PerturbationSpec(
        name="heldout_low_friction",
        contact_friction_scale=0.35,
    ),
    "heldout_dim_lighting": PerturbationSpec(
        name="heldout_dim_lighting",
        light_intensity_scale=0.35,
        light_pos_jitter=0.25,
    ),
    "heldout_texture_tint": PerturbationSpec(
        name="heldout_texture_tint",
        color_shift_strength=0.45,
    ),
    "heldout_combined": PerturbationSpec(
        name="heldout_combined",
        hinge_damping_scale=2.5,
        hinge_frictionloss_add=1.0,
        contact_friction_scale=0.4,
        light_intensity_scale=0.4,
        color_shift_strength=0.35,
    ),
}


class PerturbationWrapper(gym.Wrapper):
    """Applies repeatable physical and visual perturbations to a RoboCasa env on reset."""

    def __init__(self, env, perturbation_spec: PerturbationSpec = PRESETS["nominal"]):
        super().__init__(env)
        self.perturbation_spec = perturbation_spec
        self._baseline = None
        self._model_id = None
        self._episode_count = 0
        self.last_applied = {}

    @property
    def kitchen(self):
        return self.env.unwrapped._env.env

    def _snapshot_baseline(self, sim):
        model = sim.model
        self._model_id = id(model)
        self._baseline = {
            "dof_damping": model.dof_damping.copy(),
            "dof_frictionloss": model.dof_frictionloss.copy(),
            "geom_friction": model.geom_friction.copy(),
            "geom_rgba": model.geom_rgba.copy(),
            "mat_rgba": model.mat_rgba.copy() if model.nmat > 0 else None,
            "light_diffuse": model.light_diffuse.copy() if model.nlight > 0 else None,
            "light_ambient": model.light_ambient.copy() if model.nlight > 0 else None,
            "light_pos": model.light_pos.copy() if model.nlight > 0 else None,
        }

    def _restore_baseline(self, sim):
        model = sim.model
        if self._baseline is None or self._model_id != id(model):
            self._snapshot_baseline(sim)
            return
        model.dof_damping[:] = self._baseline["dof_damping"]
        model.dof_frictionloss[:] = self._baseline["dof_frictionloss"]
        model.geom_friction[:] = self._baseline["geom_friction"]
        model.geom_rgba[:] = self._baseline["geom_rgba"]
        if self._baseline["mat_rgba"] is not None:
            model.mat_rgba[:] = self._baseline["mat_rgba"]
        if self._baseline["light_diffuse"] is not None:
            model.light_diffuse[:] = self._baseline["light_diffuse"]
            model.light_ambient[:] = self._baseline["light_ambient"]
            model.light_pos[:] = self._baseline["light_pos"]

    def _fixture_dof_indices(self, kitchen):
        sim = kitchen.sim
        joint_names = []
        fxtr = getattr(kitchen, "fxtr", None)
        if fxtr is not None:
            for attr in ("_fridge_door_joint_names", "_door_joint_names", "door_joint_names"):
                names = getattr(fxtr, attr, None)
                if names:
                    joint_names.extend(names)
        dof_indices = []
        for name in set(joint_names):
            if name in sim.model.joint_names:
                jnt_id = sim.model.joint_name2id(name)
                dof_adr = sim.model.jnt_dofadr[jnt_id]
                dof_indices.append(int(dof_adr))
        return sorted(set(dof_indices))

    def _contact_geom_indices(self, kitchen):
        # Only the manipulated fixture and robot gripper; never floors, wheels,
        # unrelated fixtures, or all scene contacts.
        fixture = getattr(kitchen, "fxtr", None)
        if fixture is None:
            raise ValueError("Contact perturbation requires an explicit task fixture")
        names = list(fixture.contact_geoms)
        for robot in kitchen.robots:
            for gripper in robot.gripper.values():
                names.extend(gripper.contact_geoms)
        ids = sorted({int(kitchen.sim.model.geom_name2id(name)) for name in names})
        if not ids:
            raise ValueError("No target contact geometries found")
        return ids

    def _apply_perturbations(self, rng: np.random.Generator):
        kitchen = self.kitchen
        sim = kitchen.sim
        self._restore_baseline(sim)
        model = sim.model
        spec = self.perturbation_spec

        if spec.randomize_physical:
            damping_scale = float(rng.uniform(0.8, 1.8))
            frictionloss_add = float(rng.uniform(0.0, 0.35))
            friction_scale = float(rng.uniform(0.7, 1.4))
        else:
            damping_scale = spec.hinge_damping_scale
            frictionloss_add = spec.hinge_frictionloss_add
            friction_scale = spec.contact_friction_scale

        # 1. Hinge resistance on fixture joints
        dof_ids = self._fixture_dof_indices(kitchen)
        if not dof_ids and (damping_scale != 1 or frictionloss_add != 0):
            raise ValueError("No task joints found for resistance perturbation")
        for dof_id in dof_ids:
            model.dof_damping[dof_id] *= damping_scale
            model.dof_frictionloss[dof_id] += frictionloss_add

        # 2. Contact friction scaling
        # Identity must be exact: a lighting/hinge-only test must not alter friction.
        geom_ids = self._contact_geom_indices(kitchen) if friction_scale != 1.0 else []
        if geom_ids:
            model.geom_friction[geom_ids, 0] *= friction_scale

        # 3. Lighting perturbation
        if model.nlight > 0:
            model.light_diffuse[:] = np.clip(
                model.light_diffuse * spec.light_intensity_scale, 0.0, 2.0
            )
            model.light_ambient[:] = np.clip(
                model.light_ambient * spec.light_intensity_scale, 0.0, 2.0
            )
            if spec.light_pos_jitter > 0:
                jitter = rng.uniform(
                    -spec.light_pos_jitter,
                    spec.light_pos_jitter,
                    size=model.light_pos.shape,
                )
                model.light_pos[:] += jitter

        # 4. Texture / Material RGB appearance shift
        if spec.color_shift_strength > 0:
            alpha = spec.color_shift_strength
            tint = rng.uniform(0.15, 0.95, size=(1, 3)).astype(np.float32)
            model.geom_rgba[:, :3] = np.clip(
                (1.0 - alpha) * model.geom_rgba[:, :3] + alpha * tint, 0.0, 1.0
            )
            if model.nmat > 0:
                mat_tint = rng.uniform(0.15, 0.95, size=(model.nmat, 3)).astype(np.float32)
                model.mat_rgba[:, :3] = np.clip(
                    (1.0 - alpha) * model.mat_rgba[:, :3] + alpha * mat_tint, 0.0, 1.0
                )

        sim.forward()
        self.last_applied = {
            "preset": spec.name,
            "hinge_dofs": dof_ids,
            "contact_geom_ids": geom_ids,
            "actual_damping": model.dof_damping[dof_ids].tolist(),
            "actual_frictionloss": model.dof_frictionloss[dof_ids].tolist(),
            "actual_contact_friction": model.geom_friction[geom_ids].tolist(),
            "damping_scale": damping_scale,
            "frictionloss_add": frictionloss_add,
            "contact_friction_scale": friction_scale,
            "light_intensity_scale": spec.light_intensity_scale,
            "color_shift_strength": spec.color_shift_strength,
        }

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        spec = self.perturbation_spec
        base_seed = int(info.get("episode_seed", 0 if seed is None else seed))
        rng_seed = base_seed + self._episode_count if spec.randomize_physical else base_seed
        self._episode_count += 1
        rng = np.random.default_rng(rng_seed)
        self._apply_perturbations(rng)
        if spec.light_intensity_scale != 1.0 or spec.color_shift_strength > 0 or spec.light_pos_jitter > 0:
            raw_env = self.env.unwrapped
            gym_env = raw_env._env
            kitchen = gym_env.env
            raw = (kitchen.viewer._get_observations(force_update=True)
                   if kitchen.viewer_get_obs else kitchen._get_observations(force_update=True))
            obs = raw_env._format_raw_obs(gym_env.get_observation(raw))
        self.last_applied["physics_seed"] = rng_seed
        info = dict(info, perturbation=self.last_applied)
        return obs, info
