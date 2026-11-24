"""Task shaping from simulator state; privileged residual inputs are explicitly configured."""

import numpy as np


def _mean_joint_openness(kitchen, joint_names, label="door"):
    if not joint_names:
        raise ValueError(f"no {label} joints found")
    states = kitchen.fxtr.get_joint_state(env=kitchen, joint_names=joint_names)
    values = np.asarray([states[name] for name in joint_names], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite {label} joint state")
    return float(np.clip(values, 0.0, 1.0).mean())


class CloseFridgeReward:
    """5 * signed mean door closure progress + 10 * success - action_weight * ||delta||².

    Signed progress penalizes reopening and telescopes without discounting; it is
    not claimed to preserve the optimal policy under TD3's discounted objective.
    The upstream predicate requires every fridge door to close. Freezer excluded.
    """

    def __init__(self, action_weight: float = 0.001):
        self.action_weight = float(action_weight)
        self.previous = 0.0

    @staticmethod
    def openness(env):
        kitchen = env.unwrapped._env.env
        if getattr(env.unwrapped, "task", None) != "CloseFridge":
            raise ValueError("CloseFridgeReward only supports CloseFridge")
        names = kitchen.fxtr._fridge_door_joint_names
        return _mean_joint_openness(kitchen, names, label="fridge door")

    def reset(self, env):
        self.previous = self.openness(env)

    def step(self, env, success, correction):
        current = self.openness(env)
        terms = {
            "door_openness": current,
            "progress": 5.0 * (self.previous - current),
            "success_bonus": 10.0 * bool(success),
            "residual_cost": -self.action_weight * float(np.square(correction).sum()),
        }
        self.previous = current
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms


class OpenFridgeReward:
    """5 * signed mean door opening progress + 10 * success - action_weight * ||delta||²."""

    def __init__(self, action_weight: float = 0.001):
        self.action_weight = float(action_weight)
        self.previous = 0.0

    @staticmethod
    def openness(env):
        kitchen = env.unwrapped._env.env
        if getattr(env.unwrapped, "task", None) != "OpenFridge":
            raise ValueError("OpenFridgeReward only supports OpenFridge")
        names = kitchen.fxtr._fridge_door_joint_names
        return _mean_joint_openness(kitchen, names, label="fridge door")

    def reset(self, env):
        self.previous = self.openness(env)

    def step(self, env, success, correction):
        current = self.openness(env)
        terms = {
            "door_openness": current,
            "progress": 5.0 * (current - self.previous),
            "success_bonus": 10.0 * bool(success),
            "residual_cost": -self.action_weight * float(np.square(correction).sum()),
        }
        self.previous = current
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms


class OpenCabinetReward:
    """5 * signed mean cabinet door opening progress + 10 * success - action_weight * ||delta||²."""

    def __init__(self, action_weight: float = 0.001):
        self.action_weight = float(action_weight)
        self.previous = 0.0

    @staticmethod
    def openness(env):
        kitchen = env.unwrapped._env.env
        task = getattr(env.unwrapped, "task", None)
        if task not in ("OpenCabinet", "OpenSingleDoor", "OpenDoubleDoor"):
            raise ValueError(f"OpenCabinetReward does not support {task}")
        fxtr = kitchen.fxtr
        names = (
            getattr(fxtr, "door_joint_names", None)
            or getattr(fxtr, "_door_joint_names", None)
        )
        return _mean_joint_openness(kitchen, names, label="cabinet door")

    def reset(self, env):
        self.previous = self.openness(env)

    def step(self, env, success, correction):
        current = self.openness(env)
        terms = {
            "door_openness": current,
            "progress": 5.0 * (current - self.previous),
            "success_bonus": 10.0 * bool(success),
            "residual_cost": -self.action_weight * float(np.square(correction).sum()),
        }
        self.previous = current
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms


class PlaceDishInSinkReward:
    """5 * signed dish-to-sink XY/3D distance reduction + 10 * success - action_weight * ||delta||²."""

    def __init__(self, action_weight: float = 0.001):
        self.action_weight = float(action_weight)
        self.previous = 0.0

    @staticmethod
    def distance(env):
        kitchen = env.unwrapped._env.env
        obj_pos = None
        for key in ("obj", "dish", "item"):
            if hasattr(kitchen, "objects") and key in kitchen.objects:
                body_id = kitchen.obj_body_id[key]
                obj_pos = np.asarray(kitchen.sim.data.body_xpos[body_id], dtype=np.float64)
                break
        if obj_pos is None and hasattr(kitchen, "obj_body_id") and kitchen.obj_body_id:
            first_key = next(iter(kitchen.obj_body_id))
            obj_pos = np.asarray(
                kitchen.sim.data.body_xpos[kitchen.obj_body_id[first_key]], dtype=np.float64
            )
        sink = getattr(kitchen, "sink", None) or getattr(kitchen, "fxtr", None)
        if obj_pos is None or sink is None or not hasattr(sink, "pos"):
            raise ValueError("unable to resolve dish and sink positions")
        sink_pos = np.asarray(sink.pos, dtype=np.float64)
        dist = float(np.linalg.norm(obj_pos - sink_pos))
        if not np.isfinite(dist):
            raise ValueError("nonfinite dish-to-sink distance")
        return dist

    def reset(self, env):
        self.previous = self.distance(env)

    def step(self, env, success, correction):
        current = self.distance(env)
        terms = {
            "dish_to_sink_distance": current,
            "progress": 5.0 * (self.previous - current),
            "success_bonus": 10.0 * bool(success),
            "residual_cost": -self.action_weight * float(np.square(correction).sum()),
        }
        self.previous = current
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms


REWARD_BY_TASK = {
    "CloseFridge": CloseFridgeReward,
    "OpenFridge": OpenFridgeReward,
    "OpenCabinet": OpenCabinetReward,
    "PlaceDishInSink": PlaceDishInSinkReward,
}


def make_reward(task: str, action_weight: float = 0.05):
    if task in ("SlideDishwasherRack", "RecycleBottlesByType"):
        from vla_rl.task_interfaces import TaskProgressReward
        return TaskProgressReward(task, action_weight)
    if task not in REWARD_BY_TASK:
        raise ValueError(f"Unsupported task {task!r}; expected one of {sorted(REWARD_BY_TASK)}")
    return REWARD_BY_TASK[task](action_weight=action_weight)