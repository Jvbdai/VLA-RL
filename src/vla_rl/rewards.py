"""CloseFridge training reward. Privileged state never enters policy observations."""

import numpy as np


class CloseFridgeReward:
    """5 * signed mean door closure progress + 10 * success - .001 * ||delta||².

    Signed progress penalizes reopening and telescopes without discounting; it is
    not claimed to preserve the optimal policy under TD3's discounted objective.
    The upstream predicate requires every fridge door to close. Freezer excluded.
    """

    @staticmethod
    def openness(env):
        kitchen = env.unwrapped._env.env
        if getattr(env.unwrapped, "task", None) != "CloseFridge":
            raise ValueError("CloseFridgeReward only supports CloseFridge")
        names = kitchen.fxtr._fridge_door_joint_names
        if not names:
            raise ValueError("no fridge door joints found")
        # Upstream handles opposite hinge directions and different joint ranges.
        states = kitchen.fxtr.get_joint_state(env=kitchen, joint_names=names)
        values = np.asarray([states[name] for name in names], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite fridge joint state")
        return float(np.clip(values, 0, 1).mean())

    def reset(self, env):
        self.previous = self.openness(env)

    def step(self, env, success, correction):
        current = self.openness(env)
        terms = {
            "door_openness": current,
            "progress": 5.0 * (self.previous - current),
            "success_bonus": 10.0 * bool(success),
            "residual_cost": -0.001 * float(np.square(correction).sum()),
        }
        self.previous = current
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms
