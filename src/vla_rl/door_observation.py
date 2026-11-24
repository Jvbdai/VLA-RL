"""Explicit simulator-state ablation: fridge geometry supplied to the residual only.

Each of two door slots has: valid mask, normalized openness, angular velocity,
door-body origin relative to the hand (3), and door rotation relative to hand (9).
Missing slots are zero. This is NOT camera-based pose estimation.
"""
import numpy as np


class FridgeDoorObservation:
    size = 30
    schema = "fridge_ground_truth_v1"

    def __call__(self, env):
        kitchen = env.unwrapped._env.env
        names = sorted(kitchen.fxtr._fridge_door_joint_names)
        if not 1 <= len(names) <= 2:
            raise ValueError(f"Expected one or two fridge doors, found {names}")
        sim = kitchen.sim
        robot = kitchen.robots[0]
        site = robot.eef_site_id["right"]
        hand_pos = np.asarray(sim.data.site_xpos[site])
        hand_rot = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
        openness = kitchen.fxtr.get_joint_state(kitchen, names)
        result = np.zeros((2, 15), dtype=np.float32)
        for i, name in enumerate(names):
            joint = sim.model.joint_name2id(name)
            body = sim.model.jnt_bodyid[joint]
            body_rot = np.asarray(sim.data.body_xmat[body]).reshape(3, 3)
            relative_pos = hand_rot.T @ (sim.data.body_xpos[body] - hand_pos)
            relative_rot = hand_rot.T @ body_rot
            result[i] = np.concatenate(([1., openness[name], float(sim.data.get_joint_qvel(name))],
                                        relative_pos, relative_rot.ravel()))
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite fridge observations")
        return result.ravel()
