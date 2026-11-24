"""Privileged state and progress shaping for approved non-fridge tasks.

Success is always provided by the simulator, never inferred from these errors.
"""
import numpy as np

TASK_HORIZONS = {"CloseFridge": 900, "SlideDishwasherRack": 450, "RecycleBottlesByType": 2850}
BOTTLES = ("bottle_plastic1", "bottle_plastic2", "bottle_plastic_middle",
           "bottle_glass1", "bottle_glass2", "bottle_glass_middle", "mystery_middle")


def kitchen_of(env):
    return env.unwrapped._env.env


def rack_state(kitchen):
    names = kitchen.dishwasher._joint_names
    values = kitchen.dishwasher.get_joint_state(kitchen, [names["rack"], names["door"]])
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Nonfinite dishwasher state")
    target = float(kitchen.should_pull)
    position = float(values[names["rack"]])
    return {"rack_position_fraction": position, "rack_target_fraction": target,
            "rack_remaining_fraction": abs(target - position),
            "dishwasher_door_openness": float(values[names["door"]])}


def sorting_state(kitchen):
    if kitchen.choice not in ("alcohol", "bottled_water"):
        raise ValueError(f"Unknown mystery bottle category: {kitchen.choice}")
    positions = {name: np.asarray(kitchen.sim.data.body_xpos[kitchen.obj_body_id[name]]) for name in BOTTLES}
    if not all(p.shape == (3,) and np.isfinite(p).all() for p in positions.values()):
        raise ValueError("Invalid bottle positions")
    categories = {"bottle_plastic_middle": "plastic", "bottle_glass_middle": "glass",
                  "mystery_middle": "glass" if kitchen.choice == "alcohol" else "plastic"}
    distances = {name: min(float(np.linalg.norm(positions[name][:2] - positions[f"bottle_{category}{i}"][:2]))
                           for i in (1, 2)) for name, category in categories.items()}
    excess = {name: max(0., distance - .30) for name, distance in distances.items()}
    return {"nearest_correct_cluster_distance_m": distances,
            "placement_excess_m": excess, "mean_placement_excess_m": float(np.mean(list(excess.values()))),
            "max_placement_excess_m": max(excess.values()), "mystery_is_glass": kitchen.choice == "alcohol"}


def hand_frame(kitchen):
    site = kitchen.robots[0].eef_site_id["right"]
    return np.asarray(kitchen.sim.data.site_xpos[site]), np.asarray(kitchen.sim.data.site_xmat[site]).reshape(3, 3)


def relative_body_pose(kitchen, body, hand_pos, hand_rot):
    data = kitchen.sim.data
    return np.concatenate((hand_rot.T @ (data.body_xpos[body] - hand_pos),
                           (hand_rot.T @ np.asarray(data.body_xmat[body]).reshape(3, 3)).ravel()))


class RackObservation:
    size = 16
    schema = "dishwasher_rack_ground_truth_v1"

    def __call__(self, env):
        kitchen = kitchen_of(env)
        state = rack_state(kitchen)
        joint_name = kitchen.dishwasher._joint_names["rack"]
        joint = kitchen.sim.model.joint_name2id(joint_name)
        pose = relative_body_pose(kitchen, kitchen.sim.model.jnt_bodyid[joint], *hand_frame(kitchen))
        return np.concatenate(([state["rack_target_fraction"], state["rack_position_fraction"],
                                float(kitchen.sim.data.get_joint_qvel(joint_name)),
                                state["dishwasher_door_openness"]], pose)).astype(np.float32)


class SortingObservation:
    size = 88
    schema = "bottle_sort_ground_truth_v1"

    def __call__(self, env):
        kitchen = kitchen_of(env)
        state = sorting_state(kitchen)
        frame = hand_frame(kitchen)
        poses = [relative_body_pose(kitchen, kitchen.obj_body_id[name], *frame) for name in BOTTLES]
        return np.concatenate((*poses, [float(state["mystery_is_glass"])],
                               list(state["nearest_correct_cluster_distance_m"].values()))).astype(np.float32)


class TaskProgressReward:
    def __init__(self, task, action_weight=.05):
        if task not in ("SlideDishwasherRack", "RecycleBottlesByType"):
            raise ValueError(f"Unsupported progress task: {task}")
        self.task, self.action_weight = task, action_weight
        self.previous = None

    def state(self, env):
        kitchen = kitchen_of(env)
        if self.task == "SlideDishwasherRack":
            state = rack_state(kitchen)
            error = state["rack_remaining_fraction"]
        else:
            state = sorting_state(kitchen)
            error = state["mean_placement_excess_m"]
        if not np.isfinite(error):
            raise ValueError("Nonfinite task error")
        return state, error

    def reset(self, env):
        _, self.previous = self.state(env)

    def step(self, env, success, correction):
        state, error = self.state(env)
        terms = dict(state, task_error=error, progress=5 * (self.previous - error),
                     success_bonus=10 * bool(success),
                     residual_cost=-self.action_weight * float(np.square(correction).sum()))
        self.previous = error
        return sum(terms[k] for k in ("progress", "success_bonus", "residual_cost")), terms


def task_observation(task, mode):
    if mode == "proprio":
        return None
    if task in ("CloseFridge", "OpenFridge"):
        from vla_rl.door_observation import FridgeDoorObservation
        return FridgeDoorObservation()
    if mode != "task_ground_truth":
        raise ValueError("Non-fridge tasks require task_ground_truth or proprio")
    return {"SlideDishwasherRack": RackObservation, "RecycleBottlesByType": SortingObservation}[task]()


def task_diagnostics(task):
    if task == "CloseFridge":
        from vla_rl.diagnostics import fridge_diagnostics
        return fridge_diagnostics
    if task == "SlideDishwasherRack":
        def rack(env):
            k = kitchen_of(env)
            return dict(rack_state(k), gripper_fixture_contact=any(
                k.check_contact(g, k.dishwasher) for r in k.robots for g in r.gripper.values()))
        return rack
    if task == "RecycleBottlesByType":
        def sorting(env):
            import robocasa.utils.object_utils as OU
            k = kitchen_of(env)
            return dict(sorting_state(k),
                        bottles_on_table={n: bool(OU.check_obj_fixture_contact(k, n, k.dining_counter)) for n in BOTTLES},
                        gripper_far_from_bottles={n: bool(OU.gripper_obj_far(k, obj_name=n)) for n in BOTTLES})
        return sorting
    return None
