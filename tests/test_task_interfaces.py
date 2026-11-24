from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rl.task_interfaces import BOTTLES, rack_state, sorting_state, TaskProgressReward, RackObservation, SortingObservation


class TaskInterfaceTests(unittest.TestCase):
    def fixture(self):
        positions = np.zeros((7, 3))
        positions[0:2, 0] = -1
        positions[3:5, 0] = 1
        data = NS(body_xpos=positions, body_xmat=np.tile(np.eye(3).ravel(), (7, 1)),
                  site_xpos=np.zeros((1, 3)), site_xmat=np.eye(3).reshape(1, 9), get_joint_qvel=lambda n: .1)
        return NS(sim=NS(data=data, model=NS(joint_name2id=lambda n: 0, jnt_bodyid=[0])),
                  obj_body_id={name: i for i, name in enumerate(BOTTLES)}, choice="alcohol",
                  robots=[NS(eef_site_id={"right": 0})])

    def env(self, kitchen):
        return NS(unwrapped=NS(_env=NS(env=kitchen)))

    def test_rack_direction_changes_error_and_observation(self):
        k = self.fixture()
        values = {"rack": .2, "door": 1.}
        k.dishwasher = NS(_joint_names={"rack": "rack", "door": "door"}, get_joint_state=lambda *a: values)
        k.should_pull = True
        self.assertAlmostEqual(rack_state(k)["rack_remaining_fraction"], .8)
        reward = TaskProgressReward("SlideDishwasherRack")
        reward.reset(self.env(k))
        values["rack"] = .4
        self.assertAlmostEqual(reward.step(self.env(k), False, np.zeros(6))[0], 1.)
        k.should_pull = False
        self.assertAlmostEqual(rack_state(k)["rack_remaining_fraction"], .4)
        self.assertEqual(RackObservation()(self.env(k)).shape, (16,))

    def test_sorting_uses_correct_category_and_zero_error_is_not_success(self):
        k = self.fixture()
        k.sim.data.body_xpos[2, 0] = -.8
        k.sim.data.body_xpos[5:7, 0] = .8
        env = self.env(k)
        self.assertEqual(sorting_state(k)["mean_placement_excess_m"], 0.)
        reward = TaskProgressReward("RecycleBottlesByType")
        reward.reset(env)
        _, terms = reward.step(env, False, np.zeros(6))
        self.assertEqual(terms["success_bonus"], 0.)
        k.choice = "bottled_water"
        self.assertGreater(sorting_state(k)["mean_placement_excess_m"], 0.)
        self.assertEqual(SortingObservation()(env).shape, (88,))
        k.choice = "unknown"
        with self.assertRaises(ValueError):
            sorting_state(k)
