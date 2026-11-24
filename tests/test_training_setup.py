import random
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rl.door_observation import FridgeDoorObservation
from vla_rl.seed_schedule import EpisodeSeedSchedule, seeded_reset
from vla_rl.residual import ResidualEnv
from vla_rl.rewards import CloseFridgeReward
from test_residual import KitchenStub, ChunkStub


class SetupTests(unittest.TestCase):
    def test_contact_targets_exclude_unrelated_geometry(self):
        from vla_rl.perturbations import PerturbationWrapper
        model = SimpleNamespace(geom_name2id={"door": 1, "finger": 3}.__getitem__)
        kitchen = SimpleNamespace(sim=SimpleNamespace(model=model),
            fxtr=SimpleNamespace(contact_geoms=["door"]),
            robots=[SimpleNamespace(gripper={"right": SimpleNamespace(contact_geoms=["finger"])})])
        self.assertEqual(PerturbationWrapper._contact_geom_indices(None, kitchen), [1, 3])

    def test_seed_pool_cycles_without_repeating_first_seed_every_reset(self):
        schedule = EpisodeSeedSchedule([200, 201, 202])
        self.assertEqual([schedule.next() for _ in range(5)], [200, 201, 202, 200, 201])

    def test_lazy_construction_does_not_shift_first_episode_rng(self):
        class Lazy:
            initialized = False
            def _ensure_env(self):
                if not self.initialized:
                    np.random.random(100)
                    random.random()
                    self.initialized = True
            def reset(self, seed):
                self._ensure_env()
                return [float(np.random.random()), random.random()], {}
        env = Lazy()
        first, _ = seeded_reset(env.reset, 100)
        second, _ = seeded_reset(env.reset, 100)
        self.assertEqual(first, second)

    def test_environment_reset_does_not_reset_exploration_rng(self):
        random.seed(7)
        np.random.seed(7)
        state = random.getstate()
        numpy_state = np.random.get_state()
        def reset(seed):
            return [random.random(), float(np.random.random())], {}
        a, info = seeded_reset(reset, 123)
        b, _ = seeded_reset(reset, 123)
        self.assertEqual(a, b)
        self.assertEqual(info["episode_seed"], 123)
        self.assertEqual(random.getstate(), state)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])

    def test_extra_observations_do_not_change_base_actions(self):
        class Features:
            size = 2
            def __call__(self, env):
                return np.array(list(env.unwrapped.doors.values()))
        plain = ResidualEnv(KitchenStub(), ChunkStub(), CloseFridgeReward())
        enhanced = ResidualEnv(KitchenStub(), ChunkStub(), CloseFridgeReward(), object_observation=Features())
        a, _ = plain.reset(seed=0)
        b, _ = enhanced.reset(seed=0)
        self.assertEqual(b.shape, (31,))
        np.testing.assert_array_equal(a, b[:29])
        for _ in range(3):
            a, *rest_a = plain.step(np.zeros(6))
            b, *rest_b = enhanced.step(np.zeros(6))
            np.testing.assert_array_equal(rest_a[-1]["executed_action"], rest_b[-1]["executed_action"])
            np.testing.assert_array_equal(a, b[:29])

    def test_door_features_track_geometry_and_pad_absent_second_door(self):
        data = SimpleNamespace(site_xpos=np.array([[1., 0, 0]]),
            site_xmat=np.eye(3).reshape(1, 9), body_xpos=np.array([[2., 3., 0]]),
            body_xmat=np.eye(3).reshape(1, 9), get_joint_qvel=lambda n: .2)
        model = SimpleNamespace(joint_name2id=lambda n: 0, jnt_bodyid=[0])
        fxtr = SimpleNamespace(_fridge_door_joint_names=["door"],
                              get_joint_state=lambda env, names: {"door": .5})
        kitchen = SimpleNamespace(sim=SimpleNamespace(data=data, model=model), fxtr=fxtr,
                                  robots=[SimpleNamespace(eef_site_id={"right": 0})])
        env = SimpleNamespace(unwrapped=SimpleNamespace(_env=SimpleNamespace(env=kitchen)))
        a = FridgeDoorObservation()(env)
        np.testing.assert_allclose(a[:6], [1, .5, .2, 1, 3, 0])
        np.testing.assert_array_equal(a[15:], np.zeros(15))
        data.body_xpos[0, 0] += 1
        b = FridgeDoorObservation()(env)
        self.assertEqual(float(b[3] - a[3]), 1.)


if __name__ == "__main__":
    unittest.main()
