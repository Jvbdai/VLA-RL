"""Behavioral tests without loading cameras, model weights, or simulator assets."""

from pathlib import Path
from types import SimpleNamespace
import importlib.util
import sys
import unittest

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rl.residual import ResidualEnv, compose_action
from vla_rl.rewards import CloseFridgeReward


class KitchenStub(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (12,), np.float32)
    observation_space = gym.spaces.Dict({"agent_pos": gym.spaces.Box(-np.inf, np.inf, (16,), np.float32)})
    task = "CloseFridge"

    def __init__(self):
        self.doors = {"left": 1.0, "right": 1.0}
        fixture = SimpleNamespace(_fridge_door_joint_names=["left", "right"],
                                  get_joint_state=lambda **kwargs: self.doors.copy())
        self._env = SimpleNamespace(env=SimpleNamespace(fxtr=fixture))
        self.actions = []

    def observation(self):
        return {"agent_pos": np.full(16, self.time, np.float32), "privileged": self.doors.copy()}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.doors = {"left": 1.0, "right": 1.0}
        self.actions = []
        return self.observation(), {}

    def step(self, action):
        self.actions.append(action.copy())
        self.time += 1
        self.doors = {k: max(0, v - .1) for k, v in self.doors.items()}
        return self.observation(), 0., False, False, {"is_success": False}


class ChunkStub:
    def reset(self, seed=None):
        self.calls = 0

    def action(self, obs):
        position = (self.calls % 3) / 2
        self.calls += 1
        return np.full(12, self.calls / 10, np.float32), position


class ResidualTests(unittest.TestCase):
    def make_env(self, **kwargs):
        return ResidualEnv(KitchenStub(), ChunkStub(), CloseFridgeReward(), **kwargs)

    def test_zero_residual_exactly_matches_direct_rollout(self):
        direct, base = KitchenStub(), ChunkStub()
        raw, _ = direct.reset(seed=3)
        base.reset(3)
        wrapped = self.make_env(horizon=10)
        wrapped.reset(seed=3)
        for _ in range(6):
            action, _ = base.action(raw)
            raw, *_ = direct.step(action)
            obs, *_ = wrapped.step(np.zeros(6))
            np.testing.assert_array_equal(wrapped.env.actions[-1], direct.actions[-1])
            np.testing.assert_array_equal(obs[:16], raw["agent_pos"])

    def test_only_arm_changes_and_is_bounded_at_saturation(self):
        base = np.linspace(-1, 1, 12, dtype=np.float32)
        base[5:11] = [.95, -.95, 1.5, -1.5, 0, 0]
        out = compose_action(base, np.array([1, -1, -1, 1, 1, -1]), .1,
                             np.full(12, -1), np.full(12, 1))
        np.testing.assert_array_equal(out[[0, 1, 2, 3, 4, 11]], base[[0, 1, 2, 3, 4, 11]])
        self.assertTrue(np.all(np.abs(out[5:11] - base[5:11]) <= .100001))
        self.assertEqual(out[5], 1)
        self.assertEqual(out[6], -1)
        np.testing.assert_array_equal(compose_action(base, np.zeros(6), .1,
                                                     np.full(12, -1), np.full(12, 1)), base)

    def test_observation_describes_current_pending_action_and_chunk(self):
        env = self.make_env()
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (29,))
        np.testing.assert_array_equal(obs[16:28], np.full(12, .1, np.float32))
        self.assertEqual(obs[-1], 0)
        for expected in (.5, 1, 0):
            obs, *_ = env.step(np.zeros(6))
            self.assertEqual(obs[-1], expected)
            np.testing.assert_array_equal(obs[16:28], env.pending_action)

    def test_privileged_changes_do_not_change_observation(self):
        a, b = self.make_env(), self.make_env()
        a.reset()
        b.reset()
        b.env.doors = {"left": 0, "right": 0}
        np.testing.assert_array_equal(a._observation(a.env.observation()),
                                      b._observation(b.env.observation()))

    def test_timeout_retains_next_action_and_requires_explicit_reset(self):
        env = self.make_env(horizon=1)
        env.reset()
        obs, _, term, trunc, _ = env.step(np.zeros(6))
        self.assertFalse(term)
        self.assertTrue(trunc)
        np.testing.assert_array_equal(obs[:16], np.ones(16))
        np.testing.assert_array_equal(obs[16:28], np.full(12, .2, np.float32))
        with self.assertRaises(RuntimeError):
            env.step(np.zeros(6))
        obs, _ = env.reset()
        self.assertEqual(obs[-1], 0)

    def test_success_is_terminal_even_at_horizon(self):
        env = self.make_env(horizon=1)
        env.reset()
        original = env.env.step
        def success(action):
            raw, reward, term, trunc, info = original(action)
            return raw, reward, term, trunc, dict(info, is_success=True)
        env.env.step = success
        _, reward, term, trunc, info = env.step(np.zeros(6))
        self.assertTrue(term)
        self.assertFalse(trunc)
        self.assertAlmostEqual(reward, 10.5)

    def test_invalid_actions_fail_before_simulator_step(self):
        env = self.make_env()
        env.reset()
        for action in (np.ones(5), np.full(6, np.nan), np.full(6, 1.01)):
            with self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(len(env.env.actions), 0)

    def test_bad_scale_and_horizon_rejected(self):
        for kwargs in ({"scale": -1}, {"scale": float("nan")}, {"horizon": 0}):
            with self.assertRaises(ValueError):
                self.make_env(**kwargs)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.env = KitchenStub()
        self.env.reset()
        self.reward = CloseFridgeReward()
        self.reward.reset(self.env)

    def update(self, left, right, success=False):
        self.env.doors = {"left": left, "right": right}
        return self.reward.step(self.env, success, np.zeros(6))[0]

    def test_closure_progress_stationary_and_reopening(self):
        self.assertAlmostEqual(self.update(.5, .5), 2.5)
        self.assertEqual(self.update(.5, .5), 0)
        self.assertAlmostEqual(self.update(1, 1), -2.5)

    def test_multiple_doors_and_success_bonus(self):
        self.assertAlmostEqual(self.update(0, 1), 2.5)
        self.assertAlmostEqual(self.update(0, 0, True), 12.5)

    def test_reward_reset_does_not_leak_across_episodes(self):
        self.update(0, 0)
        self.env.reset()
        self.reward.reset(self.env)
        self.assertEqual(self.update(1, 1), 0)

    def test_progress_telescopes_and_missing_state_fails(self):
        total = sum(self.update(x, x) for x in (.8, .9, .3, .1))
        self.assertAlmostEqual(total, 4.5)
        self.env._env.env.fxtr._fridge_door_joint_names = []
        with self.assertRaises(ValueError):
            self.reward.openness(self.env)


@unittest.skipUnless(importlib.util.find_spec("stable_baselines3"), "SB3 is not installed")
class SB3IntegrationTests(unittest.TestCase):
    def test_builtin_replay_uses_residuals_and_final_timeout_observation(self):
        from stable_baselines3 import TD3
        from stable_baselines3.common.buffers import ReplayBuffer

        env = ResidualEnv(KitchenStub(), ChunkStub(), CloseFridgeReward(), horizon=2)
        model = TD3("MlpPolicy", env, buffer_size=100, learning_starts=4,
                    batch_size=4, train_freq=1, gradient_steps=1,
                    policy_kwargs={"net_arch": [16, 16]}, device="cpu", seed=2)
        model.learn(total_timesteps=12)
        buffer = model.replay_buffer
        self.assertIs(type(buffer), ReplayBuffer)
        self.assertEqual(buffer.actions.shape[-1], 6)
        self.assertTrue(np.all(np.abs(buffer.actions[:12]) <= 1))
        self.assertGreater(model._n_updates, 0)
        self.assertEqual(buffer.timeouts[1, 0], 1)
        self.assertEqual(buffer.dones[1, 0], 1)
        np.testing.assert_array_equal(buffer.next_observations[1, 0, :16], np.full(16, 2))
        np.testing.assert_array_equal(buffer.next_observations[1, 0, 16:28], np.full(12, .3, np.float32))

class MultiTaskAndPerturbationTests(unittest.TestCase):
    def test_make_reward_supports_all_four_study_tasks(self):
        from vla_rl.rewards import (
            CloseFridgeReward,
            OpenCabinetReward,
            OpenFridgeReward,
            PlaceDishInSinkReward,
            make_reward,
        )

        self.assertIsInstance(make_reward("CloseFridge"), CloseFridgeReward)
        self.assertIsInstance(make_reward("OpenFridge"), OpenFridgeReward)
        self.assertIsInstance(make_reward("OpenCabinet"), OpenCabinetReward)
        self.assertIsInstance(make_reward("PlaceDishInSink"), PlaceDishInSinkReward)
        with self.assertRaises(ValueError):
            make_reward("UnknownTask")

    def test_open_fridge_reward_sign_is_opposite_close_fridge(self):
        from types import SimpleNamespace
        from vla_rl.rewards import OpenFridgeReward

        state = {"door_jnt": 0.2}
        fxtr = SimpleNamespace(
            _fridge_door_joint_names=["door_jnt"],
            get_joint_state=lambda env, joint_names: {k: state[k] for k in joint_names},
        )
        kitchen = SimpleNamespace(fxtr=fxtr)
        env = SimpleNamespace(
            unwrapped=SimpleNamespace(task="OpenFridge", _env=SimpleNamespace(env=kitchen))
        )
        reward_fn = OpenFridgeReward()
        reward_fn.reset(env)
        # Opening from 0.2 -> 0.8 (+0.6 change) should yield +3.0 progress
        state["door_jnt"] = 0.8
        r_open, terms = reward_fn.step(env, success=False, correction=np.zeros(6))
        self.assertAlmostEqual(r_open, 3.0, places=6)
        self.assertAlmostEqual(terms["progress"], 3.0, places=6)
        # Closing back from 0.8 -> 0.4 (-0.4 change) should penalize by -2.0
        state["door_jnt"] = 0.4
        r_close, _ = reward_fn.step(env, success=False, correction=np.zeros(6))
        self.assertAlmostEqual(r_close, -2.0, places=6)

    def test_perturbation_wrapper_restores_baseline_and_never_compounds(self):
        from types import SimpleNamespace
        import gymnasium as gym
        from vla_rl.perturbations import PRESETS, PerturbationWrapper

        class DummyKitchenEnv(gym.Env):
            def __init__(self):
                self.action_space = gym.spaces.Box(-1, 1, (12,), dtype=np.float32)
                self.observation_space = gym.spaces.Dict({})
                model = SimpleNamespace(
                    dof_damping=np.array([1.0, 2.0], dtype=np.float64),
                    dof_frictionloss=np.array([0.1, 0.2], dtype=np.float64),
                    geom_friction=np.array([[1.0, 0.005, 0.0001]], dtype=np.float64),
                    geom_rgba=np.array([[0.5, 0.5, 0.5, 1.0]], dtype=np.float32),
                    nmat=1,
                    mat_rgba=np.array([[0.4, 0.4, 0.4, 1.0]], dtype=np.float32),
                    nlight=1,
                    light_diffuse=np.array([[0.8, 0.8, 0.8]], dtype=np.float32),
                    light_ambient=np.array([[0.2, 0.2, 0.2]], dtype=np.float32),
                    light_pos=np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
                    joint_names=["door_jnt"],
                    joint_name2id=lambda name: 0,
                    jnt_dofadr=np.array([1]),
                )
                sim = SimpleNamespace(model=model, forward=lambda: None)
                fxtr = SimpleNamespace(_fridge_door_joint_names=["door_jnt"])
                self._env = SimpleNamespace(env=SimpleNamespace(sim=sim, fxtr=fxtr))

            def reset(self, *, seed=None, options=None):
                return {}, {}

            def step(self, action):
                return {}, 0.0, False, False, {}

        base_env = DummyKitchenEnv()
        # A hinge-only perturbation must preserve even very small surface friction.
        base_env._env.env.sim.model.geom_friction[0, 0] = 0.001
        base_env._env.env.sim.model.light_diffuse[0, 0] = 3.0
        wrapped = PerturbationWrapper(base_env, PRESETS["heldout_high_resistance"])
        wrapped.reset(seed=0)
        self.assertAlmostEqual(base_env._env.env.sim.model.dof_damping[1], 6.0)
        self.assertEqual(base_env._env.env.sim.model.geom_friction[0, 0], 0.001)
        self.assertEqual(base_env._env.env.sim.model.light_diffuse[0, 0], 3.0)
        # Resetting a second time must restore baseline (2.0) before multiplying by 3.0 -> still 6.0, not 18.0
        wrapped.reset(seed=1)
        self.assertAlmostEqual(base_env._env.env.sim.model.dof_damping[1], 6.0)

if __name__ == "__main__":
    unittest.main()
