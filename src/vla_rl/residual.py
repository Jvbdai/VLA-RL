"""Bounded residual control, with optional explicitly privileged door features."""

import hashlib
import gymnasium as gym
import numpy as np


def compose_action(base, residual, scale, low, high):
    """Correct only arm indices 5:11. Zero is an exact identity, even at limits."""
    base = np.asarray(base, dtype=np.float32)
    residual = np.asarray(residual, dtype=np.float32)
    if base.shape != (12,) or not np.isfinite(base).all():
        raise ValueError("base action must have 12 finite components")
    if residual.shape != (6,) or not np.isfinite(residual).all():
        raise ValueError("residual must have six finite components")
    if np.any(np.abs(residual) > 1):
        raise ValueError("residual must lie in [-1, 1]")
    result = base.copy()
    delta = residual * scale
    # Preserve zero corrections exactly; never clip the rest of the VLA action.
    # A pretrained action can exceed the nominal Box. Preserve its baseline
    # semantics while never extending that excursion or exceeding residual bounds.
    corrected = np.clip(base[5:11] + delta, np.minimum(low[5:11], base[5:11]),
                        np.maximum(high[5:11], base[5:11]))
    result[5:11] = np.where(delta == 0, base[5:11], corrected)
    return result


class ResidualEnv(gym.Wrapper):
    """Base provider implements reset(seed), action(obs)->(action, chunk_position).

    Returned observations describe the pending action, including after truncation,
    so SB3 can bootstrap from the final pre-reset observation. No auto-reset here.
    The reward provider implements reset(env), step(env, success, correction).
    """

    def __init__(self, env, base, reward, *, scale=0.1, horizon=900, object_observation=None, diagnostics=None):
        super().__init__(env)
        self.scale = np.broadcast_to(np.asarray(scale, np.float32), (6,)).copy()
        if not np.isfinite(self.scale).all() or np.any(self.scale < 0):
            raise ValueError("scale must be finite and nonnegative")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if env.action_space.shape != (12,):
            raise ValueError("expected PandaOmron's 12-component action space")
        self.base, self.reward_provider, self.horizon = base, reward, horizon
        self.action_space = gym.spaces.Box(-1, 1, (6,), dtype=np.float32)
        self.object_observation = object_observation
        self.diagnostics = diagnostics
        extra = 0 if object_observation is None else object_observation.size
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (29 + extra,), dtype=np.float32)
        self._done = True

    def _observation(self, raw):
        proprio = np.asarray(raw["agent_pos"], dtype=np.float32)
        if proprio.shape != (16,) or not np.isfinite(proprio).all():
            raise ValueError("expected 16 finite proprioceptive components")
        action, position = self.base.action(raw)
        self.pending_action = np.asarray(action, dtype=np.float32).copy()
        if self.pending_action.shape != (12,) or not np.isfinite(self.pending_action).all():
            raise ValueError("invalid VLA action")
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("invalid normalized chunk position")
        parts = [proprio, self.pending_action, [position]]
        if self.object_observation is not None:
            features = np.asarray(self.object_observation(self.env), dtype=np.float32)
            if features.shape != (self.object_observation.size,) or not np.isfinite(features).all():
                raise ValueError("Invalid object observation")
            parts.append(features)
        return np.concatenate(parts).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        raw, info = self.env.reset(seed=seed, options=options)
        policy_seed = (options or {}).get("policy_seed", info.get("episode_seed", seed))
        self.base.reset(policy_seed)
        info = dict(info, policy_seed=policy_seed,
                    initial_camera_sha256={k: hashlib.sha256(v.tobytes()).hexdigest()
                                           for k, v in raw.get("pixels", {}).items()})
        self.reward_provider.reset(self.env)
        self._steps, self._done = 0, False
        return self._observation(raw), info

    def step(self, residual):
        if self._done:
            raise RuntimeError("reset required before stepping")
        base_action = self.pending_action.copy()
        action = compose_action(base_action, residual, self.scale,
                                self.env.action_space.low, self.env.action_space.high)
        raw, sparse, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        success = bool(info.get("is_success", False))
        terminated = bool(terminated or success)
        truncated = bool(truncated or (self._steps >= self.horizon and not terminated))
        correction = action[5:11] - base_action[5:11]
        reward, terms = self.reward_provider.step(self.env, success, correction)
        self._done = terminated or truncated
        info = dict(info, sparse_reward=float(sparse), reward_terms=terms,
                    base_action=base_action, executed_action=action.copy(),
                    residual_saturation_fraction=float(np.mean(np.abs(residual) >= .99)),
                    executed_correction_norm=float(np.linalg.norm(correction)),
                    diagnostics=self.diagnostics(self.env) if self.diagnostics is not None else {})
        return self._observation(raw), float(reward), terminated, truncated, info
