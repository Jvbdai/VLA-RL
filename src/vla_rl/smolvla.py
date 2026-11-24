"""Frozen, single-environment adapter using the installed LeRobot processors."""

from contextlib import nullcontext

import numpy as np
import torch


class FrozenSmolVLA:
    def __init__(self, checkpoint, env, device="cuda", instruction="CloseFridge"):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import RoboCasaEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(checkpoint)
        config.device = device
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint, config=config).to(device)
        self.policy.eval().requires_grad_(False)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=config, pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": {
                    "observation.images.robot0_agentview_left": "observation.images.camera1",
                    "observation.images.robot0_agentview_right": "observation.images.camera2",
                    "observation.images.robot0_eye_in_hand": "observation.images.camera3",
                }},
            },
        )
        self.env_pre, self.env_post = make_env_pre_post_processors(RoboCasaEnvConfig(), config)
        self.env, self.instruction, self.device = env, instruction, device
        from vla_rl.policy_rng import PolicyRNG
        self.rng = PolicyRNG(device)

    def reset(self, seed=None):
        if seed is not None:
            self.rng.reset(seed)
        self.policy.reset()
        for processor in (self.pre, self.post, self.env_pre, self.env_post):
            processor.reset()

    def action(self, obs):
        from lerobot.envs.utils import preprocess_observation

        # Explicit allowlist excludes privileged fields even if the env adds them.
        batch = preprocess_observation({"pixels": obs["pixels"], "agent_pos": obs["agent_pos"]})
        batch["task"] = [self.instruction or self.env.unwrapped.task_description]
        batch = self.pre(self.env_pre(batch))
        amp = torch.autocast(device_type=torch.device(self.device).type) if self.policy.config.use_amp else nullcontext()
        with self.rng.use(), torch.inference_mode(), amp:
            action = self.post(self.policy.select_action(batch))
            action = self.env_post({"action": action})["action"]
        # Read the actual queue, rather than assuming chunk_size == n_action_steps.
        remaining = len(self.policy._queues["action"])
        length = self.policy.config.n_action_steps
        position = (length - remaining - 1) / max(length - 1, 1)
        return action[0].detach().cpu().numpy().astype(np.float32), position
