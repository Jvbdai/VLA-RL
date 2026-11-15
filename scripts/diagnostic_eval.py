"""Process-local instrumentation of the upstream evaluator; no upstream files modified."""
import hashlib
import json
import os
from pathlib import Path
import runpy

import cv2
import numpy as np
from lerobot.sim.integrations.robocasa import RoboCasaEnv

CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
original_reset = RoboCasaEnv.reset
original_step = RoboCasaEnv.step
original_close = RoboCasaEnv.close


def sample(env, action=None, success=False):
    kitchen = env._env.env
    joints = getattr(kitchen.fxtr, "_fridge_door_joint_names", [])
    if not joints:
        raise RuntimeError("No fridge door joints found; cannot report door angles reliably")
    angles = {name: float(kitchen.sim.data.get_joint_qpos(name)) for name in joints}
    row = {"step": env._diag_step, "door_angles_rad": angles,
           "success": bool(success), "state": env._diag_obs["agent_pos"].tolist(),
           "action": None if action is None else np.asarray(action).tolist()}
    env._diag_log.write(json.dumps(row) + "\n")
    env._diag_log.flush()
    env._diag_video.write(cv2.cvtColor(render(env), cv2.COLOR_RGB2BGR))
    return angles


def reset(self, *args, **kwargs):
    obs, info = original_reset(self, *args, **kwargs)
    if hasattr(self, "_diag_log"):
        self._diag_log.close()
        self._diag_video.release()
    self._diag_episode = getattr(self, "_diag_episode", -1) + 1
    folder = Path(os.environ["VLA_DIAGNOSTIC_DIR"]) / f"episode_{self.episode_index}_{self._diag_episode}"
    folder.mkdir(parents=True, exist_ok=True)
    self._diag_log = (folder / "steps.jsonl").open("w", encoding="utf-8")
    self._diag_obs, self._diag_step = obs, 0
    height, width = obs["pixels"][CAMERAS[0]].shape[:2]
    self._diag_video = cv2.VideoWriter(str(folder / "three_cameras.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"), 20, (width * 3, height))
    if not self._diag_video.isOpened():
        raise RuntimeError("Could not open diagnostic video writer")
    original_instruction = self.task_description
    self.task_description = os.environ.get("VLA_INSTRUCTION", original_instruction)
    kitchen = self._env.env
    image_hashes = {}
    for cam in CAMERAS:
        frame = obs["pixels"][cam]
        image_hashes[cam] = hashlib.sha256(frame.tobytes()).hexdigest()
        cv2.imwrite(str(folder / f"initial_{cam}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    angles = sample(self)
    manifest = {"seed": kwargs.get("seed"), "original_instruction": original_instruction,
                "instruction": self.task_description, "initial_door_angles_rad": angles,
                "initial_qpos": kitchen.sim.data.qpos.tolist(),
                "initial_qvel": kitchen.sim.data.qvel.tolist(),
                "initial_image_hashes": image_hashes,
                "ep_meta": kitchen.get_ep_meta(), "camera_order": CAMERAS}
    (folder / "initial_state.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print("Diagnostic instruction:", self.task_description, flush=True)
    return obs, info


def step(self, action):
    result = original_step(self, action)
    self._diag_obs = result[0]
    self._diag_step += 1
    sample(self, action, result[4].get("is_success", False))
    return result


def render(self):
    frames = []
    for cam in CAMERAS:
        frame = self._diag_obs["pixels"][cam].copy()
        cv2.putText(frame, cam.replace("robot0_", ""), (5, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 0), 1)
        frames.append(frame)
    return np.concatenate(frames, axis=1)


def close(self):
    if hasattr(self, "_diag_log"):
        self._diag_log.close()
        self._diag_video.release()
    original_close(self)


if __name__ == "__main__":
    RoboCasaEnv.reset = reset
    RoboCasaEnv.step = step
    RoboCasaEnv.render = render
    RoboCasaEnv.close = close
    runpy.run_module("lerobot.scripts.lerobot_eval", run_name="__main__")
