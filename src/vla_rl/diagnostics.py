"""Read-only task diagnostics. These never change the reward or success test."""
import numpy as np


def fridge_diagnostics(env):
    kitchen = env.unwrapped._env.env
    fixture = kitchen.fxtr
    names = sorted(fixture._fridge_door_joint_names)
    openness = fixture.get_joint_state(kitchen, names)
    return {
        "door_openness_by_joint": {name: float(openness[name]) for name in names},
        "door_angle_rad_by_joint": {name: float(kitchen.sim.data.get_joint_qpos(name)) for name in names},
        "max_door_openness": max(float(openness[name]) for name in names),
        "gripper_fixture_contact": any(kitchen.check_contact(gripper, fixture)
                                       for robot in kitchen.robots for gripper in robot.gripper.values()),
    }


class EpisodeDiagnostics:
    def __init__(self):
        self.steps = self.contact_steps = self.contact_losses = self.progress_steps = 0
        self.saturation_sum = self.correction_norm_sum = 0.
        self.previous_contact = False
        self.final = {}

    def update(self, info):
        self.steps += 1
        self.final = info.get("diagnostics", {})
        contact = self.final.get("gripper_fixture_contact", False)
        self.contact_steps += int(contact)
        self.contact_losses += int(self.previous_contact and not contact)
        self.previous_contact = contact
        self.progress_steps += int(info["reward_terms"]["progress"] > 1e-6)
        self.saturation_sum += info.get("residual_saturation_fraction", 0.)
        self.correction_norm_sum += info.get("executed_correction_norm", 0.)

    def summary(self):
        return dict(self.final, contact_steps=self.contact_steps,
                    contact_loss_events=self.contact_losses, positive_progress_steps=self.progress_steps,
                    mean_residual_saturation_fraction=self.saturation_sum / max(self.steps, 1),
                    mean_executed_correction_norm=self.correction_norm_sum / max(self.steps, 1))
