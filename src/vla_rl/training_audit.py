"""Periodic training artifacts without extra simulator or VLA inference calls."""
import json
import time
from stable_baselines3.common.callbacks import BaseCallback
from vla_rl.diagnostics import EpisodeDiagnostics


class TrainingAudit(BaseCallback):
    def __init__(self, output, checkpoint_every=3000):
        super().__init__()
        self.output = output
        self.checkpoint_every = checkpoint_every
        self.episode = EpisodeDiagnostics()

    def _on_training_start(self):
        self.started = time.perf_counter()
        self.model.save(str(self.output / "td3_step_0"))

    def record(self):
        metrics = {"steps": self.num_timesteps, "updates": self.model._n_updates,
                   "elapsed_s": time.perf_counter() - self.started}
        for name in ("train/actor_loss", "train/critic_loss"):
            value = self.model.logger.name_to_value.get(name)
            if value is not None:
                metrics[name] = float(value)
        with (self.output / "learning_metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")

    def _on_step(self):
        info = self.locals["infos"][0]
        self.episode.update(info)
        if self.locals["dones"][0]:
            record = dict(self.episode.summary(), steps=self.num_timesteps,
                          success=bool(info["is_success"]), episode=info.get("episode"))
            with (self.output / "training_episodes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            self.episode = EpisodeDiagnostics()
        if self.num_timesteps % 100 == 0:
            self.record()
        if self.num_timesteps % self.checkpoint_every == 0:
            self.model.save(str(self.output / f"td3_step_{self.num_timesteps}"))
        return True

    def _on_training_end(self):
        self.record()
