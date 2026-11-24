"""Separate reproducible environment reset streams from learner exploration RNGs."""
import random
import numpy as np


class EpisodeSeedSchedule:
    def __init__(self, seeds):
        self.seeds = tuple(int(s) for s in seeds)
        if not self.seeds or min(self.seeds) < 0:
            raise ValueError("Provide nonnegative training seeds")
        self.index = 0

    def next(self, explicit=None):
        # An explicit first reset (e.g. SB3) does not replace the configured pool.
        value = self.seeds[self.index % len(self.seeds)]
        self.index += 1
        return value


def seeded_reset(reset, seed, **kwargs):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        # LeRobot lazily constructs RoboCasa on the first reset. Construction
        # consumes global RNG draws (including geometry colors). Give it its
        # own seeded phase, then restart the episode stream on EVERY reset.
        owner = getattr(reset, "__self__", None)
        ensure = getattr(owner, "_ensure_env", None)
        if ensure is not None:
            ensure()
            random.seed(seed)
            np.random.seed(seed)
        obs, info = reset(seed=seed, **kwargs)
        return obs, dict(info, episode_seed=int(seed))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
