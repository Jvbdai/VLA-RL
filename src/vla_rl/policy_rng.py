"""A private Torch RNG stream, leaving TD3's global stream unchanged."""
from contextlib import contextmanager
import torch


class PolicyRNG:
    def __init__(self, device, seed=0):
        device = torch.device(device)
        self.devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        self.reset(seed)

    def reset(self, seed):
        with torch.random.fork_rng(devices=self.devices):
            torch.random.default_generator.manual_seed(int(seed))
            for device in self.devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
            self.cpu = torch.get_rng_state()
            self.cuda = {d: torch.cuda.get_rng_state(d) for d in self.devices}

    @contextmanager
    def use(self):
        with torch.random.fork_rng(devices=self.devices):
            torch.set_rng_state(self.cpu)
            for device, state in self.cuda.items():
                torch.cuda.set_rng_state(state, device)
            try:
                yield
            finally:
                self.cpu = torch.get_rng_state()
                self.cuda = {d: torch.cuda.get_rng_state(d) for d in self.devices}
