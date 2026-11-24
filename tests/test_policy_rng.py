import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rl.policy_rng import PolicyRNG

class PolicyRNGTests(unittest.TestCase):
    def test_stream_is_reproducible_and_does_not_touch_learner(self):
        before = torch.get_rng_state().clone()
        rng = PolicyRNG("cpu", 42)
        with rng.use():
            first = torch.randn(10)
        torch.testing.assert_close(torch.get_rng_state(), before)
        torch.randn(100)
        learner = torch.get_rng_state().clone()
        rng.reset(42)
        with rng.use():
            again = torch.randn(10)
        torch.testing.assert_close(first, again)
        torch.testing.assert_close(torch.get_rng_state(), learner)
        with rng.use():
            second = torch.randn(10)
        self.assertFalse(torch.equal(first, second))
