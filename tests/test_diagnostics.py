from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from vla_rl.diagnostics import EpisodeDiagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_contact_loss_progress_and_saturation_aggregation(self):
        audit = EpisodeDiagnostics()
        for contact, progress, saturation in [(False, 0, 0), (True, .1, .5), (False, -.1, 1)]:
            audit.update({'diagnostics': {'gripper_fixture_contact': contact},
                          'reward_terms': {'progress': progress},
                          'residual_saturation_fraction': saturation, 'executed_correction_norm': .1})
        result = audit.summary()
        self.assertEqual(result['contact_steps'], 1)
        self.assertEqual(result['contact_loss_events'], 1)
        self.assertEqual(result['positive_progress_steps'], 1)
        self.assertEqual(result['mean_residual_saturation_fraction'], .5)
        self.assertAlmostEqual(result['mean_executed_correction_norm'], .1)
