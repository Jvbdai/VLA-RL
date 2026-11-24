from copy import deepcopy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compare_study import paired_outcomes


class PairedOutcomeTests(unittest.TestCase):
    def reports(self):
        episode = dict(seed=1, policy_seed=2, repeat=0, initial_numeric_model={'sha256': 'model', 'schema': 'v1'},
                       initial_state_sha256='state', instruction='CloseFridge', perturbation_parameters=None,
                       initial_camera_sha256={'left': 'pixels'}, success=False, completion_steps=None, final_task_error=.5)
        base = dict(method='base_vla', task='CloseFridge', horizon=900, checkpoint='snapshot', scale=.1,
                    observation_mode='fridge_ground_truth', perturbation='nominal', instruction_override=None, episodes=[episode])
        residual = deepcopy(base)
        residual['method'] = 'residual_td3'
        residual['episodes'][0]['success'] = True
        residual['episodes'][0]['completion_steps'] = 500
        return base, residual

    def test_outcome_requires_matching_numeric_model_and_state(self):
        base, residual = self.reports()
        self.assertEqual(paired_outcomes(base, residual)[0]['outcome'], 'rescue_outcome')
        residual['episodes'][0]['initial_numeric_model']['sha256'] = 'different'
        self.assertEqual(paired_outcomes(base, residual)[0]['outcome'], 'unmatched')

    def test_legacy_hashes_and_different_horizons_are_rejected(self):
        base, residual = self.reports()
        residual['horizon'] = 300
        with self.assertRaises(ValueError): paired_outcomes(base, residual)
        residual['horizon'] = 900
        del residual['episodes'][0]['initial_numeric_model']
        with self.assertRaises(ValueError): paired_outcomes(base, residual)

    def test_camera_difference_is_visible_and_not_claimed_as_causal(self):
        base, residual = self.reports()
        residual['episodes'][0]['initial_camera_sha256']['left'] = 'different'
        self.assertFalse(paired_outcomes(base, residual)[0]['exact_initial_cameras'])
