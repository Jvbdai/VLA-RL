from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_rl.model_identity import array_digest, numeric_model_identity


class ModelIdentityTests(unittest.TestCase):
    def test_numeric_identity_ignores_xml_but_detects_physics_and_visuals(self):
        model = SimpleNamespace(geom_rgba=np.ones((2, 4)), dof_damping=np.ones(2),
                                opt=SimpleNamespace(timestep=0.002), xml="first")
        first = numeric_model_identity(model)
        model.xml = "different serialization"
        self.assertEqual(first, numeric_model_identity(model))
        model.dof_damping[0] = 2
        changed = numeric_model_identity(model)
        self.assertNotEqual(first["sha256"], changed["sha256"])
        self.assertEqual(first["fields"]["geom_rgba"], changed["fields"]["geom_rgba"])
        model.geom_rgba[0, 0] = 0
        self.assertNotEqual(changed["sha256"], numeric_model_identity(model)["sha256"])
        model.opt.timestep = 0.003
        self.assertNotEqual(first["fields"]["opt.timestep"], numeric_model_identity(model)["fields"]["opt.timestep"])

    def test_nan_payload_and_signed_zero_are_canonical_but_shape_matters(self):
        a = np.array([0., np.nan, np.inf, -np.inf])
        b = np.array([-0., np.nan, np.inf, -np.inf])
        self.assertEqual(array_digest(a), array_digest(b))
        self.assertNotEqual(array_digest(a), array_digest(a.reshape(2, 2)))
        b[2] = -np.inf
        self.assertNotEqual(array_digest(a), array_digest(b))
