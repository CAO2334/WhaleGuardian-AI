from __future__ import annotations

import unittest
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.evaluate_fast_ensemble import integer_compositions, search_probability_ensemble


class FastEnsembleTests(unittest.TestCase):
    def test_integer_compositions_cover_weight_grid(self) -> None:
        values = list(integer_compositions(4, 3))
        self.assertEqual(len(values), 15)
        self.assertTrue(all(sum(value) == 4 for value in values))
        self.assertIn((0, 0, 4), values)
        self.assertIn((2, 1, 1), values)

    def test_validation_search_can_select_complementary_models(self) -> None:
        labels = np.array([0, 0, 1, 1])
        probabilities = {
            "a": {
                "none": np.array(
                    [[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.1, 0.9]], dtype=np.float32
                )
            },
            "b": {
                "none": np.array(
                    [[0.9, 0.1], [0.4, 0.6], [0.1, 0.9], [0.1, 0.9]], dtype=np.float32
                )
            },
        }
        selection, table = search_probability_ensemble(probabilities, labels, 2, 0.25)
        self.assertAlmostEqual(float(selection["val_macro_f1"]), 1.0)
        self.assertFalse(bool(selection["test_set_used_for_selection"]))
        self.assertGreater(len(table), 0)


if __name__ == "__main__":
    unittest.main()
