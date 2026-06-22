import unittest

import torch

from utils.predicted_source_depth import (
    SOURCE_GEOMETRY_TAG,
    require_predicted_source_depths,
    require_saved_predicted_source_depths,
)


class PredictedSourceDepthTest(unittest.TestCase):
    def test_export_requires_model_output(self):
        with self.assertRaises(RuntimeError):
            require_predicted_source_depths({}, (8, 8), expected_views=3)

    def test_export_resizes_per_source_predictions(self):
        depths = torch.ones(1, 3, 4, 4)
        result = require_predicted_source_depths(
            {"src_mvs_depths": depths}, (8, 8), expected_views=3
        )
        self.assertEqual(tuple(result.shape), (1, 3, 8, 8))

    def test_refinement_rejects_legacy_alias_only(self):
        with self.assertRaises(RuntimeError):
            require_saved_predicted_source_depths(
                {"source_depths": torch.ones(3, 1, 8, 8), "cond_num": 3}
            )

    def test_refinement_accepts_tagged_predictions(self):
        depths = torch.ones(3, 1, 8, 8)
        result = require_saved_predicted_source_depths(
            {
                "source_geometry_source": SOURCE_GEOMETRY_TAG,
                "source_mvs_depths": depths,
                "cond_num": 3,
            }
        )
        self.assertIs(result, depths)


if __name__ == "__main__":
    unittest.main()
