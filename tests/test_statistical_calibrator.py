import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.reacr_rca_statistical_calibrator import (
    StatisticalEvidenceCalibrator,
    extract_temporal_statistics,
    fuse_case_scores,
)


class StatisticalCalibratorTest(unittest.TestCase):
    def test_temporal_statistics_are_node_aligned(self):
        features = torch.tensor(
            [[[[1.0, 2.0], [10.0, 20.0]], [[3.0, 4.0], [14.0, 28.0]]]]
        )
        stats = extract_temporal_statistics(features)
        self.assertEqual(stats.shape, (1, 2, 12))
        np.testing.assert_allclose(stats[0, 0, :2], [2.0, 3.0])
        np.testing.assert_allclose(stats[0, 0, -2:], [2.0, 2.0])
        np.testing.assert_allclose(stats[0, 1, -2:], [4.0, 8.0])

    def test_fusion_preserves_shape_and_weight_endpoints(self):
        base = np.array([[1.0, 2.0, 3.0]])
        statistical = np.array([[3.0, 2.0, 1.0]])
        only_base = fuse_case_scores(base, statistical, 1.0)
        only_statistical = fuse_case_scores(base, statistical, 0.0)
        self.assertEqual(only_base.shape, base.shape)
        self.assertGreater(only_base[0, 2], only_base[0, 0])
        self.assertGreater(only_statistical[0, 0], only_statistical[0, 2])

    def test_extra_trees_fit_predict_and_round_trip(self):
        features = torch.zeros(4, 3, 3, 2)
        labels = torch.zeros(4, 3)
        for case_idx in range(4):
            root_idx = case_idx % 3
            labels[case_idx, root_idx] = 1.0
            features[case_idx, :, root_idx, 0] = 5.0
        calibrator = StatisticalEvidenceCalibrator(
            method='extra_trees',
            seed=0,
            n_estimators=20,
            max_depth=2,
            min_samples_leaf=1,
        ).fit(features, labels)
        scores = calibrator.predict(features)
        self.assertEqual(scores.shape, (4, 3))
        self.assertTrue(np.all(scores.argmax(axis=1) == labels.argmax(dim=1).numpy()))
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'calibrator.pkl'
            calibrator.save(path)
            restored = StatisticalEvidenceCalibrator.load(path)
            np.testing.assert_allclose(restored.predict(features), scores)


if __name__ == '__main__':
    unittest.main()
