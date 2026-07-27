import unittest
from pathlib import Path

import torch
import yaml

from models.reacr_rca_dsbg_dataset import (
    build_root_cause_label,
    resolve_root_cause_node_indices,
    root_label_matches_node_name,
)
from models.reacr_rca_modality_gate import (
    ModalityDimensions, ModalityReliabilityGate, parse_modality_dims, valid_modality_dims,
)

ROOT = Path(__file__).resolve().parents[1]


class RootCauseLabelMappingTest(unittest.TestCase):
    def test_shared_name_matcher_is_boundary_safe(self):
        self.assertTrue(root_label_matches_node_name('paymentservice', 'paymentservice-0'))
        self.assertFalse(root_label_matches_node_name('paymentservice', 'paymentservice2-0'))

    def test_boundary_safe_service_mapping_and_multi_positive_replicas(self):
        nodes = {'service': 0, 'service-0': 1, 'service-1': 2, 'service2-0': 3}
        self.assertEqual(resolve_root_cause_node_indices('service', nodes), [0, 1, 2])
        torch.testing.assert_close(
            build_root_cause_label('service', nodes, 4),
            torch.tensor([1.0, 1.0, 1.0, 0.0]),
        )

    def test_zero_match_policy(self):
        with self.assertRaisesRegex(ValueError, 'No graph node matches'):
            build_root_cause_label('missing', {'service-0': 0}, 1)
        torch.testing.assert_close(
            build_root_cause_label('missing', {'service-0': 0}, 1, 'ignore'),
            torch.zeros(1),
        )


class ModalityLayoutTest(unittest.TestCase):
    def load(self, name):
        with (ROOT / 'config' / f'{name}.yaml').open(encoding='utf-8-sig') as stream:
            return yaml.safe_load(stream)

    def test_d1_non_contiguous_metric_and_raw_residual_order(self):
        layout = parse_modality_dims(self.load('D1'))
        self.assertTrue(valid_modality_dims(layout, 130))
        gate = ModalityReliabilityGate(layout, 130, 4, 0.0, False, 'raw_residual', 0.0)
        x = torch.arange(130, dtype=torch.float32).reshape(1, 1, 1, 130)
        splits = gate.split_modalities(x)
        torch.testing.assert_close(splits['metric'], torch.cat((x[..., :53], x[..., 73:]), -1))
        torch.testing.assert_close(splits['trace'], x[..., 53:62])
        torch.testing.assert_close(splits['log'], x[..., 62:73])
        torch.testing.assert_close(gate(x)['fused_features'], x)

    def test_d2_contiguous_layout(self):
        layout = parse_modality_dims(self.load('D2'))
        self.assertTrue(valid_modality_dims(layout, 266))
        gate = ModalityReliabilityGate(layout, 8, 4, 0.0, False)
        x = torch.arange(266, dtype=torch.float32).reshape(1, 1, 1, 266)
        splits = gate.split_modalities(x)
        torch.testing.assert_close(splits['metric'], x[..., :256])
        torch.testing.assert_close(splits['trace'], x[..., 256:258])
        torch.testing.assert_close(splits['log'], x[..., 258:266])

    def test_layout_validation_rejects_equal_total_with_overlap(self):
        invalid = ModalityDimensions(
            {'metric': 2, 'trace': 1, 'log': 1},
            {'metric': [0, 1], 'trace': [1], 'log': [3]},
        )
        self.assertFalse(valid_modality_dims(invalid, 4))
        with self.assertRaisesRegex(ValueError, 'partition'):
            ModalityReliabilityGate(invalid, 4)


if __name__ == '__main__':
    unittest.main()
