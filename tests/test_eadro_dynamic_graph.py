import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import dgl
import pandas as pd
import torch

from models.reacr_rca_causal_head import RunInspiredCausalHead
from models.reacr_rca_dsbg_dataset import build_case_windows, build_sliding_windows
from models.reacr_rca_dynamic_backbone import WeightedSpatialLayer
from models.reacr_rca_trainer import apply_observation_prior, topology_parameter_diagnostics
from models.reacr_rca_trainer import DyCauRCAModel
from reacr_rca_pipeline import split_cases_for_training
from tools.convert_eadro_dynamic import EdgeAccumulator, _build_graph, iter_json_array


def sample(timestamp, feature_value=0.0):
    graph = dgl.graph(([], []), num_nodes=2)
    graph.edata['weight'] = torch.empty(0, dtype=torch.float32)
    graph.edata['edge_attr'] = torch.empty(0, 5, dtype=torch.float32)
    features = torch.full((2, 1), float(feature_value))
    return timestamp, graph, features


class DynamicConverterTest(unittest.TestCase):
    def test_streaming_json_array_handles_objects_across_small_chunks(self):
        values = [{'id': index, 'payload': 'x' * 19} for index in range(7)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'spans.json'
            path.write_text(json.dumps(values), encoding='utf-8')
            observed = list(iter_json_array(path, chunk_chars=11))
        self.assertEqual(observed, values)

    def test_graph_uses_only_observed_edges_and_preserves_empty_bucket(self):
        empty = _build_graph(node_num=3, bucket_seconds=10, edge_stats={})
        self.assertEqual(empty.num_nodes(), 3)
        self.assertEqual(empty.num_edges(), 0)
        self.assertEqual(tuple(empty.edata['edge_attr'].shape), (0, 5))

        stats = EdgeAccumulator()
        stats.add(latency_ms=10.0, is_error=False)
        stats.add(latency_ms=30.0, is_error=True)
        graph = _build_graph(3, 10, {(0, 2): stats})
        self.assertEqual(graph.edges()[0].tolist(), [0])
        self.assertEqual(graph.edges()[1].tolist(), [2])
        self.assertEqual(graph.edata['weight'].tolist(), [2.0])
        self.assertEqual(tuple(graph.edata['edge_attr'].shape), (1, 5))
        self.assertAlmostEqual(float(graph.edata['edge_attr'][0, 4]), 0.5)


class SessionAwareWindowTest(unittest.TestCase):
    def test_sliding_windows_never_cross_session_sized_time_gap(self):
        samples = [
            sample(0), sample(10), sample(20),
            sample(100), sample(110), sample(120),
        ]
        windows = build_sliding_windows(
            samples,
            window_size=3,
            max_gap_seconds=15,
            incomplete_policy='drop',
        )
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].timestamps, [0, 10, 20])
        self.assertEqual(windows[1].timestamps, [100, 110, 120])

    def test_case_window_uses_explicit_failure_interval(self):
        samples = [sample(ts) for ts in (0, 10, 90, 100, 110, 120, 500)]
        cases = pd.DataFrame([
            {'case_id': 7, 'timestamp': 100, 'cmdb_id': 'svc-a'}
        ])
        windows = build_case_windows(
            cases,
            samples,
            node_hash={'svc-a': 0, 'svc-b': 1},
            window_size=4,
            case_pre_seconds=10,
            case_post_seconds=30,
            max_gap_seconds=15,
            incomplete_policy='error',
        )
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].timestamps, [90, 100, 110, 120])

    def test_group_split_has_no_session_leakage(self):
        cases = pd.DataFrame([
            {'timestamp': 10, 'cmdb_id': 'a', 'session_id': 's1'},
            {'timestamp': 20, 'cmdb_id': 'b', 'session_id': 's1'},
            {'timestamp': 30, 'cmdb_id': 'a', 'session_id': 's2'},
            {'timestamp': 40, 'cmdb_id': 'b', 'session_id': 's2'},
            {'timestamp': 50, 'cmdb_id': 'a', 'session_id': 's3'},
            {'timestamp': 60, 'cmdb_id': 'b', 'session_id': 's3'},
        ])
        train, validation, test, metadata = split_cases_for_training(
            cases,
            feedback_sample_ratio=0.34,
            default_ratio=0.3,
            protocol='group',
            test_start_ratio=0.67,
            group_column='session_id',
        )
        train_groups = set(train.session_id)
        validation_groups = set(validation.session_id)
        test_groups = set(test.session_id)
        self.assertFalse(train_groups & validation_groups)
        self.assertFalse(train_groups & test_groups)
        self.assertFalse(validation_groups & test_groups)
        self.assertEqual(metadata['train_groups'], ['s1'])
        self.assertEqual(metadata['val_groups'], ['s2'])
        self.assertEqual(metadata['test_groups'], ['s3'])


class EdgeEvidenceModelTest(unittest.TestCase):
    def test_spatial_layer_uses_absolute_edge_mass(self):
        layer = WeightedSpatialLayer(1, 1, edge_feature_dim=5)
        with torch.no_grad():
            layer.self_linear.weight.zero_()
            layer.self_linear.bias.zero_()
            layer.neigh_linear.weight.zero_()
            layer.neigh_linear.bias.zero_()
            layer.edge_mass_linear.weight.fill_(1.0)
            layer.edge_context_linear.weight.zero_()

        def run(weight):
            graph = dgl.graph(([0], [1]), num_nodes=2)
            graph.edata['weight'] = torch.tensor([weight])
            graph.edata['edge_attr'] = torch.zeros(1, 5)
            return float(layer(graph, torch.zeros(2, 1))[1, 0])

        self.assertGreater(run(4.0), run(1.0))

        graph = dgl.graph(([0], [1]), num_nodes=2)
        graph.edata['weight'] = torch.tensor([2.0])
        graph.edata['edge_attr'] = torch.ones(1, 5)
        layer(graph, torch.zeros(2, 1)).sum().backward()
        model = SimpleNamespace(
            backbone=SimpleNamespace(
                spatial_layers=[layer], topology_encoder=None, topology_scale=None
            )
        )
        diagnostics = topology_parameter_diagnostics(model)
        self.assertIsNotNone(diagnostics['edge_evidence_gradient_norm'])
        self.assertGreater(diagnostics['edge_evidence_gradient_norm'], 0.0)

    def test_observation_prior_stabilizes_extreme_feature_scale(self):
        rank_output = {
            'final_score': torch.tensor([[1.0, 2.0, 3.0]]),
            'outgoing_score': torch.tensor([[3.0, 2.0, 1.0]]),
        }
        features = torch.zeros(1, 2, 3, 2)
        features[0, 1, 1, 0] = 1_000_000.0
        anomaly_output = {
            'recon_error': torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
        }
        output = apply_observation_prior(
            rank_output,
            anomaly_output,
            features,
            {
                'enabled': True,
                'base_score_key': 'outgoing_score',
                'metric_feature_indices': [0],
                'all_feature_indices': [0, 1],
                'weights': {
                    'base': 0.25,
                    'metric_peak': 0.75,
                    'temporal_range': 0.0,
                    'reconstruction_peak': 0.0,
                },
            },
        )
        self.assertTrue(torch.isfinite(output['final_score']).all())
        self.assertEqual(int(torch.argmax(output['final_score'], dim=1)), 1)
        self.assertLess(float(output['final_score'].abs().max()), 10.0)
        self.assertTrue(torch.equal(
            output['model_score_before_observation_prior'], rank_output['final_score']
        ))

    def test_model_clips_features_before_backbone(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.seen = None

            def forward(self, graphs, features):
                self.seen = features
                return {'temporal_states': torch.zeros(1, 1, 3, 2)}

        class Anomaly(torch.nn.Module):
            def forward(self, states, features):
                return {
                    'anomaly_score': torch.zeros(1, 1, 3),
                    'recon_error': torch.zeros(1, 1, 3),
                }

        class Causal(torch.nn.Module):
            def forward(self, scores, graphs=None):
                return {'causal_matrix': torch.zeros(1, 3, 3)}

        class Ranker(torch.nn.Module):
            def forward(self, graphs, scores, matrix, lag_weights=None):
                return {'final_score': torch.zeros(1, 3)}

        backbone = Backbone()
        model = DyCauRCAModel(
            backbone, Anomaly(), Causal(), Ranker(), feature_value_clip=20.0
        )
        graph = dgl.graph(([], []), num_nodes=3)
        features = torch.tensor([[[[1_000_000.0], [-1_000_000.0], [0.0]]]])
        model([graph], features)
        self.assertEqual(float(backbone.seen.abs().max()), 20.0)


    def test_causal_head_uses_relative_dynamic_edge_strength(self):
        graph = dgl.graph(([0, 1], [2, 2]), num_nodes=3)
        graph.edata['weight'] = torch.tensor([1.0, 3.0])
        head = RunInspiredCausalHead(
            node_num=3,
            window_size=1,
            topology_mode='directed',
            topology_weight_gamma=0.5,
        )
        strength = head._build_topology_strength([graph], 1, 3, torch.device('cpu'), 1)
        self.assertAlmostEqual(float(strength[0, 2, 0]), 1.0 / 3.0, places=5)
