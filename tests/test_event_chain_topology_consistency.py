import unittest

import dgl
import torch

from models.reacr_rca_event_chain_explainer import (
    _build_outgoing_edges,
    _path_lag_predictions,
    _window_edge_union,
)


def make_graph(edges, node_num=3):
    sources = [source for source, _ in edges]
    targets = [target for _, target in edges]
    return dgl.graph((sources, targets), num_nodes=node_num)


class EventChainTopologyConsistencyTest(unittest.TestCase):
    def test_zero_relation_mass_never_creates_a_path_edge(self):
        outgoing = _build_outgoing_edges(
            torch.zeros(3, 3),
            edge_topk=2,
            min_strength=0.0,
        )
        self.assertEqual(outgoing, {0: [], 1: [], 2: []})

    def test_positive_relation_mass_survives_threshold(self):
        relation = torch.zeros(3, 3)
        relation[2, 0] = 0.25
        outgoing = _build_outgoing_edges(
            relation,
            edge_topk=2,
            min_strength=0.0,
        )
        self.assertEqual(outgoing[0], [(2, 0.25)])

    def test_window_validation_uses_all_observed_snapshots(self):
        steps = [
            [make_graph([(0, 1)])],
            [make_graph([(1, 2)])],
        ]
        forward, reverse = _window_edge_union(steps, sample_idx=0)
        self.assertEqual(forward, {(0, 1), (1, 2)})
        self.assertEqual(reverse, {(1, 0), (2, 1)})

    def test_pair_specific_lags_are_exported_per_path_edge(self):
        weights = torch.zeros(3, 3, 3)
        weights[:, 1, 0] = torch.tensor([0.1, 0.8, 0.1])
        weights[:, 2, 1] = torch.tensor([0.7, 0.2, 0.1])
        summary = _path_lag_predictions(
            [(0, 1, 0.5), (1, 2, 0.4)],
            weights,
        )
        self.assertEqual(summary['argmax_lags'], [2, 1])
        self.assertAlmostEqual(summary['expected_lags'][0], 2.0, places=6)
        self.assertAlmostEqual(summary['expected_lags'][1], 1.4, places=6)
        self.assertAlmostEqual(summary['confidences'][0], 0.8, places=6)


if __name__ == '__main__':
    unittest.main()
