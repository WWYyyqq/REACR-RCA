import unittest

import dgl
import torch

from models.reacr_rca_topology_dynamics import (
    TOPOLOGY_DYNAMIC_FEATURE_DIM,
    TOPOLOGY_FEATURE_DIM,
    TopologyDynamicsEncoder,
    extract_topology_dynamics,
)


def make_graph(edges, node_num=4):
    sources = [source for source, _ in edges]
    destinations = [destination for _, destination in edges]
    return dgl.graph((sources, destinations), num_nodes=node_num)


class TopologyDynamicsFeatureTest(unittest.TestCase):
    def test_fixed_topology_has_exactly_zero_change_features(self):
        graphs = [
            make_graph([(0, 1), (1, 2)]),
            make_graph([(0, 1), (1, 2)]),
            make_graph([(0, 1), (1, 2)]),
        ]

        features = extract_topology_dynamics(graphs)

        self.assertEqual(tuple(features.shape), (1, 3, 4, TOPOLOGY_FEATURE_DIM))
        expected_in_degree = torch.tensor([0.0, 1.0, 1.0, 0.0])
        expected_out_degree = torch.tensor([1.0, 1.0, 0.0, 0.0])
        for step_idx in range(3):
            self.assertTrue(torch.equal(features[0, step_idx, :, 0], expected_in_degree))
            self.assertTrue(torch.equal(features[0, step_idx, :, 1], expected_out_degree))
        self.assertEqual(
            torch.count_nonzero(features[..., 2:]).item(),
            0,
        )

    def test_edge_swap_is_visible_when_degree_delta_is_zero(self):
        graphs = [
            make_graph([(0, 1), (2, 3)]),
            make_graph([(0, 3), (2, 1)]),
        ]

        features = extract_topology_dynamics(graphs)
        change = features[0, 1]

        # The swap preserves every node's degree.
        self.assertTrue(torch.equal(change[:, 2], torch.zeros(4)))
        self.assertTrue(torch.equal(change[:, 3], torch.zeros(4)))
        # Incoming endpoints 1 and 3 each receive one added and one removed edge.
        self.assertTrue(
            torch.equal(change[:, 4], torch.tensor([0.0, 1.0, 0.0, 1.0]))
        )
        self.assertTrue(
            torch.equal(change[:, 5], torch.tensor([0.0, 1.0, 0.0, 1.0]))
        )
        # Sources 0 and 2 each redirect one outgoing edge.
        self.assertTrue(
            torch.equal(change[:, 6], torch.tensor([1.0, 0.0, 1.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(change[:, 7], torch.tensor([1.0, 0.0, 1.0, 0.0]))
        )

    def test_batched_graphs_compare_each_sample_only_with_itself(self):
        fixed_step_0 = make_graph([(0, 1)])
        fixed_step_1 = make_graph([(0, 1)])
        changing_step_0 = make_graph([(0, 1)])
        changing_step_1 = make_graph([(2, 3)])
        graphs = [
            dgl.batch([fixed_step_0, changing_step_0]),
            dgl.batch([fixed_step_1, changing_step_1]),
        ]

        features = extract_topology_dynamics(graphs)

        self.assertEqual(tuple(features.shape), (2, 2, 4, TOPOLOGY_FEATURE_DIM))
        self.assertEqual(torch.count_nonzero(features[0, :, :, 2:]).item(), 0)
        self.assertGreater(torch.count_nonzero(features[1, 1, :, 2:]).item(), 0)

        encoder = TopologyDynamicsEncoder(output_dim=5)
        details = encoder(graphs, return_details=True)
        self.assertEqual(tuple(details['encoded_features'].shape), (2, 2, 4, 5))
        self.assertEqual(
            tuple(details['dynamic_features'].shape),
            (2, 2, 4, TOPOLOGY_DYNAMIC_FEATURE_DIM),
        )
        self.assertEqual(
            torch.count_nonzero(details['dynamic_embedding'][0]).item(),
            0,
        )

    def test_parallel_edge_addition_is_counted(self):
        graphs = [
            make_graph([(0, 1)]),
            make_graph([(0, 1), (0, 1)]),
        ]

        features = extract_topology_dynamics(graphs)
        change = features[0, 1]

        self.assertEqual(change[1, 2].item(), 1.0)
        self.assertEqual(change[0, 3].item(), 1.0)
        self.assertEqual(change[1, 4].item(), 1.0)
        self.assertEqual(change[0, 6].item(), 1.0)
        self.assertEqual(change[:, 5].sum().item(), 0.0)
        self.assertEqual(change[:, 7].sum().item(), 0.0)

    def test_feature_modes_isolate_static_and_dynamic_evidence(self):
        fixed_graphs = [
            make_graph([(0, 1)]),
            make_graph([(0, 1)]),
        ]
        changing_graphs = [
            make_graph([(0, 1)]),
            make_graph([(2, 3)]),
        ]

        dynamic_only = TopologyDynamicsEncoder(
            output_dim=3,
            feature_mode='dynamic_only',
        )
        static_only = TopologyDynamicsEncoder(
            output_dim=3,
            feature_mode='static_only',
        )

        fixed_dynamic = dynamic_only(fixed_graphs, return_details=True)
        changed_dynamic = dynamic_only(changing_graphs, return_details=True)
        fixed_static = static_only(fixed_graphs, return_details=True)

        self.assertEqual(
            torch.count_nonzero(fixed_dynamic['encoded_features']).item(),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(changed_dynamic['encoded_features'][:, 1]).item(),
            0,
        )
        self.assertGreater(
            torch.count_nonzero(fixed_static['encoded_features']).item(),
            0,
        )


if __name__ == '__main__':
    unittest.main()
