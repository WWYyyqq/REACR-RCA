import unittest

import dgl
import torch

from models.reacr_rca_topology_dynamics import canonical_edge_multiset
from models.reacr_rca_trainer import (
    observed_topology_change_mask,
    prepare_graph_sequence,
)


def make_graph(edges, node_num=3):
    sources = [source for source, _ in edges]
    targets = [target for _, target in edges]
    return dgl.graph((sources, targets), num_nodes=node_num)


class GraphSequenceAblationTest(unittest.TestCase):
    def test_observed_change_mask_is_independent_of_model_ablation(self):
        fixed = [make_graph([(0, 1)]), make_graph([(0, 1)])]
        changed = [make_graph([(0, 1)]), make_graph([(1, 2)])]
        batched_steps = [
            dgl.batch([fixed[step], changed[step]])
            for step in range(2)
        ]
        self.assertTrue(
            torch.equal(
                observed_topology_change_mask(batched_steps),
                torch.tensor([False, True]),
            )
        )

    def test_observed_mode_retains_real_topology_change(self):
        graphs = [make_graph([(0, 1)]), make_graph([(1, 2)])]
        selected = prepare_graph_sequence(graphs, mode='observed')
        self.assertNotEqual(
            canonical_edge_multiset(selected[0]),
            canonical_edge_multiset(selected[1]),
        )

    def test_freeze_first_removes_change_without_adding_edges(self):
        graphs = [make_graph([(0, 1)]), make_graph([(1, 2)])]
        selected = prepare_graph_sequence(graphs, mode='freeze_first')
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            canonical_edge_multiset(selected[0]),
            canonical_edge_multiset(selected[1]),
        )
        self.assertEqual(canonical_edge_multiset(selected[1]), ((0, 1),))


if __name__ == '__main__':
    unittest.main()
