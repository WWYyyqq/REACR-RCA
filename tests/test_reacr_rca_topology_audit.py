import unittest

import dgl
import torch

from models.reacr_rca_dsbg_dataset import DSBGWindow
from tools.reacr_rca_topology_audit import (
    audit_graph_sequence,
    audit_windows,
    window_has_topology_change,
)


def make_graph(edges, node_num=4):
    sources = [source for source, _ in edges]
    destinations = [destination for _, destination in edges]
    return dgl.graph((sources, destinations), num_nodes=node_num)


class TopologyAuditTest(unittest.TestCase):
    def test_fixed_topology_and_weight_availability(self):
        graphs = []
        for _ in range(3):
            graph = make_graph([(0, 1), (1, 2)])
            graph.edata['weight'] = torch.tensor([0.25, 0.75])
            graphs.append(graph)

        report = audit_graph_sequence(
            graphs,
            timestamps=[10, 20, 30],
            include_edge_sets=True,
        )

        self.assertEqual(report['total_snapshots'], 3)
        self.assertEqual(report['unique_edge_sets'], 1)
        self.assertEqual(report['topology_switches'], 0)
        self.assertTrue(report['fixed_edge_topology'])
        self.assertTrue(report['edge_weight_availability']['available'])
        self.assertEqual(
            report['edge_weight_availability']['snapshots_with_weight_like_data'],
            3,
        )
        self.assertEqual(report['edge_churn']['total_added_edges'], 0)
        self.assertEqual(report['edge_churn']['total_removed_edges'], 0)
        self.assertEqual(
            report['unique_edge_set_catalog'][0]['edges'],
            [[0, 1], [1, 2]],
        )
        self.assertFalse(report['provenance']['self_loops_added_by_audit'])
        self.assertFalse(report['provenance']['catalog_edges_added_by_audit'])

    def test_changed_topology_reports_exact_churn_and_partial_edge_data(self):
        graph_a_0 = make_graph([(0, 1), (2, 3)])
        graph_b = make_graph([(0, 3), (2, 1)])
        graph_a_1 = make_graph([(0, 1), (2, 3)])
        graph_b.edata['edge_type'] = torch.tensor([1, 2])

        report = audit_graph_sequence(
            [graph_a_0, graph_b, graph_a_1],
            timestamps=[100, 160, 220],
        )

        self.assertEqual(report['unique_edge_sets'], 2)
        self.assertEqual(report['topology_switches'], 2)
        self.assertFalse(report['fixed_edge_topology'])
        self.assertEqual(report['edge_churn']['total_added_edges'], 4)
        self.assertEqual(report['edge_churn']['total_removed_edges'], 4)
        self.assertEqual(report['edge_churn']['max_symmetric_difference'], 4)
        self.assertEqual(
            len(report['edge_churn']['changed_transition_records']),
            2,
        )
        self.assertEqual(
            report['edge_data']['keys']['edge_type']['snapshots_present'],
            1,
        )
        self.assertFalse(report['edge_weight_availability']['available'])

    def test_audit_rejects_batched_graph_as_one_temporal_snapshot(self):
        batched = dgl.batch(
            [
                make_graph([(0, 1)]),
                make_graph([(2, 3)]),
            ]
        )
        with self.assertRaisesRegex(ValueError, 'DGL batch'):
            audit_graph_sequence([batched])

    def test_window_audit_reports_only_changes_seen_by_model_windows(self):
        fixed = DSBGWindow(
            timestamps=[0, 60],
            graphs=[make_graph([(0, 1)]), make_graph([(0, 1)])],
            features=torch.zeros(2, 4, 1),
            case_id=7,
            case_timestamp=60,
        )
        changing = DSBGWindow(
            timestamps=[120, 180],
            graphs=[make_graph([(0, 1)]), make_graph([(2, 3)])],
            features=torch.zeros(2, 4, 1),
            case_id=8,
            case_timestamp=180,
        )

        report = audit_windows([fixed, changing])

        self.assertEqual(report['total_windows'], 2)
        self.assertEqual(report['windows_with_topology_changes'], 1)
        self.assertEqual(report['changed_within_window_transitions'], 1)
        self.assertEqual(report['changed_window_records'][0]['case_id'], 8)
        self.assertFalse(report['fixed_topology_within_every_window'])
        self.assertFalse(window_has_topology_change(fixed))
        self.assertTrue(window_has_topology_change(changing))


if __name__ == '__main__':
    unittest.main()
