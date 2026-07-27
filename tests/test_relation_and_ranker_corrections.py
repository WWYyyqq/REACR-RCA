import unittest

import dgl
import torch

from models.reacr_rca_causal_head import RunInspiredCausalHead
from models.reacr_rca_causal_ranker import CausalPropagationRanker
from models.reacr_rca_trainer import PairwiseRankingLoss, root_symptom_loss
from models.layers import RankingLoss as DeepHuntRankingLoss


class RelationHeadCorrectionTests(unittest.TestCase):
    def test_even_moving_average_kernel_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            RunInspiredCausalHead(
                node_num=2,
                window_size=4,
                hidden_dim=4,
                max_lag=1,
                kernel_size=4,
            )
        with self.assertRaisesRegex(ValueError, 'positive odd'):
            RunInspiredCausalHead(
                node_num=2,
                window_size=4,
                hidden_dim=4,
                max_lag=1,
                kernel_size=5.5,
            )

    def test_lag_support_is_aligned_to_target_snapshot_topology(self):
        no_edge = dgl.graph(([], []), num_nodes=3)
        edge = dgl.graph(([0], [1]), num_nodes=3)
        graphs = [
            dgl.batch([no_edge]),
            dgl.batch([edge]),
            dgl.batch([no_edge]),
            dgl.batch([no_edge]),
        ]
        score = torch.rand(1, 4, 3)
        head = RunInspiredCausalHead(
            node_num=3,
            window_size=4,
            hidden_dim=8,
            max_lag=2,
            topology_mode='directed',
        )
        output = head(score, graphs=graphs)
        self.assertTrue(output['lag_topology_mask'][0, 0, 1, 0])
        self.assertFalse(output['lag_topology_mask'][0, 1, 1, 0])
        self.assertTrue(output['lag_alignment_topology_mask'][0, 0, 1, 1, 0])
        self.assertFalse(output['lag_alignment_topology_mask'][0, 0, 2, 1, 0])
        self.assertGreater(output['lag_weights'][0, 0, 1, 0].item(), 0.0)
        self.assertEqual(output['lag_weights'][0, 1, 1, 0].item(), 0.0)

    def test_lag_affinity_uses_only_edge_present_target_times(self):
        head = RunInspiredCausalHead(
            node_num=2,
            window_size=4,
            hidden_dim=4,
            max_lag=1,
            topology_mode='directed',
        )
        score = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
        alignment = torch.zeros(1, 1, 4, 2, 2, dtype=torch.bool)
        alignment[0, 0, 2, 1, 0] = True
        lag_support = alignment.any(dim=2)

        _, strengths, weights = head._lag_affinity(
            score,
            lag_topology_mask=lag_support,
            lag_alignment_topology_mask=alignment,
        )

        # Only target time t=2 is valid: target[2] * source[1] = 30 * 2.
        self.assertAlmostEqual(strengths[0, 0, 1, 0].item(), 60.0, places=6)
        self.assertAlmostEqual(weights[0, 0, 1, 0].item(), 1.0, places=6)

    def test_null_source_handles_no_observed_dependency(self):
        graph = dgl.graph(([], []), num_nodes=3)
        batched_steps = [dgl.batch([graph]) for _ in range(4)]
        score = torch.rand(1, 4, 3)
        head = RunInspiredCausalHead(
            node_num=3,
            window_size=4,
            hidden_dim=8,
            max_lag=2,
            topology_mode='undirected',
            use_null_source=True,
        )
        output = head(score, graphs=batched_steps)
        self.assertTrue(torch.equal(output['causal_matrix'], torch.zeros_like(output['causal_matrix'])))
        self.assertTrue(
            torch.allclose(
                output['null_source_probability'],
                torch.ones_like(output['null_source_probability']),
            )
        )

    def test_relation_respects_observed_topology_and_probability_mass(self):
        graph = dgl.graph(([0], [1]), num_nodes=3)
        batched_steps = [dgl.batch([graph]) for _ in range(4)]
        score = torch.rand(1, 4, 3)
        head = RunInspiredCausalHead(
            node_num=3,
            window_size=4,
            hidden_dim=8,
            max_lag=2,
            topology_mode='directed',
            use_null_source=True,
        )
        output = head(score, graphs=batched_steps)
        relation = output['causal_matrix']
        allowed = output['topology_mask']
        self.assertTrue(torch.equal(relation.masked_select(~allowed), torch.zeros_like(relation.masked_select(~allowed))))
        mass = relation.sum(dim=-1) + output['null_source_probability']
        self.assertTrue(torch.allclose(mass, torch.ones_like(mass), atol=1e-6))
        self.assertEqual(list(output['lag_weights'].shape), [1, 2, 3, 3])


class RankerCorrectionTests(unittest.TestCase):
    def test_graph_prior_uses_normalized_weighted_mean(self):
        graph = dgl.graph(([0, 1], [2, 2]), num_nodes=3)
        graph.edata['weight'] = torch.tensor([1.0, 3.0])
        ranker = CausalPropagationRanker(
            window_size=2,
            hidden_dim=4,
            branch_hidden_dim=4,
        )

        features = ranker._graph_prior_features(
            graph,
            torch.tensor([2.0, 10.0, 0.0]),
        )

        self.assertAlmostEqual(features[2, 0].item(), 8.0, places=6)

    def test_temporal_graph_prior_uses_earlier_graph_snapshots(self):
        edge_graph = dgl.batch([dgl.graph(([0], [1]), num_nodes=2)])
        empty_graph = dgl.batch([dgl.graph(([], []), num_nodes=2)])
        ranker = CausalPropagationRanker(
            window_size=2,
            hidden_dim=4,
            branch_hidden_dim=4,
        )
        anomaly = torch.tensor([[[4.0, 0.0], [0.0, 0.0]]])

        features = ranker._temporal_graph_prior_features(
            [edge_graph, empty_graph],
            anomaly,
        )

        self.assertAlmostEqual(features[0, 1, 0].item(), 2.0, places=6)

    def test_pairwise_loss_treats_replica_labels_as_any_hit_bag(self):
        criterion = PairwiseRankingLoss(margin=1.0)
        scores = torch.tensor([[2.0, 0.5, 0.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 1.0, 0.0]])

        loss = criterion(scores, labels)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_deephunt_loss_uses_same_candidate_bag_semantics_and_is_finite(self):
        criterion = DeepHuntRankingLoss(rank_range=3)
        labels = torch.tensor([1.0, 1.0, 0.0])

        bag_loss = criterion(torch.tensor([2.0, 0.5, 0.0]), labels)
        constant_scores = torch.ones(3, requires_grad=True)
        constant_loss = criterion(constant_scores, labels)
        constant_loss.backward()
        near_tie_scores = torch.tensor(
            [1.0, 1.0 + 1e-7, 1.0 - 1e-7],
            requires_grad=True,
        )
        near_tie_loss = criterion(near_tie_scores, labels)
        near_tie_loss.backward()

        self.assertAlmostEqual(bag_loss.item(), 0.0, places=6)
        self.assertTrue(torch.isfinite(constant_loss))
        self.assertTrue(torch.all(torch.isfinite(constant_scores.grad)))
        self.assertLessEqual(float(constant_scores.grad.abs().max()), 1.0)
        self.assertTrue(torch.all(torch.isfinite(near_tie_scores.grad)))
        self.assertLessEqual(float(near_tie_scores.grad.abs().max()), 1.0)

    def setUp(self):
        torch.manual_seed(7)
        self.ranker = CausalPropagationRanker(
            window_size=4,
            hidden_dim=8,
            branch_hidden_dim=8,
            dropout=0.0,
            ranker_mode='source_hypothesis',
        )

    def test_roles_are_independently_parameterised_probabilities(self):
        graph = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
        graphs = [dgl.batch([graph]) for _ in range(4)]
        anomaly = torch.rand(1, 4, 2)
        relation = torch.tensor([[[0.0, 0.3], [0.6, 0.0]]])
        lag_weights = torch.full((1, 2, 2, 2), 0.5)
        output = self.ranker(graphs, anomaly, relation, lag_weights)
        source = output['source_role_score']
        response = output['response_role_score']
        self.assertTrue(torch.all((0.0 <= source) & (source <= 1.0)))
        self.assertTrue(torch.all((0.0 <= response) & (response <= 1.0)))
        self.assertFalse(torch.allclose(response, -source))
        self.assertTrue(torch.allclose(output['role_margin'], source - response))

    def test_residual_is_not_coverage_complement_and_penalises_over_explanation(self):
        anomaly = torch.tensor(
            [[[10.0, 1.0], [10.0, 1.0], [10.0, 1.0], [10.0, 1.0]]]
        )
        relation = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        lag_weights = torch.tensor([1.0])
        coverage, residual = self.ranker._source_hypothesis_evidence(
            anomaly,
            relation,
            lag_weights,
        )
        self.assertFalse(torch.allclose(coverage + residual, torch.ones_like(coverage)))
        self.assertGreater(float(residual[0, 0]), 0.0)

    def test_lag_prefix_without_history_is_excluded_from_evidence(self):
        ranker = CausalPropagationRanker(
            window_size=3,
            hidden_dim=4,
            branch_hidden_dim=4,
            ranker_mode='source_hypothesis',
        )
        anomaly = torch.tensor(
            [[[5.0, 0.0], [0.0, 5.0], [0.0, 0.0]]]
        )
        relation = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])

        coverage, residual = ranker._source_hypothesis_evidence(
            anomaly,
            relation,
            lag_weights=torch.tensor([1.0]),
        )

        self.assertAlmostEqual(coverage[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(residual[0, 0].item(), 0.0, places=6)

    def test_role_loss_uses_hard_negative_response_candidates(self):
        rank_output = {
            'source_role_score': torch.tensor([[0.7, 0.2, 0.4]], requires_grad=True),
            'response_role_score': torch.tensor([[0.2, 0.8, 0.6]], requires_grad=True),
            'root_role_score': torch.tensor([[0.7, 0.2, 0.4]]),
            'symptom_role_score': torch.tensor([[0.2, 0.8, 0.6]]),
            'victim_score': torch.tensor([[0.1, 0.9, 0.6]]),
            'incoming_explainedness': torch.tensor([[0.1, 0.9, 0.5]]),
        }
        labels = torch.tensor([[1.0, 0.0, 0.0]])
        loss = root_symptom_loss(rank_output, labels, hard_negative_topk=1)
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(rank_output['source_role_score'].grad)
        self.assertIsNotNone(rank_output['response_role_score'].grad)

    def test_role_loss_does_not_force_positive_response_to_zero_without_pseudo_candidates(self):
        source = torch.tensor([[0.7, 0.8]], requires_grad=True)
        response = torch.tensor([[0.9, 0.9]], requires_grad=True)
        rank_output = {
            'source_role_score': source,
            'response_role_score': response,
            'root_role_score': source,
            'symptom_role_score': response,
            'victim_score': torch.zeros(1, 2),
            'incoming_explainedness': torch.zeros(1, 2),
        }
        loss = root_symptom_loss(rank_output, torch.ones(1, 2))
        loss.backward()
        self.assertIsNotNone(source.grad)
        self.assertIsNone(response.grad)

    def test_role_loss_does_not_assign_absolute_response_target_to_pseudo_candidate(self):
        source = torch.tensor([[0.95, 0.05]], requires_grad=True)
        response = torch.tensor([[0.05, 0.95]], requires_grad=True)
        rank_output = {
            'source_role_score': source,
            'response_role_score': response,
            'root_role_score': source,
            'symptom_role_score': response,
            'victim_score': torch.tensor([[0.0, 1.0]]),
            'incoming_explainedness': torch.tensor([[0.0, 1.0]]),
        }

        loss = root_symptom_loss(
            rank_output,
            torch.tensor([[1.0, 0.0]]),
            margin=0.2,
            hard_negative_topk=1,
        )
        loss.backward()

        self.assertAlmostEqual(response.grad[0, 1].item(), 0.0, places=7)


if __name__ == '__main__':
    unittest.main()
