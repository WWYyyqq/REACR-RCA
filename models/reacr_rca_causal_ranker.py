from typing import Dict

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalPropagationRanker(nn.Module):
    def __init__(
        self,
        window_size: int,
        hidden_dim: int = 64,
        branch_hidden_dim: int = 64,
        dropout: float = 0.1,
        eps: float = 1e-6,
        anchor_init_scale: float = 1.0,
        branch_init_scale: float = 0.1,
        ranker_mode: str = 'legacy',
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.eps = eps
        self.ranker_mode = ranker_mode
        if self.ranker_mode not in {'legacy', 'root_symptom', 'counterfactual'}:
            raise ValueError(f'unsupported ranker_mode: {self.ranker_mode}')

        self.self_temporal = nn.GRU(input_size=1, hidden_size=branch_hidden_dim, batch_first=True)
        self.self_proj = nn.Linear(branch_hidden_dim, 1)

        self.incoming_proj = nn.Sequential(
            nn.Linear(2, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.outgoing_proj = nn.Sequential(
            nn.Linear(2, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(2, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.final_proj = nn.Sequential(
            nn.Linear(4, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )

        self.anchor_linear = nn.Linear(2, 1, bias=False)
        self.anchor_linear.weight.data.copy_(torch.tensor([[0.5, 0.5]], dtype=torch.float32))
        self.branch_scale = nn.Parameter(torch.full((4,), float(branch_init_scale), dtype=torch.float32))
        self.anchor_scale = nn.Parameter(torch.tensor(float(anchor_init_scale), dtype=torch.float32))
        self.root_role_scale = nn.Parameter(torch.tensor(float(branch_init_scale), dtype=torch.float32))
        self.symptom_role_scale = nn.Parameter(torch.tensor(float(branch_init_scale), dtype=torch.float32))
        self.counterfactual_scale = nn.Parameter(torch.tensor(float(branch_init_scale), dtype=torch.float32))

    def _graph_prior_features(self, graph: dgl.DGLGraph, node_score: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['feat'] = node_score.unsqueeze(-1)
            if graph.num_edges() == 0:
                zeros = torch.zeros_like(node_score)
                return torch.stack([zeros, zeros], dim=-1)

            if 'weight' in graph.edata:
                graph.edata['w'] = graph.edata['weight'].float().view(-1, 1)
                graph.update_all(fn.u_mul_e('feat', 'w', 'm'), fn.mean('m', 'in_feat'))
                graph_rev = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
                graph_rev.update_all(fn.u_mul_e('feat', 'w', 'm'), fn.mean('m', 'out_feat'))
            else:
                graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'in_feat'))
                graph_rev = dgl.reverse(graph, copy_ndata=True)
                graph_rev.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'out_feat'))

            incoming = torch.nan_to_num(graph.ndata['in_feat'].squeeze(-1), nan=0.0)
            outgoing = torch.nan_to_num(graph_rev.ndata['out_feat'].squeeze(-1), nan=0.0)
            return torch.stack([incoming, outgoing], dim=-1)

    def _counterfactual_evidence(
        self,
        anomaly_score: torch.Tensor,
        causal_matrix: torch.Tensor,
        lag_weights: torch.Tensor = None,
    ):
        batch_size, window_size, node_num = anomaly_score.shape
        effective_lags = min(self.window_size - 1, window_size - 1)
        if lag_weights is not None:
            effective_lags = min(effective_lags, int(lag_weights.numel()))
        if effective_lags <= 0:
            zeros = torch.zeros(batch_size, node_num, device=anomaly_score.device, dtype=anomaly_score.dtype)
            return zeros, torch.ones_like(zeros)

        if lag_weights is None:
            weights = torch.full(
                (effective_lags,),
                1.0 / effective_lags,
                device=anomaly_score.device,
                dtype=anomaly_score.dtype,
            )
        else:
            weights = lag_weights[:effective_lags].to(device=anomaly_score.device, dtype=anomaly_score.dtype)
            weights = weights / (weights.sum() + self.eps)

        shifted_sources = []
        for lag_idx in range(1, effective_lags + 1):
            pad = anomaly_score[:, :1, :].expand(-1, lag_idx, -1)
            shifted = torch.cat([pad, anomaly_score[:, :-lag_idx, :]], dim=1)
            shifted_sources.append(shifted)
        source_signal = torch.stack(shifted_sources, dim=0)
        source_signal = torch.einsum('l,lbwn->bwn', weights, source_signal)

        # causal_matrix[target, source]; each source explains downstream target anomalies.
        explained = source_signal.unsqueeze(2) * causal_matrix.unsqueeze(1)
        observed = anomaly_score.unsqueeze(-1)
        observed_sum = observed.sum(dim=(1, 2)).clamp_min(self.eps)
        coverage = torch.minimum(explained, observed).sum(dim=(1, 2)) / observed_sum
        residual = F.relu(observed - explained).sum(dim=(1, 2)) / observed_sum
        return coverage, residual

    def forward(
        self,
        graphs,
        anomaly_score: torch.Tensor,
        causal_matrix: torch.Tensor,
        lag_weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, window_size, node_num = anomaly_score.shape
        current_graph = graphs[-1]

        self_branch_input = anomaly_score.permute(0, 2, 1).reshape(batch_size * node_num, window_size, 1)
        _, self_last = self.self_temporal(self_branch_input)
        self_score = self.self_proj(self_last[-1]).view(batch_size, node_num)

        node_mean = anomaly_score.mean(dim=1)
        node_peak = anomaly_score.max(dim=1).values
        anchor_input = torch.stack([node_mean, node_peak], dim=-1)
        anchor_score = self.anchor_linear(anchor_input).squeeze(-1)

        incoming_causal = torch.bmm(causal_matrix, node_mean.unsqueeze(-1)).squeeze(-1)
        outgoing_causal = torch.bmm(causal_matrix.transpose(1, 2), node_mean.unsqueeze(-1)).squeeze(-1)
        incoming_explainedness = incoming_causal / (node_mean.abs() + self.eps)

        incoming_feat = torch.stack([incoming_causal, node_peak], dim=-1)
        outgoing_feat = torch.stack([outgoing_causal, node_peak], dim=-1)
        incoming_score = self.incoming_proj(incoming_feat).squeeze(-1)
        outgoing_score = self.outgoing_proj(outgoing_feat).squeeze(-1)

        flat_mean = node_mean.reshape(batch_size * node_num)
        graph_feat = self._graph_prior_features(current_graph, flat_mean).view(batch_size, node_num, 2)
        graph_score = self.graph_proj(graph_feat).squeeze(-1)

        final_input = torch.stack([self_score, incoming_score, outgoing_score, graph_score], dim=-1)
        correction = self.final_proj(final_input).squeeze(-1)

        scaled_self = self.branch_scale[0] * self_score
        scaled_incoming = self.branch_scale[1] * incoming_score
        scaled_outgoing = self.branch_scale[2] * outgoing_score
        scaled_graph = self.branch_scale[3] * graph_score
        final_score = (
            self.anchor_scale * anchor_score
            + scaled_self
            + scaled_incoming
            + scaled_outgoing
            + scaled_graph
            + correction
        )
        source_score = outgoing_score + outgoing_causal
        victim_score = incoming_score + incoming_causal + incoming_explainedness
        root_role_score = source_score - victim_score
        symptom_role_score = victim_score - source_score

        counterfactual_coverage, counterfactual_residual = self._counterfactual_evidence(
            anomaly_score,
            causal_matrix,
            lag_weights=lag_weights,
        )

        if self.ranker_mode in {'root_symptom', 'counterfactual'}:
            final_score = (
                self.anchor_scale * anchor_score
                + scaled_self
                + scaled_outgoing
                - torch.abs(self.branch_scale[1]) * incoming_score
                + scaled_graph
                + correction
                + self.root_role_scale * root_role_score
                - torch.abs(self.symptom_role_scale) * symptom_role_score
            )

        if self.ranker_mode == 'counterfactual':
            final_score = (
                final_score
                + self.counterfactual_scale * counterfactual_coverage
                - torch.abs(self.counterfactual_scale) * counterfactual_residual
                - torch.abs(self.symptom_role_scale) * incoming_explainedness
            )

        return {
            'anchor_score': anchor_score,
            'self_score': self_score,
            'incoming_score': incoming_score,
            'outgoing_score': outgoing_score,
            'graph_score': graph_score,
            'source_score': source_score,
            'victim_score': victim_score,
            'incoming_explainedness': incoming_explainedness,
            'root_role_score': root_role_score,
            'symptom_role_score': symptom_role_score,
            'counterfactual_coverage': counterfactual_coverage,
            'counterfactual_residual': counterfactual_residual,
            'final_score': final_score,
        }


def summarize_rank_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        'anchor_score_shape': list(output['anchor_score'].shape),
        'self_score_shape': list(output['self_score'].shape),
        'incoming_score_shape': list(output['incoming_score'].shape),
        'outgoing_score_shape': list(output['outgoing_score'].shape),
        'graph_score_shape': list(output['graph_score'].shape),
        'source_score_shape': list(output['source_score'].shape),
        'victim_score_shape': list(output['victim_score'].shape),
        'incoming_explainedness_shape': list(output['incoming_explainedness'].shape),
        'root_role_score_shape': list(output['root_role_score'].shape),
        'symptom_role_score_shape': list(output['symptom_role_score'].shape),
        'counterfactual_coverage_shape': list(output['counterfactual_coverage'].shape),
        'counterfactual_residual_shape': list(output['counterfactual_residual'].shape),
        'final_score_shape': list(output['final_score'].shape),
    }
