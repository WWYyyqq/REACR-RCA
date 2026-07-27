import math
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
        if self.ranker_mode not in {'legacy', 'root_symptom', 'counterfactual', 'source_hypothesis'}:
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
        self.source_role_head = nn.Sequential(
            nn.Linear(3, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )
        self.response_role_head = nn.Sequential(
            nn.Linear(4, branch_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(branch_hidden_dim, 1),
        )

        def inverse_softplus(value: float) -> float:
            value = max(float(value), 1e-4)
            return math.log(math.expm1(value))

        self.anchor_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        self.branch_scale = nn.Parameter(
            torch.full((4,), inverse_softplus(branch_init_scale), dtype=torch.float32)
        )
        self.anchor_scale = nn.Parameter(
            torch.tensor(inverse_softplus(anchor_init_scale), dtype=torch.float32)
        )
        self.root_role_scale = nn.Parameter(
            torch.tensor(inverse_softplus(branch_init_scale), dtype=torch.float32)
        )
        self.symptom_role_scale = nn.Parameter(
            torch.tensor(inverse_softplus(branch_init_scale), dtype=torch.float32)
        )
        self.counterfactual_scale = nn.Parameter(
            torch.tensor(inverse_softplus(branch_init_scale), dtype=torch.float32)
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(inverse_softplus(branch_init_scale), dtype=torch.float32)
        )

    def _graph_prior_features(self, graph: dgl.DGLGraph, node_score: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['feat'] = node_score.unsqueeze(-1)
            if graph.num_edges() == 0:
                zeros = torch.zeros_like(node_score)
                return torch.stack([zeros, zeros], dim=-1)

            if 'weight' in graph.edata:
                edge_weight = graph.edata['weight'].float()
                if edge_weight.numel() != graph.num_edges():
                    raise ValueError(
                        'graph.edata["weight"] must contain one scalar per edge'
                    )
                graph.edata['w'] = edge_weight.view(-1, 1)
                graph.update_all(fn.u_mul_e('feat', 'w', 'm'), fn.sum('m', 'in_weighted_sum'))
                graph.update_all(fn.copy_e('w', 'wm'), fn.sum('wm', 'in_weight_sum'))
                graph.ndata['in_feat'] = (
                    graph.ndata['in_weighted_sum']
                    / graph.ndata['in_weight_sum'].clamp_min(self.eps)
                )
                graph_rev = dgl.reverse(graph, copy_ndata=True, copy_edata=True)
                graph_rev.update_all(fn.u_mul_e('feat', 'w', 'm'), fn.sum('m', 'out_weighted_sum'))
                graph_rev.update_all(fn.copy_e('w', 'wm'), fn.sum('wm', 'out_weight_sum'))
                graph_rev.ndata['out_feat'] = (
                    graph_rev.ndata['out_weighted_sum']
                    / graph_rev.ndata['out_weight_sum'].clamp_min(self.eps)
                )
            else:
                graph.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'in_feat'))
                graph_rev = dgl.reverse(graph, copy_ndata=True)
                graph_rev.update_all(fn.copy_u('feat', 'm'), fn.mean('m', 'out_feat'))

            incoming = torch.nan_to_num(graph.ndata['in_feat'].squeeze(-1), nan=0.0)
            outgoing = torch.nan_to_num(graph_rev.ndata['out_feat'].squeeze(-1), nan=0.0)
            return torch.stack([incoming, outgoing], dim=-1)

    def _temporal_graph_prior_features(
        self,
        graphs,
        anomaly_score: torch.Tensor,
    ) -> torch.Tensor:
        """Aggregate graph-neighbour evidence over the full observed window."""

        batch_size, window_size, node_num = anomaly_score.shape
        if len(graphs) != window_size:
            raise ValueError(
                f'graph sequence length {len(graphs)} does not match anomaly window {window_size}'
            )
        step_features = []
        for step_idx, graph in enumerate(graphs):
            flat_step_score = anomaly_score[:, step_idx, :].reshape(batch_size * node_num)
            graph_step = self._graph_prior_features(graph, flat_step_score).view(
                batch_size,
                node_num,
                2,
            )
            step_features.append(graph_step)
        return torch.stack(step_features, dim=1).mean(dim=1)

    def _source_hypothesis_evidence(
        self,
        anomaly_score: torch.Tensor,
        relation_matrix: torch.Tensor,
        lag_weights: torch.Tensor = None,
    ):
        """Evaluate how well each candidate source explains observed evidence.

        This is a model-internal source hypothesis score, not a do-intervention.
        The residual is a symmetric normalized explanation error, so it is not
        algebraically redundant with coverage and it penalises over-explanation.
        """
        batch_size, window_size, node_num = anomaly_score.shape
        effective_lags = min(self.window_size - 1, window_size - 1)
        if lag_weights is not None:
            if lag_weights.ndim == 1:
                effective_lags = min(effective_lags, int(lag_weights.numel()))
            elif lag_weights.ndim == 4:
                effective_lags = min(effective_lags, int(lag_weights.shape[1]))
            else:
                raise ValueError(
                    'lag_weights must be [L] or [B, L, target, source], '
                    f'got shape {tuple(lag_weights.shape)}'
                )
        if effective_lags <= 0:
            zeros = torch.zeros(batch_size, node_num, device=anomaly_score.device, dtype=anomaly_score.dtype)
            return zeros, torch.ones_like(zeros)

        shifted_sources = []
        lag_validity = []
        for lag_idx in range(1, effective_lags + 1):
            # No predecessor exists in the first ``lag_idx`` positions.  Zero
            # padding plus an explicit validity mask prevents those positions
            # from becoming fabricated explanatory evidence.
            pad = anomaly_score.new_zeros(batch_size, lag_idx, node_num)
            shifted = torch.cat([pad, anomaly_score[:, :-lag_idx, :]], dim=1)
            shifted_sources.append(shifted)
            valid = torch.arange(window_size, device=anomaly_score.device) >= lag_idx
            lag_validity.append(valid)
        shifted_sources = torch.stack(shifted_sources, dim=1)  # [B, L, W, source]
        lag_validity = torch.stack(lag_validity, dim=0).to(anomaly_score.dtype)  # [L, W]

        if lag_weights is None:
            weights = torch.full(
                (effective_lags,),
                1.0 / effective_lags,
                device=anomaly_score.device,
                dtype=anomaly_score.dtype,
            )
            valid_mass = torch.einsum('l,lw->w', weights, lag_validity)
            source_signal = torch.einsum(
                'l,blws,lw->bws',
                weights,
                shifted_sources,
                lag_validity,
            ) / valid_mass.view(1, window_size, 1).clamp_min(self.eps)
            explained = source_signal.unsqueeze(2) * relation_matrix.unsqueeze(1)
            valid_mask = (valid_mass > self.eps).view(1, window_size, 1, 1)
        elif lag_weights.ndim == 1:
            weights = lag_weights[:effective_lags].to(
                device=anomaly_score.device,
                dtype=anomaly_score.dtype,
            )
            weights = weights.clamp_min(0.0)
            if float(weights.sum().detach().item()) <= self.eps:
                weights = torch.full_like(weights, 1.0 / effective_lags)
            else:
                weights = weights / weights.sum()
            valid_mass = torch.einsum('l,lw->w', weights, lag_validity)
            source_signal = torch.einsum(
                'l,blws,lw->bws',
                weights,
                shifted_sources,
                lag_validity,
            ) / valid_mass.view(1, window_size, 1).clamp_min(self.eps)
            explained = source_signal.unsqueeze(2) * relation_matrix.unsqueeze(1)
            valid_mask = (valid_mass > self.eps).view(1, window_size, 1, 1)
        else:
            pair_weights = lag_weights[:, :effective_lags].to(
                device=anomaly_score.device,
                dtype=anomaly_score.dtype,
            )
            pair_weights = pair_weights.clamp_min(0.0)
            pair_sum = pair_weights.sum(dim=1, keepdim=True)
            uniform = torch.full_like(pair_weights, 1.0 / effective_lags)
            pair_weights = torch.where(
                pair_sum > self.eps,
                pair_weights / pair_sum.clamp_min(self.eps),
                uniform,
            )
            valid_mass = torch.einsum('blts,lw->bwts', pair_weights, lag_validity)
            explained = torch.einsum(
                'blws,blts,lw,bts->bwts',
                shifted_sources,
                pair_weights,
                lag_validity,
                relation_matrix,
            ) / valid_mass.clamp_min(self.eps)
            valid_mask = valid_mass > self.eps

        observed = anomaly_score.unsqueeze(-1)
        observed_expanded = observed.expand_as(explained)
        valid_mask = valid_mask.expand_as(explained)
        valid_observed = torch.where(valid_mask, observed_expanded, torch.zeros_like(observed_expanded))
        valid_explained = torch.where(valid_mask, explained, torch.zeros_like(explained))
        observed_sum = valid_observed.sum(dim=(1, 2)).clamp_min(self.eps)
        explained_sum = valid_explained.sum(dim=(1, 2))
        coverage = torch.minimum(valid_explained, valid_observed).sum(dim=(1, 2)) / observed_sum
        residual = torch.abs(valid_observed - valid_explained).sum(dim=(1, 2))
        residual = residual / (observed_sum + explained_sum).clamp_min(self.eps)
        return coverage, residual

    # Backward-compatible internal alias.
    _counterfactual_evidence = _source_hypothesis_evidence

    def forward(
        self,
        graphs,
        anomaly_score: torch.Tensor,
        causal_matrix: torch.Tensor,
        lag_weights: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, window_size, node_num = anomaly_score.shape

        self_branch_input = anomaly_score.permute(0, 2, 1).reshape(batch_size * node_num, window_size, 1)
        _, self_last = self.self_temporal(self_branch_input)
        self_score = F.softplus(self.self_proj(self_last[-1]).view(batch_size, node_num))

        node_mean = anomaly_score.mean(dim=1)
        node_peak = anomaly_score.max(dim=1).values
        anchor_input = torch.stack([node_mean, node_peak], dim=-1)
        anchor_weights = torch.softmax(self.anchor_logits, dim=0)
        anchor_score = torch.sum(anchor_input * anchor_weights.view(1, 1, 2), dim=-1)

        incoming_causal = torch.bmm(causal_matrix, node_mean.unsqueeze(-1)).squeeze(-1)
        outgoing_causal = torch.bmm(causal_matrix.transpose(1, 2), node_mean.unsqueeze(-1)).squeeze(-1)
        incoming_explainedness = incoming_causal / (
            incoming_causal + node_mean.abs() + self.eps
        )

        incoming_feat = torch.stack([incoming_causal, node_peak], dim=-1)
        outgoing_feat = torch.stack([outgoing_causal, node_peak], dim=-1)
        incoming_score = F.softplus(self.incoming_proj(incoming_feat).squeeze(-1))
        outgoing_score = F.softplus(self.outgoing_proj(outgoing_feat).squeeze(-1))

        graph_feat = self._temporal_graph_prior_features(graphs, anomaly_score)
        graph_score = F.softplus(self.graph_proj(graph_feat).squeeze(-1))

        final_input = torch.stack([self_score, incoming_score, outgoing_score, graph_score], dim=-1)
        correction = self.final_proj(final_input).squeeze(-1)

        positive_branch_scale = F.softplus(self.branch_scale)
        positive_anchor_scale = F.softplus(self.anchor_scale)
        scaled_self = positive_branch_scale[0] * self_score
        scaled_incoming = positive_branch_scale[1] * incoming_score
        scaled_outgoing = positive_branch_scale[2] * outgoing_score
        scaled_graph = positive_branch_scale[3] * graph_score
        final_score = (
            positive_anchor_scale * anchor_score
            + scaled_self
            + scaled_incoming
            + scaled_outgoing
            + scaled_graph
            + correction
        )

        source_hypothesis_coverage, source_hypothesis_residual = self._source_hypothesis_evidence(
            anomaly_score,
            causal_matrix,
            lag_weights=lag_weights,
        )
        source_score = outgoing_score + outgoing_causal + source_hypothesis_coverage
        victim_score = incoming_score + incoming_causal + incoming_explainedness
        source_role_input = torch.stack(
            [outgoing_score, outgoing_causal, source_hypothesis_coverage],
            dim=-1,
        )
        response_role_input = torch.stack(
            [incoming_score, incoming_causal, incoming_explainedness, node_peak],
            dim=-1,
        )
        source_role_score = torch.sigmoid(self.source_role_head(source_role_input).squeeze(-1))
        response_role_score = torch.sigmoid(self.response_role_head(response_role_input).squeeze(-1))
        role_margin = source_role_score - response_role_score

        # Backward-compatible aliases. They are now independently parameterised
        # source/response evidence rather than exact negatives of each other.
        root_role_score = source_role_score
        symptom_role_score = response_role_score

        if self.ranker_mode in {'root_symptom', 'counterfactual', 'source_hypothesis'}:
            final_score = (
                positive_anchor_scale * anchor_score
                + scaled_self
                + scaled_outgoing
                - scaled_incoming
                + scaled_graph
                + F.softplus(self.root_role_scale) * source_role_score
                - F.softplus(self.symptom_role_scale) * response_role_score
            )

        if self.ranker_mode in {'counterfactual', 'source_hypothesis'}:
            final_score = (
                final_score
                + F.softplus(self.counterfactual_scale) * source_hypothesis_coverage
                - F.softplus(self.residual_scale) * source_hypothesis_residual
                - F.softplus(self.symptom_role_scale) * incoming_explainedness
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
            'source_role_score': source_role_score,
            'response_role_score': response_role_score,
            'role_margin': role_margin,
            'root_role_score': root_role_score,
            'symptom_role_score': symptom_role_score,
            'source_hypothesis_coverage': source_hypothesis_coverage,
            'source_hypothesis_residual': source_hypothesis_residual,
            # Backward-compatible aliases for artifact readers.
            'counterfactual_coverage': source_hypothesis_coverage,
            'counterfactual_residual': source_hypothesis_residual,
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
        'source_role_score_shape': list(output['source_role_score'].shape),
        'response_role_score_shape': list(output['response_role_score'].shape),
        'role_margin_shape': list(output['role_margin'].shape),
        'root_role_score_shape': list(output['root_role_score'].shape),
        'symptom_role_score_shape': list(output['symptom_role_score'].shape),
        'counterfactual_coverage_shape': list(output['counterfactual_coverage'].shape),
        'counterfactual_residual_shape': list(output['counterfactual_residual'].shape),
        'final_score_shape': list(output['final_score'].shape),
    }
