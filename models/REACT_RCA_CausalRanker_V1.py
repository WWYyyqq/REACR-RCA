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
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.branch_hidden_dim = branch_hidden_dim
        self.eps = eps

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

    def forward(self, graphs, anomaly_score: torch.Tensor, causal_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
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

        return {
            'anchor_score': anchor_score,
            'self_score': self_score,
            'incoming_score': incoming_score,
            'outgoing_score': outgoing_score,
            'graph_score': graph_score,
            'final_score': final_score,
        }


def summarize_rank_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        'anchor_score_shape': list(output['anchor_score'].shape),
        'self_score_shape': list(output['self_score'].shape),
        'incoming_score_shape': list(output['incoming_score'].shape),
        'outgoing_score_shape': list(output['outgoing_score'].shape),
        'graph_score_shape': list(output['graph_score'].shape),
        'final_score_shape': list(output['final_score'].shape),
    }
