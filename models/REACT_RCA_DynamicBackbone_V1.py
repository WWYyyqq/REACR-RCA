from typing import Dict, List

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedSpatialLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['h'] = features
            if graph.num_edges() == 0:
                neigh = torch.zeros_like(features)
            else:
                if 'weight' in graph.edata:
                    edge_weight = graph.edata['weight'].float().view(-1, 1)
                    graph.edata['w'] = edge_weight
                    graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.mean('m', 'h_neigh'))
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                neigh = graph.ndata['h_neigh']

            out = self.self_linear(features) + self.neigh_linear(neigh)
            out = F.relu(out)
            out = self.dropout(out)
            return out


class DynamicSpatialTemporalBackbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        spatial_hidden_dim: int = 64,
        temporal_hidden_dim: int = 64,
        num_spatial_layers: int = 2,
        dropout: float = 0.1,
        num_temporal_heads: int = 1,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_spatial_layers = num_spatial_layers
        self.use_temporal_attention = use_temporal_attention

        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_spatial_layers):
            layers.append(WeightedSpatialLayer(current_dim, spatial_hidden_dim, dropout=dropout))
            current_dim = spatial_hidden_dim
        self.spatial_layers = nn.ModuleList(layers)

        self.temporal_gru = nn.GRU(
            input_size=spatial_hidden_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )

        if self.use_temporal_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=temporal_hidden_dim,
                num_heads=num_temporal_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_norm = nn.LayerNorm(temporal_hidden_dim)
        else:
            self.temporal_attention = None
            self.temporal_norm = None

    def encode_spatial_step(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        hidden = features
        for layer in self.spatial_layers:
            hidden = layer(graph, hidden)
        return hidden

    def forward(self, graphs: List[dgl.DGLGraph], features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, window_size, node_num, feat_dim = features.shape
        spatial_steps = []

        for step_idx in range(window_size):
            graph_t = graphs[step_idx]
            feat_t = features[:, step_idx, :, :].reshape(batch_size * node_num, feat_dim)
            spatial_t = self.encode_spatial_step(graph_t, feat_t)
            spatial_steps.append(spatial_t.view(batch_size, node_num, self.spatial_hidden_dim))

        spatial_states = torch.stack(spatial_steps, dim=1)  # [B, W, N, Hs]
        temporal_input = spatial_states.permute(0, 2, 1, 3).reshape(batch_size * node_num, window_size, self.spatial_hidden_dim)

        temporal_states, final_hidden = self.temporal_gru(temporal_input)
        if self.temporal_attention is not None:
            attn_out, attn_weights = self.temporal_attention(temporal_states, temporal_states, temporal_states)
            temporal_states = self.temporal_norm(temporal_states + attn_out)
        else:
            attn_weights = None

        temporal_states = temporal_states.view(batch_size, node_num, window_size, self.temporal_hidden_dim).permute(0, 2, 1, 3)
        final_node_repr = temporal_states[:, -1, :, :]
        pooled_window_repr = final_node_repr.mean(dim=1)

        return {
            'spatial_states': spatial_states,
            'temporal_states': temporal_states,
            'final_node_repr': final_node_repr,
            'window_repr': pooled_window_repr,
            'attn_weights': attn_weights,
        }


def summarize_backbone_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    summary = {
        'spatial_states_shape': list(output['spatial_states'].shape),
        'temporal_states_shape': list(output['temporal_states'].shape),
        'final_node_repr_shape': list(output['final_node_repr'].shape),
        'window_repr_shape': list(output['window_repr'].shape),
    }
    if output['attn_weights'] is not None:
        summary['attn_weights_shape'] = list(output['attn_weights'].shape)
    else:
        summary['attn_weights_shape'] = None
    return summary
