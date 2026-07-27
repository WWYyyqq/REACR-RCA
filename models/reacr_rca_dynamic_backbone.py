from typing import Dict, List

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reacr_rca_topology_dynamics import TopologyDynamicsEncoder


class WeightedSpatialLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        edge_feature_dim: int = 0,
    ):
        super().__init__()
        self.edge_feature_dim = int(edge_feature_dim)
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neigh_linear = nn.Linear(in_dim, out_dim)
        self.edge_mass_linear = nn.Linear(1, out_dim, bias=False) if self.edge_feature_dim > 0 else None
        self.edge_context_linear = nn.Linear(self.edge_feature_dim, out_dim, bias=False) if self.edge_feature_dim > 0 else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        with graph.local_scope():
            graph.ndata['h'] = features
            edge_mass = None
            edge_context = None
            if graph.num_edges() == 0:
                neigh = torch.zeros_like(features)
            else:
                if 'weight' in graph.edata:
                    edge_weight = graph.edata['weight'].float().clamp_min(0.0).view(-1, 1)
                    graph.edata['w'] = edge_weight
                    graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'weighted_sum'))
                    graph.update_all(fn.copy_e('w', 'weight_message'), fn.sum('weight_message', 'weight_sum'))
                    neigh = graph.ndata['weighted_sum'] / graph.ndata['weight_sum'].clamp_min(1e-6)
                    edge_mass = torch.log1p(graph.ndata['weight_sum'])
                else:
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                    neigh = graph.ndata['h_neigh']
                if self.edge_feature_dim > 0:
                    if 'edge_attr' not in graph.edata:
                        raise ValueError('edge_attr is required when edge_feature_dim > 0')
                    edge_attr = graph.edata['edge_attr'].float()
                    if edge_attr.ndim != 2 or edge_attr.shape[1] != self.edge_feature_dim:
                        raise ValueError(
                            'graph.edata["edge_attr"] must be [E, edge_feature_dim], '
                            f'got {tuple(edge_attr.shape)} with edge_feature_dim={self.edge_feature_dim}'
                        )
                    graph.edata['edge_context'] = edge_attr
                    graph.update_all(
                        fn.copy_e('edge_context', 'edge_context_message'),
                        fn.mean('edge_context_message', 'edge_context_mean'),
                    )
                    edge_context = graph.ndata['edge_context_mean']

            out = self.self_linear(features) + self.neigh_linear(neigh)
            if edge_mass is not None and self.edge_mass_linear is not None:
                out = out + self.edge_mass_linear(edge_mass)
            if edge_context is not None and self.edge_context_linear is not None:
                out = out + self.edge_context_linear(edge_context)
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
        causal_temporal_attention: bool = True,
        use_topology_dynamics: bool = True,
        topology_dropout: float = 0.0,
        edge_feature_dim: int = 0,
        topology_feature_mode: str = 'full',
    ):
        super().__init__()
        self.in_dim = in_dim
        self.spatial_hidden_dim = spatial_hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        self.num_spatial_layers = num_spatial_layers
        self.use_temporal_attention = use_temporal_attention
        self.causal_temporal_attention = causal_temporal_attention
        self.use_topology_dynamics = use_topology_dynamics
        self.edge_feature_dim = int(edge_feature_dim)
        if self.use_topology_dynamics:
            self.topology_encoder = TopologyDynamicsEncoder(
                output_dim=spatial_hidden_dim,
                dropout=topology_dropout,
                feature_mode=topology_feature_mode,
            )
            self.topology_scale = nn.Parameter(torch.tensor(-2.1972246))
        else:
            self.topology_encoder = None
            self.topology_scale = None

        layers: List[nn.Module] = []
        current_dim = in_dim
        for _ in range(num_spatial_layers):
            layers.append(
                WeightedSpatialLayer(
                    current_dim,
                    spatial_hidden_dim,
                    dropout=dropout,
                    edge_feature_dim=self.edge_feature_dim,
                )
            )
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

    @staticmethod
    def _causal_attention_mask(window_size: int, device: torch.device) -> torch.Tensor:
        """Mask keys later than each query while retaining the current step."""
        return torch.triu(
            torch.ones(window_size, window_size, dtype=torch.bool, device=device),
            diagonal=1,
        )

    def forward(self, graphs: List[dgl.DGLGraph], features: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, window_size, node_num, feat_dim = features.shape
        spatial_steps = []
        topology_output = None
        if self.topology_encoder is not None:
            topology_output = self.topology_encoder(graphs, return_details=True)
            topology_encoded = topology_output['encoded_features'].to(
                device=features.device,
                dtype=features.dtype,
            )
        else:
            topology_encoded = None

        for step_idx in range(window_size):
            graph_t = graphs[step_idx]
            feat_t = features[:, step_idx, :, :].reshape(batch_size * node_num, feat_dim)
            spatial_t = self.encode_spatial_step(graph_t, feat_t)
            if topology_encoded is not None:
                topology_t = topology_encoded[:, step_idx].reshape(
                    batch_size * node_num,
                    self.spatial_hidden_dim,
                )
                spatial_t = spatial_t + torch.sigmoid(self.topology_scale) * topology_t
            spatial_steps.append(spatial_t.view(batch_size, node_num, self.spatial_hidden_dim))

        spatial_states = torch.stack(spatial_steps, dim=1)  # [B, W, N, Hs]
        temporal_input = spatial_states.permute(0, 2, 1, 3).reshape(batch_size * node_num, window_size, self.spatial_hidden_dim)

        temporal_states, final_hidden = self.temporal_gru(temporal_input)
        if self.temporal_attention is not None:
            attention_mask = None
            if self.causal_temporal_attention:
                attention_mask = self._causal_attention_mask(window_size, temporal_states.device)
            attn_out, attn_weights = self.temporal_attention(
                temporal_states,
                temporal_states,
                temporal_states,
                attn_mask=attention_mask,
            )
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
            'topology_output': topology_output,
        }


def summarize_backbone_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    summary = {
        'spatial_states_shape': list(output['spatial_states'].shape),
        'temporal_states_shape': list(output['temporal_states'].shape),
        'final_node_repr_shape': list(output['final_node_repr'].shape),
        'window_repr_shape': list(output['window_repr'].shape),
        'topology_features_shape': (
            None
            if output.get('topology_output') is None
            else list(output['topology_output']['raw_features'].shape)
        ),
        'topology_dynamic_nonzero_count': (
            0
            if output.get('topology_output') is None
            else int(torch.count_nonzero(output['topology_output']['dynamic_features']).item())
        ),
    }
    if output['attn_weights'] is not None:
        summary['attn_weights_shape'] = list(output['attn_weights'].shape)
    else:
        summary['attn_weights_shape'] = None
    return summary
