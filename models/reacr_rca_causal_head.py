import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        try:
            normalized_kernel_size = int(kernel_size)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f'kernel_size must be a positive odd integer, got {kernel_size!r}'
            ) from exc
        if (
            normalized_kernel_size != kernel_size
            or normalized_kernel_size <= 0
            or normalized_kernel_size % 2 == 0
        ):
            raise ValueError(
                f'kernel_size must be a positive odd integer, got {kernel_size!r}'
            )
        self.kernel_size = normalized_kernel_size
        self.avg = nn.AvgPool1d(
            kernel_size=normalized_kernel_size,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, N]
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        x_avg = self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)
        return x_avg


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class RunInspiredCausalHead(nn.Module):
    """Learn case-conditioned directed lagged relation evidence.

    The historical class name is retained for API compatibility, but the
    returned matrix is not treated as an identified causal graph. Candidate
    sources are constrained by observed graph snapshots and an explicit null
    source allows a target to have no incoming relation.
    """

    def __init__(
        self,
        node_num: int,
        window_size: int,
        hidden_dim: int = 64,
        max_lag: int = 3,
        kernel_size: int = 5,
        gamma: float = 0.5,
        min_strength: float = 0.0,
        topology_mode: str = 'undirected',
        use_null_source: bool = True,
        lag_temperature: float = 1.0,
        topology_weight_gamma: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.node_num = node_num
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.gamma = gamma
        self.min_strength = min_strength
        self.topology_mode = topology_mode
        self.use_null_source = use_null_source
        self.lag_temperature = max(float(lag_temperature), eps)
        self.topology_weight_gamma = float(topology_weight_gamma)
        self.eps = eps
        if self.topology_mode not in {'none', 'directed', 'undirected'}:
            raise ValueError(f'unsupported topology_mode: {self.topology_mode}')

        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)
        self.source_encoder = nn.Sequential(
            nn.Linear(window_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(window_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.null_source_head = nn.Linear(hidden_dim, 1)
        self.lag_prior = nn.Parameter(torch.zeros(max_lag))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)

    def _lag_affinity(
        self,
        score: torch.Tensor,
        lag_topology_mask: Optional[torch.Tensor] = None,
        lag_alignment_topology_mask: Optional[torch.Tensor] = None,
    ):
        # score: [B, N, W]
        effective_lags = min(self.max_lag, max(score.shape[-1] - 1, 0))
        if effective_lags <= 0:
            empty = score.new_zeros(
                score.shape[0],
                0,
                score.shape[1],
                score.shape[1],
            )
            return score.new_zeros(score.shape[0], score.shape[1], score.shape[1]), empty, empty
        if lag_alignment_topology_mask is not None:
            expected_shape = (
                score.shape[0],
                effective_lags,
                score.shape[-1],
                score.shape[1],
                score.shape[1],
            )
            if tuple(lag_alignment_topology_mask.shape) != expected_shape:
                raise ValueError(
                    'lag_alignment_topology_mask must be '
                    f'{expected_shape}, got {tuple(lag_alignment_topology_mask.shape)}'
                )
        lag_strengths = []
        for lag_idx in range(1, effective_lags + 1):
            src = score[:, :, :-lag_idx]
            dst = score[:, :, lag_idx:]
            aligned_products = dst.unsqueeze(2) * src.unsqueeze(1)
            if lag_alignment_topology_mask is None:
                affinity = aligned_products.mean(dim=-1)
            else:
                # [B, W-lag, target, source] -> [B, target, source, W-lag]
                alignment = lag_alignment_topology_mask[
                    :, lag_idx - 1, lag_idx:, :, :
                ].permute(0, 2, 3, 1)
                alignment_weight = alignment.to(dtype=aligned_products.dtype)
                valid_count = alignment_weight.sum(dim=-1)
                affinity = (
                    (aligned_products * alignment_weight).sum(dim=-1)
                    / valid_count.clamp_min(1.0)
                )
                affinity = torch.where(
                    valid_count > 0,
                    affinity,
                    torch.zeros_like(affinity),
                )
            lag_strengths.append(affinity)
        lag_tensor = torch.stack(lag_strengths, dim=1)
        lag_logits = (
            lag_tensor / self.lag_temperature
            + self.lag_prior[:effective_lags].view(1, effective_lags, 1, 1)
        )
        if lag_topology_mask is not None:
            lag_logits = lag_logits.masked_fill(~lag_topology_mask, float('-inf'))
        # Case- and pair-conditioned lag distribution: [B, L, target, source].
        weights = torch.softmax(lag_logits, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        lag_affinity = torch.sum(weights * lag_tensor, dim=1)
        return lag_affinity, lag_tensor, weights

    def _build_topology_support(
        self,
        graphs: Optional[List],
        batch_size: int,
        node_num: int,
        device: torch.device,
        effective_lags: int,
        window_size: int,
    ):
        """Return window-, lag-, and aligned-time observed relation support.

        For lag ``l``, the product for ``(t-l, t)`` is retained only when the
        edge is present in target snapshot ``t``.  Lag- and window-level masks
        are unions of this time-resolved support.
        """

        lag_alignment_allowed = torch.zeros(
            batch_size,
            effective_lags,
            window_size,
            node_num,
            node_num,
            dtype=torch.bool,
            device=device,
        )
        if self.topology_mode == 'none' or graphs is None:
            for lag_idx in range(1, effective_lags + 1):
                lag_alignment_allowed[:, lag_idx - 1, lag_idx:, :, :] = True
            lag_allowed = torch.any(lag_alignment_allowed, dim=2)
            allowed = torch.any(lag_allowed, dim=1) if effective_lags > 0 else torch.zeros(
                batch_size, node_num, node_num, dtype=torch.bool, device=device
            )
            return allowed, lag_allowed, lag_alignment_allowed

        if len(graphs) != window_size:
            raise ValueError(
                f'graph sequence length {len(graphs)} does not match score window {window_size}'
            )
        expected_nodes = batch_size * node_num
        for step_idx, graph in enumerate(graphs):
            if int(graph.num_nodes()) != expected_nodes:
                raise ValueError(
                    f'expected a batched graph with {expected_nodes} nodes, '
                    f'got {int(graph.num_nodes())}'
                )
            source, target = graph.edges()
            source = source.to(device=device, dtype=torch.long)
            target = target.to(device=device, dtype=torch.long)
            source_batch = torch.div(source, node_num, rounding_mode='floor')
            target_batch = torch.div(target, node_num, rounding_mode='floor')
            same_sample = source_batch == target_batch
            source_batch = source_batch[same_sample]
            local_source = source[same_sample] % node_num
            local_target = target[same_sample] % node_num
            for lag_idx in range(1, effective_lags + 1):
                if step_idx < lag_idx:
                    continue
                lag_alignment_allowed[
                    source_batch,
                    lag_idx - 1,
                    step_idx,
                    local_target,
                    local_source,
                ] = True
                if self.topology_mode == 'undirected':
                    lag_alignment_allowed[
                        source_batch,
                        lag_idx - 1,
                        step_idx,
                        local_source,
                        local_target,
                    ] = True
        lag_allowed = torch.any(lag_alignment_allowed, dim=2)
        allowed = torch.any(lag_allowed, dim=1) if effective_lags > 0 else torch.zeros(
            batch_size, node_num, node_num, dtype=torch.bool, device=device
        )
        return allowed, lag_allowed, lag_alignment_allowed

    def _build_topology_strength(
        self,
        graphs: Optional[List],
        batch_size: int,
        node_num: int,
        device: torch.device,
        window_size: int,
    ) -> torch.Tensor:
        """Aggregate observed scalar edge intensity in relation orientation."""

        strength = torch.zeros(
            batch_size,
            node_num,
            node_num,
            dtype=torch.float32,
            device=device,
        )
        if self.topology_mode == 'none' or graphs is None:
            return strength
        if len(graphs) != window_size:
            raise ValueError(
                f'graph sequence length {len(graphs)} does not match score window {window_size}'
            )

        expected_nodes = batch_size * node_num
        for graph in graphs:
            if int(graph.num_nodes()) != expected_nodes:
                raise ValueError(
                    f'expected a batched graph with {expected_nodes} nodes, '
                    f'got {int(graph.num_nodes())}'
                )
            source, target = graph.edges()
            source = source.to(device=device, dtype=torch.long)
            target = target.to(device=device, dtype=torch.long)
            source_batch = torch.div(source, node_num, rounding_mode='floor')
            target_batch = torch.div(target, node_num, rounding_mode='floor')
            same_sample = source_batch == target_batch
            if 'weight' in graph.edata:
                edge_weight = (
                    graph.edata['weight'].to(device=device, dtype=torch.float32)
                    .clamp_min(0.0)
                )
            else:
                edge_weight = torch.ones(source.shape[0], device=device)
            source_batch = source_batch[same_sample]
            local_source = source[same_sample] % node_num
            local_target = target[same_sample] % node_num
            edge_weight = edge_weight[same_sample]
            strength.index_put_(
                (source_batch, local_target, local_source),
                edge_weight,
                accumulate=True,
            )
            if self.topology_mode == 'undirected':
                strength.index_put_(
                    (source_batch, local_source, local_target),
                    edge_weight,
                    accumulate=True,
                )

        strength = strength / float(max(window_size, 1))
        row_scale = strength.amax(dim=-1, keepdim=True)
        normalized = strength / row_scale.clamp_min(self.eps)
        return torch.where(row_scale > 0, normalized, strength)

    def forward(
        self,
        anomaly_score: torch.Tensor,
        graphs: Optional[List] = None,
    ) -> Dict[str, torch.Tensor]:
        # anomaly_score: [B, W, N]
        batch_size, window_size, node_num = anomaly_score.shape
        seasonal, trend = self.decomposition(anomaly_score)

        source_feature = torch.cat([seasonal, trend], dim=1).permute(0, 2, 1)  # [B, N, 2W]
        source_repr = self.source_encoder(source_feature)  # [B, N, H]
        target_repr = self.target_encoder(source_feature)  # [B, N, H]

        learned_attention = (
            torch.einsum('bih,bjh->bij', target_repr, source_repr)
            / math.sqrt(float(self.hidden_dim))
        )

        normalized_score = self._normalize(anomaly_score.permute(0, 2, 1))  # [B, N, W]
        effective_lags = min(self.max_lag, max(window_size - 1, 0))
        topology_mask, lag_topology_mask, lag_alignment_topology_mask = self._build_topology_support(
            graphs,
            batch_size=batch_size,
            node_num=node_num,
            device=learned_attention.device,
            effective_lags=effective_lags,
            window_size=window_size,
        )
        topology_strength = self._build_topology_strength(
            graphs,
            batch_size=batch_size,
            node_num=node_num,
            device=learned_attention.device,
            window_size=window_size,
        )
        eye = torch.eye(node_num, device=learned_attention.device, dtype=torch.bool).unsqueeze(0)
        topology_mask = topology_mask & ~eye
        lag_topology_mask = lag_topology_mask & ~eye.unsqueeze(1)
        lag_alignment_topology_mask = (
            lag_alignment_topology_mask & ~eye.unsqueeze(1).unsqueeze(1)
        )
        lag_affinity, lag_tensor, lag_weights = self._lag_affinity(
            normalized_score,
            lag_topology_mask=lag_topology_mask,
            lag_alignment_topology_mask=lag_alignment_topology_mask,
        )

        relation_logits = (
            learned_attention
            + self.gamma * lag_affinity
            + self.topology_weight_gamma * torch.log1p(topology_strength)
        )
        relation_logits = relation_logits.masked_fill(~topology_mask, float('-inf'))

        if self.use_null_source:
            null_logits = self.null_source_head(target_repr).squeeze(-1).unsqueeze(-1)
            full_logits = torch.cat([relation_logits, null_logits], dim=-1)
            full_relation = torch.softmax(full_logits, dim=-1)
            relation_matrix = full_relation[..., :node_num]
            null_source_probability = full_relation[..., node_num]
        else:
            relation_matrix = torch.softmax(relation_logits, dim=-1)
            null_source_probability = torch.zeros(
                batch_size,
                node_num,
                device=relation_logits.device,
                dtype=relation_logits.dtype,
            )
        relation_matrix = torch.nan_to_num(
            relation_matrix,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if self.min_strength > 0:
            relation_matrix = torch.where(
                relation_matrix >= self.min_strength,
                relation_matrix,
                torch.zeros_like(relation_matrix),
            )

        return {
            'directed_relation_matrix': relation_matrix,
            # Backward-compatible alias used by exporters and older scripts.
            'causal_matrix': relation_matrix,
            'null_source_probability': null_source_probability,
            'topology_mask': topology_mask,
            'topology_strength': topology_strength,
            'lag_topology_mask': lag_topology_mask,
            'lag_alignment_topology_mask': lag_alignment_topology_mask,
            'lag_strengths': lag_tensor,
            'lag_weights': lag_weights,
            'learned_attention': learned_attention,
            'seasonal_component': seasonal,
            'trend_component': trend,
        }


def summarize_causal_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        'causal_matrix_shape': list(output['causal_matrix'].shape),
        'null_source_probability_shape': list(output['null_source_probability'].shape),
        'topology_mask_shape': list(output['topology_mask'].shape),
        'topology_strength_shape': list(output['topology_strength'].shape),
        'lag_topology_mask_shape': list(output['lag_topology_mask'].shape),
        'lag_alignment_topology_mask_shape': list(
            output['lag_alignment_topology_mask'].shape
        ),
        'lag_strengths_shape': list(output['lag_strengths'].shape),
        'lag_weights_shape': list(output['lag_weights'].shape),
        'learned_attention_shape': list(output['learned_attention'].shape),
        'seasonal_component_shape': list(output['seasonal_component'].shape),
        'trend_component_shape': list(output['trend_component'].shape),
    }


LagAwareRelationHead = RunInspiredCausalHead
