from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

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
    def __init__(
        self,
        node_num: int,
        window_size: int,
        hidden_dim: int = 64,
        max_lag: int = 3,
        kernel_size: int = 5,
        gamma: float = 0.5,
        min_strength: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.node_num = node_num
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.gamma = gamma
        self.min_strength = min_strength
        self.eps = eps

        self.decomposition = SeriesDecomposition(kernel_size=kernel_size)
        self.source_encoder = nn.Sequential(
            nn.Linear(window_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.target_queries = nn.Parameter(torch.randn(node_num, hidden_dim))
        self.target_bias = nn.Parameter(torch.zeros(node_num, node_num))
        self.lag_weights = nn.Parameter(torch.ones(max_lag))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)

    def _lag_affinity(self, score: torch.Tensor):
        # score: [B, N, W]
        effective_lags = min(self.max_lag, max(score.shape[-1] - 1, 1))
        weights = torch.softmax(self.lag_weights[:effective_lags], dim=0)
        lag_strengths = []
        for lag_idx in range(1, effective_lags + 1):
            src = score[:, :, :-lag_idx]
            dst = score[:, :, lag_idx:]
            affinity = torch.einsum('bjk,bik->bij', src, dst) / max(src.shape[-1], 1)
            lag_strengths.append(affinity)
        lag_tensor = torch.stack(lag_strengths, dim=1)
        lag_affinity = torch.einsum('l,blij->bij', weights, lag_tensor)
        return lag_affinity, lag_tensor, weights

    def forward(self, anomaly_score: torch.Tensor) -> Dict[str, torch.Tensor]:
        # anomaly_score: [B, W, N]
        batch_size, window_size, node_num = anomaly_score.shape
        seasonal, trend = self.decomposition(anomaly_score)

        source_feature = torch.cat([seasonal, trend], dim=1).permute(0, 2, 1)  # [B, N, 2W]
        source_repr = self.source_encoder(source_feature)  # [B, N, H]
        target_query = self.target_queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, H]

        learned_attention = torch.einsum('bih,bjh->bij', target_query, source_repr)
        learned_attention = learned_attention + self.target_bias.unsqueeze(0)

        normalized_score = self._normalize(anomaly_score.permute(0, 2, 1))  # [B, N, W]
        lag_affinity, lag_tensor, lag_weights = self._lag_affinity(normalized_score)

        causal_logits = learned_attention + self.gamma * lag_affinity
        eye = torch.eye(node_num, device=causal_logits.device).unsqueeze(0)
        causal_logits = causal_logits.masked_fill(eye.bool(), float('-inf'))
        causal_matrix = torch.softmax(causal_logits, dim=-1)
        causal_matrix = torch.nan_to_num(causal_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        if self.min_strength > 0:
            causal_matrix = torch.where(
                causal_matrix >= self.min_strength,
                causal_matrix,
                torch.zeros_like(causal_matrix),
            )
            row_sum = causal_matrix.sum(dim=-1, keepdim=True)
            causal_matrix = causal_matrix / (row_sum + self.eps)

        return {
            'causal_matrix': causal_matrix,
            'lag_strengths': lag_tensor,
            'lag_weights': lag_weights,
            'learned_attention': learned_attention,
            'seasonal_component': seasonal,
            'trend_component': trend,
        }


def summarize_causal_output(output: Dict[str, torch.Tensor]) -> Dict[str, object]:
    return {
        'causal_matrix_shape': list(output['causal_matrix'].shape),
        'lag_strengths_shape': list(output['lag_strengths'].shape),
        'lag_weights_shape': list(output['lag_weights'].shape),
        'learned_attention_shape': list(output['learned_attention'].shape),
        'seasonal_component_shape': list(output['seasonal_component'].shape),
        'trend_component_shape': list(output['trend_component'].shape),
    }
