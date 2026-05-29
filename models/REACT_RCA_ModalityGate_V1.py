from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityReliabilityGate(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        output_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        use_gaussian_bias: bool = True,
        fusion_mode: str = 'encoded',
        residual_alpha: float = 0.2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.modality_names = [name for name in ('metric', 'trace', 'log') if modality_dims.get(name, 0) > 0]
        self.modality_dims = {name: int(modality_dims[name]) for name in self.modality_names}
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_gaussian_bias = use_gaussian_bias
        self.fusion_mode = fusion_mode
        self.residual_alpha = residual_alpha
        self.eps = eps

        if len(self.modality_names) == 0:
            raise ValueError('At least one modality dimension must be positive.')
        if fusion_mode not in {'encoded', 'raw_residual'}:
            raise ValueError(f'Unsupported modality fusion mode: {fusion_mode}')
        if fusion_mode == 'raw_residual' and output_dim != sum(self.modality_dims.values()):
            raise ValueError('raw_residual fusion requires output_dim to match the original feature dimension.')

        self.encoders = nn.ModuleDict()
        for name in self.modality_names:
            self.encoders[name] = nn.Sequential(
                nn.Linear(self.modality_dims[name], hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

        self.gate_network = nn.Sequential(
            nn.Linear(output_dim * len(self.modality_names), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(self.modality_names)),
        )
        self.sigma = nn.Parameter(torch.tensor(1.0))

    def split_modalities(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        splits = {}
        start = 0
        for name in self.modality_names:
            dim = self.modality_dims[name]
            splits[name] = features[..., start:start + dim]
            start += dim
        return splits

    def _gaussian_reliability_bias(self, encoded_stack: torch.Tensor) -> torch.Tensor:
        # encoded_stack: [B, W, N, M, D]
        if encoded_stack.shape[-2] == 1:
            return torch.zeros(encoded_stack.shape[:-1], device=encoded_stack.device, dtype=encoded_stack.dtype)

        center = encoded_stack.mean(dim=-2, keepdim=True)
        dist = torch.mean((encoded_stack - center) ** 2, dim=-1)
        sigma = F.softplus(self.sigma) + self.eps
        return torch.exp(-dist / (2.0 * sigma * sigma))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # features: [B, W, N, F]
        splits = self.split_modalities(features)
        encoded: List[torch.Tensor] = [self.encoders[name](splits[name]) for name in self.modality_names]
        encoded_stack = torch.stack(encoded, dim=-2)  # [B, W, N, M, D]

        gate_input = torch.cat(encoded, dim=-1)
        logits = self.gate_network(gate_input)
        if self.use_gaussian_bias:
            logits = logits + self._gaussian_reliability_bias(encoded_stack)

        weights = torch.softmax(logits, dim=-1)
        if self.fusion_mode == 'raw_residual':
            scaled_splits = []
            modality_count = float(len(self.modality_names))
            for idx, name in enumerate(self.modality_names):
                reliability = 1.0 + self.residual_alpha * (modality_count * weights[..., idx:idx + 1] - 1.0)
                scaled_splits.append(splits[name] * reliability)
            fused = torch.cat(scaled_splits, dim=-1)
        else:
            fused = torch.sum(encoded_stack * weights.unsqueeze(-1), dim=-2)

        return {
            'fused_features': fused,
            'modality_weights': weights,
            'modality_embeddings': encoded_stack,
            'modality_names': self.modality_names,
            'fusion_mode': self.fusion_mode,
        }


def summarize_modality_gate_output(output: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, object]]:
    if output is None:
        return None
    return {
        'fused_features_shape': list(output['fused_features'].shape),
        'modality_weights_shape': list(output['modality_weights'].shape),
        'modality_names': list(output['modality_names']),
        'fusion_mode': output['fusion_mode'],
    }


def parse_modality_dims(config: Dict) -> Dict[str, int]:
    feat_dim = config.get('feat_dim', {})
    return {
        'metric': int(feat_dim.get('metric', 0)),
        'trace': int(feat_dim.get('trace', 0)),
        'log': int(feat_dim.get('log', 0)),
    }


def valid_modality_dims(modality_dims: Dict[str, int], total_dim: int) -> bool:
    return sum(int(value) for value in modality_dims.values()) == int(total_dim)
