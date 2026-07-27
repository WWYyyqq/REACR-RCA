from typing import Dict, List, Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityDimensions(dict):
    """Dimension mapping carrying an optional explicit feature-index layout."""

    def __init__(self, dimensions: Mapping[str, int], indices=None):
        super().__init__((name, int(dim)) for name, dim in dimensions.items())
        self.indices = None if indices is None else {
            name: tuple(int(index) for index in values)
            for name, values in indices.items()
        }


def _normalize_modality_indices(modality_dims, modality_indices):
    names = [name for name in ('metric', 'trace', 'log') if int(modality_dims.get(name, 0)) > 0]
    if modality_indices is None:
        normalized = {}
        start = 0
        for name in names:
            end = start + int(modality_dims[name])
            normalized[name] = list(range(start, end))
            start = end
        return normalized
    missing = [name for name in names if name not in modality_indices]
    unexpected = [name for name in modality_indices if name not in names]
    if missing or unexpected:
        raise ValueError(
            f'Modality index keys mismatch: missing={missing}, unexpected={unexpected}.'
        )
    return {name: [int(index) for index in modality_indices[name]] for name in names}


def _indices_from_inclusive_spans(modality_spans):
    indices = {}
    for name, spans in modality_spans.items():
        values = []
        for span in spans:
            if len(span) != 2:
                raise ValueError(f'Expected inclusive [start, end] span for {name}, got {span!r}.')
            start, end = int(span[0]), int(span[1])
            if start < 0 or end < start:
                raise ValueError(f'Invalid inclusive modality span {span!r} for {name}.')
            values.extend(range(start, end + 1))
        indices[name] = values
    return indices


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
        modality_indices: Optional[Mapping[str, Sequence[int]]] = None,
    ):
        super().__init__()
        self.modality_names = [name for name in ('metric', 'trace', 'log') if modality_dims.get(name, 0) > 0]
        self.modality_dims = {name: int(modality_dims[name]) for name in self.modality_names}
        if modality_indices is None:
            modality_indices = getattr(modality_dims, 'indices', None)
        self.modality_indices = _normalize_modality_indices(self.modality_dims, modality_indices)
        self.input_dim = sum(self.modality_dims.values())
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
        layout = ModalityDimensions(self.modality_dims, self.modality_indices)
        if not valid_modality_dims(layout, self.input_dim):
            raise ValueError(
                f'Modality indices must partition [0, {self.input_dim}) without overlap '
                f'and match {self.modality_dims}; received {self.modality_indices}.'
            )
        if fusion_mode == 'raw_residual' and output_dim != self.input_dim:
            raise ValueError('raw_residual fusion requires output_dim to match the original feature dimension.')

        for name in self.modality_names:
            self.register_buffer(
                f'_indices_{name}',
                torch.tensor(self.modality_indices[name], dtype=torch.long),
                persistent=False,
            )
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
        if features.shape[-1] != self.input_dim:
            raise ValueError(f'Expected feature dimension {self.input_dim}, got {features.shape[-1]}.')
        return {
            name: torch.index_select(features, -1, getattr(self, f'_indices_{name}'))
            for name in self.modality_names
        }

    def _gaussian_reliability_bias(self, encoded_stack: torch.Tensor) -> torch.Tensor:
        if encoded_stack.shape[-2] == 1:
            return torch.zeros(encoded_stack.shape[:-1], device=encoded_stack.device, dtype=encoded_stack.dtype)
        center = encoded_stack.mean(dim=-2, keepdim=True)
        dist = torch.mean((encoded_stack - center) ** 2, dim=-1)
        sigma = F.softplus(self.sigma) + self.eps
        return torch.exp(-dist / (2.0 * sigma * sigma))

    def forward(self, features: torch.Tensor) -> Dict[str, object]:
        splits = self.split_modalities(features)
        encoded: List[torch.Tensor] = [self.encoders[name](splits[name]) for name in self.modality_names]
        encoded_stack = torch.stack(encoded, dim=-2)
        logits = self.gate_network(torch.cat(encoded, dim=-1))
        if self.use_gaussian_bias:
            logits = logits + self._gaussian_reliability_bias(encoded_stack)
        weights = torch.softmax(logits, dim=-1)
        if self.fusion_mode == 'raw_residual':
            fused = torch.zeros_like(features)
            modality_count = float(len(self.modality_names))
            for idx, name in enumerate(self.modality_names):
                reliability = 1.0 + self.residual_alpha * (modality_count * weights[..., idx:idx + 1] - 1.0)
                fused = torch.index_copy(
                    fused, -1, getattr(self, f'_indices_{name}'), splits[name] * reliability
                )
        else:
            fused = torch.sum(encoded_stack * weights.unsqueeze(-1), dim=-2)
        return {
            'fused_features': fused,
            'modality_weights': weights,
            'modality_embeddings': encoded_stack,
            'modality_names': self.modality_names,
            'modality_indices': self.modality_indices,
            'fusion_mode': self.fusion_mode,
        }


def summarize_modality_gate_output(output: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if output is None:
        return None
    return {
        'fused_features_shape': list(output['fused_features'].shape),
        'modality_weights_shape': list(output['modality_weights'].shape),
        'modality_names': list(output['modality_names']),
        'modality_indices': {name: list(values) for name, values in output['modality_indices'].items()},
        'fusion_mode': output['fusion_mode'],
    }


def parse_modality_dims(config: Dict) -> ModalityDimensions:
    feat_dim = config.get('feat_dim', {})
    dimensions = {
        'metric': int(feat_dim.get('metric', 0)),
        'trace': int(feat_dim.get('trace', 0)),
        'log': int(feat_dim.get('log', 0)),
    }
    spans = config.get('modality_spans')
    indices = config.get('modality_indices')
    if spans is not None and indices is not None:
        raise ValueError('Configure only one of modality_spans or modality_indices.')
    if spans is not None:
        indices = _indices_from_inclusive_spans(spans)
    normalized = None if indices is None else _normalize_modality_indices(dimensions, indices)
    parsed = ModalityDimensions(dimensions, normalized)
    if normalized is not None:
        for name, dim in dimensions.items():
            if dim > 0 and len(normalized[name]) != dim:
                raise ValueError(
                    f'Modality {name!r} declares {dim} dimensions but layout has '
                    f'{len(normalized[name])} indices.'
                )
    return parsed


def valid_modality_dims(modality_dims: Dict[str, int], total_dim: int) -> bool:
    dimensions = {name: int(modality_dims.get(name, 0)) for name in ('metric', 'trace', 'log')}
    if sum(dimensions.values()) != int(total_dim):
        return False
    try:
        indices = _normalize_modality_indices(dimensions, getattr(modality_dims, 'indices', None))
    except (TypeError, ValueError):
        return False
    flattened = [index for name in ('metric', 'trace', 'log') for index in indices.get(name, [])]
    if any(len(indices[name]) != dimensions[name] for name in indices):
        return False
    return sorted(flattened) == list(range(int(total_dim)))
