from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_statistics(features: torch.Tensor) -> torch.Tensor:
    # [B,W,N,F] -> [B,N,6F]
    x = features.permute(0, 2, 1, 3)
    return torch.cat(
        [x.mean(2), x.std(2, unbiased=False), x.amax(2), x.amin(2),
         x.amax(2) - x.amin(2), x[:, :, -1] - x[:, :, 0]], dim=-1
    )


def pagerank(personalization: torch.Tensor, adjacency: torch.Tensor, alpha: float = 0.85, steps: int = 20):
    p = torch.softmax(personalization, dim=-1)
    score = p
    # adjacency rows aggregate predecessors, matching callee-to-caller propagation evidence.
    for _ in range(steps):
        score = (1.0 - alpha) * p + alpha * torch.bmm(adjacency, score.unsqueeze(-1)).squeeze(-1)
        score = score / score.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return score


class TemporalBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, n, f = x.shape
        sequence = x.permute(0, 2, 1, 3).reshape(b * n, w, f)
        _, hidden = self.gru(sequence)
        return hidden[-1].reshape(b, n, -1)


class ConvTemporalBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernels=(3, 5)):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2, groups=hidden_dim)
            for kernel in kernels
        ])
        self.scale_logits = nn.Parameter(torch.zeros(len(kernels)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, w, n, _ = x.shape
        value = self.input(x).permute(0, 2, 3, 1).reshape(b * n, -1, w)
        weights = torch.softmax(self.scale_logits, dim=0)
        value = sum(weight * conv(value) for weight, conv in zip(weights, self.convs))
        return F.gelu(value).mean(-1).reshape(b, n, -1)


class GraphLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_linear = nn.Linear(dim, dim)
        self.neighbor_linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighbor = torch.bmm(adjacency, h)
        return self.norm(F.gelu(self.self_linear(h) + self.neighbor_linear(neighbor)))


class BaseRCLModel(nn.Module):
    aux_loss: torch.Tensor | None = None

    def extra_loss(self) -> torch.Tensor:
        if self.aux_loss is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        return self.aux_loss


class DejaVuModel(BaseRCLModel):
    def __init__(self, feature_dim: int, hidden: int = 64):
        super().__init__()
        self.temporal = TemporalBranch(feature_dim, hidden)
        self.graph = GraphLayer(hidden)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, adjacency):
        self.aux_loss = None
        return self.score(self.graph(self.temporal(features), adjacency)).squeeze(-1)


class EadroModel(BaseRCLModel):
    def __init__(self, modality_indices: Dict[str, List[int]], hidden: int = 48):
        super().__init__()
        self.indices = modality_indices
        self.branches = nn.ModuleDict({name: TemporalBranch(len(index), hidden) for name, index in self.indices.items()})
        self.gate = nn.Linear(hidden * len(self.indices), len(self.indices))
        self.graph = GraphLayer(hidden)
        self.score = nn.Linear(hidden, 1)

    def modality_embeddings(self, features):
        return [self.branches[name](features[..., index]) for name, index in self.indices.items()]

    def forward(self, features, adjacency):
        values = self.modality_embeddings(features)
        gate = torch.softmax(self.gate(torch.cat(values, dim=-1)), dim=-1)
        fused = sum(gate[..., i:i+1] * value for i, value in enumerate(values))
        self.aux_loss = None
        return self.score(self.graph(fused, adjacency)).squeeze(-1)


class MicroIRCModel(BaseRCLModel):
    def __init__(self, metric_indices: List[int], hidden: int = 64):
        super().__init__()
        self.metric_indices = metric_indices
        self.temporal = TemporalBranch(len(metric_indices), hidden)
        self.graph1 = GraphLayer(hidden)
        self.graph2 = GraphLayer(hidden)
        self.weight = nn.Linear(hidden, 1)
        self.alpha_logit = nn.Parameter(torch.tensor(1.735))  # sigmoid ~= 0.85

    def forward(self, features, adjacency):
        h = self.temporal(features[..., self.metric_indices])
        h = self.graph2(self.graph1(h, adjacency), adjacency)
        weights = self.weight(h).squeeze(-1)
        alpha = torch.sigmoid(self.alpha_logit)
        self.aux_loss = None
        return torch.log(pagerank(weights, adjacency, float(alpha.detach()), steps=20).clamp_min(1e-8))


class MODIFyModel(BaseRCLModel):
    def __init__(self, modality_indices: Dict[str, List[int]], hidden: int = 64):
        super().__init__()
        self.indices = modality_indices
        self.branches = nn.ModuleDict({name: ConvTemporalBranch(len(index), hidden) for name, index in self.indices.items()})
        self.project = nn.Linear(hidden * len(self.indices), hidden * 2)
        self.channel = nn.Sequential(nn.Linear(hidden, max(hidden // 4, 1)), nn.ReLU(), nn.Linear(max(hidden // 4, 1), hidden), nn.Sigmoid())
        self.graph = GraphLayer(hidden)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, features, adjacency):
        values = []
        for name, index in self.indices.items():
            x = features[..., index]
            if name == 'trace' and self.training:
                x = 0.98 * x + 0.02 * torch.randn_like(x)
            values.append(self.branches[name](x))
        value, gate = self.project(torch.cat(values, dim=-1)).chunk(2, dim=-1)
        fused = F.gelu(value) * torch.sigmoid(gate)
        fused = fused * self.channel(fused.mean(dim=1)).unsqueeze(1)
        self.aux_loss = None
        return self.score(self.graph(fused, adjacency)).squeeze(-1)


class MicroInspectModel(BaseRCLModel):
    def __init__(self, modality_indices: Dict[str, List[int]], hidden: int = 48):
        super().__init__()
        self.indices = modality_indices
        self.shared = nn.ModuleDict({name: TemporalBranch(len(index), hidden) for name, index in self.indices.items()})
        self.unique = nn.ModuleDict({name: TemporalBranch(len(index), hidden) for name, index in self.indices.items()})
        expert_dim = hidden * 2
        self.experts = nn.ModuleList([nn.Linear(expert_dim, hidden) for _ in self.indices])
        self.router = nn.Linear(expert_dim * len(self.indices), len(self.indices))
        self.graph = GraphLayer(hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, features, adjacency):
        decomposed = []
        for name, index in self.indices.items():
            x = features[..., index]
            decomposed.append(torch.cat([self.shared[name](x), self.unique[name](x)], dim=-1))
        routing = torch.softmax(self.router(torch.cat(decomposed, dim=-1)), dim=-1)
        expert_values = [F.gelu(expert(value)) for expert, value in zip(self.experts, decomposed)]
        fused = sum(routing[..., i:i+1] * value for i, value in enumerate(expert_values))
        self.aux_loss = None
        return self.score(self.graph(fused, adjacency)).squeeze(-1)


class TVDiagModel(EadroModel):
    def __init__(self, modality_indices: Dict[str, List[int]], hidden: int = 48, temperature: float = 0.3):
        super().__init__(modality_indices, hidden)
        self.temperature = temperature
        self.modality_scorers = nn.ModuleList([nn.Linear(hidden, 1) for _ in self.indices])

    def forward(self, features, adjacency):
        values = self.modality_embeddings(features)
        gate = torch.softmax(self.gate(torch.cat(values, dim=-1)), dim=-1)
        fused = sum(gate[..., i:i+1] * value for i, value in enumerate(values))
        fused = self.graph(fused, adjacency)
        # View-invariant co-learning: align normalized node representations.
        contrast = 0.0
        pairs = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                contrast = contrast + (1.0 - (F.normalize(values[i], dim=-1) * F.normalize(values[j], dim=-1)).sum(-1)).mean()
                pairs += 1
        self.aux_loss = 0.1 * contrast / max(pairs, 1)
        task_scores = torch.stack([head(value).squeeze(-1) for head, value in zip(self.modality_scorers, values)]).mean(0)
        return self.score(fused).squeeze(-1) + 0.2 * task_scores


class UDARCLModel(BaseRCLModel):
    def __init__(self, feature_dim: int, hidden: int = 32):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(feature_dim * 6, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, features, adjacency):
        event_features = temporal_statistics(features)
        node_weight = self.weight(event_features).squeeze(-1)
        alpha = 0.5 + 0.45 * torch.sigmoid(self.alpha_logit)
        p = torch.softmax(node_weight, dim=-1)
        score = p
        for _ in range(30):
            score = (1.0 - alpha) * p + alpha * torch.bmm(adjacency, score.unsqueeze(-1)).squeeze(-1)
        self.aux_loss = None
        return torch.log(score.clamp_min(1e-8))


def build_model(method: str, feature_dim: int, modality_indices: Dict[str, List[int]], hidden: int = 64):
    if method == 'DejaVu':
        return DejaVuModel(feature_dim, hidden)
    if method == 'Eadro':
        return EadroModel(modality_indices, max(hidden // 1, 32))
    if method == 'MicroIRC':
        return MicroIRCModel(modality_indices['metric'], hidden)
    if method == 'MODIFy':
        return MODIFyModel(modality_indices, hidden)
    if method == 'MicroInspect':
        return MicroInspectModel(modality_indices, max(hidden // 1, 32))
    if method == 'TVDiag':
        return TVDiagModel(modality_indices, max(hidden // 1, 32))
    if method == 'UDA-RCL':
        return UDARCLModel(feature_dim, max(hidden // 2, 16))
    raise ValueError(f'unsupported trainable baseline: {method}')


def baro_scores(features: torch.Tensor, normal_mean: torch.Tensor, normal_std: torch.Tensor, metric_indices: List[int]):
    metric = features[..., metric_indices]
    mean = normal_mean[:, metric_indices]
    scale = normal_std[:, metric_indices].clamp_min(1e-4)
    deviation = ((metric - mean) / scale).abs().amax(dim=1).amax(dim=-1)
    return deviation


def dycause_scores(features: torch.Tensor, adjacency: torch.Tensor, normal_mean: torch.Tensor, normal_std: torch.Tensor, all_indices: List[int]):
    x = (features[..., all_indices] - normal_mean[:, all_indices]) / normal_std[:, all_indices].clamp_min(1e-4)
    anomaly = x.abs().mean(dim=-1)  # [B,W,N]
    if anomaly.shape[1] > 1:
        source = anomaly[:, :-1]
        target = anomaly[:, 1:]
        source = source - source.mean(dim=1, keepdim=True)
        target = target - target.mean(dim=1, keepdim=True)
        covariance = torch.einsum('bwi,bwj->bij', target, source) / max(source.shape[1], 1)
        covariance = covariance.relu() * adjacency
        covariance = covariance / covariance.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    else:
        covariance = adjacency
    personal = anomaly.amax(dim=1)
    return torch.log(pagerank(personal, covariance + 1e-6, alpha=0.85, steps=30).clamp_min(1e-8))
