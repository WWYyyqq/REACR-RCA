from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.reacr_rca_topology_dynamics import extract_topology_dynamics


class PairwiseRankingLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        losses = []
        for score_vec, label_vec in zip(scores, labels):
            pos_scores = score_vec[label_vec > 0]
            neg_scores = score_vec[label_vec <= 0]
            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue
            # A service-level label may map to several replica nodes.  The
            # dataset confirms only that one member of this positive bag is a
            # valid localization target, so align training with any-hit A@K.
            positive_bag_score = torch.max(pos_scores)
            pairwise = self.margin - positive_bag_score + neg_scores
            losses.append(F.relu(pairwise).mean())
        if len(losses) == 0:
            return scores.mean() * 0.0
        return torch.stack(losses).mean()


class DyCauRCAModel(nn.Module):
    def __init__(
        self,
        backbone,
        anomaly_head,
        causal_head,
        ranker,
        modality_gate=None,
        graph_sequence_mode: str = 'observed',
        observation_prior=None,
        feature_value_clip: Optional[float] = None,
    ):
        super().__init__()
        if graph_sequence_mode not in {'observed', 'freeze_first'}:
            raise ValueError(
                f'unsupported graph_sequence_mode={graph_sequence_mode!r}; '
                "expected 'observed' or 'freeze_first'"
            )
        self.modality_gate = modality_gate
        self.backbone = backbone
        self.anomaly_head = anomaly_head
        self.causal_head = causal_head
        self.ranker = ranker
        self.graph_sequence_mode = graph_sequence_mode
        self.observation_prior = dict(observation_prior or {})
        self.feature_value_clip = (
            None if feature_value_clip is None else float(feature_value_clip)
        )
        if self.feature_value_clip is not None and self.feature_value_clip <= 0:
            raise ValueError('feature_value_clip must be positive when provided')

    def forward(self, graphs, features) -> Dict[str, Dict[str, torch.Tensor]]:
        model_graphs = prepare_graph_sequence(graphs, self.graph_sequence_mode)
        input_features = (
            features.clamp(-self.feature_value_clip, self.feature_value_clip)
            if self.feature_value_clip is not None
            else features
        )
        gate_output = (
            self.modality_gate(input_features)
            if self.modality_gate is not None
            else None
        )
        model_features = (
            gate_output['fused_features']
            if gate_output is not None
            else input_features
        )

        backbone_output = self.backbone(model_graphs, model_features)
        anomaly_output = self.anomaly_head(backbone_output['temporal_states'], model_features)
        causal_output = self.causal_head(
            anomaly_output['anomaly_score'],
            graphs=model_graphs,
        )
        rank_output = self.ranker(
            model_graphs,
            anomaly_output['anomaly_score'],
            causal_output['causal_matrix'],
            lag_weights=causal_output.get('lag_weights'),
        )
        rank_output = apply_observation_prior(
            rank_output,
            anomaly_output,
            input_features,
            self.observation_prior,
        )

        return {
            'modality_gate_output': gate_output,
            'backbone_output': backbone_output,
            'anomaly_output': anomaly_output,
            'causal_output': causal_output,
            'rank_output': rank_output,
        }


def prepare_graph_sequence(graphs, mode: str = 'observed'):
    """Select the observed or a deliberately frozen topology sequence.

    ``freeze_first`` is an experiment-only counterfactual control: every time
    step uses the first observed graph in that same window.  It disables edge
    evolution consistently for spatial message passing, explicit topology
    features, relation masking, and graph-prior ranking without fabricating any
    new edge.
    """

    if mode == 'observed':
        return graphs
    if mode == 'freeze_first':
        if len(graphs) == 0:
            raise ValueError('cannot freeze an empty graph sequence')
        return [graphs[0] for _ in graphs]
    raise ValueError(f'unsupported graph sequence mode: {mode}')


def _standardize_node_score(score: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Make heterogeneous score branches comparable within each case."""

    centered = score - score.mean(dim=1, keepdim=True)
    scale = centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return torch.nan_to_num(centered / scale, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-10.0, 10.0)


def apply_observation_prior(
    rank_output: Dict[str, torch.Tensor],
    anomaly_output: Dict[str, torch.Tensor],
    original_features: torch.Tensor,
    config: Optional[Dict] = None,
) -> Dict[str, torch.Tensor]:
    """Fuse stable raw-observation evidence with a graph-aware score branch.

    SN/TT features are normalized using normal sessions, so their positive peak
    and within-window range provide service-comparable controlled-fault evidence.
    Every component is standardized across nodes before fusion; this prevents an
    extreme CPU z-score from creating the unbounded TT ranking values seen with
    raw squared reconstruction errors.
    """

    config = dict(config or {})
    if not config.get('enabled', False):
        return rank_output

    feature_dim = int(original_features.shape[-1])

    def validate_indices(name, values):
        indices = [int(value) for value in values]
        if not indices:
            raise ValueError(f'observation_prior {name} cannot be empty')
        if min(indices) < 0 or max(indices) >= feature_dim:
            raise ValueError(
                f'observation_prior {name} contains an index outside [0, {feature_dim})'
            )
        return indices

    metric_indices = validate_indices(
        'metric_feature_indices',
        config.get('metric_feature_indices', range(feature_dim)),
    )
    all_indices = validate_indices(
        'all_feature_indices',
        config.get('all_feature_indices', range(feature_dim)),
    )
    metric_features = original_features[..., metric_indices]
    all_features = original_features[..., all_indices]
    metric_peak = metric_features.amax(dim=(1, 3))
    temporal_range = (
        all_features.amax(dim=1) - all_features.amin(dim=1)
    ).abs().mean(dim=-1)
    reconstruction_peak = anomaly_output['recon_error'].amax(dim=1)

    base_key = str(config.get('base_score_key', 'final_score'))
    if base_key not in rank_output:
        raise KeyError(f'observation_prior base_score_key={base_key!r} is unavailable')
    components = {
        'base': rank_output[base_key],
        'metric_peak': metric_peak,
        'temporal_range': temporal_range,
        'reconstruction_peak': reconstruction_peak,
    }
    weights = {
        str(key): float(value)
        for key, value in config.get('weights', {'base': 1.0}).items()
    }
    unknown = set(weights) - set(components)
    if unknown:
        raise ValueError(f'unknown observation_prior components: {sorted(unknown)}')
    if any(value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError('observation_prior weights must be non-negative with positive sum')

    normalized = {key: _standardize_node_score(value) for key, value in components.items()}
    total_weight = sum(weights.values())
    fused = sum(weights.get(key, 0.0) * value for key, value in normalized.items()) / total_weight
    output = dict(rank_output)
    output['model_score_before_observation_prior'] = rank_output['final_score']
    output['observation_metric_peak'] = metric_peak
    output['observation_temporal_range'] = temporal_range
    output['observation_reconstruction_peak'] = reconstruction_peak
    output['final_score'] = fused
    return output


def compute_topk_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5)) -> Dict[str, float]:
    metrics = {f'A@{k}': 0.0 for k in ks}
    batch_size = scores.shape[0]
    for sample_idx in range(batch_size):
        order = torch.argsort(scores[sample_idx], descending=True)
        positives = labels[sample_idx] > 0
        for k in ks:
            topk = order[:k]
            if torch.any(positives[topk]):
                metrics[f'A@{k}'] += 1.0
    for k in ks:
        metrics[f'A@{k}'] /= max(batch_size, 1)
    metrics['_sample_count'] = float(batch_size)
    return metrics


def aggregate_metric_dict(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if len(metric_dicts) == 0:
        return {}
    keys = [key for key in metric_dicts[0].keys() if key != '_sample_count']
    weights = np.array([metric.get('_sample_count', 1.0) for metric in metric_dicts], dtype=float)
    if float(weights.sum()) <= 0:
        weights = np.ones(len(metric_dicts), dtype=float)
    return {
        key: float(np.average([metric[key] for metric in metric_dicts], weights=weights))
        for key in keys
    }


def _zero_like_loss(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.mean() * 0.0


def candidate_bag_relation_signature(
    relation_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Build a relation signature without averaging unverified replicas.

    ``labels`` denotes a positive candidate bag.  Max aggregation asks whether
    any candidate instance supports each relation dimension, matching the
    any-hit localization target used by A@K.
    """

    if relation_matrix.ndim != 3 or labels.ndim != 2:
        raise ValueError('relation_matrix must be [B,N,N] and labels must be [B,N]')
    if relation_matrix.shape[:2] != labels.shape or relation_matrix.shape[2] != labels.shape[1]:
        raise ValueError(
            f'incompatible relation/label shapes: {tuple(relation_matrix.shape)} '
            f'and {tuple(labels.shape)}'
        )
    candidates = labels > 0
    source_mask = candidates.unsqueeze(1)
    target_mask = candidates.unsqueeze(2)
    from_candidate = relation_matrix.masked_fill(~source_mask, float('-inf')).max(dim=2).values
    into_candidate = relation_matrix.masked_fill(~target_mask, float('-inf')).max(dim=1).values
    valid = candidates.any(dim=1, keepdim=True)
    from_candidate = torch.where(valid, from_candidate, torch.zeros_like(from_candidate))
    into_candidate = torch.where(valid, into_candidate, torch.zeros_like(into_candidate))
    from_candidate = torch.nan_to_num(from_candidate, nan=0.0, neginf=0.0, posinf=0.0)
    into_candidate = torch.nan_to_num(into_candidate, nan=0.0, neginf=0.0, posinf=0.0)
    return torch.cat([from_candidate, into_candidate], dim=-1)


def _sample_response_candidate_indices(
    rank_output: Dict[str, torch.Tensor],
    anomaly_score: torch.Tensor,
    labels: torch.Tensor,
    sample_idx: int,
    topk: int,
) -> torch.Tensor:
    positives = labels[sample_idx] > 0
    negatives = ~positives
    if not torch.any(negatives):
        return torch.empty(0, dtype=torch.long, device=labels.device)

    anomaly_peak = anomaly_score[sample_idx].max(dim=0).values
    incoming = rank_output['incoming_explainedness'][sample_idx]
    victim = rank_output['victim_score'][sample_idx]
    final_score = rank_output['final_score'][sample_idx]

    def zscore(values: torch.Tensor) -> torch.Tensor:
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)

    # A mined response candidate is a plausible false localization candidate,
    # not a dataset-provided symptom label. Requiring high final rank avoids
    # regularising nodes that cannot displace the labelled candidate bag.
    hard_score = zscore(anomaly_peak) + zscore(incoming) + zscore(victim) + zscore(final_score)
    negative_incoming = incoming[negatives]
    negative_victim = victim[negatives]
    negative_final = final_score[negatives]
    reliable = (
        negatives
        & (incoming >= negative_incoming.median())
        & (victim >= negative_victim.median())
        & (final_score >= negative_final.median())
    )
    candidate_mask = reliable if torch.any(reliable) else negatives
    hard_score = hard_score.masked_fill(~candidate_mask, float('-inf'))
    candidate_count = int(candidate_mask.sum().item())
    if candidate_count <= 0:
        return torch.empty(0, dtype=torch.long, device=labels.device)
    k = max(1, min(int(topk), candidate_count))
    return torch.topk(hard_score, k=k).indices


# Deprecated internal alias retained for historical imports.
_sample_hard_symptom_indices = _sample_response_candidate_indices


def root_symptom_loss(
    rank_output: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    margin: float = 0.2,
    hard_negative_topk: int = 3,
) -> torch.Tensor:
    """Weakly separate labelled sources from mined high-response candidates.

    The negative candidates are latent response-role pseudo labels, not
    dataset-provided symptom truth.  The loss uses relative role comparisons;
    it does not force source and response probabilities to be mutually exclusive
    for the same node.
    """
    source_role = rank_output.get('source_role_score', rank_output['root_role_score'])
    response_role = rank_output.get('response_role_score', rank_output['symptom_role_score'])
    role_margin_score = source_role - response_role
    victim_score = rank_output['victim_score']
    incoming_explainedness = rank_output['incoming_explainedness']
    losses = []
    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        negatives = ~positives
        if not torch.any(positives):
            continue

        positive_indices = torch.where(positives)[0]
        positive_source_values = source_role[sample_idx, positive_indices]
        selected_positive = positive_indices[torch.argmax(positive_source_values)]
        pos_source = source_role[sample_idx, selected_positive]
        pos_response = response_role[sample_idx, selected_positive]
        pos_role_margin = role_margin_score[sample_idx, selected_positive]
        sample_losses = [
            F.binary_cross_entropy(pos_source, torch.ones_like(pos_source)),
        ]

        if torch.any(negatives):
            hard_score = (
                victim_score[sample_idx].detach()
                + incoming_explainedness[sample_idx].detach()
            ).masked_fill(~negatives, float('-inf'))
            candidate_count = int(negatives.sum().item())
            topk = max(1, min(int(hard_negative_topk), candidate_count))
            hard_indices = torch.topk(hard_score, k=topk).indices
            neg_source = source_role[sample_idx, hard_indices]
            neg_response = response_role[sample_idx, hard_indices]
            sample_losses.extend([
                F.relu(
                    margin
                    - pos_source
                    + neg_source
                ).mean(),
                F.relu(
                    margin
                    + pos_response
                    - neg_response
                ).mean(),
                F.relu(
                    margin
                    - pos_role_margin
                    + role_margin_score[sample_idx, hard_indices]
                ).mean(),
            ])
        losses.append(torch.stack(sample_losses).mean())
    if len(losses) == 0:
        return _zero_like_loss(source_role)
    return torch.stack(losses).mean()


def counterfactual_loss(rank_output: Dict[str, torch.Tensor], labels: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    coverage = rank_output.get(
        'source_hypothesis_coverage',
        rank_output['counterfactual_coverage'],
    )
    residual = rank_output.get(
        'source_hypothesis_residual',
        rank_output['counterfactual_residual'],
    )
    losses = []
    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        negatives = labels[sample_idx] <= 0
        if not torch.any(positives) or not torch.any(negatives):
            continue
        positive_indices = torch.where(positives)[0]
        positive_support = (
            coverage[sample_idx, positive_indices]
            - residual[sample_idx, positive_indices]
        )
        selected_positive = positive_indices[torch.argmax(positive_support)]
        pos_cov = coverage[sample_idx, selected_positive]
        neg_cov = coverage[sample_idx, negatives]
        pos_res = residual[sample_idx, selected_positive]
        neg_res = residual[sample_idx, negatives]
        cov_pairwise = margin - pos_cov + neg_cov
        res_pairwise = margin + pos_res - neg_res
        losses.append(F.relu(cov_pairwise).mean() + F.relu(res_pairwise).mean())
    if len(losses) == 0:
        return _zero_like_loss(coverage)
    return torch.stack(losses).mean()


def event_chain_consistency_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    evidence_topk: int = 5,
    margin: float = 0.05,
) -> torch.Tensor:
    causal_matrix = outputs['causal_output']['causal_matrix']
    anomaly_score = outputs['anomaly_output']['anomaly_score']
    lag_strengths = outputs['causal_output'].get('lag_strengths')
    lag_weights = outputs['causal_output'].get('lag_weights')
    anomaly_peak = anomaly_score.max(dim=1).values
    node_num = anomaly_peak.shape[1]
    topk = max(1, min(evidence_topk, node_num))
    losses = []
    if lag_strengths is not None and lag_weights is not None:
        if lag_weights.ndim == 1:
            weighted_lag = torch.einsum(
                'l,blij->bij',
                lag_weights[: lag_strengths.shape[1]],
                lag_strengths,
            )
        elif lag_weights.ndim == 4:
            effective_lags = min(lag_weights.shape[1], lag_strengths.shape[1])
            weighted_lag = torch.sum(
                lag_weights[:, :effective_lags] * lag_strengths[:, :effective_lags],
                dim=1,
            )
        else:
            raise ValueError(f'unsupported lag_weights shape: {tuple(lag_weights.shape)}')
    else:
        weighted_lag = None

    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        negatives = labels[sample_idx] <= 0
        if not torch.any(positives) or not torch.any(negatives):
            continue
        evidence_idx = torch.topk(anomaly_peak[sample_idx], k=topk).indices
        reach = causal_matrix[sample_idx][evidence_idx, :].mean(dim=0)
        positive_indices = torch.where(positives)[0]
        selected_positive = positive_indices[torch.argmax(reach[positive_indices])]
        pos_reach = reach[selected_positive]
        neg_reach = reach[negatives]
        pairwise = margin - pos_reach + neg_reach
        sample_losses = [F.relu(pairwise).mean()]
        if weighted_lag is not None:
            lag_reach = weighted_lag[sample_idx][evidence_idx, :].mean(dim=0)
            sample_losses.append(F.relu(-lag_reach[selected_positive]))
        losses.append(torch.stack(sample_losses).mean())

    if len(losses) == 0:
        return _zero_like_loss(causal_matrix)
    return torch.stack(losses).mean()


def propagation_invariance_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    group_ids: Optional[torch.Tensor],
    temperature: float = 0.2,
) -> torch.Tensor:
    causal_matrix = outputs['causal_output']['causal_matrix']
    if group_ids is None or labels is None or labels.shape[0] < 2:
        return _zero_like_loss(causal_matrix)

    signature = candidate_bag_relation_signature(causal_matrix, labels)
    signature = F.normalize(signature, dim=-1)
    sim = torch.matmul(signature, signature.t()) / temperature
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    same_group = group_ids.unsqueeze(0) == group_ids.unsqueeze(1)
    valid_group = group_ids >= 0
    same_group = same_group & valid_group.unsqueeze(0) & valid_group.unsqueeze(1)
    positive_mask = same_group & ~eye
    if not torch.any(positive_mask):
        return _zero_like_loss(causal_matrix)
    sim = sim.masked_fill(eye, float('-inf'))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    safe_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_anchor = -safe_log_prob.sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    valid = positive_mask.any(dim=1)
    return loss_per_anchor[valid].mean()


def _move_graphs_to_device(graphs, device: torch.device):
    return [graph.to(device) for graph in graphs]


def _safe_device(device: torch.device) -> torch.device:
    if device.type == 'cuda' and not torch.cuda.is_available():
        return torch.device('cpu')
    return device


def observed_topology_change_mask(graphs) -> torch.Tensor:
    """Return one boolean per sample using the original observed graph window."""

    raw = extract_topology_dynamics(graphs, device='cpu')
    dynamic = raw[..., 2:] != 0
    return dynamic.reshape(dynamic.shape[0], -1).any(dim=1)


def topology_parameter_diagnostics(model: DyCauRCAModel) -> Dict[str, Optional[float]]:
    backbone = getattr(model, 'backbone', None)
    encoder = getattr(backbone, 'topology_encoder', None)
    scale = getattr(backbone, 'topology_scale', None)
    dynamic_projection = getattr(encoder, 'dynamic_projection', None)
    gradient = None if dynamic_projection is None else dynamic_projection.weight.grad
    edge_gradient_squares = []
    for layer in getattr(backbone, 'spatial_layers', []):
        for module_name in ('edge_mass_linear', 'edge_context_linear'):
            module = getattr(layer, module_name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.grad is not None:
                    edge_gradient_squares.append(
                        torch.sum(parameter.grad.detach().float() ** 2)
                    )
    edge_gradient_norm = None if not edge_gradient_squares else torch.sqrt(
        torch.stack(edge_gradient_squares).sum()
    )
    return {
        'fusion_scale': (
            None
            if scale is None
            else float(torch.sigmoid(scale.detach()).cpu().item())
        ),
        'dynamic_projection_gradient_norm': (
            None
            if gradient is None
            else float(torch.linalg.vector_norm(gradient.detach()).cpu().item())
        ),
        'edge_evidence_gradient_norm': (
            None
            if edge_gradient_norm is None
            else float(edge_gradient_norm.cpu().item())
        ),
    }


def predict_scores(model: DyCauRCAModel, dataloader, device: torch.device):
    model.eval()
    all_scores = []
    all_labels = []
    all_case_ids = []
    all_case_timestamps = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)
            final_score = outputs['rank_output']['final_score'].detach().cpu()
            all_scores.append(final_score)
            if batch['labels'] is not None:
                all_labels.append(batch['labels'].float().cpu())
            all_case_ids.append(batch['case_ids'].cpu())
            all_case_timestamps.append(batch['case_timestamps'].cpu())

    scores = torch.cat(all_scores, dim=0) if len(all_scores) > 0 else torch.empty(0)
    labels = torch.cat(all_labels, dim=0) if len(all_labels) > 0 else None
    case_ids = torch.cat(all_case_ids, dim=0) if len(all_case_ids) > 0 else torch.empty(0, dtype=torch.long)
    case_timestamps = torch.cat(all_case_timestamps, dim=0) if len(all_case_timestamps) > 0 else torch.empty(0, dtype=torch.long)
    return scores, labels, case_ids, case_timestamps


def build_prediction_dataframe(
    scores: torch.Tensor,
    labels: Optional[torch.Tensor],
    case_ids: torch.Tensor,
    case_timestamps: torch.Tensor,
    index_to_node: Dict[int, str],
) -> pd.DataFrame:
    rows = []
    for sample_idx in range(scores.shape[0]):
        order = torch.argsort(scores[sample_idx], descending=True)
        row = {
            'case_id': int(case_ids[sample_idx].item()) if len(case_ids) > 0 else sample_idx,
            'case_timestamp': int(case_timestamps[sample_idx].item()) if len(case_timestamps) > 0 else -1,
        }
        if labels is not None:
            positive_nodes = [index_to_node[idx] for idx in torch.where(labels[sample_idx] > 0)[0].tolist()]
            row['label_nodes'] = ','.join(positive_nodes)
        for rank_idx, node_idx in enumerate(order.tolist(), start=1):
            row[f'Top{rank_idx}'] = f"{index_to_node[node_idx]}:{scores[sample_idx, node_idx].item():.6f}"
        rows.append(row)
    return pd.DataFrame(rows)


def _binary_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels)]
    pos = [score for score, label in pairs if label > 0]
    neg = [score for score, label in pairs if label <= 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = 0.0
    total = 0.0
    for pos_score in pos:
        for neg_score in neg:
            if pos_score > neg_score:
                wins += 1.0
            elif pos_score == neg_score:
                wins += 0.5
            total += 1.0
    return wins / total if total > 0 else None


def build_challenge2_diagnostic_dataframe(
    model: DyCauRCAModel,
    dataloader,
    device: torch.device,
    index_to_node: Dict[int, str],
    hard_symptom_topk: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    device = _safe_device(device)
    model.eval()
    rows = []
    auc_scores = []
    auc_labels = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)
            rank_output = outputs['rank_output']
            anomaly_score = outputs['anomaly_output']['anomaly_score']
            final_score = rank_output['final_score']
            batch_size = labels.shape[0]

            for sample_idx in range(batch_size):
                positives = torch.where(labels[sample_idx] > 0)[0]
                if positives.numel() == 0:
                    continue
                hard_idx = _sample_response_candidate_indices(
                    rank_output,
                    anomaly_score,
                    labels,
                    sample_idx,
                    topk=hard_symptom_topk,
                )
                if hard_idx.numel() == 0:
                    continue

                root_scores = final_score[sample_idx, positives]
                root_best_pos = int(torch.argmax(root_scores).item())
                root_idx = int(positives[root_best_pos].item())
                hard_scores = final_score[sample_idx, hard_idx]
                hard_best_pos = int(torch.argmax(hard_scores).item())
                hard_node_idx = int(hard_idx[hard_best_pos].item())

                root_rank = int((final_score[sample_idx] > final_score[sample_idx, root_idx]).sum().item() + 1)
                hard_rank = int((final_score[sample_idx] > final_score[sample_idx, hard_node_idx]).sum().item() + 1)
                source_role_gap = float(
                    (
                        rank_output['source_role_score'][sample_idx, root_idx]
                        - rank_output['source_role_score'][sample_idx, hard_node_idx]
                    ).detach().cpu().item()
                )
                response_role_gap = float(
                    (
                        rank_output['response_role_score'][sample_idx, hard_node_idx]
                        - rank_output['response_role_score'][sample_idx, root_idx]
                    ).detach().cpu().item()
                )
                row = {
                    'case_id': int(batch['case_ids'][sample_idx].item()),
                    'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
                    'label_candidate_node': index_to_node.get(root_idx, str(root_idx)),
                    'label_candidate_index': root_idx,
                    'response_candidate_node': index_to_node.get(hard_node_idx, str(hard_node_idx)),
                    'response_candidate_index': hard_node_idx,
                    'label_candidate_rank': root_rank,
                    'response_candidate_rank': hard_rank,
                    'response_candidate_before_label_candidate': int(hard_rank < root_rank),
                    'label_candidate_final_score': float(final_score[sample_idx, root_idx].detach().cpu().item()),
                    'response_candidate_final_score': float(final_score[sample_idx, hard_node_idx].detach().cpu().item()),
                    'label_vs_response_candidate_margin': float((final_score[sample_idx, root_idx] - final_score[sample_idx, hard_node_idx]).detach().cpu().item()),
                    'label_candidate_anomaly_peak': float(anomaly_score[sample_idx, :, root_idx].max().detach().cpu().item()),
                    'response_candidate_anomaly_peak': float(anomaly_score[sample_idx, :, hard_node_idx].max().detach().cpu().item()),
                    # Deprecated aliases retained for historical report readers.
                    'root_node': index_to_node.get(root_idx, str(root_idx)),
                    'root_index': root_idx,
                    'hard_symptom_node': index_to_node.get(hard_node_idx, str(hard_node_idx)),
                    'hard_symptom_index': hard_node_idx,
                    'root_rank': root_rank,
                    'hard_symptom_rank': hard_rank,
                    'hard_symptom_before_root': int(hard_rank < root_rank),
                    'root_final_score': float(final_score[sample_idx, root_idx].detach().cpu().item()),
                    'hard_symptom_final_score': float(final_score[sample_idx, hard_node_idx].detach().cpu().item()),
                    'root_vs_hard_symptom_margin': float((final_score[sample_idx, root_idx] - final_score[sample_idx, hard_node_idx]).detach().cpu().item()),
                    'source_role_gap': source_role_gap,
                    'response_role_gap': response_role_gap,
                    # Deprecated alias retained for historical report readers.
                    'root_role_gap': source_role_gap,
                    'source_gap': float((rank_output['source_score'][sample_idx, root_idx] - rank_output['source_score'][sample_idx, hard_node_idx]).detach().cpu().item()),
                    'victim_gap': float((rank_output['victim_score'][sample_idx, hard_node_idx] - rank_output['victim_score'][sample_idx, root_idx]).detach().cpu().item()),
                    'incoming_explainedness_gap': float((rank_output['incoming_explainedness'][sample_idx, hard_node_idx] - rank_output['incoming_explainedness'][sample_idx, root_idx]).detach().cpu().item()),
                    'root_anomaly_peak': float(anomaly_score[sample_idx, :, root_idx].max().detach().cpu().item()),
                    'hard_symptom_anomaly_peak': float(anomaly_score[sample_idx, :, hard_node_idx].max().detach().cpu().item()),
                }
                rows.append(row)

                auc_scores.extend(rank_output['source_role_score'][sample_idx].detach().cpu().tolist())
                auc_labels.extend((labels[sample_idx] > 0).detach().cpu().int().tolist())

    df = pd.DataFrame(rows)
    candidate_membership_auc = _binary_auc(auc_scores, auc_labels)
    source_role_gap_mean = float(df['source_role_gap'].mean()) if len(df) > 0 else None
    response_candidate_error_rate = (
        float(df['response_candidate_before_label_candidate'].mean())
        if len(df) > 0
        else None
    )
    candidate_margin_mean = (
        float(df['label_vs_response_candidate_margin'].mean())
        if len(df) > 0
        else None
    )
    summary = {
        'cases': int(len(df)),
        'response_candidate_error_rate': response_candidate_error_rate,
        'source_role_gap_mean': source_role_gap_mean,
        'response_role_gap_mean': float(df['response_role_gap'].mean()) if len(df) > 0 else None,
        'source_gap_mean': float(df['source_gap'].mean()) if len(df) > 0 else None,
        'victim_gap_mean': float(df['victim_gap'].mean()) if len(df) > 0 else None,
        'incoming_explainedness_gap_mean': float(df['incoming_explainedness_gap'].mean()) if len(df) > 0 else None,
        'label_vs_response_candidate_margin_mean': candidate_margin_mean,
        'candidate_membership_auc': candidate_membership_auc,
        # Deprecated aliases retained for historical report readers.
        'hard_symptom_error_rate': response_candidate_error_rate,
        'root_vs_hard_symptom_margin_mean': candidate_margin_mean,
        'root_role_gap_mean': source_role_gap_mean,
        'role_auc': candidate_membership_auc,
    }
    return df, summary


def save_challenge2_diagnostics(
    model: DyCauRCAModel,
    dataloader,
    device: torch.device,
    index_to_node: Dict[int, str],
    csv_path: str,
    metrics_path: str,
    markdown_path: str,
    hard_symptom_topk: int = 5,
) -> Dict[str, float]:
    df, summary = build_challenge2_diagnostic_dataframe(
        model,
        dataloader,
        device=device,
        index_to_node=index_to_node,
        hard_symptom_topk=hard_symptom_topk,
    )
    df.to_csv(csv_path, index=False)
    import json
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    lines = [
        '# Source-Response Candidate Diagnostic',
        '',
        f"- Cases: {summary.get('cases')}",
        f"- response_candidate_error_rate: {summary.get('response_candidate_error_rate')}",
        f"- candidate_membership_auc: {summary.get('candidate_membership_auc')}",
        f"- source_role_gap_mean: {summary.get('source_role_gap_mean')}",
        f"- response_role_gap_mean: {summary.get('response_role_gap_mean')}",
        f"- source_gap_mean: {summary.get('source_gap_mean')}",
        f"- victim_gap_mean: {summary.get('victim_gap_mean')}",
        f"- incoming_explainedness_gap_mean: {summary.get('incoming_explainedness_gap_mean')}",
        f"- label_vs_response_candidate_margin_mean: {summary.get('label_vs_response_candidate_margin_mean')}",
    ]
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return summary


def evaluate_dycau(model: DyCauRCAModel, dataloader, device: torch.device, ks=(1, 3, 5)):
    device = _safe_device(device)
    model.eval()
    losses = []
    metrics = []
    score_stds = []
    topology_metrics = {'changed': [], 'stable': []}
    topology_counts = {'changed': 0, 'stable': 0}
    criterion = PairwiseRankingLoss()

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)
            final_score = outputs['rank_output']['final_score']
            loss = criterion(final_score, labels)
            losses.append(loss.item())
            score_stds.append(final_score.detach().float().std(dim=1).mean().item())
            metrics.append(compute_topk_metrics(final_score.cpu(), labels.cpu(), ks=ks))
            changed_mask = observed_topology_change_mask(batch['graphs'])
            cpu_scores = final_score.detach().cpu()
            cpu_labels = labels.detach().cpu()
            for stratum, mask in (
                ('changed', changed_mask),
                ('stable', ~changed_mask),
            ):
                count = int(mask.sum().item())
                topology_counts[stratum] += count
                if count > 0:
                    topology_metrics[stratum].append(
                        compute_topk_metrics(cpu_scores[mask], cpu_labels[mask], ks=ks)
                    )

    return {
        'loss': float(np.mean(losses)) if len(losses) > 0 else 0.0,
        'metrics': aggregate_metric_dict(metrics),
        'score_std': float(np.mean(score_stds)) if len(score_stds) > 0 else 0.0,
        'topology_strata': {
            stratum: {
                'sample_count': topology_counts[stratum],
                'metrics': aggregate_metric_dict(topology_metrics[stratum]),
            }
            for stratum in ('changed', 'stable')
        },
    }


def normal_dynamics_auxiliary_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    reconstruction_weight: float = 1.0,
    forecast_weight: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anomaly_output = outputs['anomaly_output']
    reconstruction = anomaly_output['reconstruction_loss']
    forecast = anomaly_output['forecast_loss']
    total = reconstruction_weight * reconstruction + forecast_weight * forecast
    return total, reconstruction, forecast


def train_dycau(
    model: DyCauRCAModel,
    train_loader,
    val_loader,
    device: torch.device,
    test_loader=None,
    epochs: int = 5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    ks=(1, 3, 5),
    patience: int = 10,
    enable_explanation_losses: bool = False,
    root_symptom_loss_weight: float = 0.0,
    counterfactual_loss_weight: float = 0.0,
    event_chain_loss_weight: float = 0.0,
    invariance_loss_weight: float = 0.0,
    normal_loader=None,
    normal_pretrain_epochs: int = 0,
    normal_auxiliary_loss_weight: float = 0.0,
    reconstruction_loss_weight: float = 1.0,
    forecast_loss_weight: float = 1.0,
):
    device = _safe_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = PairwiseRankingLoss()

    history = []
    pretrain_history = []
    best_state = None
    best_key = None
    best_epoch = -1
    patience_counter = 0

    def selection_key(eval_result):
        metrics = eval_result.get('metrics', {})
        return (
            metrics.get('A@1', 0.0),
            metrics.get('A@3', 0.0),
            metrics.get('A@5', 0.0),
            eval_result.get('score_std', 0.0),
            -eval_result.get('loss', 0.0),
        )

    model.to(device)
    if normal_loader is not None and normal_pretrain_epochs > 0:
        for pretrain_epoch in range(normal_pretrain_epochs):
            model.train()
            epoch_total = []
            epoch_reconstruction = []
            epoch_forecast = []
            epoch_topology_gradients = []
            epoch_edge_evidence_gradients = []
            epoch_changed_windows = 0
            for batch in normal_loader:
                features = batch['features'].float().to(device)
                graphs = _move_graphs_to_device(batch['graphs'], device)
                outputs = model(graphs, features)
                auxiliary, reconstruction, forecast = normal_dynamics_auxiliary_loss(
                    outputs,
                    reconstruction_weight=reconstruction_loss_weight,
                    forecast_weight=forecast_loss_weight,
                )
                optimizer.zero_grad()
                auxiliary.backward()
                topology_diagnostics = topology_parameter_diagnostics(model)
                if topology_diagnostics['dynamic_projection_gradient_norm'] is not None:
                    epoch_topology_gradients.append(
                        topology_diagnostics['dynamic_projection_gradient_norm']
                    )
                if topology_diagnostics['edge_evidence_gradient_norm'] is not None:
                    epoch_edge_evidence_gradients.append(
                        topology_diagnostics['edge_evidence_gradient_norm']
                    )
                epoch_changed_windows += int(
                    observed_topology_change_mask(batch['graphs']).sum().item()
                )
                optimizer.step()
                epoch_total.append(auxiliary.item())
                epoch_reconstruction.append(reconstruction.item())
                epoch_forecast.append(forecast.item())
            pretrain_history.append({
                'epoch': pretrain_epoch,
                'loss': float(np.mean(epoch_total)) if epoch_total else 0.0,
                'reconstruction_loss': float(np.mean(epoch_reconstruction)) if epoch_reconstruction else 0.0,
                'forecast_loss': float(np.mean(epoch_forecast)) if epoch_forecast else 0.0,
                'topology_dynamic_projection_gradient_norm': (
                    float(np.mean(epoch_topology_gradients))
                    if epoch_topology_gradients
                    else None
                ),
                'edge_evidence_gradient_norm': (
                    float(np.mean(epoch_edge_evidence_gradients))
                    if epoch_edge_evidence_gradients
                    else None
                ),
                'topology_fusion_scale': topology_parameter_diagnostics(model)['fusion_scale'],
                'sampled_windows_with_topology_changes': epoch_changed_windows,
            })

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_ranking_losses = []
        train_root_symptom_losses = []
        train_counterfactual_losses = []
        train_event_chain_losses = []
        train_invariance_losses = []
        train_normal_auxiliary_losses = []
        train_reconstruction_losses = []
        train_forecast_losses = []
        train_topology_gradients = []
        train_changed_case_windows = 0
        train_edge_evidence_gradients = []
        train_changed_normal_windows = 0
        train_metrics = []
        normal_iterator = iter(normal_loader) if normal_loader is not None else None

        for batch in train_loader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            group_ids = batch.get('group_ids')
            group_ids = group_ids.to(device) if group_ids is not None else None

            outputs = model(graphs, features)
            train_changed_case_windows += int(
                observed_topology_change_mask(batch['graphs']).sum().item()
            )
            final_score = outputs['rank_output']['final_score']
            ranking_loss = criterion(final_score, labels)
            rs_loss = root_symptom_loss(outputs['rank_output'], labels)
            cf_loss = counterfactual_loss(outputs['rank_output'], labels)
            ec_loss = event_chain_consistency_loss(outputs, labels)
            inv_loss = propagation_invariance_loss(outputs, labels, group_ids)
            loss = ranking_loss
            if enable_explanation_losses:
                loss = (
                    loss
                    + root_symptom_loss_weight * rs_loss
                    + counterfactual_loss_weight * cf_loss
                    + event_chain_loss_weight * ec_loss
                    + invariance_loss_weight * inv_loss
                )

            if normal_iterator is not None and normal_auxiliary_loss_weight > 0:
                try:
                    normal_batch = next(normal_iterator)
                except StopIteration:
                    normal_iterator = iter(normal_loader)
                    normal_batch = next(normal_iterator)
                normal_features = normal_batch['features'].float().to(device)
                normal_graphs = _move_graphs_to_device(normal_batch['graphs'], device)
                normal_outputs = model(normal_graphs, normal_features)
                train_changed_normal_windows += int(
                    observed_topology_change_mask(normal_batch['graphs']).sum().item()
                )
                normal_auxiliary, reconstruction, forecast = normal_dynamics_auxiliary_loss(
                    normal_outputs,
                    reconstruction_weight=reconstruction_loss_weight,
                    forecast_weight=forecast_loss_weight,
                )
                loss = loss + normal_auxiliary_loss_weight * normal_auxiliary
                train_normal_auxiliary_losses.append(normal_auxiliary.item())
                train_reconstruction_losses.append(reconstruction.item())
                train_forecast_losses.append(forecast.item())

            optimizer.zero_grad()
            loss.backward()
            topology_diagnostics = topology_parameter_diagnostics(model)
            if topology_diagnostics['dynamic_projection_gradient_norm'] is not None:
                train_topology_gradients.append(
                    topology_diagnostics['dynamic_projection_gradient_norm']
                )
            optimizer.step()
            if topology_diagnostics['edge_evidence_gradient_norm'] is not None:
                train_edge_evidence_gradients.append(
                    topology_diagnostics['edge_evidence_gradient_norm']
                )

            train_losses.append(loss.item())
            train_ranking_losses.append(ranking_loss.item())
            train_root_symptom_losses.append(rs_loss.item())
            train_counterfactual_losses.append(cf_loss.item())
            train_event_chain_losses.append(ec_loss.item())
            train_invariance_losses.append(inv_loss.item())
            train_metrics.append(compute_topk_metrics(final_score.detach().cpu(), labels.detach().cpu(), ks=ks))

        val_result = evaluate_dycau(model, val_loader, device, ks=ks)
        train_result = {
            'loss': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
            'ranking_loss': float(np.mean(train_ranking_losses)) if len(train_ranking_losses) > 0 else 0.0,
            'root_symptom_loss': float(np.mean(train_root_symptom_losses)) if len(train_root_symptom_losses) > 0 else 0.0,
            'counterfactual_loss': float(np.mean(train_counterfactual_losses)) if len(train_counterfactual_losses) > 0 else 0.0,
            'event_chain_loss': float(np.mean(train_event_chain_losses)) if len(train_event_chain_losses) > 0 else 0.0,
            'invariance_loss': float(np.mean(train_invariance_losses)) if len(train_invariance_losses) > 0 else 0.0,
            'normal_auxiliary_loss': float(np.mean(train_normal_auxiliary_losses)) if train_normal_auxiliary_losses else 0.0,
            'normal_reconstruction_loss': float(np.mean(train_reconstruction_losses)) if train_reconstruction_losses else 0.0,
            'normal_forecast_loss': float(np.mean(train_forecast_losses)) if train_forecast_losses else 0.0,
            'topology_dynamic_projection_gradient_norm': (
                float(np.mean(train_topology_gradients))
                if train_topology_gradients
                else None
            ),
            'topology_fusion_scale': topology_parameter_diagnostics(model)['fusion_scale'],
            'sampled_case_windows_with_topology_changes': train_changed_case_windows,
            'edge_evidence_gradient_norm': (
                float(np.mean(train_edge_evidence_gradients))
                if train_edge_evidence_gradients
                else None
            ),
            'sampled_normal_windows_with_topology_changes': train_changed_normal_windows,
            'metrics': aggregate_metric_dict(train_metrics),
        }

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_result['loss'],
            'train_ranking_loss': train_result['ranking_loss'],
            'train_root_symptom_loss': train_result['root_symptom_loss'],
            'train_counterfactual_loss': train_result['counterfactual_loss'],
            'train_event_chain_loss': train_result['event_chain_loss'],
            'train_invariance_loss': train_result['invariance_loss'],
            'train_normal_auxiliary_loss': train_result['normal_auxiliary_loss'],
            'train_normal_reconstruction_loss': train_result['normal_reconstruction_loss'],
            'train_normal_forecast_loss': train_result['normal_forecast_loss'],
            'train_topology_dynamic_projection_gradient_norm': train_result['topology_dynamic_projection_gradient_norm'],
            'train_edge_evidence_gradient_norm': train_result['edge_evidence_gradient_norm'],
            'train_topology_fusion_scale': train_result['topology_fusion_scale'],
            'train_sampled_case_windows_with_topology_changes': train_result['sampled_case_windows_with_topology_changes'],
            'train_sampled_normal_windows_with_topology_changes': train_result['sampled_normal_windows_with_topology_changes'],
            'train_metrics': train_result['metrics'],
            'val_loss': val_result['loss'],
            'val_metrics': val_result['metrics'],
        }
        history.append(epoch_record)

        current_key = selection_key(val_result)
        if best_key is None or current_key > best_key:
            best_key = current_key
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    validation_eval = evaluate_dycau(model, val_loader, device, ks=ks)
    final_eval = evaluate_dycau(model, test_loader if test_loader is not None else val_loader, device, ks=ks)
    return model, history, final_eval, validation_eval, {
        'best_epoch': best_epoch,
        'best_selection_key': list(best_key) if best_key is not None else None,
        'selection_metric_source': 'validation',
        'selection_rule': 'A@1,A@3,A@5,score_std,-loss',
        'trained_epochs': len(history),
        'device': str(device),
        'enable_explanation_losses': enable_explanation_losses,
        'root_symptom_loss_weight': root_symptom_loss_weight,
        'counterfactual_loss_weight': counterfactual_loss_weight,
        'event_chain_loss_weight': event_chain_loss_weight,
        'invariance_loss_weight': invariance_loss_weight,
        'normal_pretrain_epochs': normal_pretrain_epochs,
        'normal_auxiliary_loss_weight': normal_auxiliary_loss_weight,
        'reconstruction_loss_weight': reconstruction_loss_weight,
        'forecast_loss_weight': forecast_loss_weight,
        'pretrain_history': pretrain_history,
        'final_topology_parameter_diagnostics': topology_parameter_diagnostics(model),
    }
