from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            pairwise = self.margin - pos_scores.unsqueeze(1) + neg_scores.unsqueeze(0)
            losses.append(F.relu(pairwise).mean())
        if len(losses) == 0:
            return scores.mean() * 0.0
        return torch.stack(losses).mean()


class DyCauRCAModel(nn.Module):
    def __init__(self, backbone, anomaly_head, causal_head, ranker, modality_gate=None):
        super().__init__()
        self.modality_gate = modality_gate
        self.backbone = backbone
        self.anomaly_head = anomaly_head
        self.causal_head = causal_head
        self.ranker = ranker

    def forward(self, graphs, features) -> Dict[str, Dict[str, torch.Tensor]]:
        gate_output = self.modality_gate(features) if self.modality_gate is not None else None
        model_features = gate_output['fused_features'] if gate_output is not None else features

        backbone_output = self.backbone(graphs, model_features)
        anomaly_output = self.anomaly_head(backbone_output['temporal_states'], model_features)
        causal_output = self.causal_head(anomaly_output['anomaly_score'])
        rank_output = self.ranker(
            graphs,
            anomaly_output['anomaly_score'],
            causal_output['causal_matrix'],
            lag_weights=causal_output.get('lag_weights'),
        )
        return {
            'modality_gate_output': gate_output,
            'backbone_output': backbone_output,
            'anomaly_output': anomaly_output,
            'causal_output': causal_output,
            'rank_output': rank_output,
        }


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


def _sample_hard_symptom_indices(
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

    # A hard symptom must be a plausible false root, not merely any non-root.
    # Requiring high final rank avoids regularising nodes that cannot displace
    # the labelled root in the evaluated Top-k list.
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


def root_symptom_loss(rank_output: Dict[str, torch.Tensor], labels: torch.Tensor, margin: float = 0.2) -> torch.Tensor:
    root_role = rank_output['root_role_score']
    symptom_role = rank_output['symptom_role_score']
    source_score = rank_output['source_score']
    victim_score = rank_output['victim_score']
    losses = []
    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        if not torch.any(positives):
            continue
        role_margin = margin - root_role[sample_idx, positives] + symptom_role[sample_idx, positives]
        source_margin = margin - source_score[sample_idx, positives] + victim_score[sample_idx, positives]
        losses.append(F.relu(role_margin).mean() + F.relu(source_margin).mean())
    if len(losses) == 0:
        return _zero_like_loss(root_role)
    return torch.stack(losses).mean()


def counterfactual_loss(rank_output: Dict[str, torch.Tensor], labels: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    coverage = rank_output['counterfactual_coverage']
    residual = rank_output['counterfactual_residual']
    losses = []
    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        negatives = labels[sample_idx] <= 0
        if not torch.any(positives) or not torch.any(negatives):
            continue
        pos_cov = coverage[sample_idx, positives]
        neg_cov = coverage[sample_idx, negatives]
        pos_res = residual[sample_idx, positives]
        neg_res = residual[sample_idx, negatives]
        cov_pairwise = margin - pos_cov.unsqueeze(1) + neg_cov.unsqueeze(0)
        res_pairwise = margin + pos_res.unsqueeze(1) - neg_res.unsqueeze(0)
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
        weighted_lag = torch.einsum('l,blij->bij', lag_weights[: lag_strengths.shape[1]], lag_strengths)
    else:
        weighted_lag = None

    for sample_idx in range(labels.shape[0]):
        positives = labels[sample_idx] > 0
        negatives = labels[sample_idx] <= 0
        if not torch.any(positives) or not torch.any(negatives):
            continue
        evidence_idx = torch.topk(anomaly_peak[sample_idx], k=topk).indices
        reach = causal_matrix[sample_idx][evidence_idx, :].mean(dim=0)
        pos_reach = reach[positives]
        neg_reach = reach[negatives]
        pairwise = margin - pos_reach.unsqueeze(1) + neg_reach.unsqueeze(0)
        sample_losses = [F.relu(pairwise).mean()]
        if weighted_lag is not None:
            lag_reach = weighted_lag[sample_idx][evidence_idx, :].mean(dim=0)
            sample_losses.append(F.relu(-lag_reach[positives]).mean())
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

    label_weight = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    root_to_targets = torch.bmm(causal_matrix, label_weight.unsqueeze(-1)).squeeze(-1)
    root_outgoing = torch.bmm(causal_matrix.transpose(1, 2), label_weight.unsqueeze(-1)).squeeze(-1)
    signature = torch.cat([root_to_targets, root_outgoing], dim=-1)
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
                hard_idx = _sample_hard_symptom_indices(
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
                row = {
                    'case_id': int(batch['case_ids'][sample_idx].item()),
                    'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
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
                    'root_role_gap': float((rank_output['root_role_score'][sample_idx, root_idx] - rank_output['root_role_score'][sample_idx, hard_node_idx]).detach().cpu().item()),
                    'source_gap': float((rank_output['source_score'][sample_idx, root_idx] - rank_output['source_score'][sample_idx, hard_node_idx]).detach().cpu().item()),
                    'victim_gap': float((rank_output['victim_score'][sample_idx, hard_node_idx] - rank_output['victim_score'][sample_idx, root_idx]).detach().cpu().item()),
                    'incoming_explainedness_gap': float((rank_output['incoming_explainedness'][sample_idx, hard_node_idx] - rank_output['incoming_explainedness'][sample_idx, root_idx]).detach().cpu().item()),
                    'root_anomaly_peak': float(anomaly_score[sample_idx, :, root_idx].max().detach().cpu().item()),
                    'hard_symptom_anomaly_peak': float(anomaly_score[sample_idx, :, hard_node_idx].max().detach().cpu().item()),
                }
                rows.append(row)

                auc_scores.extend(rank_output['root_role_score'][sample_idx].detach().cpu().tolist())
                auc_labels.extend((labels[sample_idx] > 0).detach().cpu().int().tolist())

    df = pd.DataFrame(rows)
    summary = {
        'cases': int(len(df)),
        'hard_symptom_error_rate': float(df['hard_symptom_before_root'].mean()) if len(df) > 0 else None,
        'root_role_gap_mean': float(df['root_role_gap'].mean()) if len(df) > 0 else None,
        'source_gap_mean': float(df['source_gap'].mean()) if len(df) > 0 else None,
        'victim_gap_mean': float(df['victim_gap'].mean()) if len(df) > 0 else None,
        'incoming_explainedness_gap_mean': float(df['incoming_explainedness_gap'].mean()) if len(df) > 0 else None,
        'root_vs_hard_symptom_margin_mean': float(df['root_vs_hard_symptom_margin'].mean()) if len(df) > 0 else None,
        'role_auc': _binary_auc(auc_scores, auc_labels),
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
        '# Challenge 2 Role Diagnostic',
        '',
        f"- Cases: {summary.get('cases')}",
        f"- hard_symptom_error_rate: {summary.get('hard_symptom_error_rate')}",
        f"- role_auc: {summary.get('role_auc')}",
        f"- root_role_gap_mean: {summary.get('root_role_gap_mean')}",
        f"- source_gap_mean: {summary.get('source_gap_mean')}",
        f"- victim_gap_mean: {summary.get('victim_gap_mean')}",
        f"- incoming_explainedness_gap_mean: {summary.get('incoming_explainedness_gap_mean')}",
        f"- root_vs_hard_symptom_margin_mean: {summary.get('root_vs_hard_symptom_margin_mean')}",
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

    return {
        'loss': float(np.mean(losses)) if len(losses) > 0 else 0.0,
        'metrics': aggregate_metric_dict(metrics),
        'score_std': float(np.mean(score_stds)) if len(score_stds) > 0 else 0.0,
    }


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
):
    device = _safe_device(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = PairwiseRankingLoss()

    history = []
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
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_ranking_losses = []
        train_root_symptom_losses = []
        train_counterfactual_losses = []
        train_event_chain_losses = []
        train_invariance_losses = []
        train_metrics = []

        for batch in train_loader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)
            group_ids = batch.get('group_ids')
            group_ids = group_ids.to(device) if group_ids is not None else None

            outputs = model(graphs, features)
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

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
    }
