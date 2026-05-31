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
        rank_output = self.ranker(graphs, anomaly_output['anomaly_score'], causal_output['causal_matrix'])
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
    return metrics


def aggregate_metric_dict(metric_dicts: List[Dict[str, float]]) -> Dict[str, float]:
    if len(metric_dicts) == 0:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metric[key] for metric in metric_dicts])) for key in keys}


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


def evaluate_dycau(model: DyCauRCAModel, dataloader, device: torch.device, ks=(1, 3, 5)):
    device = _safe_device(device)
    model.eval()
    losses = []
    metrics = []
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
            metrics.append(compute_topk_metrics(final_score.cpu(), labels.cpu(), ks=ks))

    return {
        'loss': float(np.mean(losses)) if len(losses) > 0 else 0.0,
        'metrics': aggregate_metric_dict(metrics),
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
            -eval_result.get('loss', 0.0),
        )

    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_metrics = []

        for batch in train_loader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().to(device)
            graphs = _move_graphs_to_device(batch['graphs'], device)

            outputs = model(graphs, features)
            final_score = outputs['rank_output']['final_score']
            loss = criterion(final_score, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_metrics.append(compute_topk_metrics(final_score.detach().cpu(), labels.detach().cpu(), ks=ks))

        val_result = evaluate_dycau(model, val_loader, device, ks=ks)
        train_result = {
            'loss': float(np.mean(train_losses)) if len(train_losses) > 0 else 0.0,
            'metrics': aggregate_metric_dict(train_metrics),
        }

        epoch_record = {
            'epoch': epoch,
            'train_loss': train_result['loss'],
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
        'selection_rule': 'A@1,A@3,A@5,-loss',
        'trained_epochs': len(history),
        'device': str(device),
    }
