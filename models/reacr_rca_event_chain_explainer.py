import os
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import numpy as np
import pandas as pd
import torch

from models.dycau_trainer import _move_graphs_to_device, _safe_device


def _safe_node_name(index_to_node: Dict[int, str], node_idx: int) -> str:
    return index_to_node.get(int(node_idx), str(int(node_idx)))


def _static_edges(graph: dgl.DGLGraph) -> Tuple[set, set]:
    src, dst = graph.edges()
    forward = set(zip(src.cpu().tolist(), dst.cpu().tolist()))
    reverse = {(v, u) for u, v in forward}
    return forward, reverse


def _edge_direction(source: int, target: int, forward_edges: set, reverse_edges: set) -> str:
    if source == target:
        return 'self'
    if (source, target) in forward_edges:
        return 'downstream'
    if (source, target) in reverse_edges:
        return 'upstream'
    return 'causal_inferred'


def _build_outgoing_edges(causal_matrix: torch.Tensor, edge_topk: int, min_strength: float) -> Dict[int, List[Tuple[int, float]]]:
    node_num = causal_matrix.shape[0]
    edge_topk = max(1, min(edge_topk, node_num - 1))
    outgoing = {node_idx: [] for node_idx in range(node_num)}
    for source_idx in range(node_num):
        # causal_matrix[target, source] follows the causal head convention.
        strengths = causal_matrix[:, source_idx].clone()
        strengths[source_idx] = 0.0
        values, targets = torch.topk(strengths, k=edge_topk)
        for target_idx, strength in zip(targets.tolist(), values.tolist()):
            if strength >= min_strength and target_idx != source_idx:
                outgoing[source_idx].append((target_idx, float(strength)))
    return outgoing


def _best_path(
    root_idx: int,
    evidence_nodes: Sequence[int],
    outgoing: Dict[int, List[Tuple[int, float]]],
    max_depth: int,
    allow_fallback: bool = False,
) -> Tuple[List[Tuple[int, int, float]], str, bool]:
    evidence_set = set(int(node_idx) for node_idx in evidence_nodes if int(node_idx) != int(root_idx))
    if not evidence_set:
        return [], 'none', False

    best_path = []
    best_score = -1.0
    queue = [(int(root_idx), [], 1.0, {int(root_idx)})]
    while queue:
        current, path, path_score, visited = queue.pop(0)
        if len(path) >= max_depth:
            continue
        for nxt, strength in outgoing.get(current, []):
            if nxt in visited:
                continue
            new_path = path + [(current, nxt, strength)]
            new_score = path_score * max(float(strength), 1e-8)
            if nxt in evidence_set and new_score > best_score:
                best_path = new_path
                best_score = new_score
            queue.append((nxt, new_path, new_score, visited | {nxt}))

    if best_path:
        return best_path, 'strict', False

    first_hop = outgoing.get(int(root_idx), [])
    if allow_fallback and first_hop:
        target_idx, strength = max(first_hop, key=lambda item: item[1])
        return [(int(root_idx), int(target_idx), float(strength))], 'fallback_one_hop', True
    return [], 'none', False


def _path_nodes(root_idx: int, path: Sequence[Tuple[int, int, float]]) -> List[int]:
    nodes = [int(root_idx)]
    for _, target_idx, _ in path:
        nodes.append(int(target_idx))
    return nodes


def _format_chain(nodes: Sequence[int], index_to_node: Dict[int, str]) -> str:
    return ' -> '.join(_safe_node_name(index_to_node, node_idx) for node_idx in nodes)


def _parse_catalog_nodes(text: object) -> List[str]:
    if pd.isna(text):
        return []
    return [item.strip() for item in str(text).split('->') if item.strip()]


def _parse_catalog_edges(text: object) -> List[Tuple[str, str]]:
    if pd.isna(text):
        return []
    edges = []
    for item in str(text).split('|'):
        parts = [part.strip() for part in item.split('->')]
        if len(parts) == 2 and parts[0] and parts[1]:
            edges.append((parts[0], parts[1]))
    return edges


def _parse_catalog_lags(text: object) -> List[float]:
    if pd.isna(text):
        return []
    values = []
    for item in str(text).split('|'):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            continue
    return values


def _contains_contiguous_path(path: Sequence[str], query: Sequence[str]) -> bool:
    if not query or len(query) > len(path):
        return False
    q = list(query)
    p = list(path)
    for start_idx in range(0, len(p) - len(q) + 1):
        if p[start_idx:start_idx + len(q)] == q:
            return True
    return False


def _load_call_chain_catalog(
    path: Optional[str],
    case_ids: Optional[Sequence[int]] = None,
    chunk_size: int = 200000,
) -> Dict[int, Dict[str, object]]:
    if not path or not os.path.exists(path):
        return {}
    wanted = None if case_ids is None else {int(case_id) for case_id in case_ids if int(case_id) >= 0}
    catalog: Dict[int, Dict[str, object]] = {}
    for df in pd.read_csv(path, chunksize=chunk_size):
        if 'case_id' not in df.columns:
            continue
        df = df.dropna(subset=['case_id'])
        df['case_id'] = df['case_id'].astype(int)
        if wanted is not None:
            df = df[df['case_id'].isin(wanted)]
        if df.empty:
            continue
        for _, row in df.iterrows():
            case_id = int(row['case_id'])
            nodes = _parse_catalog_nodes(row.get('nodes'))
            edges = _parse_catalog_edges(row.get('edges'))
            lags = _parse_catalog_lags(row.get('edge_lags_ms'))
            entry = catalog.setdefault(
                case_id,
                {
                    'paths': [],
                    'edge_set': set(),
                    'reachability': set(),
                    'edge_lags': {},
                },
            )
            if nodes:
                entry['paths'].append(nodes)
                for source_idx in range(len(nodes)):
                    for target_idx in range(source_idx + 1, len(nodes)):
                        entry['reachability'].add((nodes[source_idx], nodes[target_idx]))
            for edge_idx, edge in enumerate(edges):
                entry['edge_set'].add(edge)
                if edge_idx < len(lags):
                    entry['edge_lags'].setdefault(edge, []).append(float(lags[edge_idx]))
    return catalog


def _call_chain_match_features(case_catalog: Optional[Dict[str, object]], chain_nodes: Sequence[str]) -> Dict[str, object]:
    edges = list(zip(chain_nodes[:-1], chain_nodes[1:]))
    if not case_catalog or not chain_nodes:
        return {
            'exact_call_path_exists': None,
            'all_edges_exist_in_observed_trace': None,
            'edge_presence_rate_in_case_window': None,
            'root_to_evidence_reachable_in_trace': None,
            'matched_trace_count': 0,
            'temporal_span_order_valid': None,
            'mean_span_lag_ms': None,
        }

    paths = case_catalog['paths']
    edge_set = case_catalog['edge_set']
    reachability = case_catalog['reachability']
    edge_lags = case_catalog['edge_lags']

    exact_count = sum(1 for path in paths if _contains_contiguous_path(path, chain_nodes))
    edge_hits = [edge in edge_set for edge in edges]
    lag_values = []
    for edge in edges:
        lag_values.extend(edge_lags.get(edge, []))
    if len(edges) == 0:
        edge_presence_rate = 0.0
        all_edges_exist = False
    else:
        edge_presence_rate = float(np.mean(edge_hits))
        all_edges_exist = bool(all(edge_hits))

    root_to_evidence = False
    if len(chain_nodes) > 1:
        root_to_evidence = (chain_nodes[0], chain_nodes[-1]) in reachability

    return {
        'exact_call_path_exists': int(exact_count > 0),
        'all_edges_exist_in_observed_trace': int(all_edges_exist),
        'edge_presence_rate_in_case_window': edge_presence_rate,
        'root_to_evidence_reachable_in_trace': int(root_to_evidence),
        'matched_trace_count': int(exact_count),
        'temporal_span_order_valid': None if not lag_values else int(all(lag >= 0 for lag in lag_values)),
        'mean_span_lag_ms': None if not lag_values else float(np.mean(lag_values)),
    }


def _format_events(
    nodes: Sequence[int],
    final_score: torch.Tensor,
    anomaly_mean: torch.Tensor,
    anomaly_peak: torch.Tensor,
    index_to_node: Dict[int, str],
) -> str:
    events = []
    for node_idx in nodes:
        events.append(
            (
                f"{_safe_node_name(index_to_node, node_idx)}"
                f"(rank_score={final_score[node_idx].item():.6f},"
                f"mean_anom={anomaly_mean[node_idx].item():.6f},"
                f"peak_anom={anomaly_peak[node_idx].item():.6f})"
            )
        )
    return ' | '.join(events)


def _format_edges(
    path: Sequence[Tuple[int, int, float]],
    forward_edges: set,
    reverse_edges: set,
    index_to_node: Dict[int, str],
) -> str:
    formatted = []
    for source_idx, target_idx, strength in path:
        direction = _edge_direction(source_idx, target_idx, forward_edges, reverse_edges)
        formatted.append(
            (
                f"{_safe_node_name(index_to_node, source_idx)}"
                f"->{_safe_node_name(index_to_node, target_idx)}"
                f"[{direction},causal={strength:.6f}]"
            )
        )
    return ' | '.join(formatted)


def _node_onset_steps(anomaly_score: torch.Tensor) -> torch.Tensor:
    # anomaly_score: [W, N]. Use observed anomaly dynamics, not labels.
    threshold = anomaly_score.mean(dim=0) + anomaly_score.std(dim=0)
    above = anomaly_score >= threshold.unsqueeze(0)
    fallback = torch.argmax(anomaly_score, dim=0)
    onset = []
    for node_idx in range(anomaly_score.shape[1]):
        hits = torch.where(above[:, node_idx])[0]
        onset.append(int(hits[0].item()) if len(hits) else int(fallback[node_idx].item()))
    return torch.tensor(onset, dtype=torch.long)


def _path_validation_features(
    root_idx: int,
    path: Sequence[Tuple[int, int, float]],
    evidence_nodes: Sequence[int],
    anomaly_score: torch.Tensor,
    causal_lag_limit: int,
    forward_edges: set,
    reverse_edges: set,
) -> Dict[str, float]:
    nodes = _path_nodes(root_idx, path)
    evidence_set = set(int(node_idx) for node_idx in evidence_nodes)
    chain_set = set(nodes)
    observed_anomaly_coverage = float(len(chain_set & evidence_set) / max(len(evidence_set), 1))

    if not path:
        return {
            'topology_edge_hit_rate': 0.0,
            'direction_consistency_rate': 0.0,
            'causal_inferred_ratio': 0.0,
            'temporal_order_rate': 0.0,
            'lag_consistency_rate': 0.0,
            'mean_observed_lag_steps': 0.0,
            'observed_anomaly_coverage': observed_anomaly_coverage,
        }

    directions = [_edge_direction(source_idx, target_idx, forward_edges, reverse_edges) for source_idx, target_idx, _ in path]
    topology_hits = [direction != 'causal_inferred' for direction in directions]
    direction_hits = [direction in {'self', 'downstream'} for direction in directions]
    inferred_hits = [direction == 'causal_inferred' for direction in directions]

    onset = _node_onset_steps(anomaly_score)
    lags = [int(onset[target_idx].item() - onset[source_idx].item()) for source_idx, target_idx, _ in path]
    temporal_hits = [lag >= 0 for lag in lags]
    lag_hits = [0 <= lag <= max(causal_lag_limit, 0) for lag in lags]

    return {
        'topology_edge_hit_rate': float(np.mean(topology_hits)),
        'direction_consistency_rate': float(np.mean(direction_hits)),
        'causal_inferred_ratio': float(np.mean(inferred_hits)),
        'temporal_order_rate': float(np.mean(temporal_hits)),
        'lag_consistency_rate': float(np.mean(lag_hits)),
        'mean_observed_lag_steps': float(np.mean(lags)),
        'observed_anomaly_coverage': observed_anomaly_coverage,
    }


def _batch_signature_consistency(causal_matrices: torch.Tensor, labels: Optional[torch.Tensor], group_ids: Optional[torch.Tensor]) -> List[float]:
    if labels is None or group_ids is None or causal_matrices.shape[0] < 2:
        return [float('nan')] * causal_matrices.shape[0]
    label_weight = labels / labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    root_to_targets = torch.bmm(causal_matrices, label_weight.unsqueeze(-1)).squeeze(-1)
    root_outgoing = torch.bmm(causal_matrices.transpose(1, 2), label_weight.unsqueeze(-1)).squeeze(-1)
    signature = torch.cat([root_to_targets, root_outgoing], dim=-1)
    signature = torch.nn.functional.normalize(signature, dim=-1)
    sim = torch.matmul(signature, signature.t())
    values = []
    for sample_idx in range(causal_matrices.shape[0]):
        same_group = (group_ids == group_ids[sample_idx]) & (group_ids >= 0)
        same_group[sample_idx] = False
        if torch.any(same_group):
            values.append(float(sim[sample_idx, same_group].mean().item()))
        else:
            values.append(float('nan'))
    return values


def _aggregate_metrics(rows: List[Dict[str, object]], topk: int) -> Dict[str, float]:
    if not rows:
        return {
            'cases': 0,
            'explanations': 0,
        }

    df = pd.DataFrame(rows)
    case_count = max(int(df['case_id'].nunique()), 1)
    label_rows = df[df['rank'] <= topk]
    strict_rows = df[df['path_found'] > 0]
    fallback_rows = df[df['fallback_used'] > 0]
    strict_label_rows = label_rows[label_rows['path_found'] > 0]
    return {
        'cases': case_count,
        'explanations': int(len(df)),
        'topk_per_case': int(topk),
        'strict_chain_coverage': float((df['path_found'] > 0).mean()),
        'strict_chain_coverage_at_topk': float((label_rows['path_found'] > 0).mean()) if len(label_rows) else 0.0,
        'fallback_chain_coverage': float((df['fallback_used'] > 0).mean()),
        'fallback_chain_coverage_at_topk': float((label_rows['fallback_used'] > 0).mean()) if len(label_rows) else 0.0,
        'label_chain_hit_at_topk_strict': float(label_rows.groupby('case_id')['label_in_chain_strict'].max().mean()) if len(label_rows) else 0.0,
        'label_root_hit_at_topk': float(label_rows.groupby('case_id')['label_is_root'].max().mean()) if len(label_rows) else 0.0,
        'strict_explanations': int(len(strict_rows)),
        'fallback_explanations': int(len(fallback_rows)),
        'avg_path_length': float(strict_rows['path_length'].mean()) if len(strict_rows) else 0.0,
        'avg_path_strength': float(strict_rows['path_strength'].mean()) if len(strict_rows) else 0.0,
        'avg_directional_edge_ratio': float(strict_rows['directional_edge_ratio'].mean()) if len(strict_rows) else 0.0,
    }


def _aggregate_validation_metrics(rows: List[Dict[str, object]], topk: int) -> Dict[str, float]:
    if not rows:
        return {'cases': 0, 'validated_explanations': 0}
    df = pd.DataFrame(rows)
    label_rows = df[df['rank'] <= topk]
    numeric_cols = [
        'root_hit',
        'label_in_chain',
        'topology_edge_hit_rate',
        'direction_consistency_rate',
        'causal_inferred_ratio',
        'temporal_order_rate',
        'lag_consistency_rate',
        'observed_anomaly_coverage',
        'counterfactual_coverage',
        'counterfactual_residual',
        'same_group_signature_consistency',
        'exact_call_path_exists',
        'all_edges_exist_in_observed_trace',
        'edge_presence_rate_in_case_window',
        'root_to_evidence_reachable_in_trace',
        'matched_trace_count',
        'temporal_span_order_valid',
        'mean_span_lag_ms',
    ]
    metrics = {
        'cases': int(df['case_id'].nunique()),
        'validated_explanations': int(len(df)),
        'topk_per_case': int(topk),
        'root_hit_at_topk': float(label_rows.groupby('case_id')['root_hit'].max().mean()) if len(label_rows) else 0.0,
        'label_in_chain_at_topk': float(label_rows.groupby('case_id')['label_in_chain'].max().mean()) if len(label_rows) else 0.0,
    }
    for col in numeric_cols:
        if col in df.columns:
            mean_value = pd.to_numeric(df[col], errors='coerce').mean()
            metrics[f'mean_{col}'] = None if pd.isna(mean_value) else float(mean_value)
    return metrics


def export_event_chains(
    model,
    dataloader,
    device: torch.device,
    index_to_node: Dict[int, str],
    csv_path: str,
    metrics_path: str,
    validation_csv_path: str = None,
    validation_metrics_path: str = None,
    call_chain_catalog_path: str = None,
    topk: int = 3,
    max_depth: int = 3,
    edge_topk: int = 3,
    evidence_topk: int = 5,
    min_strength: float = 0.0,
    allow_fallback: bool = False,
) -> Dict[str, float]:
    device = _safe_device(device)
    model.eval()
    rows: List[Dict[str, object]] = []
    validation_rows: List[Dict[str, object]] = []
    target_case_ids = set()
    if call_chain_catalog_path:
        for batch in dataloader:
            if 'case_ids' in batch:
                target_case_ids.update(int(case_id.item()) for case_id in batch['case_ids'] if int(case_id.item()) >= 0)
    call_chain_catalog = _load_call_chain_catalog(call_chain_catalog_path, case_ids=target_case_ids)

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().cpu() if batch['labels'] is not None else None
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)

            final_scores = outputs['rank_output']['final_score'].detach().cpu()
            rank_output = {key: value.detach().cpu() for key, value in outputs['rank_output'].items()}
            anomaly_scores = outputs['anomaly_output']['anomaly_score'].detach().cpu()
            causal_matrices = outputs['causal_output']['causal_matrix'].detach().cpu()
            causal_lag_limit = int(outputs['causal_output'].get('lag_weights', torch.empty(0)).numel())
            graph_list = dgl.unbatch(batch['graphs'][-1])
            group_ids = batch.get('group_ids')
            group_ids = group_ids.cpu() if group_ids is not None else None
            signature_consistency = _batch_signature_consistency(causal_matrices, labels, group_ids)

            for sample_idx in range(final_scores.shape[0]):
                final_score = final_scores[sample_idx]
                anomaly_score = anomaly_scores[sample_idx]
                anomaly_mean = anomaly_score.mean(dim=0)
                anomaly_peak = anomaly_score.max(dim=0).values
                causal_matrix = causal_matrices[sample_idx]

                top_roots = torch.argsort(final_score, descending=True)[:topk].tolist()
                evidence_nodes = torch.argsort(anomaly_peak, descending=True)[:evidence_topk].tolist()
                outgoing = _build_outgoing_edges(causal_matrix, edge_topk=edge_topk, min_strength=min_strength)
                forward_edges, reverse_edges = _static_edges(graph_list[sample_idx])
                label_nodes = []
                if labels is not None:
                    label_nodes = torch.where(labels[sample_idx] > 0)[0].tolist()
                label_set = set(label_nodes)

                for rank_idx, root_idx in enumerate(top_roots, start=1):
                    path, path_type, fallback_used = _best_path(
                        root_idx,
                        evidence_nodes,
                        outgoing,
                        max_depth=max_depth,
                        allow_fallback=allow_fallback,
                    )
                    nodes = _path_nodes(root_idx, path)
                    strengths = [edge[2] for edge in path]
                    path_found = int(path_type == 'strict')
                    directional_count = sum(
                        1
                        for source_idx, target_idx, _ in path
                        if _edge_direction(source_idx, target_idx, forward_edges, reverse_edges) != 'causal_inferred'
                    )
                    path_strength = float(np.prod(strengths)) if strengths else 0.0
                    label_in_chain_strict = int(path_found and any(node_idx in label_set for node_idx in nodes))
                    validation_features = _path_validation_features(
                        root_idx,
                        path,
                        evidence_nodes,
                        anomaly_score,
                        causal_lag_limit=causal_lag_limit,
                        forward_edges=forward_edges,
                        reverse_edges=reverse_edges,
                    )
                    chain_node_names = [_safe_node_name(index_to_node, node_idx) for node_idx in nodes]
                    case_id = int(batch['case_ids'][sample_idx].item())
                    call_chain_features = _call_chain_match_features(
                        call_chain_catalog.get(case_id),
                        chain_node_names,
                    )
                    rows.append(
                        {
                            'case_id': case_id,
                            'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
                            'rank': int(rank_idx),
                            'root_node': _safe_node_name(index_to_node, root_idx),
                            'root_index': int(root_idx),
                            'root_score': float(final_score[root_idx].item()),
                            'root_role_score': float(rank_output['root_role_score'][sample_idx, root_idx].item()),
                            'symptom_role_score': float(rank_output['symptom_role_score'][sample_idx, root_idx].item()),
                            'source_score': float(rank_output['source_score'][sample_idx, root_idx].item()),
                            'victim_score': float(rank_output['victim_score'][sample_idx, root_idx].item()),
                            'incoming_explainedness': float(rank_output['incoming_explainedness'][sample_idx, root_idx].item()),
                            'counterfactual_coverage': float(rank_output['counterfactual_coverage'][sample_idx, root_idx].item()),
                            'counterfactual_residual': float(rank_output['counterfactual_residual'][sample_idx, root_idx].item()),
                            'label_nodes': ','.join(_safe_node_name(index_to_node, idx) for idx in label_nodes),
                            'label_is_root': int(root_idx in label_set),
                            'label_in_chain': label_in_chain_strict,
                            'label_in_chain_strict': label_in_chain_strict,
                            'path_found': path_found,
                            'path_type': path_type,
                            'fallback_used': int(fallback_used),
                            'path_length': int(len(path)),
                            'path_strength': path_strength,
                            'directional_edge_ratio': float(directional_count / max(len(path), 1)),
                            'chain': _format_chain(nodes, index_to_node),
                            'event_evidence': _format_events(nodes, final_score, anomaly_mean, anomaly_peak, index_to_node),
                            'edge_evidence': _format_edges(path, forward_edges, reverse_edges, index_to_node),
                        }
                    )
                    validation_rows.append(
                        {
                            'case_id': case_id,
                            'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
                            'rank': int(rank_idx),
                            'pred_root': _safe_node_name(index_to_node, root_idx),
                            'pred_root_index': int(root_idx),
                            'true_roots': ','.join(_safe_node_name(index_to_node, idx) for idx in label_nodes),
                            'root_hit': int(root_idx in label_set),
                            'label_in_chain': label_in_chain_strict,
                            'chain': _format_chain(nodes, index_to_node),
                            'path_type': path_type,
                            'path_length': int(len(path)),
                            'path_strength': path_strength,
                            'topology_edge_hit_rate': validation_features['topology_edge_hit_rate'],
                            'direction_consistency_rate': validation_features['direction_consistency_rate'],
                            'causal_inferred_ratio': validation_features['causal_inferred_ratio'],
                            'temporal_order_rate': validation_features['temporal_order_rate'],
                            'lag_consistency_rate': validation_features['lag_consistency_rate'],
                            'mean_observed_lag_steps': validation_features['mean_observed_lag_steps'],
                            'observed_anomaly_coverage': validation_features['observed_anomaly_coverage'],
                            'root_role_score': float(rank_output['root_role_score'][sample_idx, root_idx].item()),
                            'symptom_role_score': float(rank_output['symptom_role_score'][sample_idx, root_idx].item()),
                            'incoming_explainedness': float(rank_output['incoming_explainedness'][sample_idx, root_idx].item()),
                            'counterfactual_coverage': float(rank_output['counterfactual_coverage'][sample_idx, root_idx].item()),
                            'counterfactual_residual': float(rank_output['counterfactual_residual'][sample_idx, root_idx].item()),
                            'same_group_signature_consistency': signature_consistency[sample_idx],
                            'exact_call_path_exists': call_chain_features['exact_call_path_exists'],
                            'all_edges_exist_in_observed_trace': call_chain_features['all_edges_exist_in_observed_trace'],
                            'edge_presence_rate_in_case_window': call_chain_features['edge_presence_rate_in_case_window'],
                            'root_to_evidence_reachable_in_trace': call_chain_features['root_to_evidence_reachable_in_trace'],
                            'matched_trace_count': call_chain_features['matched_trace_count'],
                            'temporal_span_order_valid': call_chain_features['temporal_span_order_valid'],
                            'mean_span_lag_ms': call_chain_features['mean_span_lag_ms'],
                            'validation_note': (
                                'root/topology labels come from dataset metadata; '
                                'temporal/anomaly coverage uses observed window anomaly scores; '
                                'call-chain fields come from observed traceId/id/pid span trees when available; '
                                'chain and counterfactual fields are model outputs.'
                            ),
                        }
                    )

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metrics = _aggregate_metrics(rows, topk=topk)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        import json

        json.dump(metrics, f, indent=2)
    if validation_csv_path is not None:
        pd.DataFrame(validation_rows).to_csv(validation_csv_path, index=False)
    if validation_metrics_path is not None:
        validation_metrics = _aggregate_validation_metrics(validation_rows, topk=topk)
        with open(validation_metrics_path, 'w', encoding='utf-8') as f:
            import json

            json.dump(validation_metrics, f, indent=2)
    return metrics
