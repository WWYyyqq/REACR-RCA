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


def export_event_chains(
    model,
    dataloader,
    device: torch.device,
    index_to_node: Dict[int, str],
    csv_path: str,
    metrics_path: str,
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

    with torch.no_grad():
        for batch in dataloader:
            features = batch['features'].float().to(device)
            labels = batch['labels'].float().cpu() if batch['labels'] is not None else None
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, features)

            final_scores = outputs['rank_output']['final_score'].detach().cpu()
            anomaly_scores = outputs['anomaly_output']['anomaly_score'].detach().cpu()
            causal_matrices = outputs['causal_output']['causal_matrix'].detach().cpu()
            graph_list = dgl.unbatch(batch['graphs'][-1])

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
                    rows.append(
                        {
                            'case_id': int(batch['case_ids'][sample_idx].item()),
                            'case_timestamp': int(batch['case_timestamps'][sample_idx].item()),
                            'rank': int(rank_idx),
                            'root_node': _safe_node_name(index_to_node, root_idx),
                            'root_index': int(root_idx),
                            'root_score': float(final_score[root_idx].item()),
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

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    metrics = _aggregate_metrics(rows, topk=topk)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        import json

        json.dump(metrics, f, indent=2)
    return metrics
