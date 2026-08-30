"""Audit the topology that is actually stored in REACR-RCA graph snapshots.

The audit reads graph objects from ``train_samples.pkl`` and
``test_samples.pkl``.  It does not read a graph catalogue to fill missing
edges, and it does not add self-loops.  Reported edge sets, edge attributes,
and churn therefore describe only the serialized DGL snapshots used by the
model.
"""

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import torch
import yaml

from models.reacr_rca_topology_dynamics import (
    canonical_edge_multiset,
)


def _edge_set_digest(signature: Tuple[Tuple[int, int], ...]) -> str:
    payload = json.dumps(signature, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _ensure_unbatched(graph: dgl.DGLGraph, snapshot_idx: int) -> None:
    if int(graph.batch_size) != 1:
        raise ValueError(
            f'snapshot {snapshot_idx} is a DGL batch of {graph.batch_size} graphs; '
            'an audit sequence must contain individual temporal snapshots'
        )


def _counter_difference_size(left: Counter, right: Counter) -> int:
    return int(sum((left - right).values()))


def _multiset_union_size(left: Counter, right: Counter) -> int:
    return int(sum((left | right).values()))


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_max(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def _finalize_edge_data_stats(
    stats,
    total_snapshots: int,
) -> Dict[str, Dict[str, object]]:
    result: Dict[str, Dict[str, object]] = {}
    for key in sorted(stats):
        item = stats[key]
        snapshots_present = int(item['snapshots_present'])
        result[key] = {
            'snapshots_present': snapshots_present,
            'snapshot_fraction': (
                float(snapshots_present / total_snapshots)
                if total_snapshots > 0
                else 0.0
            ),
            'leading_dimension_matches_edge_count': int(
                item['leading_dimension_matches_edge_count']
            ),
            'leading_dimension_mismatches': int(item['leading_dimension_mismatches']),
            'total_values': int(item['total_values']),
            'nonfinite_values': int(item['nonfinite_values']),
            'dtype_counts': dict(sorted(item['dtype_counts'].items())),
            'shape_counts': dict(sorted(item['shape_counts'].items())),
        }
    return result


def audit_graph_sequence(
    graphs: Sequence[dgl.DGLGraph],
    timestamps: Optional[Sequence[int]] = None,
    max_transition_records: int = 100,
    include_edge_sets: bool = False,
) -> Dict[str, object]:
    """Audit an ordered sequence of individual DGL snapshots.

    Parallel edges are retained, so topology signatures and churn are based on
    directed edge multisets rather than deduplicated endpoint pairs.
    """

    if timestamps is not None and len(timestamps) != len(graphs):
        raise ValueError(
            f'timestamps length {len(timestamps)} does not match graphs length {len(graphs)}'
        )
    if max_transition_records < 0:
        raise ValueError('max_transition_records must be non-negative')

    signatures: List[Tuple[Tuple[int, int], ...]] = []
    edge_counters: List[Counter] = []
    node_counts: List[int] = []
    edge_counts: List[int] = []
    edge_data_stats = defaultdict(
        lambda: {
            'snapshots_present': 0,
            'leading_dimension_matches_edge_count': 0,
            'leading_dimension_mismatches': 0,
            'total_values': 0,
            'nonfinite_values': 0,
            'dtype_counts': Counter(),
            'shape_counts': Counter(),
        }
    )
    snapshots_with_any_edge_data = 0
    snapshots_with_weight_like_data = 0

    for snapshot_idx, graph in enumerate(graphs):
        _ensure_unbatched(graph, snapshot_idx)
        signature = canonical_edge_multiset(graph)
        signatures.append(signature)
        edge_counters.append(Counter(signature))
        node_count = int(graph.num_nodes())
        edge_count = int(graph.num_edges())
        node_counts.append(node_count)
        edge_counts.append(edge_count)

        edge_keys = list(graph.edata.keys())
        if edge_keys:
            snapshots_with_any_edge_data += 1
        if any(key == 'w' or 'weight' in key.lower() for key in edge_keys):
            snapshots_with_weight_like_data += 1

        for key in edge_keys:
            tensor = graph.edata[key]
            item = edge_data_stats[key]
            item['snapshots_present'] += 1
            item['dtype_counts'][str(tensor.dtype)] += 1
            item['shape_counts'][str(list(tensor.shape))] += 1
            item['total_values'] += int(tensor.numel())
            if tensor.ndim > 0 and int(tensor.shape[0]) == edge_count:
                item['leading_dimension_matches_edge_count'] += 1
            else:
                item['leading_dimension_mismatches'] += 1
            if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
                item['nonfinite_values'] += int(
                    torch.count_nonzero(~torch.isfinite(tensor)).item()
                )

    total_snapshots = len(graphs)
    signature_counts = Counter(signatures)
    unique_catalog = []
    first_indices = {}
    for index, signature in enumerate(signatures):
        first_indices.setdefault(signature, index)
    for signature, occurrence_count in sorted(
        signature_counts.items(),
        key=lambda item: first_indices[item[0]],
    ):
        entry = {
            'edge_set_sha256': _edge_set_digest(signature),
            'edge_count': len(signature),
            'occurrence_count': int(occurrence_count),
            'first_snapshot_index': int(first_indices[signature]),
        }
        if include_edge_sets:
            entry['edges'] = [list(edge) for edge in signature]
        unique_catalog.append(entry)

    added_counts: List[float] = []
    removed_counts: List[float] = []
    symmetric_difference_counts: List[float] = []
    jaccard_distances: List[float] = []
    edge_mass_churn_rates: List[float] = []
    transition_records = []
    topology_switches = 0
    node_count_changes = 0

    for current_idx in range(1, total_snapshots):
        previous = edge_counters[current_idx - 1]
        current = edge_counters[current_idx]
        added = _counter_difference_size(current, previous)
        removed = _counter_difference_size(previous, current)
        symmetric_difference = added + removed
        union_size = _multiset_union_size(previous, current)
        jaccard_distance = (
            float(symmetric_difference / union_size) if union_size > 0 else 0.0
        )
        previous_edge_count = int(sum(previous.values()))
        current_edge_count = int(sum(current.values()))
        total_edge_mass = previous_edge_count + current_edge_count
        edge_mass_churn_rate = (
            float(symmetric_difference / total_edge_mass)
            if total_edge_mass > 0
            else 0.0
        )
        changed = signatures[current_idx] != signatures[current_idx - 1]
        if changed:
            topology_switches += 1
        if node_counts[current_idx] != node_counts[current_idx - 1]:
            node_count_changes += 1

        added_counts.append(float(added))
        removed_counts.append(float(removed))
        symmetric_difference_counts.append(float(symmetric_difference))
        jaccard_distances.append(jaccard_distance)
        edge_mass_churn_rates.append(edge_mass_churn_rate)

        if changed and len(transition_records) < max_transition_records:
            record = {
                'from_snapshot_index': current_idx - 1,
                'to_snapshot_index': current_idx,
                'previous_edge_count': previous_edge_count,
                'current_edge_count': current_edge_count,
                'added_edges': added,
                'removed_edges': removed,
                'symmetric_difference': symmetric_difference,
                'multiset_jaccard_distance': jaccard_distance,
                'edge_mass_churn_rate': edge_mass_churn_rate,
            }
            if timestamps is not None:
                previous_timestamp = int(timestamps[current_idx - 1])
                current_timestamp = int(timestamps[current_idx])
                record.update(
                    {
                        'from_timestamp': previous_timestamp,
                        'to_timestamp': current_timestamp,
                        'timestamp_gap': current_timestamp - previous_timestamp,
                    }
                )
            transition_records.append(record)

    finalized_edge_data = _finalize_edge_data_stats(
        edge_data_stats,
        total_snapshots,
    )
    weight_like_keys = sorted(
        key
        for key in finalized_edge_data
        if key == 'w' or 'weight' in key.lower()
    )

    return {
        'total_snapshots': total_snapshots,
        'unique_edge_sets': len(signature_counts),
        'topology_switches': topology_switches,
        'fixed_edge_topology': topology_switches == 0,
        'transition_count': max(total_snapshots - 1, 0),
        'node_count': {
            'unique_values': sorted(set(node_counts)),
            'changes': node_count_changes,
        },
        'edge_count': {
            'unique_values': sorted(set(edge_counts)),
            'minimum': int(min(edge_counts)) if edge_counts else 0,
            'maximum': int(max(edge_counts)) if edge_counts else 0,
            'mean': _safe_mean(edge_counts),
        },
        'edge_data': {
            'snapshots_with_any_edge_data': snapshots_with_any_edge_data,
            'snapshot_fraction_with_any_edge_data': (
                float(snapshots_with_any_edge_data / total_snapshots)
                if total_snapshots > 0
                else 0.0
            ),
            'keys': finalized_edge_data,
        },
        'edge_weight_availability': {
            'available': bool(weight_like_keys),
            'weight_like_keys': weight_like_keys,
            'snapshots_with_weight_like_data': snapshots_with_weight_like_data,
            'snapshot_fraction': (
                float(snapshots_with_weight_like_data / total_snapshots)
                if total_snapshots > 0
                else 0.0
            ),
            'standard_weight_key_available': 'weight' in finalized_edge_data,
        },
        'edge_churn': {
            'changed_transition_count': topology_switches,
            'total_added_edges': int(sum(added_counts)),
            'total_removed_edges': int(sum(removed_counts)),
            'mean_added_edges_per_transition': _safe_mean(added_counts),
            'mean_removed_edges_per_transition': _safe_mean(removed_counts),
            'mean_symmetric_difference': _safe_mean(symmetric_difference_counts),
            'max_symmetric_difference': int(_safe_max(symmetric_difference_counts)),
            'mean_multiset_jaccard_distance': _safe_mean(jaccard_distances),
            'max_multiset_jaccard_distance': _safe_max(jaccard_distances),
            'mean_edge_mass_churn_rate': _safe_mean(edge_mass_churn_rates),
            'max_edge_mass_churn_rate': _safe_max(edge_mass_churn_rates),
            'changed_transition_records': transition_records,
            'changed_transition_records_truncated': (
                topology_switches > len(transition_records)
            ),
        },
        'unique_edge_set_catalog': unique_catalog,
        'provenance': {
            'edge_source': 'serialized_dgl_snapshots_only',
            'self_loops_added_by_audit': False,
            'catalog_edges_added_by_audit': False,
            'parallel_edges_retained': True,
        },
    }


def _load_samples(sample_dir: Path):
    with (sample_dir / 'train_samples.pkl').open('rb') as file:
        train_samples = pickle.load(file)
    with (sample_dir / 'test_samples.pkl').open('rb') as file:
        test_samples = pickle.load(file)
    return train_samples, test_samples


def audit_samples(
    samples,
    max_transition_records: int = 100,
    include_edge_sets: bool = False,
) -> Dict[str, object]:
    ordered_samples = sorted(samples, key=lambda sample: int(sample[0]))
    timestamps = [int(sample[0]) for sample in ordered_samples]
    graphs = [sample[1] for sample in ordered_samples]
    report = audit_graph_sequence(
        graphs,
        timestamps=timestamps,
        max_transition_records=max_transition_records,
        include_edge_sets=include_edge_sets,
    )
    report['timestamp'] = {
        'minimum': min(timestamps) if timestamps else None,
        'maximum': max(timestamps) if timestamps else None,
        'duplicate_count': len(timestamps) - len(set(timestamps)),
        'non_monotonic_after_sort': sum(
            timestamps[idx] < timestamps[idx - 1]
            for idx in range(1, len(timestamps))
        ),
    }
    return report


def audit_windows(
    windows,
    max_window_records: int = 100,
) -> Dict[str, object]:
    """Summarize topology changes inside the exact windows consumed by a model.

    A raw sample sequence may contain edge changes that no training or case
    window ever crosses.  This audit therefore reports window-level exposure in
    addition to the dataset-level snapshot audit.  It accepts ``DSBGWindow``
    objects by interface rather than importing the dataset module.
    """

    if max_window_records < 0:
        raise ValueError('max_window_records must be non-negative')

    changed_window_records = []
    windows_with_changes = 0
    changed_transitions = 0
    total_transitions = 0

    for window_idx, window in enumerate(windows):
        graphs = list(window.graphs)
        signatures = [canonical_edge_multiset(graph) for graph in graphs]
        switch_indices = [
            step_idx
            for step_idx in range(1, len(signatures))
            if signatures[step_idx] != signatures[step_idx - 1]
        ]
        transition_count = max(len(signatures) - 1, 0)
        total_transitions += transition_count
        changed_transitions += len(switch_indices)
        if not switch_indices:
            continue

        windows_with_changes += 1
        if len(changed_window_records) >= max_window_records:
            continue
        timestamps = [int(value) for value in getattr(window, 'timestamps', [])]
        record = {
            'window_index': int(window_idx),
            'case_id': getattr(window, 'case_id', None),
            'case_timestamp': getattr(window, 'case_timestamp', None),
            'topology_switches': len(switch_indices),
            'switch_step_indices': switch_indices,
            'timestamps': timestamps,
        }
        changed_window_records.append(record)

    total_windows = len(windows)
    return {
        'total_windows': total_windows,
        'windows_with_topology_changes': windows_with_changes,
        'window_fraction_with_topology_changes': (
            float(windows_with_changes / total_windows) if total_windows > 0 else 0.0
        ),
        'total_within_window_transitions': total_transitions,
        'changed_within_window_transitions': changed_transitions,
        'fixed_topology_within_every_window': windows_with_changes == 0,
        'changed_window_records': changed_window_records,
        'changed_window_records_truncated': (
            windows_with_changes > len(changed_window_records)
        ),
        'provenance': {
            'edge_source': 'graphs_in_exact_model_windows',
            'self_loops_added_by_audit': False,
            'catalog_edges_added_by_audit': False,
        },
    }


def window_has_topology_change(window) -> bool:
    signatures = [canonical_edge_multiset(graph) for graph in window.graphs]
    return any(
        signatures[step_idx] != signatures[step_idx - 1]
        for step_idx in range(1, len(signatures))
    )


def _resolve_repo_path(path_value: str, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else repo_root / path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Audit actual topology dynamics in REACR-RCA DGL snapshots'
    )
    parser.add_argument(
        '--dataset',
        default='D1',
        help='dataset/config stem, e.g. D1, D1_weighted, D2, SN, or TT',
    )
    parser.add_argument(
        '--config',
        default='',
        help='YAML config path; defaults to config/<dataset>.yaml',
    )
    parser.add_argument(
        '--sample-dir',
        default='',
        help='override config path.sample_dir',
    )
    parser.add_argument('--output', default='', help='optional JSON output path')
    parser.add_argument('--max-transition-records', type=int, default=100)
    parser.add_argument(
        '--include-edge-sets',
        action='store_true',
        help='include exact edge endpoint lists in the unique-set catalogue',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = _resolve_repo_path(
        args.config or f'config/{args.dataset}.yaml',
        repo_root,
    )
    with config_path.open('r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    sample_dir_value = args.sample_dir or config['path']['sample_dir']
    sample_dir = _resolve_repo_path(sample_dir_value, repo_root)
    train_samples, test_samples = _load_samples(sample_dir)
    combined_samples = list(train_samples) + list(test_samples)

    report = {
        'dataset': args.dataset,
        'config': str(config_path),
        'sample_dir': str(sample_dir),
        'train': audit_samples(
            train_samples,
            max_transition_records=args.max_transition_records,
            include_edge_sets=args.include_edge_sets,
        ),
        'test': audit_samples(
            test_samples,
            max_transition_records=args.max_transition_records,
            include_edge_sets=args.include_edge_sets,
        ),
        'combined': audit_samples(
            combined_samples,
            max_transition_records=args.max_transition_records,
            include_edge_sets=args.include_edge_sets,
        ),
    }

    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        output_path = _resolve_repo_path(args.output, repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + '\n', encoding='utf-8')
    print(payload)


if __name__ == '__main__':
    main()
