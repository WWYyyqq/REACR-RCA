import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


Sample = Tuple[int, dgl.DGLGraph, torch.Tensor]


@dataclass
class DSBGWindow:
    timestamps: List[int]
    graphs: List[dgl.DGLGraph]
    features: torch.Tensor  # [W, N, F]
    label: Optional[torch.Tensor] = None  # [N]
    case_id: Optional[int] = None
    case_timestamp: Optional[int] = None
    group_id: Optional[int] = None


def _sort_samples(samples: Sequence[Sample]) -> List[Sample]:
    return sorted(samples, key=lambda x: x[0])


def _split_contiguous_samples(
    samples: Sequence[Sample],
    max_gap_seconds: Optional[float],
) -> List[List[Sample]]:
    """Split observations without synthesising continuity across time gaps."""

    sorted_samples = _sort_samples(samples)
    if len(sorted_samples) == 0:
        return []
    if max_gap_seconds is None or max_gap_seconds <= 0:
        return [sorted_samples]

    segments: List[List[Sample]] = [[sorted_samples[0]]]
    for sample in sorted_samples[1:]:
        previous = segments[-1][-1]
        if float(sample[0]) - float(previous[0]) > float(max_gap_seconds):
            segments.append([sample])
        else:
            segments[-1].append(sample)
    return segments


def _pad_series(series: List[Sample], window_size: int) -> List[Sample]:
    if len(series) == 0:
        raise ValueError('cannot build a DSBG window from an empty series')
    if len(series) >= window_size:
        return series[-window_size:]
    return [series[0]] * (window_size - len(series)) + series


def _series_to_window(
    series: Sequence[Sample],
    label: Optional[torch.Tensor] = None,
    case_id: Optional[int] = None,
    case_timestamp: Optional[int] = None,
    group_id: Optional[int] = None,
) -> DSBGWindow:
    timestamps = [sample[0] for sample in series]
    graphs = [sample[1] for sample in series]
    features = torch.stack([sample[2] for sample in series], dim=0)
    return DSBGWindow(
        timestamps=timestamps,
        graphs=graphs,
        features=features,
        label=label,
        case_id=case_id,
        case_timestamp=case_timestamp,
        group_id=group_id,
    )


def build_sliding_windows(
    samples: Sequence[Sample],
    window_size: int,
    stride: int = 1,
    max_windows: Optional[int] = None,
    max_gap_seconds: Optional[float] = None,
    incomplete_policy: str = 'pad',
) -> List[DSBGWindow]:
    if incomplete_policy not in {'pad', 'drop', 'error'}:
        raise ValueError(
            f'unsupported incomplete_policy={incomplete_policy!r}; '
            "expected 'pad', 'drop', or 'error'"
        )

    windows: List[DSBGWindow] = []
    for segment in _split_contiguous_samples(samples, max_gap_seconds):
        if len(segment) < window_size:
            if incomplete_policy == 'pad':
                windows.append(_series_to_window(_pad_series(segment, window_size)))
            elif incomplete_policy == 'error':
                raise ValueError(
                    f'contiguous segment has {len(segment)} samples, fewer than '
                    f'window_size={window_size}'
                )
            if max_windows is not None and len(windows) >= max_windows:
                return windows
            continue

        for end_idx in range(window_size, len(segment) + 1, max(stride, 1)):
            series = segment[end_idx - window_size:end_idx]
            windows.append(_series_to_window(series))
            if max_windows is not None and len(windows) >= max_windows:
                return windows
    return windows


def build_case_windows(
    cases,
    samples: Sequence[Sample],
    node_hash: Dict[str, int],
    window_size: int,
    max_cases: Optional[int] = None,
    unmatched_label_policy: str = 'error',
    label_separator: str = '-',
    sample_interval_seconds: float = 60.0,
    case_pre_seconds: Optional[float] = None,
    case_post_seconds: Optional[float] = None,
    max_gap_seconds: Optional[float] = None,
    incomplete_policy: str = 'pad',
) -> List[DSBGWindow]:
    sorted_samples = _sort_samples(samples)
    node_num = sorted_samples[0][1].num_nodes() if len(sorted_samples) > 0 else 0
    windows: List[DSBGWindow] = []
    group_hash: Dict[str, int] = {}
    segments = _split_contiguous_samples(sorted_samples, max_gap_seconds)
    if incomplete_policy not in {'pad', 'drop', 'error'}:
        raise ValueError(
            f'unsupported incomplete_policy={incomplete_policy!r}; '
            "expected 'pad', 'drop', or 'error'"
        )

    for row_case_id, case in cases.reset_index(drop=True).iterrows():
        case_id = int(case['case_id']) if 'case_id' in case.index else int(row_case_id)
        if case_pre_seconds is None and case_post_seconds is None:
            half_window = (window_size * float(sample_interval_seconds)) / 2.0
            start_ts = float(case['timestamp']) - half_window
            end_ts = float(case['timestamp']) + half_window
        else:
            pre_seconds = 0.0 if case_pre_seconds is None else float(case_pre_seconds)
            post_seconds = 0.0 if case_post_seconds is None else float(case_post_seconds)
            start_ts = float(case['timestamp']) - pre_seconds
            end_ts = float(case['timestamp']) + post_seconds
            if end_ts <= start_ts:
                raise ValueError(
                    f'invalid case interval [{start_ts}, {end_ts}) for case_id={case_id}'
                )

        label = build_root_cause_label(
            root_label=case['cmdb_id'],
            node_hash=node_hash,
            node_num=node_num,
            unmatched_label_policy=unmatched_label_policy,
            label_separator=label_separator,
            case_id=case_id,
        )
        group_key = str(case['cmdb_id'])
        if 'failure_type' in case.index and case.get('failure_type') == case.get('failure_type'):
            group_key = f"{group_key}|{case.get('failure_type')}"
        if group_key not in group_hash:
            group_hash[group_key] = len(group_hash)

        candidate_segment: Sequence[Sample] = sorted_samples
        if segments:
            candidate_segment = min(
                segments,
                key=lambda segment: min(
                    abs(float(sample[0]) - float(case['timestamp']))
                    for sample in segment
                ),
            )
        series = [
            sample
            for sample in candidate_segment
            if start_ts <= float(sample[0]) < end_ts
        ]
        if len(series) == 0 and len(candidate_segment) > 0:
            nearest = min(
                candidate_segment,
                key=lambda x: abs(float(x[0]) - float(case['timestamp'])),
            )
            series = [nearest]
        if len(series) < window_size:
            if incomplete_policy == 'drop':
                continue
            if incomplete_policy == 'error':
                raise ValueError(
                    f'case_id={case_id} has only {len(series)} observations in '
                    f'[{start_ts}, {end_ts}), fewer than window_size={window_size}'
                )
            series = _pad_series(series, window_size)
        else:
            series = series[-window_size:]

        window = _series_to_window(
            series,
            label=label,
            case_id=case_id,
            case_timestamp=int(case['timestamp']),
            group_id=group_hash[group_key],
        )
        windows.append(window)
        if max_cases is not None and len(windows) >= max_cases:
            break
    return windows


def resolve_root_cause_node_indices(
    root_label,
    node_hash: Dict[str, int],
    label_separator: str = '-',
) -> List[int]:
    """Resolve a root label to exact and label-plus-separator graph nodes."""
    if root_label is None:
        raise ValueError('root-cause label cannot be None')
    normalized_label = str(root_label).strip()
    if not normalized_label:
        raise ValueError('root-cause label cannot be empty')
    if not label_separator:
        raise ValueError('label_separator cannot be empty')
    return sorted({
        int(node_idx)
        for node, node_idx in node_hash.items()
        if root_label_matches_node_name(
            normalized_label,
            node,
            label_separator=label_separator,
        )
    })


def root_label_matches_node_name(
    root_label,
    node_name,
    label_separator: str = '-',
) -> bool:
    """Boundary-safe match shared by training and every evaluation path."""

    if not label_separator:
        raise ValueError('label_separator cannot be empty')
    normalized_label = str(root_label).strip()
    normalized_node = str(node_name).strip()
    if not normalized_label:
        return False
    return (
        normalized_node == normalized_label
        or normalized_node.startswith(f'{normalized_label}{label_separator}')
    )


def build_root_cause_label(
    root_label,
    node_hash: Dict[str, int],
    node_num: int,
    unmatched_label_policy: str = 'error',
    label_separator: str = '-',
    case_id: Optional[int] = None,
) -> torch.Tensor:
    """Build a multi-positive label with an explicit zero-match policy."""
    supported_policies = {'error', 'warn', 'ignore'}
    if unmatched_label_policy not in supported_policies:
        raise ValueError(
            f'Unsupported unmatched_label_policy={unmatched_label_policy!r}; '
            f'expected one of {sorted(supported_policies)}.'
        )
    matched_indices = resolve_root_cause_node_indices(root_label, node_hash, label_separator)
    if not matched_indices:
        case_context = '' if case_id is None else f' for case_id={case_id}'
        message = (
            f'No graph node matches root-cause label {str(root_label)!r}{case_context}. '
            f'Expected an exact node name or a name beginning with '
            f'{str(root_label) + label_separator!r}.'
        )
        if unmatched_label_policy == 'error':
            raise ValueError(message)
        if unmatched_label_policy == 'warn':
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    label = torch.zeros(int(node_num), dtype=torch.float32)
    for node_idx in matched_indices:
        if node_idx < 0 or node_idx >= node_num:
            raise ValueError(
                f'node_hash index {node_idx} is outside [0, {node_num}) '
                f'for root-cause label {str(root_label)!r}.'
            )
        label[node_idx] = 1.0
    return label


class DSBGWindowDataset(Dataset):
    def __init__(self, windows: Sequence[DSBGWindow], with_labels: bool):
        self.windows = list(windows)
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> DSBGWindow:
        return self.windows[idx]


def collate_dsbg_windows(batch: Sequence[DSBGWindow]):
    if len(batch) == 0:
        raise ValueError('empty batch')

    window_size = len(batch[0].graphs)
    batched_graphs = []
    for step_idx in range(window_size):
        step_graphs = [sample.graphs[step_idx] for sample in batch]
        batched_graphs.append(dgl.batch(step_graphs))

    features = torch.stack([sample.features for sample in batch], dim=0)  # [B, W, N, F]
    timestamps = torch.tensor([sample.timestamps for sample in batch], dtype=torch.long)
    case_ids = torch.tensor(
        [-1 if sample.case_id is None else sample.case_id for sample in batch],
        dtype=torch.long,
    )
    case_timestamps = torch.tensor(
        [-1 if sample.case_timestamp is None else sample.case_timestamp for sample in batch],
        dtype=torch.long,
    )
    group_ids = torch.tensor(
        [-1 if sample.group_id is None else sample.group_id for sample in batch],
        dtype=torch.long,
    )

    labels = None
    if batch[0].label is not None:
        labels = torch.stack([sample.label for sample in batch], dim=0)

    return {
        'graphs': batched_graphs,
        'features': features,
        'timestamps': timestamps,
        'labels': labels,
        'case_ids': case_ids,
        'case_timestamps': case_timestamps,
        'group_ids': group_ids,
        'window_size': window_size,
        'batch_size': len(batch),
        'node_num': features.shape[2],
        'feat_dim': features.shape[3],
    }


def create_dsbg_dataloader(
    windows: Sequence[DSBGWindow],
    batch_size: int,
    shuffle: bool = False,
    with_labels: bool = False,
    sample_weights: Optional[Sequence[float]] = None,
) -> DataLoader:
    dataset = DSBGWindowDataset(windows, with_labels=with_labels)
    sampler = None
    if sample_weights is not None:
        if len(sample_weights) != len(dataset):
            raise ValueError(
                f'sample_weights length {len(sample_weights)} does not match '
                f'dataset length {len(dataset)}'
            )
        weights = torch.as_tensor(sample_weights, dtype=torch.double)
        if torch.any(weights <= 0):
            raise ValueError('all sample_weights must be positive')
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_dsbg_windows,
    )


def summarize_windows(windows: Sequence[DSBGWindow]) -> Dict[str, float]:
    if len(windows) == 0:
        return {'num_windows': 0}

    feat_shape = list(windows[0].features.shape)
    positives = 0
    if windows[0].label is not None:
        positives = int(sum(window.label.sum().item() for window in windows))

    return {
        'num_windows': len(windows),
        'window_size': feat_shape[0],
        'node_num': feat_shape[1],
        'feat_dim': feat_shape[2],
        'positive_labels': positives,
    }


def batch_summary(batch) -> Dict[str, object]:
    labels_shape = None if batch['labels'] is None else list(batch['labels'].shape)
    return {
        'num_graph_steps': len(batch['graphs']),
        'batched_graph_nodes_per_step': [int(g.num_nodes()) for g in batch['graphs']],
        'features_shape': list(batch['features'].shape),
        'timestamps_shape': list(batch['timestamps'].shape),
        'labels_shape': labels_shape,
        'window_size': int(batch['window_size']),
        'batch_size': int(batch['batch_size']),
        'node_num': int(batch['node_num']),
        'feat_dim': int(batch['feat_dim']),
    }
