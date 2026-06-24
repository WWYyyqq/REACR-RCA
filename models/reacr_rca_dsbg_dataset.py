import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dgl
import torch
from torch.utils.data import DataLoader, Dataset


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
) -> List[DSBGWindow]:
    sorted_samples = _sort_samples(samples)
    if len(sorted_samples) == 0:
        return []

    windows: List[DSBGWindow] = []
    if len(sorted_samples) < window_size:
        windows.append(_series_to_window(_pad_series(sorted_samples, window_size)))
        return windows

    for end_idx in range(window_size, len(sorted_samples) + 1, max(stride, 1)):
        series = sorted_samples[end_idx - window_size:end_idx]
        windows.append(_series_to_window(series))
        if max_windows is not None and len(windows) >= max_windows:
            break
    return windows


def build_case_windows(
    cases,
    samples: Sequence[Sample],
    node_hash: Dict[str, int],
    window_size: int,
    max_cases: Optional[int] = None,
) -> List[DSBGWindow]:
    sorted_samples = _sort_samples(samples)
    node_num = sorted_samples[0][1].num_nodes() if len(sorted_samples) > 0 else 0
    windows: List[DSBGWindow] = []
    group_hash: Dict[str, int] = {}

    for row_case_id, case in cases.reset_index(drop=True).iterrows():
        case_id = int(case['case_id']) if 'case_id' in case.index else int(row_case_id)
        start_ts = case['timestamp'] - (window_size * 60) / 2
        end_ts = case['timestamp'] + (window_size * 60) / 2

        label = torch.zeros(node_num, dtype=torch.float32)
        for node, node_idx in node_hash.items():
            if node.startswith(case['cmdb_id']):
                label[node_idx] = 1.0
        group_key = str(case['cmdb_id'])
        if 'failure_type' in case.index and case.get('failure_type') == case.get('failure_type'):
            group_key = f"{group_key}|{case.get('failure_type')}"
        if group_key not in group_hash:
            group_hash[group_key] = len(group_hash)

        series = [sample for sample in sorted_samples if start_ts <= sample[0] < end_ts]
        if len(series) == 0 and len(sorted_samples) > 0:
            nearest = min(sorted_samples, key=lambda x: abs(x[0] - case['timestamp']))
            series = [nearest]

        window = _series_to_window(
            _pad_series(series, window_size),
            label=label,
            case_id=case_id,
            case_timestamp=int(case['timestamp']),
            group_id=group_hash[group_key],
        )
        windows.append(window)
        if max_cases is not None and len(windows) >= max_cases:
            break
    return windows


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
) -> DataLoader:
    dataset = DSBGWindowDataset(windows, with_labels=with_labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_dsbg_windows)


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
