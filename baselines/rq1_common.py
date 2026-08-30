from __future__ import annotations

import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import yaml

from models.reacr_rca_dsbg_dataset import build_case_windows
from reacr_rca_pipeline import split_cases_for_training
from utils.public_functions import load_init, load_samples


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CaseTensor:
    case_id: int
    features: torch.Tensor  # [W,N,F]
    adjacency: torch.Tensor  # [N,N], row-normalized incoming adjacency
    label: torch.Tensor  # [N]


@dataclass
class RQ1Data:
    dataset: str
    train: List[CaseTensor]
    validation: List[CaseTensor]
    test: List[CaseTensor]
    normal_mean: torch.Tensor  # [N,F]
    normal_std: torch.Tensor  # [N,F]
    modality_indices: Dict[str, List[int]]
    node_names: List[str]
    split_meta: Dict[str, object]
    config: Dict[str, object]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _indices_from_spans(spans) -> List[int]:
    values: List[int] = []
    for start, end in spans:
        values.extend(range(int(start), int(end) + 1))
    return values


def _window_adjacency(graphs: Sequence, node_num: int) -> torch.Tensor:
    # DGL stores source -> destination. Rows below represent destinations so
    # adj @ h aggregates callers/callees observed as incoming evidence.
    adj = torch.zeros(node_num, node_num, dtype=torch.float32)
    for graph in graphs:
        src, dst = graph.edges()
        if src.numel() == 0:
            continue
        weight = torch.ones(src.numel(), dtype=torch.float32)
        if 'weight' in graph.edata:
            weight = graph.edata['weight'].detach().cpu().float().reshape(-1)
        adj.index_put_((dst.cpu().long(), src.cpu().long()), weight, accumulate=True)
    adj /= max(len(graphs), 1)
    adj += torch.eye(node_num)
    return adj / adj.sum(dim=1, keepdim=True).clamp_min(1e-8)


def _convert_windows(windows) -> List[CaseTensor]:
    result: List[CaseTensor] = []
    for window in windows:
        result.append(
            CaseTensor(
                case_id=int(window.case_id),
                features=window.features.detach().cpu().float(),
                adjacency=_window_adjacency(window.graphs, int(window.features.shape[1])),
                label=window.label.detach().cpu().float(),
            )
        )
    return result


def _normal_statistics(samples) -> tuple[torch.Tensor, torch.Tensor]:
    count = 0
    total = None
    total_sq = None
    for _, _, feature in samples:
        value = feature.detach().cpu().double()
        total = value.clone() if total is None else total + value
        total_sq = value.square() if total_sq is None else total_sq + value.square()
        count += 1
    if count == 0:
        raise ValueError('normal sample collection is empty')
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float()


def load_rq1_data(dataset: str, feedback_ratio: float = 0.3, test_start_ratio: float = 0.5) -> RQ1Data:
    config_path = ROOT / 'config' / f'{dataset}.yaml'
    config = yaml.load(config_path.read_text(encoding='utf-8-sig'), Loader=yaml.FullLoader)
    paths = config['path']
    train_samples, test_samples = load_samples(str(ROOT / paths['sample_dir']))
    cases = pd.read_csv(ROOT / paths['case_dir'])
    node_hash, _, _ = load_init(str(ROOT / paths['graph_dir']))
    data_settings = config.get('data_settings', {})
    default_ratio = config.get('feedback', {}).get('sample_num', feedback_ratio)
    if not isinstance(default_ratio, float):
        default_ratio = feedback_ratio
    train_cases, val_cases, test_cases, split_meta = split_cases_for_training(
        cases,
        feedback_ratio,
        float(default_ratio),
        protocol=data_settings.get('case_split_protocol', 'time_strict'),
        test_start_ratio=test_start_ratio,
        group_column=data_settings.get('case_group_column'),
    )
    kwargs = {
        'sample_interval_seconds': float(data_settings.get('sample_interval_seconds', 60.0)),
        'case_pre_seconds': data_settings.get('case_pre_seconds'),
        'case_post_seconds': data_settings.get('case_post_seconds'),
        'max_gap_seconds': data_settings.get('max_gap_seconds'),
        'incomplete_policy': data_settings.get('case_incomplete_policy', 'pad'),
    }
    window_size = int(data_settings.get('window_size', 10))
    build = lambda subset: _convert_windows(
        build_case_windows(subset, test_samples, node_hash, window_size=window_size, **kwargs)
    )
    normal_mean, normal_std = _normal_statistics(train_samples)
    modality_indices = {
        name: _indices_from_spans(spans)
        for name, spans in config.get('modality_spans', {}).items()
    }
    index_to_name = {int(index): str(name) for name, index in node_hash.items()}
    node_names = [index_to_name[index] for index in range(len(index_to_name))]
    return RQ1Data(
        dataset=dataset,
        train=build(train_cases),
        validation=build(val_cases),
        test=build(test_cases),
        normal_mean=normal_mean,
        normal_std=normal_std,
        modality_indices=modality_indices,
        node_names=node_names,
        split_meta=split_meta,
        config=config,
    )


def batch_cases(cases: Sequence[CaseTensor], indices: Sequence[int], device: torch.device):
    selected = [cases[int(index)] for index in indices]
    return {
        'features': torch.stack([case.features for case in selected]).to(device),
        'adjacency': torch.stack([case.adjacency for case in selected]).to(device),
        'labels': torch.stack([case.label for case in selected]).to(device),
        'case_ids': [case.case_id for case in selected],
    }


def iter_batches(cases: Sequence[CaseTensor], batch_size: int, shuffle: bool, seed: int):
    indices = np.arange(len(cases))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


def bag_classification_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_prob = torch.log_softmax(scores, dim=-1)
    positive_log_prob = torch.logsumexp(log_prob.masked_fill(labels <= 0, -1e9), dim=-1)
    return -positive_log_prob.mean()


def topk_metrics(scores: torch.Tensor, labels: torch.Tensor, ks=(1, 3, 5)) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for k in ks:
        effective_k = min(int(k), int(scores.shape[1]))
        top = scores.topk(effective_k, dim=1).indices
        hit = torch.gather(labels, 1, top).gt(0).any(dim=1).float().mean()
        result[f'A@{k}'] = float(hit.item() * 100.0)
    return result


def validation_objective(metrics: Dict[str, float]) -> float:
    return 3.0 * metrics['A@1'] + 2.0 * metrics['A@3'] + metrics['A@5']


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
