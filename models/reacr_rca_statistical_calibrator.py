"""Service-independent statistical evidence calibration for zero-shot RCA.

SN/TT session-level splits contain disjoint root-cause services. This module
learns one shared node classifier from temporal statistics instead of service
identities, then fuses it with the dynamic-graph neural score.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def extract_temporal_statistics(features) -> np.ndarray:
    """Return [B, N, 6F] service-independent temporal summary features."""
    if torch.is_tensor(features):
        values = features.detach().cpu().float().numpy()
    else:
        values = np.asarray(features, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(f'features must have shape [B,W,N,F], got {values.shape}')
    node_series = np.transpose(values, (0, 2, 1, 3))
    statistics = np.concatenate(
        [
            node_series.mean(axis=2),
            node_series.std(axis=2),
            node_series.max(axis=2),
            node_series.min(axis=2),
            np.ptp(node_series, axis=2),
            node_series[:, :, -1] - node_series[:, :, 0],
        ],
        axis=-1,
    )
    return np.nan_to_num(statistics, nan=0.0, posinf=1e30, neginf=-1e30).astype(
        np.float32, copy=False
    )


def standardize_case_scores(scores, eps: float = 1e-8) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'scores must have shape [B,N], got {values.shape}')
    centered = values - values.mean(axis=1, keepdims=True)
    scale = centered.std(axis=1, keepdims=True)
    return np.nan_to_num(centered / np.maximum(scale, eps), nan=0.0)


def fuse_case_scores(base_scores, statistical_scores, base_weight: float) -> np.ndarray:
    weight = float(base_weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError('base_weight must be in [0,1]')
    base = standardize_case_scores(base_scores)
    statistical = standardize_case_scores(statistical_scores)
    if base.shape != statistical.shape:
        raise ValueError(
            f'base/statistical score shapes differ: {base.shape} vs {statistical.shape}'
        )
    return weight * base + (1.0 - weight) * statistical


class StatisticalEvidenceCalibrator:
    """Fit a shared node-level classifier without node identity features."""

    def __init__(
        self,
        method: str = 'extra_trees',
        seed: int = 0,
        n_estimators: int = 500,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 1,
        max_features: float = 1.0,
        svc_c: float = 0.03,
    ):
        if method not in {'extra_trees', 'linear_svc'}:
            raise ValueError(f'unsupported statistical calibrator method: {method}')
        self.method = method
        self.seed = int(seed)
        self.n_estimators = int(n_estimators)
        self.max_depth = None if max_depth in (None, 0) else int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = float(max_features)
        self.svc_c = float(svc_c)
        self.estimator = None
        self.feature_dim = None

    def _build_estimator(self):
        if self.method == 'extra_trees':
            from sklearn.ensemble import ExtraTreesClassifier

            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight='balanced',
                random_state=self.seed,
                n_jobs=-1,
            )
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC

        return make_pipeline(
            StandardScaler(),
            SVC(C=self.svc_c, kernel='linear', class_weight='balanced'),
        )

    def fit(self, features, labels):
        statistics = extract_temporal_statistics(features)
        target = np.asarray(
            labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
        )
        if target.shape != statistics.shape[:2]:
            raise ValueError(
                f'labels must have shape {statistics.shape[:2]}, got {target.shape}'
            )
        flat_x = statistics.reshape(-1, statistics.shape[-1])
        flat_y = (target.reshape(-1) > 0).astype(np.int64)
        if np.unique(flat_y).size != 2:
            raise ValueError('calibrator requires positive and negative nodes')
        self.estimator = self._build_estimator()
        self.estimator.fit(flat_x, flat_y)
        self.feature_dim = int(flat_x.shape[-1])
        return self

    def predict(self, features) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError('statistical calibrator has not been fitted')
        statistics = extract_temporal_statistics(features)
        if statistics.shape[-1] != self.feature_dim:
            raise ValueError(
                f'expected feature dim {self.feature_dim}, got {statistics.shape[-1]}'
            )
        flat = statistics.reshape(-1, statistics.shape[-1])
        if hasattr(self.estimator, 'predict_proba'):
            score = self.estimator.predict_proba(flat)[:, 1]
        else:
            score = self.estimator.decision_function(flat)
        return np.asarray(score).reshape(statistics.shape[:2])

    def fit_loader(self, dataloader):
        features = []
        labels = []
        for batch in dataloader:
            features.append(batch['features'].float().cpu())
            labels.append(batch['labels'].float().cpu())
        if not features:
            raise ValueError('cannot fit calibrator from an empty dataloader')
        return self.fit(torch.cat(features, dim=0), torch.cat(labels, dim=0))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('wb') as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        with Path(path).open('rb') as handle:
            calibrator = pickle.load(handle)
        if not isinstance(calibrator, cls):
            raise TypeError(f'{path} does not contain {cls.__name__}')
        return calibrator

    def config_dict(self) -> Dict[str, object]:
        return {
            'method': self.method,
            'seed': self.seed,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'svc_c': self.svc_c,
            'feature_dim': self.feature_dim,
        }


def predict_calibrated_scores(model, calibrator, dataloader, device, base_weight):
    from models.reacr_rca_trainer import (
        _move_graphs_to_device,
        observed_topology_change_mask,
    )

    model.eval()
    values = {
        'scores': [],
        'labels': [],
        'case_ids': [],
        'case_timestamps': [],
        'changed_mask': [],
    }
    with torch.no_grad():
        for batch in dataloader:
            cpu_features = batch['features'].float().cpu()
            graphs = _move_graphs_to_device(batch['graphs'], device)
            outputs = model(graphs, cpu_features.to(device))
            base = outputs['rank_output']['final_score'].detach().cpu().numpy()
            statistical = calibrator.predict(cpu_features)
            fused = fuse_case_scores(base, statistical, base_weight)
            values['scores'].append(torch.as_tensor(fused, dtype=torch.float32))
            values['labels'].append(batch['labels'].float().cpu())
            values['case_ids'].append(batch['case_ids'].cpu())
            values['case_timestamps'].append(batch['case_timestamps'].cpu())
            values['changed_mask'].append(
                observed_topology_change_mask(batch['graphs']).cpu()
            )
    if not values['scores']:
        return {
            'scores': torch.empty(0),
            'labels': torch.empty(0),
            'case_ids': torch.empty(0, dtype=torch.long),
            'case_timestamps': torch.empty(0, dtype=torch.long),
            'changed_mask': torch.empty(0, dtype=torch.bool),
        }
    return {key: torch.cat(items) for key, items in values.items()}


def evaluate_calibrated_bundle(bundle, ks=(1, 3, 5)):
    from models.reacr_rca_trainer import PairwiseRankingLoss, compute_topk_metrics

    scores = bundle['scores']
    labels = bundle['labels']
    changed = bundle['changed_mask'].bool()
    criterion = PairwiseRankingLoss()
    metrics = compute_topk_metrics(scores, labels, ks=ks) if len(scores) else {}
    metrics.pop('_sample_count', None)
    result = {
        'loss': float(criterion(scores, labels).item()) if len(scores) else 0.0,
        'metrics': metrics,
        'score_std': (
            float(scores.std(dim=1, unbiased=False).mean().item())
            if len(scores)
            else 0.0
        ),
        'topology_strata': {},
        'score_source': 'dynamic_neural_plus_statistical_calibrator',
    }
    for name, mask in (('changed', changed), ('stable', ~changed)):
        stratum = (
            compute_topk_metrics(scores[mask], labels[mask], ks=ks)
            if mask.any()
            else {}
        )
        stratum.pop('_sample_count', None)
        result['topology_strata'][name] = {
            'sample_count': int(mask.sum().item()),
            'metrics': stratum,
        }
    return result
