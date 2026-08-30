"""Validation-only statistical calibration search for locked REACR runs.

The search reads validation scores from the already completed five-seed REACR
neural runs. Test score files are not opened until a configuration has been
locked and the validation-search artifact has been written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from baselines.rq1_common import ROOT, load_rq1_data, save_json, topk_metrics, validation_objective
from models.reacr_rca_statistical_calibrator import StatisticalEvidenceCalibrator, fuse_case_scores


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SN', 'TT', 'D1_weighted'])
    parser.add_argument('--base-dir', default='res/reacr_rca_full_20260824')
    parser.add_argument('--variant', default='source_hypothesis_losses')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--output-dir', default='res/reacr_rq1_optimized_20260824')
    return parser.parse_args()


def feature_and_label(cases):
    return torch.stack([case.features for case in cases]), torch.stack([case.label for case in cases])


def read_score_matrix(path: Path, cases, node_names):
    by_case = {}
    name_to_index = {name: index for index, name in enumerate(node_names)}
    with path.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            score = np.full(len(node_names), -1e9, dtype=np.float64)
            for key, value in row.items():
                if not key.startswith('Top') or not value:
                    continue
                name, raw_score = value.rsplit(':', 1)
                score[name_to_index[name]] = float(raw_score)
            by_case[int(row['case_id'])] = score
    missing = [case.case_id for case in cases if case.case_id not in by_case]
    if missing:
        raise ValueError(f'{path} misses case ids {missing}')
    return np.stack([by_case[case.case_id] for case in cases])


def metric_bundle(scores, labels):
    return topk_metrics(torch.as_tensor(scores, dtype=torch.float32), labels)


def candidate_grid():
    values = []
    for depth in (None, 3, 5, 8):
        for leaf in (1, 2, 4):
            for max_features in (0.3, 0.6, 1.0):
                values.append({
                    'method': 'extra_trees', 'n_estimators': 300,
                    'max_depth': depth, 'min_samples_leaf': leaf,
                    'max_features': max_features, 'svc_c': 0.03, 'seed': 0,
                })
    for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        values.append({
            'method': 'linear_svc', 'n_estimators': 300,
            'max_depth': None, 'min_samples_leaf': 1,
            'max_features': 1.0, 'svc_c': c, 'seed': 0,
        })
    return values


def build_calibrator(config):
    return StatisticalEvidenceCalibrator(
        method=config['method'], seed=config['seed'],
        n_estimators=config['n_estimators'], max_depth=config['max_depth'],
        min_samples_leaf=config['min_samples_leaf'], max_features=config['max_features'],
        svc_c=config['svc_c'],
    )


def search_dataset(dataset, args, output):
    data = load_rq1_data(dataset)
    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    base_root = ROOT / args.base_dir / dataset / args.variant
    validation_base = {
        seed: read_score_matrix(base_root / f'seed{seed}' / 'validation_predictions.csv', data.validation, data.node_names)
        for seed in args.seeds
    }
    records = []
    weights = [round(value, 2) for value in np.arange(0.1, 1.0001, 0.05)]
    for candidate in candidate_grid():
        calibrator = build_calibrator(candidate).fit(train_x, train_y)
        statistical = calibrator.predict(val_x)
        for weight in weights:
            per_seed = []
            for seed in args.seeds:
                fused = fuse_case_scores(validation_base[seed], statistical, weight)
                per_seed.append(metric_bundle(fused, val_y))
            mean = {key: float(np.mean([value[key] for value in per_seed])) for key in ('A@1', 'A@3', 'A@5')}
            records.append({
                'calibrator': candidate, 'neural_base_weight': weight,
                'validation_mean': mean,
                'validation_objective': validation_objective(mean),
                'validation_per_seed': per_seed,
            })
    records.sort(
        key=lambda row: (
            row['validation_objective'], row['validation_mean']['A@1'],
            row['validation_mean']['A@3'], row['validation_mean']['A@5'],
            row['neural_base_weight'],
        ), reverse=True,
    )
    lock = records[0]
    audit = {
        'dataset': dataset,
        'selection_split': 'validation_only',
        'test_files_opened_during_search': False,
        'candidate_count': len(records),
        'locked': lock,
        'top_candidates': records[:20],
    }
    save_json(output / dataset / 'validation_search.json', audit)
    return data, train_x, train_y, lock


def locked_test(dataset, data, train_x, train_y, lock, args, output):
    # This function is deliberately called only after validation_search.json exists.
    # Once the hyperparameters have been frozen on the validation split, refit the
    # lightweight calibrator on all development cases.  This is the usual final-fit
    # protocol and is especially important for SN/TT, whose validation folds are
    # deliberately small.  Test labels remain completely unseen.
    val_x, val_y = feature_and_label(data.validation)
    development_x = torch.cat([train_x, val_x], dim=0)
    development_y = torch.cat([train_y, val_y], dim=0)
    calibrator = build_calibrator(lock['calibrator']).fit(development_x, development_y)
    test_x, test_y = feature_and_label(data.test)
    statistical = calibrator.predict(test_x)
    base_root = ROOT / args.base_dir / dataset / args.variant
    per_seed = []
    for seed in args.seeds:
        base = read_score_matrix(base_root / f'seed{seed}' / 'eval_predictions.csv', data.test, data.node_names)
        fused = fuse_case_scores(base, statistical, lock['neural_base_weight'])
        per_seed.append(metric_bundle(fused, test_y))
    aggregate = {}
    for key in ('A@1', 'A@3', 'A@5'):
        values = np.asarray([row[key] for row in per_seed], dtype=float)
        aggregate[key] = {
            'values': values.tolist(), 'mean': float(values.mean()),
            'sample_std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    result = {
        'dataset': dataset,
        'variant': args.variant,
        'seeds': args.seeds,
        'locked_calibration': lock,
        'test_per_seed': per_seed,
        'test_aggregate': aggregate,
        'test_used_for_selection': False,
        'final_calibrator_fit_split': 'train_plus_validation_after_hyperparameter_lock',
        'reporting_rule': 'mean_and_sample_std_of_locked_five_seed_runs',
    }
    save_json(output / dataset / 'locked_test.json', result)
    return result


def main():
    args = parse_args()
    output = ROOT / args.output_dir
    summary = []
    for dataset in args.datasets:
        data, train_x, train_y, lock = search_dataset(dataset, args, output)
        result = locked_test(dataset, data, train_x, train_y, lock, args, output)
        row = {'dataset': dataset}
        for key, value in result['test_aggregate'].items():
            row[key] = value
        summary.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    save_json(output / 'summary.json', summary)


if __name__ == '__main__':
    main()
