from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from baselines.rq1_common import (
    ROOT,
    bag_classification_loss,
    batch_cases,
    iter_batches,
    load_rq1_data,
    save_json,
    seed_everything,
    topk_metrics,
    validation_objective,
)
from baselines.rq1_models import baro_scores, build_model, dycause_scores


TRAINABLE = ['DejaVu', 'Eadro', 'MicroIRC', 'MODIFy', 'MicroInspect', 'TVDiag', 'UDA-RCL']
DETERMINISTIC = ['DyCause', 'BARO']
ALL_METHODS = DETERMINISTIC + TRAINABLE
SOURCE_REPOS = {
    'DejaVu': ROOT.parent / 'baseline_sources' / 'DejaVu',
    'Eadro': Path(os.environ.get('REACR_EADRO_SOURCE', ROOT.parent / 'baseline_sources' / 'Eadro')),
    'BARO': Path(os.environ.get('REACR_BARO_SOURCE', ROOT.parent / 'baseline_sources' / 'BARO')),
    'MicroIRC': ROOT.parent / 'baseline_sources' / 'MicroIRC',
    'MODIFy': Path(os.environ.get('REACR_MODIFY_SOURCE', ROOT.parent / 'baseline_sources' / 'MODIFy')),
    'TVDiag': ROOT.parent / 'baseline_sources' / 'TVDiag',
    'UDA-RCL': ROOT.parent / 'baseline_sources' / 'UDA-RCL',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SN', 'TT', 'D1_weighted'])
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', default='res/rq1_baselines_20260824')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def git_commit(path: Path):
    if not (path / '.git').exists():
        return None
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=path, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def implementation_manifest(method: str):
    files = [ROOT / 'baselines' / 'rq1_common.py', ROOT / 'baselines' / 'rq1_models.py', ROOT / 'tools' / 'run_rq1_baselines.py']
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    repo = SOURCE_REPOS.get(method)
    reference_commit = git_commit(repo) if repo else None
    return {
        'method': method,
        # The executable models live in baselines/rq1_models.py.  External
        # repositories are provenance references, not imported runtime code.
        'implementation': 'paper-aligned-reimplementation',
        'reference_source_path': str(repo) if repo else None,
        'reference_source_commit': reference_commit,
        'reference_source_available': bool(reference_commit),
        'adapter_sha256': digest.hexdigest(),
        'root_label_semantics': 'positive_candidate_bag_any_hit',
        'test_used_for_model_selection': False,
    }


def evaluate_model(model, cases, device, batch_size):
    model.eval()
    scores, labels, ids = [], [], []
    with torch.no_grad():
        for indices in iter_batches(cases, batch_size, False, 0):
            batch = batch_cases(cases, indices, device)
            scores.append(model(batch['features'], batch['adjacency']).cpu())
            labels.append(batch['labels'].cpu())
            ids.extend(batch['case_ids'])
    score = torch.cat(scores)
    label = torch.cat(labels)
    return topk_metrics(score, label), score, label, ids


def train_one(method, data, seed, device, epochs, patience, batch_size):
    seed_everything(seed)
    model = build_model(method, int(data.train[0].features.shape[-1]), data.modality_indices, hidden=64).to(device)
    learning_rate = 1e-3
    weight_decay = 1e-4 if method == 'TVDiag' else 0.0
    if method == 'UDA-RCL':
        learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = copy.deepcopy(model.state_dict())
    best_objective = -float('inf')
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for indices in iter_batches(data.train, batch_size, True, seed * 1000 + epoch):
            batch = batch_cases(data.train, indices, device)
            optimizer.zero_grad()
            score = model(batch['features'], batch['adjacency'])
            loss = bag_classification_loss(score, batch['labels']) + model.extra_loss()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.item()))
        validation, _, _, _ = evaluate_model(model, data.validation, device, batch_size)
        objective = validation_objective(validation)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': validation, 'objective': objective})
        if objective > best_objective + 1e-9:
            best_objective = objective
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    validation, _, _, _ = evaluate_model(model, data.validation, device, batch_size)
    test, scores, labels, ids = evaluate_model(model, data.test, device, batch_size)
    return model, validation, test, scores, labels, ids, history, best_epoch


def deterministic_one(method, data, device):
    scores, labels, ids = [], [], []
    all_indices = sorted(set(sum(data.modality_indices.values(), [])))
    for case in data.test:
        features = case.features.unsqueeze(0).to(device)
        adjacency = case.adjacency.unsqueeze(0).to(device)
        if method == 'BARO':
            score = baro_scores(features, data.normal_mean.to(device), data.normal_std.to(device), data.modality_indices['metric'])
        else:
            score = dycause_scores(features, adjacency, data.normal_mean.to(device), data.normal_std.to(device), all_indices)
        scores.append(score.cpu())
        labels.append(case.label.unsqueeze(0))
        ids.append(case.case_id)
    score = torch.cat(scores)
    label = torch.cat(labels)
    return topk_metrics(score, label), score, label, ids


def write_predictions(path: Path, ids, scores, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'ranked_node_indices', 'label_node_indices'])
        for case_id, score, label in zip(ids, scores, labels):
            writer.writerow([case_id, ' '.join(map(str, torch.argsort(score, descending=True).tolist())), ' '.join(map(str, torch.where(label > 0)[0].tolist()))])


def summarize(output_dir: Path):
    rows = []
    for path in output_dir.glob('*/*/seed*/result.json'):
        value = json.loads(path.read_text(encoding='utf-8'))
        rows.append(value)
    grouped = {}
    for row in rows:
        key = (row['dataset'], row['method'])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (dataset, method), values in sorted(grouped.items()):
        record = {'dataset': dataset, 'method': method, 'runs': len(values)}
        for metric in ('A@1', 'A@3', 'A@5'):
            array = np.asarray([item['test_metrics'][metric] for item in values], dtype=float)
            record[metric + '_mean'] = float(array.mean())
            record[metric + '_std'] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        summary.append(record)
    save_json(output_dir / 'summary.json', summary)
    return summary


def main():
    args = parse_args()
    device = torch.device(args.device)
    base = ROOT / args.output_dir
    for dataset in args.datasets:
        data = load_rq1_data(dataset)
        save_json(base / dataset / 'split.json', data.split_meta)
        for method in args.methods:
            if method not in ALL_METHODS:
                raise ValueError(method)
            seeds = [0] if method in DETERMINISTIC else args.seeds
            for seed in seeds:
                run_dir = base / dataset / method / f'seed{seed}'
                result_path = run_dir / 'result.json'
                if result_path.exists() and not args.overwrite:
                    print('reuse', result_path, flush=True)
                    continue
                start = time.time()
                if method in DETERMINISTIC:
                    test_metrics, scores, labels, ids = deterministic_one(method, data, device)
                    validation_metrics = None
                    history = []
                    best_epoch = None
                    model = None
                else:
                    model, validation_metrics, test_metrics, scores, labels, ids, history, best_epoch = train_one(
                        method, data, seed, device, args.epochs, args.patience, args.batch_size
                    )
                run_dir.mkdir(parents=True, exist_ok=True)
                if model is not None:
                    torch.save(model.state_dict(), run_dir / 'model.pt')
                write_predictions(run_dir / 'predictions.csv', ids, scores, labels)
                result = {
                    'dataset': dataset,
                    'method': method,
                    'seed': seed,
                    'validation_metrics': validation_metrics,
                    'test_metrics': test_metrics,
                    'best_epoch': best_epoch,
                    'elapsed_seconds': time.time() - start,
                    'split_meta': data.split_meta,
                    'implementation_manifest': implementation_manifest(method),
                    'history': history,
                }
                save_json(result_path, result)
                print(json.dumps({k: result[k] for k in ('dataset', 'method', 'seed', 'validation_metrics', 'test_metrics', 'elapsed_seconds')}, ensure_ascii=False), flush=True)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(json.dumps(summarize(base), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
