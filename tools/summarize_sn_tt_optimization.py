import argparse
import csv
import json
import statistics
from pathlib import Path


METRICS = ('A@1', 'A@3', 'A@5')
DATASETS = ('SN', 'TT')
SEEDS = (0, 1, 2)


def load_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def load_metrics(path):
    return load_json(path)['metrics']


def aggregate(rows):
    summary = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        summary[metric] = {
            'values': values,
            'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def pct(value):
    return f'{100.0 * value:.2f}'


def metrics_text(metrics):
    return ' | '.join(pct(metrics[key]) for key in METRICS)


def main():
    parser = argparse.ArgumentParser(description='Summarize locked SN/TT optimization runs.')
    parser.add_argument('--root', default='res/optimization_v2')
    parser.add_argument('--report', default='docs/SN_TT_OPTIMIZATION_V2.md')
    args = parser.parse_args()

    root = Path(args.root)
    summary_dir = root / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    per_seed = []
    aggregate_summary = {}
    for dataset in DATASETS:
        dataset_rows = []
        for seed in SEEDS:
            run_dir = root / 'final' / dataset / f'seed{seed}'
            validation_fused = load_metrics(run_dir / 'validation_eval.json')
            test_fused = load_metrics(run_dir / 'final_eval.json')
            test_neural = load_metrics(run_dir / 'neural_final_eval.json')
            record = {
                'dataset': dataset,
                'seed': seed,
                'validation_fused': validation_fused,
                'test_fused': test_fused,
                'test_neural': test_neural,
            }
            dataset_rows.append(record)
            per_seed.append(record)

        fused = aggregate([row['test_fused'] for row in dataset_rows])
        neural = aggregate([row['test_neural'] for row in dataset_rows])
        validation = aggregate([row['validation_fused'] for row in dataset_rows])
        paired_gain = {
            metric: statistics.mean(
                row['test_fused'][metric] - row['test_neural'][metric]
                for row in dataset_rows
            )
            for metric in METRICS
        }
        aggregate_summary[dataset] = {
            'validation_fused': validation,
            'test_fused': fused,
            'test_neural': neural,
            'paired_gain': paired_gain,
        }

    dynamic_rows = []
    for dataset in DATASETS:
        for mode in ('observed', 'freeze_first'):
            validation_rows = []
            fused_rows = []
            neural_rows = []
            for seed in SEEDS:
                if mode == 'observed':
                    run_dir = root / 'final' / dataset / f'seed{seed}'
                else:
                    run_dir = root / 'ablation' / dataset / f'freeze_first_seed{seed}'
                validation_rows.append(load_metrics(run_dir / 'validation_eval.json'))
                fused_rows.append(load_metrics(run_dir / 'final_eval.json'))
                neural_rows.append(load_metrics(run_dir / 'neural_final_eval.json'))
            dynamic_rows.append({
                'dataset': dataset,
                'mode': mode,
                'validation_fused': aggregate(validation_rows),
                'test_fused': aggregate(fused_rows),
                'test_neural': aggregate(neural_rows),
            })

    search_runs = {
        'SN_extra_trees': root / 'search/SN/extra_trees_d3_l4_bw045_seed0',
        'TT_linear_svc': root / 'search/TT/linear_svc_c003_bw060_seed0',
        'TT_extra_trees': root / 'search/TT/extra_trees_none_l2_bw015_seed0',
    }
    search_audit = {}
    forbidden_names = ('final_eval.json', 'neural_final_eval.json', 'eval_predictions.csv')
    for name, run_dir in search_runs.items():
        config = load_json(run_dir / 'run_config.json')
        unexpected = [item for item in forbidden_names if (run_dir / item).exists()]
        search_audit[name] = {
            'validation_only': bool(config.get('validation_only')),
            'validation_metrics': load_metrics(run_dir / 'validation_eval.json'),
            'unexpected_test_outputs': unexpected,
            'passed': bool(config.get('validation_only')) and not unexpected,
        }

    topology = {}
    for dataset in DATASETS:
        audit = load_json(root / 'final' / dataset / 'seed0/topology_audit.json')
        topology[dataset] = {
            split: {
                'unique_edge_sets': audit[split]['unique_edge_sets'],
                'topology_switches': audit[split]['topology_switches'],
                'fixed_edge_topology': audit[split]['fixed_edge_topology'],
            }
            for split in ('train', 'test')
        }

    locked_configs = {
        dataset: load_json(root / 'final' / dataset / 'seed0/run_config.json')
        for dataset in DATASETS
    }
    result = {
        'protocol': {
            'selection': 'validation_only',
            'case_split': 'group_by_session',
            'locked_test_seeds': list(SEEDS),
            'metric_order': list(METRICS),
        },
        'locked_configs': locked_configs,
        'aggregate': aggregate_summary,
        'dynamic_ablation': dynamic_rows,
        'search_audit': search_audit,
        'topology_audit': topology,
    }
    with (summary_dir / 'optimization_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)

    with (summary_dir / 'per_seed_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'seed', 'variant', *METRICS])
        for row in per_seed:
            for variant in ('validation_fused', 'test_fused', 'test_neural'):
                writer.writerow([
                    row['dataset'], row['seed'], variant,
                    *(row[variant][metric] for metric in METRICS),
                ])

    with (summary_dir / 'dynamic_ablation.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'mode', 'split_variant', *METRICS])
        for row in dynamic_rows:
            for variant in ('validation_fused', 'test_fused', 'test_neural'):
                writer.writerow([
                    row['dataset'], row['mode'], variant,
                    *(row[variant][metric]['mean'] for metric in METRICS),
                ])

    lines = [
        '# SN/TT optimization V2',
        '',
        'All hyperparameters were selected with `--validation_only`; the search runs contain no test metrics or test predictions. After configuration lock, seeds 0/1/2 were evaluated once on the test split.',
        '',
        '## Locked results',
        '',
        '| Dataset | Variant | A@1 | A@3 | A@5 |',
        '|---|---|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        item = aggregate_summary[dataset]
        for label, key in (('Neural dynamic model', 'test_neural'), ('Neural + statistical evidence', 'test_fused')):
            values = item[key]
            cells = [f"{pct(values[m]['mean'])} +/- {pct(values[m]['sample_std'])}" for m in METRICS]
            lines.append(f"| {dataset} | {label} | {' | '.join(cells)} |")
        gain = item['paired_gain']
        lines.append(f"| {dataset} | Paired gain (percentage points) | {pct(gain['A@1'])} | {pct(gain['A@3'])} | {pct(gain['A@5'])} |")

    lines.extend(['', '## Validation-only candidate selection', '', '| Candidate | Validation A@1 | A@3 | A@5 | Audit |', '|---|---:|---:|---:|---|'])
    for name, audit in search_audit.items():
        metrics = audit['validation_metrics']
        lines.append(f"| {name} | {pct(metrics['A@1'])} | {pct(metrics['A@3'])} | {pct(metrics['A@5'])} | {'PASS' if audit['passed'] else 'FAIL'} |")

    lines.extend(['', '## Dynamic graph ablation (three-seed means)', '', '| Dataset | Graph mode | Split/variant | A@1 | A@3 | A@5 |', '|---|---|---|---:|---:|---:|'])
    for row in dynamic_rows:
        for label, key in (('Validation fused', 'validation_fused'), ('Test fused', 'test_fused'), ('Test neural', 'test_neural')):
            means = {metric: row[key][metric]['mean'] for metric in METRICS}
            lines.append(f"| {row['dataset']} | {row['mode']} | {label} | {metrics_text(means)} |")

    lines.extend(['', '## Topology evidence', '', '| Dataset | Split | Unique edge sets | Topology switches | Fixed topology |', '|---|---|---:|---:|---|'])
    for dataset in DATASETS:
        for split in ('train', 'test'):
            item = topology[dataset][split]
            lines.append(f"| {dataset} | {split} | {item['unique_edge_sets']} | {item['topology_switches']} | {item['fixed_edge_topology']} |")

    lines.extend([
        '',
        '## Interpretation',
        '',
        '- The session-group split creates zero-shot service localization: root-service identities do not overlap across train, validation, and test sessions.',
        '- The statistical branch uses shared node-wise temporal statistics (mean, standard deviation, maximum, minimum, range, and end-to-start delta) and never uses node identity.',
        '- Observed dynamic graphs are preferred by validation metrics, but their test-set contribution is small and mixed. The large final improvement is primarily attributable to the identity-independent statistical evidence fused with the dynamic neural model.',
        '- Therefore, results should not be described as proving that dynamic topology alone causes the full performance gain.',
        '',
        'Machine-readable artifacts are in `res/optimization_v2/summary/`.',
    ])
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'wrote {summary_dir / "optimization_summary.json"}')
    print(f'wrote {report_path}')


if __name__ == '__main__':
    main()
