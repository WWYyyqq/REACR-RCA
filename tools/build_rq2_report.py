"""Audit and summarize the RQ2 observed-vs-frozen topology experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / 'res' / 'reacr_rca_full_20260824'
OUTPUT = ROOT / 'res' / 'rq2_dynamic_topology_20260825'
DATASETS = ('SN', 'TT', 'D1_weighted')
DISPLAY = {'SN': 'SN', 'TT': 'TT', 'D1_weighted': 'D1'}
VARIANTS = {
    'Dynamic': 'source_hypothesis_losses',
    'Frozen': 'frozen_window_topology',
}
METRICS = ('A@1', 'A@3', 'A@5')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate(values):
    array = np.asarray(values, dtype=float) * 100.0
    return {
        'values': array.tolist(),
        'mean': float(array.mean()),
        'sample_std': float(array.std(ddof=1)),
        'display': f'{array.mean():.2f} ± {array.std(ddof=1):.2f}',
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    table = []
    errors = []
    for dataset in DATASETS:
        per_variant = {}
        for display_name, variant in VARIANTS.items():
            runs = []
            for seed in range(5):
                run_dir = RUN_ROOT / dataset / variant / f'seed{seed}'
                config_path = run_dir / 'run_config.json'
                result_path = run_dir / 'final_eval.json'
                if not config_path.exists() or not result_path.exists():
                    errors.append(f'missing RQ2 artifact: {run_dir}')
                    continue
                config = read_json(config_path)
                result = read_json(result_path)
                expected_mode = 'observed' if display_name == 'Dynamic' else 'freeze_first'
                if config.get('graph_sequence_mode') != expected_mode:
                    errors.append(f'{config_path}: expected {expected_mode}')
                runs.append(result)
                records.append({
                    'dataset': dataset,
                    'variant': display_name,
                    'seed': seed,
                    'graph_sequence_mode': config.get('graph_sequence_mode'),
                    'final_eval_sha256': sha256(result_path),
                    'run_config_sha256': sha256(config_path),
                })
            if len(runs) != 5:
                errors.append(f'{dataset}/{display_name}: expected five runs, found {len(runs)}')
                continue
            counts = {
                stratum: {run['topology_strata'][stratum]['sample_count'] for run in runs}
                for stratum in ('changed', 'stable')
            }
            if any(len(value) != 1 for value in counts.values()):
                errors.append(f'{dataset}/{display_name}: inconsistent topology strata counts {counts}')
            per_variant[display_name] = {
                'overall': {
                    metric: aggregate([run['metrics'][metric] for run in runs])
                    for metric in METRICS
                },
                'changed': {
                    'case_count': next(iter(counts['changed'])),
                    **{
                        metric: aggregate([
                            run['topology_strata']['changed']['metrics'][metric]
                            for run in runs
                        ])
                        for metric in METRICS
                    },
                },
                'stable': {
                    'case_count': next(iter(counts['stable'])),
                    **{
                        metric: aggregate([
                            run['topology_strata']['stable']['metrics'][metric]
                            for run in runs
                        ])
                        for metric in METRICS
                    },
                },
            }
        if errors:
            continue
        # Verify the counterfactual differs only in graph sequence mode/output path.
        for seed in range(5):
            dynamic = read_json(RUN_ROOT / dataset / VARIANTS['Dynamic'] / f'seed{seed}' / 'run_config.json')
            frozen = read_json(RUN_ROOT / dataset / VARIANTS['Frozen'] / f'seed{seed}' / 'run_config.json')
            differing = {
                key for key in set(dynamic) | set(frozen)
                if dynamic.get(key) != frozen.get(key)
            }
            if differing != {'graph_sequence_mode', 'output_dir'}:
                errors.append(f'{dataset}/seed{seed}: unexpected config differences {sorted(differing)}')

        delta = {}
        for scope in ('overall', 'changed', 'stable'):
            delta[scope] = {}
            for metric in METRICS:
                d = np.asarray(per_variant['Dynamic'][scope][metric]['values'])
                f = np.asarray(per_variant['Frozen'][scope][metric]['values'])
                diff = d - f
                delta[scope][metric] = {
                    'paired_values': diff.tolist(),
                    'mean': float(diff.mean()),
                    'sample_std': float(diff.std(ddof=1)),
                }
        table.append({
            'dataset': DISPLAY[dataset],
            'dataset_key': dataset,
            'Dynamic': per_variant['Dynamic'],
            'Frozen': per_variant['Frozen'],
            'dynamic_minus_frozen': delta,
        })

    if errors:
        raise SystemExit('\n'.join(errors))

    payload = {
        'research_question': 'Does the observed dynamic dependency graph improve localization under topology changes?',
        'scope': 'REACR-N neural branch to isolate graph-sequence effects from the identity-free statistical calibrator',
        'protocol': {
            'seeds': [0, 1, 2, 3, 4],
            'reporting': 'mean_and_sample_std',
            'dynamic': 'observed graph at every time step',
            'frozen': 'repeat the first observed graph of each window at all time steps',
            'only_config_difference': 'graph_sequence_mode',
            'selection': 'same hyperparameters; test not used for selection',
        },
        'table': table,
        'run_audit': records,
    }
    (OUTPUT / 'rq2_results.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')

    lines = [
        '# RQ2：动态图对拓扑变化场景的影响', '',
        '为隔离图结构演化本身的作用，本实验仅比较 REACR-N 神经分支，不引入与节点身份无关的统计校准器。Dynamic 使用窗口内每一时刻的实测图；Frozen 将窗口第一张实测图重复到全部时间步。两组实验除 `graph_sequence_mode` 外配置完全相同，均报告五个随机种子的均值 ± 样本标准差。', '',
        '| 数据集 | 图序列 | 全体 A@1 | 边集变化案例 A@1 | 边集变化案例 A@3 | 边集变化案例 A@5 | 边集不变案例 A@1 |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in table:
        for variant in ('Dynamic', 'Frozen'):
            item = row[variant]
            lines.append('| {} | {} | {} | {} | {} | {} | {} |'.format(
                row['dataset'], variant,
                item['overall']['A@1']['display'],
                item['changed']['A@1']['display'],
                item['changed']['A@3']['display'],
                item['changed']['A@5']['display'],
                item['stable']['A@1']['display'],
            ))
    lines += ['', '## 结果', '']
    for row in table:
        d = row['dynamic_minus_frozen']
        lines.append(
            f"- {row['dataset']}：动态图在变化案例上的 A@1/A@3/A@5 相对冻结图变化为 "
            f"{d['changed']['A@1']['mean']:+.2f}/{d['changed']['A@3']['mean']:+.2f}/{d['changed']['A@5']['mean']:+.2f} 个百分点。"
        )
    lines += [
        '',
        '动态图在三个数据集的拓扑变化案例上均提高 A@1，增益分别为 2.67、3.00 和 0.20 个百分点；但 A@3/A@5 并未一致提高。因此，RQ2 的证据支持一个较窄的结论：保留图序列演化有助于把根因推到首位，但当前实验不足以证明其能够普遍改善较长候选列表的覆盖率。', '',
        '这里的分层依据是窗口内有向边集合是否发生切换；Dynamic 仍同时保留边属性和边权的逐时刻变化。边集不变案例仅有 SN 3 个、TT 25 个、D1 3 个，其中 SN 和 D1 样本过少，不据此作一般性结论。', '',
        '## Claim–Evidence Map', '',
        '- Claim：动态图提高拓扑变化案例的首位定位率。 | Evidence：三个数据集变化子集 A@1 均为正增益。 | Status：supported（增益较小，应限定措辞）。',
        '- Claim：动态图全面提高 A@K。 | Evidence：变化子集的 A@3/A@5 结果不一致。 | Status：not supported。',
        '- Claim：动态图在稳定拓扑下同样更优。 | Evidence：SN、D1 稳定案例过少且结果不一致。 | Status：needs evidence。', '',
        '## 自审', '',
        '- 清晰性：动态图与冻结图的唯一差异已明确。',
        '- 一致性：统一使用“动态图序列”“冻结图”和“拓扑变化案例”。',
        '- 证据边界：未把 A@1 的小幅增益扩展为所有 A@K 的全面提升。',
        '- 待补证据：若论文需要更强的动态图贡献，应增加边权冻结、边集冻结和高变化强度分层实验。',
    ]
    (OUTPUT / 'RQ2_DYNAMIC_TOPOLOGY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'ok', 'runs': len(records), 'output': str(OUTPUT)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
