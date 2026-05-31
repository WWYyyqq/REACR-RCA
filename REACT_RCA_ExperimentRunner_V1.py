import argparse
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from models.evaluation import evaluation, fd_test, test as deephunt_reconstruction_test
from models.rc_scorer import feedback, naive_scorer
from models.train import train as train_deephunt
from utils.public_functions import load_init, load_samples


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_DIR = ROOT / 'res' / 'react_rca_fair'


def parse_args():
    parser = argparse.ArgumentParser(description='Fair REACT-RCA experiment runner with validation-only selection.')
    parser.add_argument('--datasets', nargs='+', default=['D1', 'D2'])
    parser.add_argument('--base_dir', default=str(DEFAULT_BASE_DIR))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--mode', choices=['all', 'run', 'summarize', 'reports'], default='all')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--skip_deephunt', action='store_true')
    parser.add_argument('--skip_dycau', action='store_true')
    parser.add_argument('--screening_seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--confirm_seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--alphas', nargs='+', type=float, default=[0.01, 0.02, 0.05, 0.10])
    parser.add_argument('--epochs_grid', nargs='+', type=int, default=[10, 20])
    parser.add_argument('--confirm_topn', type=int, default=3)
    parser.add_argument('--train_ratio', type=float, default=0.3)
    parser.add_argument('--test_start_ratio', type=float, default=0.5)
    return parser.parse_args()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def load_config(dataset: str):
    with open(ROOT / 'config' / f'{dataset}.yaml', 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data):
    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, tuple):
            return [sanitize(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(sanitize(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def method_dir(base_dir: Path, dataset: str, method: str, run_id: str) -> Path:
    return base_dir / dataset / method / run_id


def fmt_float(value: float) -> str:
    text = f'{value:.4g}'.replace('.', 'p')
    return text.replace('-', 'm')


def run_id(seed: int, epochs: int, cp_weight: float, alpha: float, causal_tag: str) -> str:
    return f's{seed}_e{epochs}_cp{fmt_float(cp_weight)}_a{fmt_float(alpha)}_c{causal_tag}'


def dataset_dycau_defaults(dataset: str):
    if dataset == 'D2':
        return {
            'batch_size': 4,
            'change_point_weight': 0.05,
            'causal_hidden_dim': 16,
            'causal_gamma': 0.2,
            'causal_kernel_size': 3,
            'dropout': 0.15,
            'branch_init_scale': 0.075,
            'train_weight_decay': 1e-5,
            'train_patience': 8,
            'causal_tag': 'h16_g0p2_k3_do0p15_wd1e5_b0p075',
        }
    return {
        'batch_size': 8,
        'change_point_weight': 1.0,
        'causal_hidden_dim': 64,
        'causal_gamma': 0.5,
        'causal_kernel_size': 5,
        'causal_tag': 'default',
    }


def top_metrics_from_res(values):
    return {
        'A@1': float(values[0]) if len(values) > 0 else math.nan,
        'A@3': float(values[2]) if len(values) > 2 else math.nan,
        'A@5': float(values[4]) if len(values) > 4 else math.nan,
    }


def split_cases_fair(cases: pd.DataFrame, train_ratio: float = 0.3, test_start_ratio: float = 0.5):
    case_count = len(cases)
    train_end = max(int(case_count * train_ratio), 1)
    test_start = max(int(case_count * test_start_ratio), train_end + 1)
    test_start = min(test_start, case_count - 1)
    train_end = min(train_end, test_start - 1)
    train_cases = cases.iloc[:train_end].reset_index(drop=True)
    val_cases = cases.iloc[train_end:test_start].reset_index(drop=True)
    test_cases = cases.iloc[test_start:].reset_index(drop=True)
    split_meta = {
        'protocol': f'fair_{round(train_ratio * 100)}_{round((test_start_ratio - train_ratio) * 100)}_{round((1 - test_start_ratio) * 100)}',
        'train_ratio': train_ratio,
        'test_start_ratio': test_start_ratio,
        'train_start': 0,
        'train_end_exclusive': train_end,
        'val_start': train_end,
        'val_end_exclusive': test_start,
        'test_start': test_start,
        'test_end_exclusive': case_count,
        'train_case_count': len(train_cases),
        'val_case_count': len(val_cases),
        'test_case_count': len(test_cases),
    }
    return train_cases, val_cases, test_cases, split_meta


def deephunt_eval_on_cases(model, cases, samples, node_hash, config):
    loss_df = deephunt_reconstruction_test(model, samples, node_hash, config['model_param']['feat_span'])
    df = naive_scorer(cases, samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    return df, top_metrics_from_res(evaluation(df, 5).tolist())


def top_hit(row, k: int, label_col: str):
    labels = [item.strip() for item in str(row[label_col]).split(',') if item.strip()]
    for rank in range(1, k + 1):
        pred = str(row.get(f'Top{rank}', ''))
        pred_node = pred.split(':', 1)[0]
        if any(label in pred_node or pred_node in label for label in labels):
            return True
    return False


def run_deephunt(dataset: str, base_dir: Path, overwrite: bool, train_ratio: float = 0.3, test_start_ratio: float = 0.5):
    out_dir = method_dir(base_dir, dataset, 'deephunt', 'fair_feedback')
    if (out_dir / 'final_eval.json').exists() and not overwrite:
        return out_dir

    ensure_clean_dir(out_dir, overwrite)
    set_random_seed(0)
    config = load_config(dataset)
    train_samples, test_samples = load_samples(config['path']['sample_dir'])
    input_samples = train_samples if config['train_samples_num'] == 'whole' else train_samples[:config['train_samples_num']]
    model = train_deephunt(input_samples, config['model_param'])
    cases = pd.read_csv(ROOT / config['path']['case_dir'])
    train_cases, val_cases, test_cases, split_meta = split_cases_fair(cases, train_ratio=train_ratio, test_start_ratio=test_start_ratio)
    node_hash, _, _ = load_init(config['path']['graph_dir'])

    loss_df = deephunt_reconstruction_test(model, test_samples, node_hash, config['model_param']['feat_span'])
    naive_val_df = naive_scorer(val_cases, test_samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    naive_test_df = naive_scorer(test_cases, test_samples, loss_df, node_hash, window_size=config['feedback']['window_size'], pre=0, suc=-0)
    fd_model = feedback(model, train_cases, test_samples, node_hash, config['feedback'])
    fd_val_df = fd_test(val_cases, fd_model, test_samples, node_hash, config['feedback']['window_size'])
    fd_test_df = fd_test(test_cases, fd_model, test_samples, node_hash, config['feedback']['window_size'])

    with open(out_dir / 'naive_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(out_dir / 'fd_model.pkl', 'wb') as f:
        pickle.dump(fd_model, f)
    naive_val_df.to_csv(out_dir / 'validation_df.csv', index=False)
    fd_val_df.to_csv(out_dir / 'fd_validation_df.csv', index=False)
    naive_test_df.to_csv(out_dir / 'test_df.csv', index=False)
    fd_test_df.to_csv(out_dir / 'fd_test_df.csv', index=False)
    res_dict = {
        'validation_naive_res_all': evaluation(naive_val_df, 5).tolist(),
        'validation_fd_res_all': evaluation(fd_val_df, 5).tolist(),
        'test_naive_res_all': evaluation(naive_test_df, 5).tolist(),
        'test_fd_res_all': evaluation(fd_test_df, 5).tolist(),
    }
    save_json(out_dir / 'res.json', res_dict)
    save_json(out_dir / 'run_config.json', {
        'dataset': dataset,
        'method': 'deephunt',
        'seed': 0,
        'case_split': split_meta,
        'selection_metric_source': 'none_baseline',
    })

    validation_eval = {
        'naive': top_metrics_from_res(res_dict['validation_naive_res_all']),
        'feedback': top_metrics_from_res(res_dict['validation_fd_res_all']),
        'primary': top_metrics_from_res(res_dict['validation_fd_res_all']),
    }
    final_eval = {
        'naive': top_metrics_from_res(res_dict['test_naive_res_all']),
        'feedback': top_metrics_from_res(res_dict['test_fd_res_all']),
        'primary': top_metrics_from_res(res_dict['test_fd_res_all']),
    }
    save_json(out_dir / 'validation_eval.json', validation_eval)
    save_json(out_dir / 'final_eval.json', final_eval)
    return out_dir


def run_dycau(
    dataset: str,
    method: str,
    seed: int,
    epochs: int,
    alpha: float,
    base_dir: Path,
    device: str,
    overwrite: bool,
    export_event_chains: bool,
    train_ratio: float = 0.3,
    test_start_ratio: float = 0.5,
):
    defaults = dataset_dycau_defaults(dataset)
    rid = run_id(seed, epochs, defaults['change_point_weight'], alpha, defaults['causal_tag'])
    out_dir = method_dir(base_dir, dataset, method, rid)
    required_done = (out_dir / 'final_eval.json').exists() and (out_dir / 'validation_eval.json').exists()
    if export_event_chains:
        required_done = required_done and (out_dir / 'event_chain_metrics.json').exists()
    if required_done and not overwrite:
        return out_dir

    ensure_clean_dir(out_dir, overwrite)
    cmd = [
        sys.executable,
        '-u',
        str(ROOT / 'REACT_RCA_Pipeline_V1.py'),
        '--dataset', dataset,
        '--device', device,
        '--seed', str(seed),
        '--batch_size', str(defaults['batch_size']),
        '--train_epochs', str(epochs),
        '--feedback_sample_ratio', str(train_ratio),
        '--test_start_ratio', str(test_start_ratio),
        '--case_split_protocol', 'fair',
        '--enable_change_point_branch',
        '--change_point_weight', str(defaults['change_point_weight']),
        '--causal_hidden_dim', str(defaults['causal_hidden_dim']),
        '--causal_gamma', str(defaults['causal_gamma']),
        '--causal_kernel_size', str(defaults['causal_kernel_size']),
        '--dropout', str(defaults.get('dropout', 0.1)),
        '--branch_init_scale', str(defaults.get('branch_init_scale', 0.1)),
        '--train_weight_decay', str(defaults.get('train_weight_decay', 0.0)),
        '--train_patience', str(defaults.get('train_patience', 10)),
        '--output_dir', str(out_dir),
    ]
    if method == 'famos_baro_coe_dycaurca':
        cmd.extend([
            '--enable_modality_gate',
            '--modality_gate_fusion_mode', 'raw_residual',
            '--modality_gate_residual_alpha', str(alpha),
        ])
    else:
        cmd.append('--disable_modality_gate')
    if export_event_chains:
        cmd.append('--export_event_chains')

    subprocess.run(cmd, cwd=str(ROOT), check=True)
    cfg_path = out_dir / 'run_config.json'
    if cfg_path.exists():
        cfg = load_json(cfg_path)
        cfg.update({
            'runner_protocol': f'fair_{round(train_ratio * 100)}_{round((test_start_ratio - train_ratio) * 100)}_{round((1 - test_start_ratio) * 100)}',
            'selection_metric_source': 'validation',
            'final_metric_source': 'test',
        })
        save_json(cfg_path, cfg)
    return out_dir


def validate_environment(device: str):
    info = {
        'python': sys.executable,
        'torch': torch.__version__,
        'torch_cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device': device,
    }
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but torch.cuda.is_available() is False.')
    if device == 'cuda':
        import dgl
        g = dgl.graph(([0, 1], [1, 2]), num_nodes=3).to('cuda')
        x = torch.randn(3, 4, device='cuda')
        g.ndata['h'] = x
        g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.mean('m', 'h2'))
        info['dgl'] = dgl.__version__
        info['gpu_name'] = torch.cuda.get_device_name(0)
    return info


def collect_run_row(dataset: str, method: str, run_dir: Path):
    final_path = run_dir / 'final_eval.json'
    if not final_path.exists():
        return None
    final_eval = load_json(final_path)
    validation_eval = load_json(run_dir / 'validation_eval.json') if (run_dir / 'validation_eval.json').exists() else {}
    if method == 'deephunt':
        metrics = final_eval['primary']
        val_metrics = validation_eval.get('primary', {})
        seed = 0
        epochs = None
        alpha = None
    else:
        metrics = final_eval['metrics']
        val_metrics = validation_eval.get('metrics', {})
        cfg = load_json(run_dir / 'run_config.json') if (run_dir / 'run_config.json').exists() else {}
        seed = cfg.get('seed')
        epochs = cfg.get('train_epochs')
        alpha = cfg.get('modality_gate_residual_alpha') if cfg.get('enable_modality_gate') else 0.0
    event_metrics = load_json(run_dir / 'event_chain_metrics.json') if (run_dir / 'event_chain_metrics.json').exists() else {}
    return {
        'dataset': dataset,
        'method': method,
        'run_id': run_dir.name,
        'run_dir': str(run_dir.relative_to(ROOT)),
        'seed': seed,
        'epochs': epochs,
        'alpha': alpha,
        'A@1': metrics.get('A@1'),
        'A@3': metrics.get('A@3'),
        'A@5': metrics.get('A@5'),
        'val_A@1': val_metrics.get('A@1'),
        'val_A@3': val_metrics.get('A@3'),
        'val_A@5': val_metrics.get('A@5'),
        'label_chain_hit_at_topk_strict': event_metrics.get('label_chain_hit_at_topk_strict'),
        'label_root_hit_at_topk': event_metrics.get('label_root_hit_at_topk'),
        'strict_chain_coverage': event_metrics.get('strict_chain_coverage'),
        'fallback_chain_coverage': event_metrics.get('fallback_chain_coverage'),
    }


def collect_metrics(base_dir: Path):
    rows = []
    for dataset_dir in sorted(path for path in base_dir.iterdir() if path.is_dir() and path.name != 'summary'):
        dataset = dataset_dir.name
        for method_dir_path in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            method = method_dir_path.name
            for run_dir in sorted(path for path in method_dir_path.iterdir() if path.is_dir()):
                row = collect_run_row(dataset, method, run_dir)
                if row is not None:
                    rows.append(row)
    return pd.DataFrame(rows)


def select_best_runs(metrics_df: pd.DataFrame):
    best = {}
    for (dataset, method), group in metrics_df.groupby(['dataset', 'method']):
        if method == 'deephunt':
            row = group.sort_values(['A@1', 'A@3', 'A@5'], ascending=False).iloc[0]
            best[f'{dataset}/{method}'] = row.to_dict()
            continue
        summary = group.groupby(['epochs', 'alpha'], dropna=False).agg(
            val_mean_a1=('val_A@1', 'mean'),
            val_mean_a3=('val_A@3', 'mean'),
            val_mean_a5=('val_A@5', 'mean'),
            val_std_a1=('val_A@1', 'std'),
            test_mean_a1=('A@1', 'mean'),
            test_mean_a3=('A@3', 'mean'),
            test_mean_a5=('A@5', 'mean'),
            test_std_a1=('A@1', 'std'),
            test_std_a3=('A@3', 'std'),
            test_std_a5=('A@5', 'std'),
            runs=('A@1', 'count'),
        ).reset_index()
        for col in ['val_std_a1', 'test_std_a1', 'test_std_a3', 'test_std_a5']:
            summary[col] = summary[col].fillna(0.0)
        max_runs = summary['runs'].max()
        summary = summary[summary['runs'] == max_runs]
        summary = summary.sort_values(
            ['val_mean_a1', 'val_mean_a3', 'val_mean_a5', 'val_std_a1'],
            ascending=[False, False, False, True],
        )
        best_cfg = summary.iloc[0].to_dict()
        cfg_runs = group[(group['epochs'] == best_cfg['epochs']) & (group['alpha'] == best_cfg['alpha'])]
        best_single = cfg_runs.sort_values(['A@1', 'A@3', 'A@5'], ascending=False).iloc[0].to_dict()
        best[f'{dataset}/{method}'] = {
            'best_config_by_validation': best_cfg,
            'best_single_run': best_single,
            'selection_metric_source': 'validation',
        }
    return best


def build_case_diff(base_dir: Path, best_runs: dict):
    rows = []
    for dataset in sorted({key.split('/')[0] for key in best_runs}):
        deep_key = f'{dataset}/deephunt'
        fbc_key = f'{dataset}/famos_baro_coe_dycaurca'
        if deep_key not in best_runs or fbc_key not in best_runs:
            continue
        deep_dir = ROOT / best_runs[deep_key]['run_dir']
        fbc_single = best_runs[fbc_key]['best_single_run']
        fbc_dir = ROOT / fbc_single['run_dir']
        deep_df_path = deep_dir / 'fd_test_df.csv'
        fbc_df_path = fbc_dir / 'eval_predictions.csv'
        if not deep_df_path.exists() or not fbc_df_path.exists():
            continue
        deep_df = pd.read_csv(deep_df_path).reset_index(drop=True)
        fbc_df = pd.read_csv(fbc_df_path).reset_index(drop=True)
        total = min(len(deep_df), len(fbc_df))
        for idx in range(total):
            deep_row = deep_df.iloc[idx]
            fbc_row = fbc_df.iloc[idx]
            deep_hit = top_hit(deep_row, 1, 'cmdb_id')
            fbc_hit = top_hit(fbc_row, 1, 'label_nodes')
            if deep_hit and fbc_hit:
                bucket = 'both_hit'
            elif deep_hit and not fbc_hit:
                bucket = 'deephunt_only'
            elif fbc_hit and not deep_hit:
                bucket = 'fbc_only'
            else:
                bucket = 'both_miss'
            rows.append({
                'dataset': dataset,
                'case_index': idx,
                'case_timestamp': fbc_row.get('case_timestamp', deep_row.get('timestamp')),
                'deephunt_label': deep_row.get('cmdb_id'),
                'fbc_label': fbc_row.get('label_nodes'),
                'deephunt_top1': str(deep_row.get('Top1', '')).split(':', 1)[0],
                'fbc_top1': str(fbc_row.get('Top1', '')).split(':', 1)[0],
                'bucket': bucket,
            })
    return pd.DataFrame(rows)


def write_reports(base_dir: Path, metrics_df: pd.DataFrame, best_runs: dict, case_diff_df: pd.DataFrame):
    summary_dir = base_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        '# FAMOS-Baro-CoE-DyCauRCA 全流程梳理',
        '',
        '## 统一流程',
        '- DeepHunt：以单时刻 SBG 为输入，GraphSAGE/GAE 学正常模式，用重构误差和 feedback scorer 做实例级 Top-k 定位。',
        '- Baro-CoE-DyCauRCA：把输入扩展为窗口级 DSBG，使用动态图时空编码器、多头异常表示、BARO change-point 分支、RUN-inspired 时滞传播矩阵、可分解根因排序头，并导出 CoE 风格审计链路。',
        '- FAMOS-Baro-CoE-DyCauRCA：在 BARO-CoE-DyCauRCA 前加入 FAMOS-inspired 模态可靠性门控，但采用 raw-residual 缩放原始 metric/trace/log 分段，避免 encoded 替换破坏原始特征分布。',
        '',
        '## 最佳结果',
    ]
    for key, value in sorted(best_runs.items()):
        if key.endswith('/deephunt'):
            report_lines.append(f'- {key}: A@1={value["A@1"]:.4f}, A@3={value["A@3"]:.4f}, A@5={value["A@5"]:.4f}')
        else:
            mean_cfg = value['best_config_by_validation']
            single = value['best_single_run']
            report_lines.append(
                f'- {key}: validation-selected epochs={mean_cfg["epochs"]}, alpha={mean_cfg["alpha"]}; '
                f'test mean A@1={mean_cfg["test_mean_a1"]:.4f}, A@3={mean_cfg["test_mean_a3"]:.4f}, '
                f'A@5={mean_cfg["test_mean_a5"]:.4f}; diagnostic best single {single["run_id"]} '
                f'A@1={single["A@1"]:.4f}, A@3={single["A@3"]:.4f}, A@5={single["A@5"]:.4f}'
            )
    report_lines.extend(['', '## DeepHunt vs FBC-DyCauRCA 性能差异'])
    for dataset in sorted({key.split('/')[0] for key in best_runs}):
        deep = best_runs.get(f'{dataset}/deephunt')
        fbc = best_runs.get(f'{dataset}/famos_baro_coe_dycaurca')
        if not deep or not fbc:
            continue
        mean_cfg = fbc['best_config_by_validation']
        single = fbc['best_single_run']
        report_lines.append(
            f'- {dataset} validation-selected test mean-over-seeds: FBC 相比 DeepHunt '
            f'ΔA@1={mean_cfg["test_mean_a1"] - deep["A@1"]:+.4f}, '
            f'ΔA@3={mean_cfg["test_mean_a3"] - deep["A@3"]:+.4f}, '
            f'ΔA@5={mean_cfg["test_mean_a5"] - deep["A@5"]:+.4f}.'
        )
        report_lines.append(
            f'- {dataset} diagnostic best-single-run: FBC 相比 DeepHunt '
            f'ΔA@1={single["A@1"] - deep["A@1"]:+.4f}, '
            f'ΔA@3={single["A@3"] - deep["A@3"]:+.4f}, '
            f'ΔA@5={single["A@5"] - deep["A@5"]:+.4f}.'
        )
    if len(case_diff_df) > 0:
        report_lines.extend(['', '## Case-level 差异'])
        case_summary = case_diff_df.groupby(['dataset', 'bucket']).size().reset_index(name='count')
        for _, row in case_summary.iterrows():
            report_lines.append(f'- {row["dataset"]} {row["bucket"]}: {int(row["count"])} cases')
    report_lines.extend([
        '',
        '## 与 DeepHunt 的核心差异',
        '- 输入粒度：DeepHunt 使用单时刻 SBG；FBC-DyCauRCA 使用窗口级动态图 DSBG。',
        '- 异常表示：DeepHunt 主要依赖重构误差；FBC-DyCauRCA 显式建模重构误差、时间漂移、预测误差和 change-point。',
        '- 传播建模：DeepHunt 使用静态一阶上下游传播模式；FBC-DyCauRCA 学习窗口条件化时滞传播矩阵。',
        '- 可解释性：DeepHunt 输出 Top-k 分数；FBC-DyCauRCA 额外导出 root/event chain、边强度和严格链路命中指标。',
    ])
    (summary_dir / 'FBC_DyCauRCA_FLOW_CN.md').write_text('\n'.join(report_lines), encoding='utf-8')

    audit_lines = [
        '# REACT-RCA Fairness Audit',
        '',
        '- Case split: chronological 30% train/feedback, 20% validation, 50% final test.',
        '- Checkpoint selection: validation only, using A@1, A@3, A@5, then validation loss.',
        '- Parameter selection: FAMOS alpha/epochs are selected from validation metrics only.',
        '- Final metrics: `final_eval.json` and `metrics_summary.csv` A@k columns are final test metrics.',
        '- Event-chain coverage: strict coverage does not use one-hop fallback; fallback is disabled by default.',
        '- Best single run is diagnostic only and is not the primary performance claim.',
    ]
    (summary_dir / 'fairness_audit.md').write_text('\n'.join(audit_lines), encoding='utf-8')

    opt_lines = [
        '# FBC-DyCauRCA 后续优化建议',
        '',
        '- 稳定性：当前 DyCauRCA 对 seed 敏感，必须报告多 seed 均值；checkpoint 选择建议改为 A@1/A@3/A@5 加权，而不是只按 A@5。',
        '- 排序目标：pairwise ranking loss 对 Top-1 不够直接，建议加入 ListMLE 或 LambdaRank-style top-heavy loss。',
        '- FAMOS 门控：保留 raw-residual，增加 residual alpha warmup 或模态权重 entropy regularization，避免门控扰乱首位排序。',
        '- 因果传播：对 causal matrix 加拓扑 mask/一致性正则，减少非拓扑传播边；对 CoE 失败链路反查边方向和强度。',
        '- 诊断导出：为每个 Top-k 节点导出 self anomaly、incoming/outgoing propagation、graph prior、modality weights、change-point score、causal edge strength。',
    ]
    (summary_dir / 'FBC_DyCauRCA_OPTIMIZATION_CN.md').write_text('\n'.join(opt_lines), encoding='utf-8')

    if len(case_diff_df) > 0:
        case_summary = case_diff_df.groupby(['dataset', 'bucket']).size().reset_index(name='count')
        case_summary.to_csv(summary_dir / 'case_diff_bucket_summary.csv', index=False)


def summarize(base_dir: Path):
    summary_dir = base_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = collect_metrics(base_dir)
    metrics_df.to_csv(summary_dir / 'metrics_summary.csv', index=False)
    best_runs = select_best_runs(metrics_df) if len(metrics_df) > 0 else {}
    save_json(summary_dir / 'best_runs.json', best_runs)
    save_json(summary_dir / 'best_configs_by_validation.json', best_runs)

    if len(metrics_df) > 0:
        cfg_df = metrics_df[metrics_df['method'] != 'deephunt']
        if len(cfg_df) > 0:
            selection_df = (
                cfg_df.groupby(['dataset', 'method', 'epochs', 'alpha'], dropna=False)
                .agg(
                    val_mean_a1=('val_A@1', 'mean'),
                    val_mean_a3=('val_A@3', 'mean'),
                    val_mean_a5=('val_A@5', 'mean'),
                    test_mean_a1=('A@1', 'mean'),
                    test_mean_a3=('A@3', 'mean'),
                    test_mean_a5=('A@5', 'mean'),
                    test_std_a1=('A@1', 'std'),
                    runs=('A@1', 'count'),
                )
                .reset_index()
                .sort_values(['dataset', 'method', 'val_mean_a1', 'val_mean_a3', 'val_mean_a5'], ascending=[True, True, False, False, False])
            )
        else:
            selection_df = pd.DataFrame()
    else:
        selection_df = pd.DataFrame()
    selection_df.to_csv(summary_dir / 'validation_selection_summary.csv', index=False)

    final_summary = {}
    for key, value in best_runs.items():
        if key.endswith('/deephunt'):
            final_summary[key] = {
                'test_A@1': value.get('A@1'),
                'test_A@3': value.get('A@3'),
                'test_A@5': value.get('A@5'),
                'selection_metric_source': 'none_baseline',
            }
        else:
            cfg = value['best_config_by_validation']
            final_summary[key] = {
                'epochs': cfg.get('epochs'),
                'alpha': cfg.get('alpha'),
                'runs': cfg.get('runs'),
                'validation_mean_A@1': cfg.get('val_mean_a1'),
                'validation_mean_A@3': cfg.get('val_mean_a3'),
                'validation_mean_A@5': cfg.get('val_mean_a5'),
                'test_mean_A@1': cfg.get('test_mean_a1'),
                'test_mean_A@3': cfg.get('test_mean_a3'),
                'test_mean_A@5': cfg.get('test_mean_a5'),
                'test_std_A@1': cfg.get('test_std_a1'),
                'test_std_A@3': cfg.get('test_std_a3'),
                'test_std_A@5': cfg.get('test_std_a5'),
                'selection_metric_source': 'validation',
            }
    save_json(summary_dir / 'final_test_summary.json', final_summary)
    case_diff_df = build_case_diff(base_dir, best_runs)
    case_diff_df.to_csv(summary_dir / 'deephunt_vs_fbc_case_diff.csv', index=False)
    write_reports(base_dir, metrics_df, best_runs, case_diff_df)
    return metrics_df, best_runs, case_diff_df


def run_all(args):
    base_dir = Path(args.base_dir).resolve()
    summary_dir = base_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    save_json(summary_dir / 'environment.json', validate_environment(args.device))

    if not args.skip_deephunt:
        for dataset in args.datasets:
            run_deephunt(dataset, base_dir, args.overwrite, train_ratio=args.train_ratio, test_start_ratio=args.test_start_ratio)

    if args.skip_dycau:
        return

    for dataset in args.datasets:
        defaults = dataset_dycau_defaults(dataset)
        for seed in args.confirm_seeds:
            run_dycau(
                dataset=dataset,
                method='baro_coe_dycaurca',
                seed=seed,
                epochs=20,
                alpha=0.0,
                base_dir=base_dir,
                device=args.device,
                overwrite=args.overwrite,
                export_event_chains=True,
                train_ratio=args.train_ratio,
                test_start_ratio=args.test_start_ratio,
            )

        screened = []
        for alpha in args.alphas:
            for epochs in args.epochs_grid:
                for seed in args.screening_seeds:
                    out_dir = run_dycau(
                        dataset=dataset,
                        method='famos_baro_coe_dycaurca',
                        seed=seed,
                        epochs=epochs,
                        alpha=alpha,
                        base_dir=base_dir,
                        device=args.device,
                        overwrite=args.overwrite,
                        export_event_chains=False,
                        train_ratio=args.train_ratio,
                        test_start_ratio=args.test_start_ratio,
                    )
                    validation_eval = load_json(out_dir / 'validation_eval.json')
                    final_eval = load_json(out_dir / 'final_eval.json')
                    screened.append({
                        'dataset': dataset,
                        'alpha': alpha,
                        'epochs': epochs,
                        'seed': seed,
                        'val_A@1': validation_eval['metrics']['A@1'],
                        'val_A@3': validation_eval['metrics']['A@3'],
                        'val_A@5': validation_eval['metrics']['A@5'],
                        'test_A@1': final_eval['metrics']['A@1'],
                        'test_A@3': final_eval['metrics']['A@3'],
                        'test_A@5': final_eval['metrics']['A@5'],
                        'selection_metric_source': 'validation',
                    })

        screen_df = pd.DataFrame(screened)
        screen_df.to_csv(summary_dir / f'{dataset}_famos_screening.csv', index=False)
        top_cfgs = (
            screen_df.groupby(['alpha', 'epochs'])
            .agg(mean_a1=('val_A@1', 'mean'), mean_a3=('val_A@3', 'mean'), mean_a5=('val_A@5', 'mean'), std_a1=('val_A@1', 'std'))
            .reset_index()
            .fillna({'std_a1': 0.0})
            .sort_values(['mean_a1', 'mean_a3', 'mean_a5', 'std_a1'], ascending=[False, False, False, True])
            .head(args.confirm_topn)
        )
        for _, cfg in top_cfgs.iterrows():
            for seed in args.confirm_seeds:
                run_dycau(
                    dataset=dataset,
                    method='famos_baro_coe_dycaurca',
                    seed=int(seed),
                    epochs=int(cfg['epochs']),
                    alpha=float(cfg['alpha']),
                    base_dir=base_dir,
                    device=args.device,
                    overwrite=args.overwrite,
                    export_event_chains=True,
                    train_ratio=args.train_ratio,
                    test_start_ratio=args.test_start_ratio,
                )


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    os.chdir(ROOT)

    if args.mode in {'all', 'run'}:
        run_all(args)
    if args.mode in {'all', 'summarize', 'reports'}:
        summarize(base_dir)


if __name__ == '__main__':
    main()
