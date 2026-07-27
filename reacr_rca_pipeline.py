import argparse
import json
import os

import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from models.reacr_rca_anomaly_head import MultiHeadAnomalyHead, summarize_anomaly_output
from models.reacr_rca_causal_head import RunInspiredCausalHead, summarize_causal_output
from models.reacr_rca_causal_ranker import CausalPropagationRanker, summarize_rank_output
from models.reacr_rca_dsbg_dataset import (
    batch_summary,
    build_case_windows,
    build_sliding_windows,
    create_dsbg_dataloader,
    summarize_windows,
)
from models.reacr_rca_dynamic_backbone import DynamicSpatialTemporalBackbone, summarize_backbone_output
from models.reacr_rca_trainer import (
    DyCauRCAModel,
    build_prediction_dataframe,
    predict_scores,
    save_challenge2_diagnostics,
    train_dycau,
)
from models.reacr_rca_event_chain_explainer import export_event_chains
from models.reacr_rca_statistical_calibrator import (
    StatisticalEvidenceCalibrator,
    evaluate_calibrated_bundle,
    predict_calibrated_scores,
)
from models.reacr_rca_artifact_exporter import export_intermediate_artifacts
from models.reacr_rca_modality_gate import (
    ModalityReliabilityGate,
    parse_modality_dims,
    summarize_modality_gate_output,
    valid_modality_dims,
)
from tools.reacr_rca_topology_audit import (
    audit_samples,
    audit_windows,
    window_has_topology_change,
)
from utils.public_functions import load_init, load_samples


def parse_args():
    parser = argparse.ArgumentParser(description='REACR-RCA full RCA trainer')
    parser.add_argument('--dataset', default='D1')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--train_stride', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_train_windows', type=int, default=0)
    parser.add_argument('--max_case_windows', type=int, default=0)
    parser.add_argument('--spatial_hidden_dim', type=int, default=64)
    parser.add_argument('--temporal_hidden_dim', type=int, default=64)
    parser.add_argument('--num_spatial_layers', type=int, default=2)
    parser.add_argument('--num_temporal_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    modality_group = parser.add_mutually_exclusive_group()
    modality_group.add_argument('--enable_modality_gate', dest='enable_modality_gate', action='store_true', default=None)
    modality_group.add_argument('--disable_modality_gate', dest='enable_modality_gate', action='store_false')
    parser.add_argument('--modality_gate_hidden_dim', type=int, default=64)
    parser.add_argument('--modality_gate_output_dim', type=int, default=0)
    parser.add_argument('--modality_gate_fusion_mode', choices=['encoded', 'raw_residual'], default='encoded')
    parser.add_argument('--modality_gate_residual_alpha', type=float, default=0.2)
    parser.add_argument('--disable_modality_gaussian_bias', action='store_true')
    parser.add_argument('--disable_temporal_attention', action='store_true')
    parser.add_argument(
        '--disable_causal_temporal_attention',
        action='store_true',
        help='Ablation only: allow temporal attention to read future window steps.',
    )
    topology_group = parser.add_mutually_exclusive_group()
    topology_group.add_argument(
        '--enable_topology_dynamics',
        dest='enable_topology_dynamics',
        action='store_true',
        default=True,
    )
    topology_group.add_argument(
        '--disable_topology_dynamics',
        dest='enable_topology_dynamics',
        action='store_false',
    )
    parser.add_argument('--topology_dropout', type=float, default=0.0)
    parser.add_argument(
        '--graph_sequence_mode',
        choices=['observed', 'freeze_first'],
        default='observed',
        help='Use observed graph snapshots or freeze each window to its first graph for ablation.',
    )
    parser.add_argument(
        '--topology_feature_mode',
        choices=['full', 'static_only', 'dynamic_only'],
        default='full',
    )
    parser.add_argument('--skip_topology_audit', action='store_true')
    parser.add_argument('--anomaly_hidden_dim', type=int, default=64)
    change_point_group = parser.add_mutually_exclusive_group()
    change_point_group.add_argument('--enable_change_point_branch', dest='enable_change_point_branch', action='store_true', default=None)
    change_point_group.add_argument('--disable_change_point_branch', dest='enable_change_point_branch', action='store_false')
    parser.add_argument('--change_point_weight', type=float, default=1.0)
    parser.add_argument('--causal_hidden_dim', type=int, default=64)
    parser.add_argument('--causal_max_lag', type=int, default=3)
    parser.add_argument('--causal_min_strength', type=float, default=0.0)
    parser.add_argument('--causal_gamma', type=float, default=0.5)
    parser.add_argument(
        '--topology_weight_gamma',
        type=float,
        default=None,
        help='Strength of observed dynamic edge intensity in causal relation logits.',
    )
    parser.add_argument('--causal_kernel_size', type=int, default=5)
    parser.add_argument(
        '--relation_topology_mode',
        choices=['none', 'directed', 'undirected'],
        default='undirected',
    )
    parser.add_argument('--disable_null_source', action='store_true')
    parser.add_argument('--causal_lag_temperature', type=float, default=1.0)
    parser.add_argument('--rank_hidden_dim', type=int, default=64)
    parser.add_argument(
        '--ranker_mode',
        choices=['legacy', 'root_symptom', 'counterfactual', 'source_hypothesis'],
        default='source_hypothesis',
    )
    parser.add_argument('--anchor_init_scale', type=float, default=1.0)
    parser.add_argument('--branch_init_scale', type=float, default=0.1)
    parser.add_argument(
        '--feature_value_clip',
        type=float,
        default=None,
        help='Optional symmetric clipping threshold applied to normalized node features.',
    )
    parser.add_argument(
        '--observation_prior_base_score_key',
        choices=['final_score', 'outgoing_score'],
        default=None,
    )
    parser.add_argument('--observation_prior_base_weight', type=float, default=None)
    parser.add_argument('--observation_prior_metric_weight', type=float, default=None)
    parser.add_argument('--observation_prior_range_weight', type=float, default=None)
    parser.add_argument('--observation_prior_reconstruction_weight', type=float, default=None)
    parser.add_argument(
        '--statistical_calibrator',
        choices=['none', 'extra_trees', 'linear_svc'],
        default='none',
    )
    parser.add_argument('--statistical_base_weight', type=float, default=0.5)
    parser.add_argument('--statistical_n_estimators', type=int, default=500)
    parser.add_argument('--statistical_max_depth', type=int, default=0)
    parser.add_argument('--statistical_min_samples_leaf', type=int, default=1)
    parser.add_argument('--statistical_max_features', type=float, default=1.0)
    parser.add_argument('--statistical_svc_c', type=float, default=0.03)
    parser.add_argument('--statistical_seed', type=int, default=-1)
    parser.add_argument('--train_epochs', type=int, default=0)
    parser.add_argument('--train_learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_weight_decay', type=float, default=0.0)
    parser.add_argument('--train_patience', type=int, default=10)
    explanation_group = parser.add_mutually_exclusive_group()
    explanation_group.add_argument(
        '--enable_explanation_losses',
        dest='enable_explanation_losses',
        action='store_true',
        default=True,
    )
    explanation_group.add_argument(
        '--disable_explanation_losses',
        dest='enable_explanation_losses',
        action='store_false',
    )
    parser.add_argument('--root_symptom_loss_weight', type=float, default=0.1)
    parser.add_argument('--counterfactual_loss_weight', type=float, default=0.1)
    parser.add_argument('--event_chain_loss_weight', type=float, default=0.05)
    parser.add_argument('--invariance_loss_weight', type=float, default=0.05)
    parser.add_argument('--normal_pretrain_epochs', type=int, default=3)
    parser.add_argument('--normal_auxiliary_loss_weight', type=float, default=0.1)
    parser.add_argument(
        '--normal_topology_change_oversample_factor',
        type=float,
        default=4.0,
        help='Relative sampling weight for normal windows that contain a real edge change.',
    )
    parser.add_argument('--reconstruction_loss_weight', type=float, default=1.0)
    parser.add_argument('--forecast_loss_weight', type=float, default=1.0)
    parser.add_argument(
        '--response_candidate_topk',
        '--hard_symptom_topk',
        dest='response_candidate_topk',
        type=int,
        default=5,
        help='Number of mined high-response candidates; hard_symptom_topk is deprecated.',
    )
    parser.add_argument('--feedback_sample_ratio', type=float, default=-1.0)
    parser.add_argument('--test_start_ratio', type=float, default=0.5)
    parser.add_argument(
        '--case_split_protocol',
        choices=['legacy', 'fair', 'time_strict', 'group'],
        default='legacy',
    )
    parser.add_argument(
        '--validation_only',
        action='store_true',
        help='Tune on train/validation only; do not evaluate or export any test-set outputs.',
    )
    parser.add_argument('--export_event_chains', action='store_true')
    parser.add_argument('--event_chain_topk', type=int, default=3)
    parser.add_argument('--event_chain_max_depth', type=int, default=3)
    parser.add_argument('--event_chain_edge_topk', type=int, default=3)
    parser.add_argument('--event_chain_evidence_topk', type=int, default=5)
    parser.add_argument('--event_chain_min_strength', type=float, default=0.0)
    parser.add_argument('--allow_event_chain_fallback', action='store_true')
    parser.add_argument('--export_intermediate_artifacts', action='store_true')
    parser.add_argument('--artifact_dir', default='')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--tag', default='')
    parser.add_argument('--output_dir', default='')
    args = parser.parse_args()
    apply_fair_defaults(args)
    return args


def _arg_was_provided(flag: str) -> bool:
    return flag in sys.argv[1:]


def apply_fair_defaults(args):
    if args.dataset == 'D2' and args.case_split_protocol == 'fair':
        recipe_defaults = {
            '--train_epochs': ('train_epochs', 20),
            '--train_patience': ('train_patience', 8),
            '--change_point_weight': ('change_point_weight', 0.05),
            '--causal_hidden_dim': ('causal_hidden_dim', 16),
            '--causal_gamma': ('causal_gamma', 0.2),
            '--causal_kernel_size': ('causal_kernel_size', 3),
            '--modality_gate_fusion_mode': ('modality_gate_fusion_mode', 'raw_residual'),
            '--modality_gate_residual_alpha': ('modality_gate_residual_alpha', 0.1),
            '--dropout': ('dropout', 0.15),
            '--branch_init_scale': ('branch_init_scale', 0.075),
            '--train_weight_decay': ('train_weight_decay', 1e-5),
        }
        for flag, (attr, value) in recipe_defaults.items():
            if not _arg_was_provided(flag):
                setattr(args, attr, value)
        if args.enable_change_point_branch is None:
            args.enable_change_point_branch = True
        if args.enable_modality_gate is None:
            args.enable_modality_gate = True
        return

    if args.enable_change_point_branch is None:
        args.enable_change_point_branch = False
    if args.enable_modality_gate is None:
        args.enable_modality_gate = False


def set_random_seed(seed: int):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def label_mapping_granularity(dataset: str) -> str:
    if str(dataset).startswith('D1'):
        return 'mixed_service_to_instance_candidate_bag'
    if str(dataset).startswith('D2'):
        return 'exact_single_instance_candidate'
    return 'audited_positive_candidate_bag'


def build_backbone(args, feat_dim: int, edge_feature_dim: int = 0) -> DynamicSpatialTemporalBackbone:
    return DynamicSpatialTemporalBackbone(
        in_dim=feat_dim,
        spatial_hidden_dim=args.spatial_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        num_spatial_layers=args.num_spatial_layers,
        dropout=args.dropout,
        num_temporal_heads=args.num_temporal_heads,
        use_temporal_attention=not args.disable_temporal_attention,
        causal_temporal_attention=not args.disable_causal_temporal_attention,
        use_topology_dynamics=args.enable_topology_dynamics,
        topology_dropout=args.topology_dropout,
        topology_feature_mode=args.topology_feature_mode,
        edge_feature_dim=edge_feature_dim,
    )


def build_model(args, feat_dim: int, node_num: int, config=None) -> DyCauRCAModel:
    modality_gate = None
    model_feat_dim = feat_dim
    if args.enable_modality_gate:
        if config is None:
            raise ValueError('config is required when modality gate is enabled.')
        modality_dims = parse_modality_dims(config)
        if not valid_modality_dims(modality_dims, feat_dim):
            raise ValueError(f'Modality dimensions {modality_dims} do not match feature dim {feat_dim}.')
        model_feat_dim = args.modality_gate_output_dim if args.modality_gate_output_dim > 0 else feat_dim
        modality_gate = ModalityReliabilityGate(
            modality_dims=modality_dims,
            output_dim=model_feat_dim,
            hidden_dim=args.modality_gate_hidden_dim,
            dropout=args.dropout,
            use_gaussian_bias=not args.disable_modality_gaussian_bias,
            fusion_mode=args.modality_gate_fusion_mode,
            residual_alpha=args.modality_gate_residual_alpha,
        )

    model_config = (config or {}).get('model_param', {})
    edge_feature_dim = int(
        model_config.get('edge_feature_dim', (config or {}).get('edge_feature_dim', 0))
    )
    backbone = build_backbone(args, feat_dim=model_feat_dim, edge_feature_dim=edge_feature_dim)
    anomaly_head = MultiHeadAnomalyHead(
        hidden_dim=args.temporal_hidden_dim,
        feat_dim=model_feat_dim,
        fusion_hidden_dim=args.anomaly_hidden_dim,
        use_change_point=args.enable_change_point_branch,
        change_point_weight=args.change_point_weight,
    )
    causal_head = RunInspiredCausalHead(
        node_num=node_num,
        window_size=args.window_size,
        hidden_dim=args.causal_hidden_dim,
        max_lag=args.causal_max_lag,
        kernel_size=args.causal_kernel_size,
        gamma=args.causal_gamma,
        min_strength=args.causal_min_strength,
        topology_mode=args.relation_topology_mode,
        use_null_source=not args.disable_null_source,
        lag_temperature=args.causal_lag_temperature,
        topology_weight_gamma=getattr(args, 'topology_weight_gamma', 0.0),
    )
    ranker = CausalPropagationRanker(
        window_size=args.window_size,
        hidden_dim=args.temporal_hidden_dim,
        branch_hidden_dim=args.rank_hidden_dim,
        dropout=args.dropout,
        anchor_init_scale=args.anchor_init_scale,
        branch_init_scale=args.branch_init_scale,
        ranker_mode=args.ranker_mode,
    )
    return DyCauRCAModel(
        backbone,
        anomaly_head,
        causal_head,
        ranker,
        modality_gate=modality_gate,
        graph_sequence_mode=args.graph_sequence_mode,
        observation_prior=model_config.get('observation_prior'),
        feature_value_clip=model_config.get('feature_value_clip'),
    )


def split_cases_for_training(
    cases,
    feedback_sample_ratio: float,
    default_ratio: float,
    protocol: str = 'legacy',
    test_start_ratio: float = 0.5,
    group_column=None,
):
    ratio = feedback_sample_ratio if 0 < feedback_sample_ratio < 1 else default_ratio
    cases = cases.reset_index(drop=True).copy()
    if 'case_id' not in cases.columns:
        cases.insert(0, 'case_id', np.arange(len(cases), dtype=int))
    if protocol == 'time_strict':
        cases = cases.sort_values(['timestamp', 'case_id']).reset_index(drop=True)
    if protocol == 'group':
        if not group_column or group_column not in cases.columns:
            raise ValueError(
                f"Group split requires case column {group_column!r}; "
                f"available columns are {list(cases.columns)}."
            )
        group_order = (
            cases.sort_values(['timestamp', 'case_id'])
            .drop_duplicates(group_column)[group_column]
            .tolist()
        )
        group_count = len(group_order)
        if group_count < 3:
            raise ValueError(
                f'Group split needs at least 3 distinct groups, found {group_count}.'
            )
        train_group_end = max(int(group_count * ratio), 1)
        test_group_start = max(int(group_count * test_start_ratio), train_group_end + 1)
        test_group_start = min(test_group_start, group_count - 1)
        train_group_end = min(train_group_end, test_group_start - 1)
        train_groups = group_order[:train_group_end]
        val_groups = group_order[train_group_end:test_group_start]
        test_groups = group_order[test_group_start:]
        select = lambda values: cases[cases[group_column].isin(values)].reset_index(drop=True)
        train_cases = select(train_groups)
        val_cases = select(val_groups)
        test_cases = select(test_groups)
        split_meta = {
            'protocol': protocol,
            'group_column': group_column,
            'feedback_sample_ratio': ratio,
            'test_start_ratio': test_start_ratio,
            'train_groups': [str(value) for value in train_groups],
            'val_groups': [str(value) for value in val_groups],
            'test_groups': [str(value) for value in test_groups],
            'train_case_count': len(train_cases),
            'val_case_count': len(val_cases),
            'test_case_count': len(test_cases),
        }
        return train_cases, val_cases, test_cases, split_meta

    case_count = len(cases)
    train_end = max(int(case_count * ratio), 1)

    if protocol in ('fair', 'time_strict'):
        test_start = max(int(case_count * test_start_ratio), train_end + 1)
        test_start = min(test_start, case_count - 1)
        train_end = min(train_end, test_start - 1)
        train_cases = cases.iloc[:train_end].reset_index(drop=True)
        val_cases = cases.iloc[train_end:test_start].reset_index(drop=True)
        test_cases = cases.iloc[test_start:].reset_index(drop=True)
        split_meta = {
            'protocol': protocol,
            'feedback_sample_ratio': ratio,
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

    test_index = -max(int(case_count * 0.7), 1)
    train_cases = cases.iloc[:train_end].reset_index(drop=True)
    eval_cases = cases.iloc[test_index:].reset_index(drop=True)
    split_meta = {
        'protocol': protocol,
        'feedback_sample_ratio': ratio,
        'train_start': 0,
        'train_end_exclusive': train_end,
        'val_start': case_count + test_index,
        'val_end_exclusive': case_count,
        'test_start': case_count + test_index,
        'test_end_exclusive': case_count,
        'train_case_count': len(train_cases),
        'val_case_count': len(eval_cases),
        'test_case_count': len(eval_cases),
    }
    return train_cases, eval_cases, eval_cases, split_meta


def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    set_random_seed(args.seed)
    config = yaml.load(open(f'config/{args.dataset}.yaml', 'r'), Loader=yaml.FullLoader)
    data_settings = config.get('data_settings', {})
    model_settings = config.get('model_param', {})
    if not _arg_was_provided('--window_size'):
        args.window_size = int(data_settings.get('window_size', args.window_size))
    if not _arg_was_provided('--train_stride'):
        args.train_stride = int(data_settings.get('train_stride', args.train_stride))
    if not _arg_was_provided('--case_split_protocol'):
        args.case_split_protocol = data_settings.get(
            'case_split_protocol', args.case_split_protocol
        )
    if args.topology_weight_gamma is None:
        args.topology_weight_gamma = float(model_settings.get('topology_weight_gamma', 0.0))
    if args.feature_value_clip is not None:
        if args.feature_value_clip <= 0:
            raise ValueError('feature_value_clip must be positive when provided')
        model_settings['feature_value_clip'] = float(args.feature_value_clip)

    observation_prior = dict(model_settings.get('observation_prior', {}))
    observation_weights = dict(observation_prior.get('weights', {}))
    if args.observation_prior_base_score_key is not None:
        observation_prior['base_score_key'] = args.observation_prior_base_score_key
    weight_overrides = {
        'base': args.observation_prior_base_weight,
        'metric_peak': args.observation_prior_metric_weight,
        'temporal_range': args.observation_prior_range_weight,
        'reconstruction_peak': args.observation_prior_reconstruction_weight,
    }
    if any(value is not None for value in weight_overrides.values()):
        observation_prior['enabled'] = True
        for key, value in weight_overrides.items():
            if value is not None:
                observation_weights[key] = float(value)
        observation_prior['weights'] = observation_weights
    if observation_prior:
        model_settings['observation_prior'] = observation_prior

    suffix = f'_{args.tag}' if args.tag else ''
    res_dir = args.output_dir if args.output_dir else f"res/{args.dataset}_reacr_rca_stage1{suffix}"
    os.makedirs(res_dir, exist_ok=True)

    summary_path = os.path.join(res_dir, 'stage1_dsbg_summary.json')
    topology_audit_path = os.path.join(res_dir, 'topology_audit.json')
    train_history_path = os.path.join(res_dir, 'train_history.json')
    validation_eval_path = os.path.join(res_dir, 'validation_eval.json')
    final_eval_path = os.path.join(res_dir, 'final_eval.json')
    neural_validation_eval_path = os.path.join(res_dir, 'neural_validation_eval.json')
    neural_final_eval_path = os.path.join(res_dir, 'neural_final_eval.json')
    statistical_calibrator_path = os.path.join(res_dir, 'statistical_calibrator.pkl')
    prediction_path = os.path.join(res_dir, 'eval_predictions.csv')
    validation_prediction_path = os.path.join(res_dir, 'validation_predictions.csv')
    event_chain_path = os.path.join(res_dir, 'event_chains.csv')
    event_chain_metrics_path = os.path.join(res_dir, 'event_chain_metrics.json')
    chain_validation_path = os.path.join(res_dir, 'chain_validation.csv')
    chain_validation_metrics_path = os.path.join(res_dir, 'chain_validation_metrics.json')
    challenge2_role_path = os.path.join(res_dir, 'challenge2_role_diagnostic.csv')
    challenge2_role_metrics_path = os.path.join(res_dir, 'challenge2_role_summary.json')
    challenge2_role_markdown_path = os.path.join(res_dir, 'challenge2_role_summary.md')
    call_chain_catalog_path = config.get('path', {}).get('call_chain_catalog', '')
    checkpoint_path = os.path.join(res_dir, 'best_model.pt')
    config_path = os.path.join(res_dir, 'run_config.json')

    run_config = dict(vars(args))
    run_config['root_label_semantics'] = 'positive_candidate_bag_any_hit'
    run_config['label_mapping_granularity'] = label_mapping_granularity(args.dataset)
    run_config['positive_bag_aggregation'] = 'max'
    run_config['effective_feature_value_clip'] = model_settings.get('feature_value_clip')
    run_config['effective_observation_prior'] = model_settings.get(
        'observation_prior', {'enabled': False}
    )
    save_json(config_path, run_config)

    train_samples, test_samples = load_samples(config['path']['sample_dir'])
    cases = pd.read_csv(config['path']['case_dir'])
    node_hash, _, _ = load_init(config['path']['graph_dir'])
    index_to_node = {idx: node for node, idx in node_hash.items()}

    topology_audit = None
    if not args.skip_topology_audit:
        train_topology_audit = audit_samples(
            train_samples,
            max_transition_records=20,
        )
        test_topology_audit = audit_samples(
            test_samples,
            max_transition_records=20,
        )
        fixed_in_both_splits = bool(
            train_topology_audit['fixed_edge_topology']
            and test_topology_audit['fixed_edge_topology']
        )
        topology_audit = {
            'train': train_topology_audit,
            'test': test_topology_audit,
            'interpretation': {
                'fixed_edge_topology_in_both_splits': fixed_in_both_splits,
                'observed_edge_changes_in_any_split': not fixed_in_both_splits,
                'topology_changes_are_never_synthesized': True,
                'dynamic_node_observations_are_distinct_from_dynamic_edges': True,
            },
        }
        save_json(topology_audit_path, topology_audit)

    max_train_windows = None if args.max_train_windows <= 0 else args.max_train_windows
    max_case_windows = None if args.max_case_windows <= 0 else args.max_case_windows
    train_window_kwargs = {
        'max_gap_seconds': data_settings.get('max_gap_seconds'),
        'incomplete_policy': data_settings.get('train_incomplete_policy', 'pad'),
    }
    case_window_kwargs = {
        'sample_interval_seconds': float(data_settings.get('sample_interval_seconds', 60.0)),
        'case_pre_seconds': data_settings.get('case_pre_seconds'),
        'case_post_seconds': data_settings.get('case_post_seconds'),
        'max_gap_seconds': data_settings.get('max_gap_seconds'),
        'incomplete_policy': data_settings.get('case_incomplete_policy', 'pad'),
    }

    train_windows = build_sliding_windows(
        train_samples,
        window_size=args.window_size,
        stride=args.train_stride,
        max_windows=max_train_windows,
        **train_window_kwargs,
    )
    case_windows = build_case_windows(
        cases,
        test_samples,
        node_hash,
        window_size=args.window_size,
        max_cases=max_case_windows,
        **case_window_kwargs,
    )

    if topology_audit is not None:
        topology_audit['model_window_coverage'] = {
            'normal_train_windows': audit_windows(train_windows, max_window_records=20),
            'all_case_windows': audit_windows(case_windows, max_window_records=20),
        }
        save_json(topology_audit_path, topology_audit)

    if args.normal_topology_change_oversample_factor < 1.0:
        raise ValueError('normal_topology_change_oversample_factor must be >= 1.0')
    normal_change_flags = [window_has_topology_change(window) for window in train_windows]
    normal_changed_window_count = int(sum(normal_change_flags))
    normal_sample_weights = None
    if normal_changed_window_count > 0 and args.normal_topology_change_oversample_factor > 1.0:
        normal_sample_weights = [
            args.normal_topology_change_oversample_factor if changed else 1.0
            for changed in normal_change_flags
        ]

    train_loader = create_dsbg_dataloader(
        train_windows,
        batch_size=args.batch_size,
        shuffle=True,
        with_labels=False,
        sample_weights=normal_sample_weights,
    )
    case_loader = create_dsbg_dataloader(case_windows, batch_size=args.batch_size, shuffle=False, with_labels=True)

    train_batch = next(iter(train_loader)) if len(train_windows) > 0 else None
    case_batch = next(iter(case_loader)) if len(case_windows) > 0 else None

    backbone_summary = None
    modality_gate_summary = None
    anomaly_summary = None
    causal_summary = None
    rank_summary = None
    training_summary = None

    if case_batch is not None:
        model = build_model(args, feat_dim=int(case_batch['feat_dim']), node_num=int(case_batch['node_num']), config=config)
        model.eval()
        with torch.no_grad():
            outputs = model([graph for graph in case_batch['graphs']], case_batch['features'].float())
        modality_gate_summary = summarize_modality_gate_output(outputs.get('modality_gate_output'))
        backbone_summary = summarize_backbone_output(outputs['backbone_output'])
        anomaly_summary = summarize_anomaly_output(outputs['anomaly_output'])
        causal_summary = summarize_causal_output(outputs['causal_output'])
        rank_summary = summarize_rank_output(outputs['rank_output'])

        if args.train_epochs > 0:
            device = torch.device(args.device)
            default_ratio = float(config['feedback']['sample_num']) if isinstance(config['feedback']['sample_num'], float) else 0.3
            train_cases, val_cases, test_cases, split_meta = split_cases_for_training(
                cases,
                args.feedback_sample_ratio,
                default_ratio,
                protocol=args.case_split_protocol,
                test_start_ratio=args.test_start_ratio,
                group_column=data_settings.get('case_group_column'),
            )
            train_case_windows = build_case_windows(
                train_cases,
                test_samples,
                node_hash,
                window_size=args.window_size,
                max_cases=max_case_windows,
                **case_window_kwargs,
            )
            val_case_windows = build_case_windows(
                val_cases,
                test_samples,
                node_hash,
                window_size=args.window_size,
                max_cases=max_case_windows,
                **case_window_kwargs,
            )
            test_case_windows = build_case_windows(
                test_cases,
                test_samples,
                node_hash,
                window_size=args.window_size,
                max_cases=max_case_windows,
                **case_window_kwargs,
            )
            if topology_audit is not None:
                topology_audit['model_window_coverage'].update({
                    'train_case_windows': audit_windows(
                        train_case_windows,
                        max_window_records=20,
                    ),
                    'validation_case_windows': audit_windows(
                        val_case_windows,
                        max_window_records=20,
                    ),
                    'test_case_windows': audit_windows(
                        test_case_windows,
                        max_window_records=20,
                    ),
                })
                save_json(topology_audit_path, topology_audit)
            train_case_loader = create_dsbg_dataloader(train_case_windows, batch_size=args.batch_size, shuffle=True, with_labels=True)
            val_case_loader = create_dsbg_dataloader(val_case_windows, batch_size=args.batch_size, shuffle=False, with_labels=True)
            test_case_loader = create_dsbg_dataloader(test_case_windows, batch_size=args.batch_size, shuffle=False, with_labels=True)

            model = build_model(args, feat_dim=int(case_batch['feat_dim']), node_num=int(case_batch['node_num']), config=config)
            model, history, final_eval, validation_eval, train_meta = train_dycau(
                model,
                train_case_loader,
                device=device,
                val_loader=val_case_loader,
                test_loader=None if args.validation_only else test_case_loader,
                epochs=args.train_epochs,
                learning_rate=args.train_learning_rate,
                weight_decay=args.train_weight_decay,
                patience=args.train_patience,
                enable_explanation_losses=args.enable_explanation_losses,
                root_symptom_loss_weight=args.root_symptom_loss_weight,
                counterfactual_loss_weight=args.counterfactual_loss_weight,
                event_chain_loss_weight=args.event_chain_loss_weight,
                invariance_loss_weight=args.invariance_loss_weight,
                normal_loader=train_loader,
                normal_pretrain_epochs=args.normal_pretrain_epochs,
                normal_auxiliary_loss_weight=args.normal_auxiliary_loss_weight,
                reconstruction_loss_weight=args.reconstruction_loss_weight,
                forecast_loss_weight=args.forecast_loss_weight,
            )

            torch.save(model.state_dict(), checkpoint_path)
            save_json(train_history_path, history)
            neural_validation_eval = validation_eval
            neural_final_eval = None if args.validation_only else final_eval
            actual_device = torch.device(train_meta['device'])
            statistical_calibrator_config = None

            if args.statistical_calibrator != 'none':
                save_json(neural_validation_eval_path, neural_validation_eval)
                if neural_final_eval is not None:
                    save_json(neural_final_eval_path, neural_final_eval)
                calibrator_seed = args.seed if args.statistical_seed < 0 else args.statistical_seed
                calibrator = StatisticalEvidenceCalibrator(
                    method=args.statistical_calibrator,
                    seed=calibrator_seed,
                    n_estimators=args.statistical_n_estimators,
                    max_depth=args.statistical_max_depth,
                    min_samples_leaf=args.statistical_min_samples_leaf,
                    max_features=args.statistical_max_features,
                    svc_c=args.statistical_svc_c,
                ).fit_loader(train_case_loader)
                calibrator.save(statistical_calibrator_path)
                statistical_calibrator_config = calibrator.config_dict()
                statistical_calibrator_config['base_weight'] = args.statistical_base_weight
                val_bundle = predict_calibrated_scores(
                    model, calibrator, val_case_loader, actual_device,
                    args.statistical_base_weight,
                )
                validation_eval = evaluate_calibrated_bundle(val_bundle)
                final_eval = None
                val_scores = val_bundle['scores']
                val_labels = val_bundle['labels']
                val_case_ids = val_bundle['case_ids']
                val_case_timestamps = val_bundle['case_timestamps']
                if not args.validation_only:
                    test_bundle = predict_calibrated_scores(
                        model, calibrator, test_case_loader, actual_device,
                        args.statistical_base_weight,
                    )
                    final_eval = evaluate_calibrated_bundle(test_bundle)
                    pred_scores = test_bundle['scores']
                    pred_labels = test_bundle['labels']
                    pred_case_ids = test_bundle['case_ids']
                    pred_case_timestamps = test_bundle['case_timestamps']
            else:
                val_scores, val_labels, val_case_ids, val_case_timestamps = predict_scores(
                    model, val_case_loader, device=actual_device,
                )
                if args.validation_only:
                    final_eval = None
                else:
                    pred_scores, pred_labels, pred_case_ids, pred_case_timestamps = predict_scores(
                        model, test_case_loader, device=actual_device,
                    )

            save_json(validation_eval_path, validation_eval)
            if final_eval is not None:
                save_json(final_eval_path, final_eval)
            val_pred_df = build_prediction_dataframe(
                val_scores, val_labels, val_case_ids, val_case_timestamps, index_to_node
            )
            val_pred_df.to_csv(validation_prediction_path, index=False)
            challenge2_metrics = None
            if not args.validation_only:
                pred_df = build_prediction_dataframe(
                    pred_scores, pred_labels, pred_case_ids, pred_case_timestamps, index_to_node
                )
                pred_df.to_csv(prediction_path, index=False)
                challenge2_metrics = save_challenge2_diagnostics(
                    model,
                    test_case_loader,
                    device=torch.device(train_meta['device']),
                    index_to_node=index_to_node,
                    csv_path=challenge2_role_path,
                    metrics_path=challenge2_role_metrics_path,
                    markdown_path=challenge2_role_markdown_path,
                    hard_symptom_topk=args.response_candidate_topk,
                )
            event_chain_metrics = None
            if args.export_event_chains and not args.validation_only:
                event_chain_metrics = export_event_chains(
                    model,
                    test_case_loader,
                    device=torch.device(train_meta['device']),
                    index_to_node=index_to_node,
                    csv_path=event_chain_path,
                    metrics_path=event_chain_metrics_path,
                    validation_csv_path=chain_validation_path,
                    validation_metrics_path=chain_validation_metrics_path,
                    call_chain_catalog_path=call_chain_catalog_path,
                    topk=args.event_chain_topk,
                    max_depth=args.event_chain_max_depth,
                    edge_topk=args.event_chain_edge_topk,
                    evidence_topk=args.event_chain_evidence_topk,
                    min_strength=args.event_chain_min_strength,
                    allow_fallback=args.allow_event_chain_fallback,
                )

            artifact_summary = None
            if args.export_intermediate_artifacts and not args.validation_only:
                artifact_dir = args.artifact_dir or os.path.join(res_dir, 'intermediate_artifacts')
                artifact_summary = export_intermediate_artifacts(
                    model,
                    test_case_loader,
                    device=torch.device(train_meta['device']),
                    index_to_node=index_to_node,
                    output_dir=artifact_dir,
                )

            training_summary = {
                'train_case_num': len(train_case_windows),
                'val_case_num': len(val_case_windows),
                'test_case_num': len(test_case_windows),
                'case_split': split_meta,
                'epochs': args.train_epochs,
                'learning_rate': args.train_learning_rate,
                'weight_decay': args.train_weight_decay,
                'patience': args.train_patience,
                'checkpoint_path': checkpoint_path,
                'train_history_path': train_history_path,
                'validation_eval_path': validation_eval_path,
                'final_eval_path': None if args.validation_only else final_eval_path,
                'neural_validation_eval_path': neural_validation_eval_path if args.statistical_calibrator != 'none' else None,
                'neural_final_eval_path': neural_final_eval_path if args.statistical_calibrator != 'none' and not args.validation_only else None,
                'statistical_calibrator_path': statistical_calibrator_path if args.statistical_calibrator != 'none' else None,
                'statistical_calibrator_config': statistical_calibrator_config,
                'neural_validation_eval': neural_validation_eval,
                'neural_final_eval': neural_final_eval,
                'validation_prediction_path': validation_prediction_path,
                'prediction_path': None if args.validation_only else prediction_path,
                'event_chain_path': event_chain_path if args.export_event_chains and not args.validation_only else None,
                'event_chain_metrics_path': event_chain_metrics_path if args.export_event_chains and not args.validation_only else None,
                'chain_validation_path': chain_validation_path if args.export_event_chains and not args.validation_only else None,
                'chain_validation_metrics_path': chain_validation_metrics_path if args.export_event_chains and not args.validation_only else None,
                'call_chain_catalog_path': call_chain_catalog_path if args.export_event_chains and not args.validation_only else None,
                'challenge2_role_path': None if args.validation_only else challenge2_role_path,
                'challenge2_role_metrics_path': None if args.validation_only else challenge2_role_metrics_path,
                'challenge2_role_markdown_path': None if args.validation_only else challenge2_role_markdown_path,
                'challenge2_role_metrics': challenge2_metrics,
                'event_chain_metrics': event_chain_metrics,
                'intermediate_artifacts': artifact_summary,
                'validation_eval': validation_eval,
                'final_eval': final_eval,
                'train_meta': train_meta,
            }

    summary = {
        'dataset': args.dataset,
        'seed': args.seed,
        'case_split_protocol': args.case_split_protocol,
        'root_label_semantics': 'positive_candidate_bag_any_hit',
        'label_mapping_granularity': label_mapping_granularity(args.dataset),
        'positive_bag_aggregation': 'max',
        'window_size': args.window_size,
        'train_stride': args.train_stride,
        'batch_size': args.batch_size,
        'test_start_ratio': args.test_start_ratio,
        'train_window_summary': summarize_windows(train_windows),
        'case_window_summary': summarize_windows(case_windows),
        'train_batch_summary': None if train_batch is None else batch_summary(train_batch),
        'case_batch_summary': None if case_batch is None else batch_summary(case_batch),
        'backbone_summary': backbone_summary,
        'modality_gate_summary': modality_gate_summary,
        'anomaly_summary': anomaly_summary,
        'causal_summary': causal_summary,
        'rank_summary': rank_summary,
        'topology_audit_path': None if topology_audit is None else topology_audit_path,
        'topology_audit': topology_audit,
        'training_summary': training_summary,
        'backbone_config': {
            'spatial_hidden_dim': args.spatial_hidden_dim,
            'temporal_hidden_dim': args.temporal_hidden_dim,
            'num_spatial_layers': args.num_spatial_layers,
            'num_temporal_heads': args.num_temporal_heads,
            'dropout': args.dropout,
            'use_temporal_attention': not args.disable_temporal_attention,
            'causal_temporal_attention': not args.disable_causal_temporal_attention,
            'enable_topology_dynamics': args.enable_topology_dynamics,
            'graph_sequence_mode': args.graph_sequence_mode,
            'topology_dropout': args.topology_dropout,
            'topology_feature_mode': args.topology_feature_mode,
            'edge_feature_dim': int(model_settings.get('edge_feature_dim', 0)),
            'observation_prior': model_settings.get('observation_prior', {'enabled': False}),
        },
        'modality_gate_config': {
            'enabled': args.enable_modality_gate,
            'hidden_dim': args.modality_gate_hidden_dim,
            'output_dim': args.modality_gate_output_dim,
            'fusion_mode': args.modality_gate_fusion_mode,
            'residual_alpha': args.modality_gate_residual_alpha,
            'use_gaussian_bias': not args.disable_modality_gaussian_bias,
        },
        'anomaly_config': {
            'fusion_hidden_dim': args.anomaly_hidden_dim,
            'enable_change_point_branch': args.enable_change_point_branch,
            'change_point_weight': args.change_point_weight,
        },
        'causal_config': {
            'hidden_dim': args.causal_hidden_dim,
            'max_lag': args.causal_max_lag,
            'min_strength': args.causal_min_strength,
            'gamma': args.causal_gamma,
            'kernel_size': args.causal_kernel_size,
            'relation_topology_mode': args.relation_topology_mode,
            'use_null_source': not args.disable_null_source,
            'lag_temperature': args.causal_lag_temperature,
            'topology_weight_gamma': args.topology_weight_gamma,
        },
        'rank_config': {
            'rank_hidden_dim': args.rank_hidden_dim,
            'ranker_mode': args.ranker_mode,
            'anchor_init_scale': args.anchor_init_scale,
            'branch_init_scale': args.branch_init_scale,
        },
        'explanation_loss_config': {
            'enabled': args.enable_explanation_losses,
            'root_symptom_loss_weight': args.root_symptom_loss_weight,
            'counterfactual_loss_weight': args.counterfactual_loss_weight,
            'event_chain_loss_weight': args.event_chain_loss_weight,
            'invariance_loss_weight': args.invariance_loss_weight,
            'response_candidate_topk': args.response_candidate_topk,
            'normal_pretrain_epochs': args.normal_pretrain_epochs,
            'normal_auxiliary_loss_weight': args.normal_auxiliary_loss_weight,
            'reconstruction_loss_weight': args.reconstruction_loss_weight,
            'forecast_loss_weight': args.forecast_loss_weight,
            'normal_topology_change_oversample_factor': args.normal_topology_change_oversample_factor,
            'normal_changed_window_count': normal_changed_window_count,
            'normal_window_count': len(train_windows),
        },
        'source_config': {
            'sample_dir': config['path']['sample_dir'],
            'case_dir': config['path']['case_dir'],
            'graph_dir': config['path']['graph_dir'],
            'data_settings': data_settings,
        },
        'event_chain_config': {
            'enabled': args.export_event_chains,
            'topk': args.event_chain_topk,
            'max_depth': args.event_chain_max_depth,
            'edge_topk': args.event_chain_edge_topk,
            'evidence_topk': args.event_chain_evidence_topk,
            'min_strength': args.event_chain_min_strength,
            'allow_fallback': args.allow_event_chain_fallback,
        },
    }

    save_json(summary_path, summary)

    print('REACR-RCA full RCA trainer completed successfully.')
    print(json.dumps(summary, indent=2))
    print(f'summary saved to {summary_path}')


if __name__ == '__main__':
    main()
