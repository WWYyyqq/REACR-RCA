import argparse
import json
import os

import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from models.REACT_RCA_AnomalyHead_V1 import MultiHeadAnomalyHead, summarize_anomaly_output
from models.REACT_RCA_CausalHead_V1 import RunInspiredCausalHead, summarize_causal_output
from models.REACT_RCA_CausalRanker_V1 import CausalPropagationRanker, summarize_rank_output
from models.REACT_RCA_DSBGDataset_V1 import (
    batch_summary,
    build_case_windows,
    build_sliding_windows,
    create_dsbg_dataloader,
    summarize_windows,
)
from models.REACT_RCA_DynamicBackbone_V1 import DynamicSpatialTemporalBackbone, summarize_backbone_output
from models.REACT_RCA_Trainer_V1 import (
    DyCauRCAModel,
    build_prediction_dataframe,
    predict_scores,
    save_challenge2_diagnostics,
    train_dycau,
)
from models.REACT_RCA_EventChainExplainer_V1 import export_event_chains
from models.REACR_RCA_ArtifactExporter_V1 import export_intermediate_artifacts
from models.REACT_RCA_ModalityGate_V1 import (
    ModalityReliabilityGate,
    parse_modality_dims,
    summarize_modality_gate_output,
    valid_modality_dims,
)
from utils.public_functions import load_init, load_samples


def parse_args():
    parser = argparse.ArgumentParser(description='REACR-RCA stage-1 full RCA trainer')
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
    parser.add_argument('--anomaly_hidden_dim', type=int, default=64)
    change_point_group = parser.add_mutually_exclusive_group()
    change_point_group.add_argument('--enable_change_point_branch', dest='enable_change_point_branch', action='store_true', default=None)
    change_point_group.add_argument('--disable_change_point_branch', dest='enable_change_point_branch', action='store_false')
    parser.add_argument('--change_point_weight', type=float, default=1.0)
    parser.add_argument('--causal_hidden_dim', type=int, default=64)
    parser.add_argument('--causal_max_lag', type=int, default=3)
    parser.add_argument('--causal_min_strength', type=float, default=0.0)
    parser.add_argument('--causal_gamma', type=float, default=0.5)
    parser.add_argument('--causal_kernel_size', type=int, default=5)
    parser.add_argument('--rank_hidden_dim', type=int, default=64)
    parser.add_argument('--ranker_mode', choices=['legacy', 'root_symptom', 'counterfactual'], default='legacy')
    parser.add_argument('--anchor_init_scale', type=float, default=1.0)
    parser.add_argument('--branch_init_scale', type=float, default=0.1)
    parser.add_argument('--train_epochs', type=int, default=0)
    parser.add_argument('--train_learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_weight_decay', type=float, default=0.0)
    parser.add_argument('--train_patience', type=int, default=10)
    parser.add_argument('--enable_explanation_losses', action='store_true')
    parser.add_argument('--root_symptom_loss_weight', type=float, default=0.0)
    parser.add_argument('--counterfactual_loss_weight', type=float, default=0.0)
    parser.add_argument('--event_chain_loss_weight', type=float, default=0.0)
    parser.add_argument('--invariance_loss_weight', type=float, default=0.0)
    parser.add_argument('--hard_symptom_topk', type=int, default=5)
    parser.add_argument('--feedback_sample_ratio', type=float, default=-1.0)
    parser.add_argument('--test_start_ratio', type=float, default=0.5)
    parser.add_argument('--case_split_protocol', choices=['legacy', 'fair', 'time_strict'], default='legacy')
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


def build_backbone(args, feat_dim: int) -> DynamicSpatialTemporalBackbone:
    return DynamicSpatialTemporalBackbone(
        in_dim=feat_dim,
        spatial_hidden_dim=args.spatial_hidden_dim,
        temporal_hidden_dim=args.temporal_hidden_dim,
        num_spatial_layers=args.num_spatial_layers,
        dropout=args.dropout,
        num_temporal_heads=args.num_temporal_heads,
        use_temporal_attention=not args.disable_temporal_attention,
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

    backbone = build_backbone(args, feat_dim=model_feat_dim)
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
    return DyCauRCAModel(backbone, anomaly_head, causal_head, ranker, modality_gate=modality_gate)


def split_cases_for_training(
    cases,
    feedback_sample_ratio: float,
    default_ratio: float,
    protocol: str = 'legacy',
    test_start_ratio: float = 0.5,
):
    ratio = feedback_sample_ratio if 0 < feedback_sample_ratio < 1 else default_ratio
    cases = cases.reset_index(drop=True).copy()
    if 'case_id' not in cases.columns:
        cases.insert(0, 'case_id', np.arange(len(cases), dtype=int))
    if protocol == 'time_strict':
        cases = cases.sort_values(['timestamp', 'case_id']).reset_index(drop=True)

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

    suffix = f'_{args.tag}' if args.tag else ''
    res_dir = args.output_dir if args.output_dir else f"res/{args.dataset}_reacr_rca_stage1{suffix}"
    os.makedirs(res_dir, exist_ok=True)

    summary_path = os.path.join(res_dir, 'stage1_dsbg_summary.json')
    train_history_path = os.path.join(res_dir, 'train_history.json')
    validation_eval_path = os.path.join(res_dir, 'validation_eval.json')
    final_eval_path = os.path.join(res_dir, 'final_eval.json')
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

    save_json(config_path, vars(args))

    train_samples, test_samples = load_samples(config['path']['sample_dir'])
    cases = pd.read_csv(config['path']['case_dir'])
    node_hash, _, _ = load_init(config['path']['graph_dir'])
    index_to_node = {idx: node for node, idx in node_hash.items()}

    max_train_windows = None if args.max_train_windows <= 0 else args.max_train_windows
    max_case_windows = None if args.max_case_windows <= 0 else args.max_case_windows

    train_windows = build_sliding_windows(
        train_samples,
        window_size=args.window_size,
        stride=args.train_stride,
        max_windows=max_train_windows,
    )
    case_windows = build_case_windows(
        cases,
        test_samples,
        node_hash,
        window_size=args.window_size,
        max_cases=max_case_windows,
    )

    train_loader = create_dsbg_dataloader(train_windows, batch_size=args.batch_size, shuffle=False, with_labels=False)
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
            )
            train_case_windows = build_case_windows(train_cases, test_samples, node_hash, window_size=args.window_size)
            val_case_windows = build_case_windows(val_cases, test_samples, node_hash, window_size=args.window_size)
            test_case_windows = build_case_windows(test_cases, test_samples, node_hash, window_size=args.window_size)
            train_case_loader = create_dsbg_dataloader(train_case_windows, batch_size=args.batch_size, shuffle=True, with_labels=True)
            val_case_loader = create_dsbg_dataloader(val_case_windows, batch_size=args.batch_size, shuffle=False, with_labels=True)
            test_case_loader = create_dsbg_dataloader(test_case_windows, batch_size=args.batch_size, shuffle=False, with_labels=True)

            model = build_model(args, feat_dim=int(case_batch['feat_dim']), node_num=int(case_batch['node_num']), config=config)
            model, history, final_eval, validation_eval, train_meta = train_dycau(
                model,
                train_case_loader,
                device=device,
                val_loader=val_case_loader,
                test_loader=test_case_loader,
                epochs=args.train_epochs,
                learning_rate=args.train_learning_rate,
                weight_decay=args.train_weight_decay,
                patience=args.train_patience,
                enable_explanation_losses=args.enable_explanation_losses,
                root_symptom_loss_weight=args.root_symptom_loss_weight,
                counterfactual_loss_weight=args.counterfactual_loss_weight,
                event_chain_loss_weight=args.event_chain_loss_weight,
                invariance_loss_weight=args.invariance_loss_weight,
            )

            torch.save(model.state_dict(), checkpoint_path)
            save_json(train_history_path, history)
            save_json(validation_eval_path, validation_eval)
            save_json(final_eval_path, final_eval)

            val_scores, val_labels, val_case_ids, val_case_timestamps = predict_scores(model, val_case_loader, device=torch.device(train_meta['device']))
            val_pred_df = build_prediction_dataframe(val_scores, val_labels, val_case_ids, val_case_timestamps, index_to_node)
            val_pred_df.to_csv(validation_prediction_path, index=False)

            pred_scores, pred_labels, pred_case_ids, pred_case_timestamps = predict_scores(model, test_case_loader, device=torch.device(train_meta['device']))
            pred_df = build_prediction_dataframe(pred_scores, pred_labels, pred_case_ids, pred_case_timestamps, index_to_node)
            pred_df.to_csv(prediction_path, index=False)
            challenge2_metrics = save_challenge2_diagnostics(
                model,
                test_case_loader,
                device=torch.device(train_meta['device']),
                index_to_node=index_to_node,
                csv_path=challenge2_role_path,
                metrics_path=challenge2_role_metrics_path,
                markdown_path=challenge2_role_markdown_path,
                hard_symptom_topk=args.hard_symptom_topk,
            )
            event_chain_metrics = None
            if args.export_event_chains:
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
            if args.export_intermediate_artifacts:
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
                'final_eval_path': final_eval_path,
                'validation_prediction_path': validation_prediction_path,
                'prediction_path': prediction_path,
                'event_chain_path': event_chain_path if args.export_event_chains else None,
                'event_chain_metrics_path': event_chain_metrics_path if args.export_event_chains else None,
                'chain_validation_path': chain_validation_path if args.export_event_chains else None,
                'chain_validation_metrics_path': chain_validation_metrics_path if args.export_event_chains else None,
                'call_chain_catalog_path': call_chain_catalog_path if args.export_event_chains else None,
                'challenge2_role_path': challenge2_role_path,
                'challenge2_role_metrics_path': challenge2_role_metrics_path,
                'challenge2_role_markdown_path': challenge2_role_markdown_path,
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
        'training_summary': training_summary,
        'backbone_config': {
            'spatial_hidden_dim': args.spatial_hidden_dim,
            'temporal_hidden_dim': args.temporal_hidden_dim,
            'num_spatial_layers': args.num_spatial_layers,
            'num_temporal_heads': args.num_temporal_heads,
            'dropout': args.dropout,
            'use_temporal_attention': not args.disable_temporal_attention,
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
            'hard_symptom_topk': args.hard_symptom_topk,
        },
        'source_config': {
            'sample_dir': config['path']['sample_dir'],
            'case_dir': config['path']['case_dir'],
            'graph_dir': config['path']['graph_dir'],
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

    print('REACR-RCA stage-1 full RCA trainer is ready.')
    print(json.dumps(summary, indent=2))
    print(f'summary saved to {summary_path}')


if __name__ == '__main__':
    main()
