import argparse
import hashlib
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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_DIR = ROOT / "res" / "reacr_rca_final_study"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.reacr_rca_dsbg_dataset import root_label_matches_node_name


def implementation_fingerprint():
    """Hash code/config inputs so corrected runs cannot reuse stale results."""

    paths = [
        ROOT / 'reacr_rca_pipeline.py',
        ROOT / 'tools' / 'reacr_rca_final_study_runner.py',
        ROOT / 'tools' / 'convert_d1_weighted_dynamic.py',
        ROOT / 'models' / 'rc_scorer.py',
        ROOT / 'models' / 'evaluation.py',
        ROOT / 'models' / 'train.py',
        ROOT / 'models' / 'layers.py',
        ROOT / 'utils' / 'public_functions.py',
        *sorted((ROOT / 'config').glob('*.yaml')),
        *sorted((ROOT / 'models').glob('reacr_rca_*.py')),
    ]
    digest = hashlib.sha256()
    relative_paths = []
    for path in sorted(set(paths), key=lambda item: str(item)):
        relative = str(path.relative_to(ROOT))
        relative_paths.append(relative)
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    version_path = ROOT / 'VERSION'
    version = version_path.read_text(encoding='utf-8').strip() if version_path.exists() else None
    return {
        'version': version,
        'sha256': digest.hexdigest(),
        'files': relative_paths,
    }


MAIN_WEIGHTS = {
    "root_symptom_loss_weight": 0.1,
    "counterfactual_loss_weight": 0.1,
    "event_chain_loss_weight": 0.05,
    "invariance_loss_weight": 0.05,
}


VARIANTS = {
    "legacy": {
        "ranker_mode": "legacy",
        "enable_explanation_losses": False,
        "weights": {},
        "role": "ablation",
    },
    "root_symptom": {
        "ranker_mode": "root_symptom",
        "enable_explanation_losses": False,
        "weights": {},
        "role": "ablation",
    },
    "source_hypothesis": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": False,
        "weights": {},
        "role": "ablation",
    },
    "source_hypothesis_losses": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "role": "main",
    },
    "no_root_symptom_loss": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": {**MAIN_WEIGHTS, "root_symptom_loss_weight": 0.0},
        "role": "ablation",
    },
    "no_source_hypothesis_loss": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": {**MAIN_WEIGHTS, "counterfactual_loss_weight": 0.0},
        "role": "ablation",
    },
    "no_event_chain_loss": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": {**MAIN_WEIGHTS, "event_chain_loss_weight": 0.0},
        "role": "ablation",
    },
    "no_invariance_loss": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": {**MAIN_WEIGHTS, "invariance_loss_weight": 0.0},
        "role": "ablation",
    },
    "no_topology_change_features": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "topology_feature_mode": "static_only",
        "role": "ablation",
    },
    "topology_dynamic_features_only": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "topology_feature_mode": "dynamic_only",
        "role": "ablation",
    },
    "frozen_window_topology": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "graph_sequence_mode": "freeze_first",
        "role": "ablation",
    },
    "no_explicit_topology_encoder": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "enable_topology_dynamics": False,
        "role": "ablation",
    },
    "no_relation_topology_constraint": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "relation_topology_mode": "none",
        "role": "ablation",
    },
    "no_lag_relation": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        # A zero lag budget makes the relation support empty and therefore
        # removes lagged directed-relation evidence without changing the graph
        # encoder, anomaly scores, ranker architecture, or training budget.
        "causal_max_lag": 0,
        "role": "challenge2",
    },
    "no_normal_dynamics_training": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "normal_pretrain_epochs": 0,
        "normal_auxiliary_loss_weight": 0.0,
        "role": "ablation",
    },
    "no_topology_change_oversampling": {
        "ranker_mode": "source_hypothesis",
        "enable_explanation_losses": True,
        "weights": MAIN_WEIGHTS,
        "normal_topology_change_oversample_factor": 1.0,
        "role": "ablation",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final REACR-RCA root-cause localization study.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SN", "TT", "D1_weighted"],
        help="Current paper datasets; D2 is retained only for historical fixed-topology checks.",
    )
    parser.add_argument("--base_dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["all", "smoke", "main", "ablations", "baseline", "summarize"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--main_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ablation_seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--baseline_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke_seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--smoke_epochs", type=int, default=1)
    parser.add_argument("--normal_pretrain_epochs", type=int, default=3)
    parser.add_argument("--normal_auxiliary_loss_weight", type=float, default=0.1)
    parser.add_argument("--normal_topology_change_oversample_factor", type=float, default=4.0)
    parser.add_argument("--feedback_sample_ratio", type=float, default=None)
    parser.add_argument("--test_start_ratio", type=float, default=0.5)
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_event_chains", action="store_true")
    return parser.parse_args()


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, tuple):
            return [clean(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(data), f, indent=2, ensure_ascii=False)


def load_config(dataset):
    with open(ROOT / "config" / f"{dataset}.yaml", "r", encoding="utf-8") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def configured_feedback_sample_ratio(dataset):
    """Return the audited feedback ratio, including integer-valued YAML input."""

    config = load_config(pipeline_dataset(dataset))
    value = config.get('feedback', {}).get('sample_num', 0.3)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 0.3


def dataset_defaults(dataset):
    if dataset in {'SN', 'TT'}:
        return {
            "batch_size": 4,
            "enable_change_point_branch": False,
            "change_point_weight": 1.0,
            "causal_hidden_dim": 64,
            "causal_gamma": 0.5,
            "causal_kernel_size": 5,
            "enable_modality_gate": False,
            "modality_gate_fusion_mode": "encoded",
            "modality_gate_residual_alpha": 0.2,
            "dropout": 0.1,
            "branch_init_scale": 0.1,
            "train_weight_decay": 0.0,
            "train_patience": 7 if dataset == 'TT' else 6,
            "train_epochs": 25 if dataset == 'TT' else 20,
        }
    if dataset == "D2":
        return {
            "batch_size": 4,
            "enable_change_point_branch": True,
            "change_point_weight": 0.05,
            "causal_hidden_dim": 16,
            "causal_gamma": 0.2,
            "causal_kernel_size": 3,
            "enable_modality_gate": True,
            "modality_gate_fusion_mode": "raw_residual",
            "modality_gate_residual_alpha": 0.1,
            "dropout": 0.15,
            "branch_init_scale": 0.075,
            "train_weight_decay": 1e-5,
            "train_patience": 8,
            "train_epochs": 20,
        }
    return {
        "batch_size": 8,
        "enable_change_point_branch": False,
        "change_point_weight": 1.0,
        "causal_hidden_dim": 64,
        "causal_gamma": 0.5,
        "causal_kernel_size": 5,
        # D1 metrics occupy two non-contiguous spans.  The corrected gate uses
        # explicit indices and writes residual features back in original order.
        "enable_modality_gate": True,
        "modality_gate_fusion_mode": "raw_residual",
        "modality_gate_residual_alpha": 0.1,
        "dropout": 0.1,
        "branch_init_scale": 0.1,
        "train_weight_decay": 0.0,
        "train_patience": 10,
        "train_epochs": 20,
    }


def resolved_train_epochs(dataset, requested_epochs):
    if requested_epochs is not None:
        return int(requested_epochs)
    return int(dataset_defaults(dataset)['train_epochs'])


def pipeline_dataset(dataset):
    # Use raw-chain validation only for the canonical D1 full run; D2 has no raw call-chain catalog.
    return "D1_raw_chain" if dataset == "D1" and (ROOT / "config" / "D1_raw_chain.yaml").exists() else dataset


def label_mapping_granularity(dataset):
    if str(dataset).startswith('D1'):
        return 'mixed_service_to_instance_candidate_bag'
    if str(dataset).startswith('D2'):
        return 'exact_single_instance_candidate'
    return 'audited_positive_candidate_bag'


def ensure_dir(path, overwrite=False):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_dir(base_dir, dataset, method, seed):
    return base_dir / dataset / method / f"seed{seed}"


def build_pipeline_cmd(dataset, variant_name, variant, seed, epochs, out_dir, args, smoke=False):
    defaults = dataset_defaults(dataset)
    pipeline_name = pipeline_dataset(dataset)
    config = load_config(pipeline_name)
    case_split_protocol = config.get('data_settings', {}).get(
        'case_split_protocol', 'time_strict'
    )
    feedback_sample_ratio = args.feedback_sample_ratio
    if feedback_sample_ratio is None:
        feedback_sample_ratio = configured_feedback_sample_ratio(dataset)
    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "reacr_rca_pipeline.py"),
        "--dataset",
        pipeline_name,
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--train_epochs",
        str(epochs),
        "--batch_size",
        str(defaults["batch_size"]),
        "--feedback_sample_ratio",
        str(feedback_sample_ratio),
        "--test_start_ratio",
        str(args.test_start_ratio),
        "--case_split_protocol",
        case_split_protocol,
        "--ranker_mode",
        variant["ranker_mode"],
        "--relation_topology_mode",
        variant.get("relation_topology_mode", "undirected"),
        "--graph_sequence_mode",
        variant.get("graph_sequence_mode", "observed"),
        "--topology_feature_mode",
        variant.get("topology_feature_mode", "full"),
        "--normal_pretrain_epochs",
        str(variant.get("normal_pretrain_epochs", args.normal_pretrain_epochs)),
        "--normal_auxiliary_loss_weight",
        str(variant.get("normal_auxiliary_loss_weight", args.normal_auxiliary_loss_weight)),
        "--normal_topology_change_oversample_factor",
        str(
            variant.get(
                "normal_topology_change_oversample_factor",
                args.normal_topology_change_oversample_factor,
            )
        ),
        "--reconstruction_loss_weight",
        "1.0",
        "--forecast_loss_weight",
        "1.0",
        "--change_point_weight",
        str(defaults["change_point_weight"]),
        "--causal_hidden_dim",
        str(defaults["causal_hidden_dim"]),
        "--causal_max_lag",
        str(variant.get("causal_max_lag", 3)),
        "--causal_gamma",
        str(defaults["causal_gamma"]),
        "--causal_kernel_size",
        str(defaults["causal_kernel_size"]),
        "--dropout",
        str(defaults["dropout"]),
        "--branch_init_scale",
        str(defaults["branch_init_scale"]),
        "--train_weight_decay",
        str(defaults["train_weight_decay"]),
        "--train_patience",
        str(defaults["train_patience"]),
        "--output_dir",
        str(out_dir),
    ]
    cmd.append(
        "--enable_topology_dynamics"
        if variant.get("enable_topology_dynamics", True)
        else "--disable_topology_dynamics"
    )
    cmd.append("--enable_change_point_branch" if defaults["enable_change_point_branch"] else "--disable_change_point_branch")
    cmd.append("--enable_modality_gate" if defaults["enable_modality_gate"] else "--disable_modality_gate")
    if defaults["enable_modality_gate"]:
        cmd.extend([
            "--modality_gate_fusion_mode",
            defaults["modality_gate_fusion_mode"],
            "--modality_gate_residual_alpha",
            str(defaults["modality_gate_residual_alpha"]),
        ])
    if variant["enable_explanation_losses"]:
        cmd.append("--enable_explanation_losses")
        weights = variant["weights"]
        for key in [
            "root_symptom_loss_weight",
            "counterfactual_loss_weight",
            "event_chain_loss_weight",
            "invariance_loss_weight",
        ]:
            cmd.extend([f"--{key}", str(weights.get(key, 0.0))])
    else:
        cmd.append("--disable_explanation_losses")
    if not args.skip_event_chains:
        cmd.extend([
            "--export_event_chains",
            "--allow_event_chain_fallback",
            "--event_chain_topk",
            "3",
            "--event_chain_max_depth",
            "3",
        ])
    if smoke:
        cmd.extend(["--max_case_windows", "8"])
    return cmd


def run_pipeline(dataset, variant_name, seed, epochs, base_dir, args, smoke=False):
    variant = VARIANTS[variant_name]
    out_dir = run_dir(base_dir, dataset, variant_name, seed)
    cmd = build_pipeline_cmd(dataset, variant_name, variant, seed, epochs, out_dir, args, smoke=smoke)
    run_manifest = {
        "cmd": cmd,
        "variant": variant_name,
        "dataset": dataset,
        "seed": seed,
        "implementation": implementation_fingerprint(),
    }
    manifest_path = out_dir / "runner_command.json"
    done = (out_dir / "final_eval.json").exists() and (out_dir / "validation_eval.json").exists()
    if not args.skip_event_chains:
        done = done and (out_dir / "event_chains.csv").exists() and (out_dir / "chain_validation.csv").exists()
    done = done and (out_dir / "challenge2_role_diagnostic.csv").exists() and (out_dir / "challenge2_role_summary.json").exists()
    if manifest_path.exists() and not args.overwrite:
        previous_manifest = load_json(manifest_path)
        if previous_manifest != run_manifest:
            raise RuntimeError(
                f'existing run at {out_dir} was produced by a different command or '
                'implementation; use --overwrite or a new --base_dir'
            )
    if done and not args.overwrite:
        if not manifest_path.exists():
            raise RuntimeError(
                f'existing run at {out_dir} has no implementation manifest; '
                'use --overwrite or a new --base_dir'
            )
        return out_dir
    ensure_dir(out_dir, overwrite=args.overwrite)
    save_json(manifest_path, run_manifest)
    with open(out_dir / "run.log", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    return out_dir


def _node_matches_root_label(node: str, label: str, separator: str = '-') -> bool:
    return root_label_matches_node_name(label, node, label_separator=separator)


def metric_from_predictions(df, ks=(1, 3, 5)):
    rows = []
    for _, row in df.iterrows():
        substring_label_match = False
        if "label_nodes" in row.index and pd.notna(row.get("label_nodes", "")):
            labels = set(str(row.get("label_nodes", "")).split(","))
        elif "cmdb_id" in row.index and pd.notna(row.get("cmdb_id", "")):
            labels = {str(row.get("cmdb_id"))}
            substring_label_match = True
        else:
            labels = set()
        labels = {item for item in labels if item}
        result = {}
        for k in ks:
            pred_values = []
            for rank in range(1, k + 1):
                value = str(row.get(f"Top{rank}", ""))
                pred_values.append(value.split(":", 1)[0])
            if substring_label_match:
                hit = any(
                    _node_matches_root_label(pred, label)
                    for label in labels
                    for pred in pred_values
                )
            else:
                hit = bool(labels.intersection(pred_values))
            result[f"A@{k}"] = float(hit)
        rows.append(result)
    if not rows:
        return {f"A@{k}": math.nan for k in ks}
    return {f"A@{k}": float(np.mean([row[f"A@{k}"] for row in rows])) for k in ks}


def topology_stratified_prediction_metrics(predictions, windows, ks=(1, 3, 5)):
    from tools.reacr_rca_topology_audit import audit_windows

    report = audit_windows(windows, max_window_records=len(windows))
    changed_case_ids = {
        int(record['case_id'])
        for record in report['changed_window_records']
        if record.get('case_id') is not None
    }
    case_ids = pd.to_numeric(predictions['case_id'], errors='coerce')
    changed_mask = case_ids.isin(changed_case_ids)
    result = {}
    for name, mask in (('changed', changed_mask), ('stable', ~changed_mask)):
        subset = predictions.loc[mask]
        result[name] = {
            'sample_count': int(len(subset)),
            'metrics': metric_from_predictions(subset, ks=ks),
        }
    return result


def run_deephunt_baseline(dataset, seed, base_dir, args):
    from models.evaluation import fd_test, test as reconstruction_test
    from models.reacr_rca_dsbg_dataset import build_case_windows
    from models.rc_scorer import feedback, naive_scorer
    from models.train import train as train_deephunt
    from reacr_rca_pipeline import split_cases_for_training
    from utils.public_functions import load_init, load_samples

    out_dir = run_dir(base_dir, dataset, "DeepHunt", seed)
    config = load_config(dataset)
    data_settings = config.get('data_settings', {})
    case_split_protocol = data_settings.get('case_split_protocol', 'time_strict')
    feedback_sample_ratio = args.feedback_sample_ratio
    if feedback_sample_ratio is None:
        feedback_sample_ratio = configured_feedback_sample_ratio(dataset)
    baseline_manifest = {
        'dataset': dataset,
        'method': 'DeepHunt',
        'seed': seed,
        'feedback_sample_ratio': feedback_sample_ratio,
        'test_start_ratio': args.test_start_ratio,
        'case_split_protocol': case_split_protocol,
        'root_label_semantics': 'positive_candidate_bag_any_hit',
        'label_mapping_granularity': label_mapping_granularity(dataset),
        'positive_bag_aggregation': 'max',
        'implementation': implementation_fingerprint(),
    }
    manifest_path = out_dir / 'runner_command.json'
    done = (out_dir / "final_eval.json").exists() and (out_dir / "validation_eval.json").exists()
    if manifest_path.exists() and not args.overwrite:
        if load_json(manifest_path) != baseline_manifest:
            raise RuntimeError(
                f'existing baseline at {out_dir} was produced by different code or settings; '
                'use --overwrite or a new --base_dir'
            )
    if done and not args.overwrite:
        if not manifest_path.exists():
            raise RuntimeError(
                f'existing baseline at {out_dir} has no implementation manifest; '
                'use --overwrite or a new --base_dir'
            )
        return out_dir
    ensure_dir(out_dir, overwrite=args.overwrite)
    save_json(manifest_path, baseline_manifest)
    set_random_seed(seed)
    train_samples, test_samples = load_samples(config["path"]["sample_dir"])
    node_hash, _, _ = load_init(config["path"]["graph_dir"])
    cases = pd.read_csv(ROOT / config["path"]["case_dir"]).reset_index(drop=True).copy()
    if "case_id" not in cases.columns:
        cases.insert(0, "case_id", np.arange(len(cases), dtype=int))
    train_cases, val_cases, test_cases, split_meta = split_cases_for_training(
        cases,
        feedback_sample_ratio,
        feedback_sample_ratio,
        protocol=case_split_protocol,
        test_start_ratio=args.test_start_ratio,
        group_column=data_settings.get('case_group_column'),
    )

    input_samples = train_samples if config["train_samples_num"] == "whole" else train_samples[: config["train_samples_num"]]
    model = train_deephunt(input_samples, config["model_param"])
    with open(out_dir / "naive_model.pkl", "wb") as f:
        pickle.dump(model, f)
    loss_df = reconstruction_test(model, test_samples, node_hash, config["model_param"]["feat_span"])
    fd_model = feedback(model, train_cases, test_samples, node_hash, config["feedback"])
    val_df = fd_test(val_cases, fd_model, test_samples, node_hash, config["feedback"]["window_size"])
    test_df = fd_test(test_cases, fd_model, test_samples, node_hash, config["feedback"]["window_size"])
    val_df.to_csv(out_dir / "validation_predictions.csv", index=False)
    test_df.to_csv(out_dir / "eval_predictions.csv", index=False)
    naive_test_df = naive_scorer(test_cases, test_samples, loss_df, node_hash, window_size=config["feedback"]["window_size"])
    naive_test_df.to_csv(out_dir / "naive_eval_predictions.csv", index=False)

    val_windows = build_case_windows(
        val_cases,
        test_samples,
        node_hash,
        config['feedback']['window_size'],
    )
    test_windows = build_case_windows(
        test_cases,
        test_samples,
        node_hash,
        config['feedback']['window_size'],
    )
    validation_eval = {
        "metrics": metric_from_predictions(val_df),
        "score_std": None,
        "loss": None,
        "topology_strata": topology_stratified_prediction_metrics(val_df, val_windows),
    }
    final_eval = {
        "metrics": metric_from_predictions(test_df),
        "score_std": None,
        "loss": None,
        "topology_strata": topology_stratified_prediction_metrics(test_df, test_windows),
    }
    save_json(out_dir / "validation_eval.json", validation_eval)
    save_json(out_dir / "final_eval.json", final_eval)
    save_json(out_dir / "run_config.json", {
        "dataset": dataset,
        "method": "DeepHunt",
        "seed": seed,
        "case_split_protocol": case_split_protocol,
        "feedback_sample_ratio": feedback_sample_ratio,
        "test_start_ratio": args.test_start_ratio,
        "root_label_semantics": "positive_candidate_bag_any_hit",
        "label_mapping_granularity": label_mapping_granularity(dataset),
        "positive_bag_aggregation": "max",
        "train_case_count": len(train_cases),
        "val_case_count": len(val_cases),
        "test_case_count": len(test_cases),
        "split_meta": split_meta,
    })
    return out_dir


def collapse_flag(metrics, score_std):
    a1 = metrics.get("A@1")
    a3 = metrics.get("A@3")
    a5 = metrics.get("A@5")
    if score_std is not None and score_std < 1e-4:
        return True
    if a1 is None or a3 is None or a5 is None:
        return False
    return a1 <= 0.05 and a3 <= 0.10 and a5 <= 0.15


def collect_row(dataset, method, run_path):
    final_path = run_path / "final_eval.json"
    if not final_path.exists():
        return None
    final_eval = load_json(final_path)
    validation_eval = load_json(run_path / "validation_eval.json") if (run_path / "validation_eval.json").exists() else {}
    run_config = load_json(run_path / "run_config.json") if (run_path / "run_config.json").exists() else {}
    metrics = final_eval.get("metrics", final_eval.get("primary", {}))
    val_metrics = validation_eval.get("metrics", validation_eval.get("primary", {}))
    chain_metrics = load_json(run_path / "chain_validation_metrics.json") if (run_path / "chain_validation_metrics.json").exists() else {}
    event_metrics = load_json(run_path / "event_chain_metrics.json") if (run_path / "event_chain_metrics.json").exists() else {}
    challenge2_metrics = load_json(run_path / "challenge2_role_summary.json") if (run_path / "challenge2_role_summary.json").exists() else {}
    score_std = final_eval.get("score_std")
    topology_strata = final_eval.get('topology_strata', {})
    changed = topology_strata.get('changed', {})
    stable = topology_strata.get('stable', {})
    changed_metrics = changed.get('metrics', {})
    stable_metrics = stable.get('metrics', {})
    try:
        display_run_dir = str(run_path.resolve().relative_to(ROOT))
    except ValueError:
        display_run_dir = str(run_path.resolve())
    return {
        "dataset": dataset,
        "method": method,
        "seed": run_config.get("seed", int(run_path.name.replace("seed", "")) if run_path.name.startswith("seed") else None),
        "split_protocol": run_config.get("case_split_protocol", "time_strict"),
        "ranker_mode": run_config.get("ranker_mode", VARIANTS.get(method, {}).get("ranker_mode", "baseline")),
        "enable_explanation_losses": run_config.get("enable_explanation_losses", VARIANTS.get(method, {}).get("enable_explanation_losses", False)),
        "A@1": metrics.get("A@1"),
        "A@3": metrics.get("A@3"),
        "A@5": metrics.get("A@5"),
        "val_A@1": val_metrics.get("A@1"),
        "val_A@3": val_metrics.get("A@3"),
        "val_A@5": val_metrics.get("A@5"),
        "score_std": score_std,
        "changed_case_count": changed.get('sample_count'),
        "changed_A@1": changed_metrics.get('A@1'),
        "changed_A@3": changed_metrics.get('A@3'),
        "changed_A@5": changed_metrics.get('A@5'),
        "stable_case_count": stable.get('sample_count'),
        "stable_A@1": stable_metrics.get('A@1'),
        "stable_A@3": stable_metrics.get('A@3'),
        "stable_A@5": stable_metrics.get('A@5'),
        "collapse": collapse_flag(metrics, score_std),
        "strict_chain_coverage": event_metrics.get("strict_chain_coverage"),
        "label_candidate_chain_hit_at_topk_strict": event_metrics.get(
            "label_candidate_chain_hit_at_topk_strict",
            event_metrics.get("label_chain_hit_at_topk_strict"),
        ),
        "label_candidate_hit_at_topk": event_metrics.get(
            "label_candidate_hit_at_topk",
            event_metrics.get("label_root_hit_at_topk"),
        ),
        # Deprecated collection aliases retained for old result directories.
        "label_chain_hit_at_topk_strict": event_metrics.get("label_chain_hit_at_topk_strict"),
        "label_root_hit_at_topk": event_metrics.get("label_root_hit_at_topk"),
        "exact_call_path_exists": chain_metrics.get("mean_exact_call_path_exists"),
        "edge_presence_rate_in_case_window": chain_metrics.get("mean_edge_presence_rate_in_case_window"),
        "challenge2_response_candidate_error_rate": challenge2_metrics.get(
            "response_candidate_error_rate",
            challenge2_metrics.get("hard_symptom_error_rate"),
        ),
        "challenge2_candidate_membership_auc": challenge2_metrics.get(
            "candidate_membership_auc",
            challenge2_metrics.get("role_auc"),
        ),
        "challenge2_source_role_gap": challenge2_metrics.get(
            "source_role_gap_mean",
            challenge2_metrics.get("root_role_gap_mean"),
        ),
        "challenge2_response_role_gap": challenge2_metrics.get("response_role_gap_mean"),
        "challenge2_label_vs_response_candidate_margin": challenge2_metrics.get(
            "label_vs_response_candidate_margin_mean",
            challenge2_metrics.get("root_vs_hard_symptom_margin_mean"),
        ),
        # Deprecated collection aliases retained for old result directories.
        "challenge2_hard_symptom_error_rate": challenge2_metrics.get("hard_symptom_error_rate"),
        "challenge2_role_auc": challenge2_metrics.get("role_auc"),
        "challenge2_root_role_gap": challenge2_metrics.get("root_role_gap_mean"),
        "challenge2_source_gap": challenge2_metrics.get("source_gap_mean"),
        "challenge2_victim_gap": challenge2_metrics.get("victim_gap_mean"),
        "challenge2_incoming_explainedness_gap": challenge2_metrics.get("incoming_explainedness_gap_mean"),
        "challenge2_root_vs_hard_symptom_margin": challenge2_metrics.get("root_vs_hard_symptom_margin_mean"),
        "run_dir": display_run_dir,
    }


def collect_results(base_dir):
    rows = []
    for dataset_dir in sorted(base_dir.iterdir() if base_dir.exists() else []):
        if not dataset_dir.is_dir() or dataset_dir.name == "summary":
            continue
        for method_dir in sorted(dataset_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            for seed_dir in sorted(method_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                row = collect_row(dataset_dir.name, method_dir.name, seed_dir)
                if row:
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_group(df):
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby(["dataset", "method"], dropna=False)
    summary = grouped.agg(
        runs=("A@1", "count"),
        A1_mean=("A@1", "mean"),
        A1_std=("A@1", "std"),
        A3_mean=("A@3", "mean"),
        A3_std=("A@3", "std"),
        A5_mean=("A@5", "mean"),
        A5_std=("A@5", "std"),
        score_std_mean=("score_std", "mean"),
        collapse_rate=("collapse", "mean"),
        val_A1_mean=("val_A@1", "mean"),
        val_A3_mean=("val_A@3", "mean"),
        val_A5_mean=("val_A@5", "mean"),
        changed_case_count=("changed_case_count", "max"),
        changed_A1_mean=("changed_A@1", "mean"),
        changed_A3_mean=("changed_A@3", "mean"),
        changed_A5_mean=("changed_A@5", "mean"),
        stable_case_count=("stable_case_count", "max"),
        stable_A1_mean=("stable_A@1", "mean"),
        stable_A3_mean=("stable_A@3", "mean"),
        stable_A5_mean=("stable_A@5", "mean"),
        exact_call_path_exists_mean=("exact_call_path_exists", "mean"),
        edge_presence_rate_mean=("edge_presence_rate_in_case_window", "mean"),
        challenge2_error_rate_mean=("challenge2_hard_symptom_error_rate", "mean"),
        challenge2_response_candidate_error_rate_mean=("challenge2_response_candidate_error_rate", "mean"),
        challenge2_candidate_membership_auc_mean=("challenge2_candidate_membership_auc", "mean"),
        challenge2_source_role_gap_mean=("challenge2_source_role_gap", "mean"),
        challenge2_response_role_gap_mean=("challenge2_response_role_gap", "mean"),
        challenge2_role_auc_mean=("challenge2_role_auc", "mean"),
        challenge2_root_role_gap_mean=("challenge2_root_role_gap", "mean"),
        challenge2_source_gap_mean=("challenge2_source_gap", "mean"),
        challenge2_victim_gap_mean=("challenge2_victim_gap", "mean"),
        challenge2_margin_mean=("challenge2_root_vs_hard_symptom_margin", "mean"),
        challenge2_candidate_margin_mean=("challenge2_label_vs_response_candidate_margin", "mean"),
    ).reset_index()
    return summary.fillna("")


def write_markdown_table(df, columns):
    if df.empty:
        return "_No rows._"
    cols = [col for col in columns if col in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(base_dir, all_runs, summary):
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    main_summary = summary[summary["method"] == "source_hypothesis_losses"] if not summary.empty else pd.DataFrame()
    ablation_summary = summary[summary["method"].isin([m for m, cfg in VARIANTS.items() if cfg["role"] == "ablation"])] if not summary.empty else pd.DataFrame()
    challenge2_summary = summary[summary["method"].isin([m for m, cfg in VARIANTS.items() if cfg["role"] == "challenge2"])] if not summary.empty else pd.DataFrame()
    baseline_summary = summary[summary["method"].isin(["DeepHunt"])] if not summary.empty else pd.DataFrame()
    try:
        display_base_dir = str(base_dir.resolve().relative_to(ROOT))
    except ValueError:
        display_base_dir = str(base_dir)

    lines = [
        "# REACR-RCA Final Experiment Report",
        "",
        "## Protocol",
        "- Task: microservice root cause localization.",
        "- Method name: REACR-RCA.",
        "- Split protocol: dataset-configured (`group` for SN/TT; `time_strict` for D1/D2).",
        "- A@K aggregation: case-weighted.",
        "- Main model: `ranker_mode=source_hypothesis` with explanation losses enabled.",
        "- Raw chain validation is reported only as interpretability limitation evidence; it is not used for training.",
        "",
        "## Main Multi-Seed Results",
        write_markdown_table(main_summary, ["dataset", "method", "runs", "A1_mean", "A1_std", "A3_mean", "A3_std", "A5_mean", "A5_std", "collapse_rate", "exact_call_path_exists_mean"]),
        "",
        "## Ablation Results",
        write_markdown_table(ablation_summary, ["dataset", "method", "runs", "A1_mean", "A3_mean", "A5_mean", "collapse_rate"]),
        "",
        "## Source-Response Candidate Diagnostic Results",
        write_markdown_table(
            challenge2_summary,
            [
                "dataset", "method", "runs", "A1_mean", "A3_mean", "A5_mean",
                "challenge2_response_candidate_error_rate_mean",
                "challenge2_candidate_membership_auc_mean",
                "challenge2_source_role_gap_mean",
                "challenge2_response_role_gap_mean",
                "challenge2_candidate_margin_mean",
            ],
        ),
        "",
        "## Baseline Results",
        write_markdown_table(baseline_summary, ["dataset", "method", "runs", "A1_mean", "A3_mean", "A5_mean", "collapse_rate"]),
        "",
        "## Topology-Stratified Results",
        "Changed-window and stable-window A@K are kept separate because D1 exposes only a small changed-window subset and D2 exposes none.",
        "",
        write_markdown_table(
            summary,
            [
                "dataset", "method", "runs",
                "changed_case_count", "changed_A1_mean", "changed_A3_mean", "changed_A5_mean",
                "stable_case_count", "stable_A1_mean", "stable_A3_mean", "stable_A5_mean",
            ],
        ),
        "",
        "## Limitation For Interpretability",
        "The current work should claim interpretable ranking evidence, not verified real propagation-chain localization. Exact raw-trace path matching is reported separately and remains weak when raw trace catalogs are available.",
        "",
        "## Reproducibility",
        f"- All run directories are under `{display_base_dir}`.",
        "- Each pipeline run stores `runner_command.json`, `run_config.json`, `train_history.json`, `validation_eval.json`, `final_eval.json`, predictions, and chain files when exported.",
    ]
    (summary_dir / "REACR_RCA_FINAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(base_dir):
    summary_dir = base_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    all_runs = collect_results(base_dir)
    all_runs.to_csv(summary_dir / "all_runs.csv", index=False)
    summary = summarize_group(all_runs)
    summary.to_csv(summary_dir / "multi_seed_summary.csv", index=False)
    main_summary = summary[summary["method"] == "source_hypothesis_losses"] if not summary.empty else pd.DataFrame()
    (summary_dir / "multi_seed_summary.md").write_text(
        write_markdown_table(
            main_summary,
            [
                "dataset", "method", "runs",
                "A1_mean", "A1_std", "A3_mean", "A3_std", "A5_mean", "A5_std",
                "changed_case_count", "changed_A1_mean", "changed_A3_mean", "changed_A5_mean",
                "stable_case_count", "stable_A1_mean", "stable_A3_mean", "stable_A5_mean",
                "collapse_rate",
            ],
        ),
        encoding="utf-8",
    )
    ablation = summary[summary["method"].isin([m for m, cfg in VARIANTS.items() if cfg["role"] == "ablation"])] if not summary.empty else pd.DataFrame()
    ablation.to_csv(summary_dir / "ablation_summary.csv", index=False)
    challenge2 = summary[summary["method"].isin([m for m, cfg in VARIANTS.items() if cfg["role"] == "challenge2"])] if not summary.empty else pd.DataFrame()
    challenge2.to_csv(summary_dir / "challenge2_summary.csv", index=False)
    write_report(base_dir, all_runs, summary)
    return all_runs, summary


def main():
    args = parse_args()
    base_dir = Path(args.base_dir).resolve()
    ensure_dir(base_dir, overwrite=False)
    if args.mode in {"all", "smoke"}:
        for dataset in args.datasets:
            for seed in args.smoke_seeds:
                run_pipeline(dataset, "source_hypothesis_losses", seed, args.smoke_epochs, base_dir / "smoke", args, smoke=True)
    if args.mode in {"all", "main"}:
        for dataset in args.datasets:
            epochs = resolved_train_epochs(dataset, args.train_epochs)
            for seed in args.main_seeds:
                run_pipeline(dataset, "source_hypothesis_losses", seed, epochs, base_dir, args)
    if args.mode in {"all", "ablations"}:
        for dataset in args.datasets:
            epochs = resolved_train_epochs(dataset, args.train_epochs)
            for method, cfg in VARIANTS.items():
                if cfg["role"] not in {"ablation", "challenge2"}:
                    continue
                for seed in args.ablation_seeds:
                    run_pipeline(dataset, method, seed, epochs, base_dir, args)
    if args.mode in {"all", "baseline"} and not args.skip_baseline:
        for dataset in args.datasets:
            for seed in args.baseline_seeds:
                run_deephunt_baseline(dataset, seed, base_dir, args)
    all_runs, summary = summarize(base_dir)
    print(f"saved all runs: {base_dir / 'summary' / 'all_runs.csv'}")
    print(f"saved summary: {base_dir / 'summary' / 'multi_seed_summary.csv'}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
