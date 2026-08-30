"""Informative validation-only RQ4 sensitivity study on D1.

Separates neural and fused metrics, adds MRR, and explicitly varies the
fusion weight together with central temporal, topology, and optimization
parameters. Test data are never evaluated or exported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from tools.optimize_reacr_rq1 import build_calibrator, feature_and_label, read_score_matrix
from tools.reacr_rca_final_study_runner import VARIANTS, build_pipeline_cmd, implementation_fingerprint


PARAMETERS = {
    "window_size": {
        "flag": "--window_size", "values": (6, 8, 10, 12, 14),
        "default": 10, "display": "时间窗口 W", "scope": "temporal_context",
    },
    "topology_weight_gamma": {
        "flag": "--topology_weight_gamma", "values": (0.0, 0.1, 0.25, 0.5, 1.0),
        "default": 0.25, "display": "动态边权系数 γ", "scope": "dynamic_topology",
    },
    "train_learning_rate": {
        "flag": "--train_learning_rate", "values": (0.0001, 0.0005, 0.001, 0.002, 0.005),
        "default": 0.001, "display": "学习率 η", "scope": "optimization",
    },
    "neural_fusion_weight": {
        "flag": None, "values": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "default": 0.2, "display": "神经融合权重 α", "scope": "score_fusion",
    },
}

DEFAULT_OUTPUT = ROOT / "res" / "rq4_informative_sensitivity_20260826"
LOCK_PATH = ROOT / "res" / "reacr_rq2_calibration_lock_20260825" / "D1_weighted" / "locked_test.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", nargs="+", choices=tuple(PARAMETERS), default=list(PARAMETERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def same_value(left, right):
    return abs(float(left) - float(right)) < 1e-12


def value_slug(value):
    return str(value).replace("-", "m").replace(".", "p")


def set_cmd_option(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        index = cmd.index(flag)
        cmd[index + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])
    return cmd


def runner_args(args):
    return SimpleNamespace(
        device=args.device, feedback_sample_ratio=None, test_start_ratio=0.5,
        normal_pretrain_epochs=3, normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0, skip_event_chains=True,
    )


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def default_run_dir(output, seed):
    return output / "D1_weighted" / "default" / f"seed{seed}"


def sensitivity_run_dir(output, parameter, value, seed):
    spec = PARAMETERS[parameter]
    if parameter == "neural_fusion_weight" or same_value(value, spec["default"]):
        return default_run_dir(output, seed)
    return output / "D1_weighted" / parameter / f"value_{value_slug(value)}" / f"seed{seed}"


def build_training_cmd(parameter, value, seed, out_dir, args):
    cmd = build_pipeline_cmd(
        "D1_weighted", "source_hypothesis_losses", VARIANTS["source_hypothesis_losses"],
        seed, args.train_epochs, out_dir, runner_args(args),
    )
    flag = PARAMETERS[parameter]["flag"]
    if flag is not None:
        cmd = set_cmd_option(cmd, flag, value)
    if "--validation_only" not in cmd:
        cmd.append("--validation_only")
    return cmd


def required_artifacts(path):
    return (
        path / "validation_eval.json", path / "validation_predictions.csv",
        path / "run_config.json", path / "best_model.pt",
    )


def run_one(output, parameter, value, seed, args):
    out_dir = sensitivity_run_dir(output, parameter, value, seed)
    # Fusion values share the newly trained default model for each seed.
    training_parameter = parameter if parameter != "neural_fusion_weight" else "window_size"
    training_value = value if parameter != "neural_fusion_weight" else PARAMETERS["window_size"]["default"]
    cmd = build_training_cmd(training_parameter, training_value, seed, out_dir, args)
    is_default = out_dir == default_run_dir(output, seed)
    manifest = {
        "research_question": "RQ4 informative parameter sensitivity",
        "selection_split": "validation_only",
        "test_outputs_exported": False,
        "parameter": "default" if is_default else parameter,
        "value": None if is_default else value,
        "seed": seed,
        "cmd": cmd,
        "implementation": implementation_fingerprint(),
        "runner_sha256": script_sha256(),
    }
    manifest_path = out_dir / "runner_command.json"
    done = all(path.exists() for path in required_artifacts(out_dir))
    if manifest_path.exists() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"stale or incompatible RQ4 run: {out_dir}")
    if done and not args.overwrite:
        return {"status": "reused", "path": str(out_dir)}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    forbidden = (out_dir / "final_eval.json", out_dir / "eval_predictions.csv")
    if any(path.exists() for path in forbidden):
        raise RuntimeError(f"validation-only run exported test artifacts: {out_dir}")
    return {"status": "trained", "path": str(out_dir)}


def reciprocal_rank(scores, labels):
    score_values = np.asarray(scores, dtype=float)
    label_values = np.asarray(labels, dtype=float)
    order = np.argsort(-score_values, axis=1, kind="stable")
    ranked_positive = np.take_along_axis(label_values > 0, order, axis=1)
    first = np.argmax(ranked_positive, axis=1)
    has_positive = ranked_positive.any(axis=1)
    values = np.where(has_positive, 1.0 / (first + 1.0), 0.0)
    return float(values.mean() * 100.0)


def metrics_from_scores(scores, labels):
    result = topk_metrics(
        torch.as_tensor(scores, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    result["MRR"] = reciprocal_rank(scores, labels)
    return result


def aggregate(rows):
    result = {}
    for key in ("A@1", "A@3", "A@5", "MRR"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def build_report(output, parameters, seeds, train_epochs):
    data = load_rq1_data("D1_weighted")
    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    labels = val_y.detach().cpu().numpy()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["locked_calibration"]
    calibrator = build_calibrator(lock["calibrator"]).fit(train_x, train_y)
    statistical = calibrator.predict(val_x)
    default_weight = float(lock["neural_base_weight"])

    rows = []
    for parameter in parameters:
        spec = PARAMETERS[parameter]
        for value in spec["values"]:
            neural_rows = []
            fused_rows = []
            sources = []
            fusion_weight = float(value) if parameter == "neural_fusion_weight" else default_weight
            for seed in seeds:
                path = sensitivity_run_dir(output, parameter, value, seed)
                missing = [str(item) for item in required_artifacts(path) if not item.exists()]
                if missing:
                    raise FileNotFoundError("missing RQ4 artifacts: " + ", ".join(missing))
                neural = read_score_matrix(path / "validation_predictions.csv", data.validation, data.node_names)
                fused = fuse_case_scores(neural, statistical, fusion_weight)
                neural_rows.append(metrics_from_scores(neural, labels))
                fused_rows.append(metrics_from_scores(fused, labels))
                sources.append({"seed": seed, "path": str(path)})
            rows.append({
                "parameter": parameter,
                "display": spec["display"],
                "scope": spec["scope"],
                "value": value,
                "default": same_value(value, spec["default"]),
                "fusion_weight": fusion_weight,
                "neural": aggregate(neural_rows),
                "complete": aggregate(fused_rows),
                "sources": sources,
            })

    payload = {
        "research_question": "Which important REACR parameters materially affect validation ranking?",
        "protocol": {
            "dataset": "D1_weighted",
            "selection_split": "validation_only",
            "validation_case_count": int(len(data.validation)),
            "test_artifacts_read": False,
            "new_test_outputs_exported": False,
            "seeds": list(seeds),
            "train_epochs": int(train_epochs),
            "default_neural_fusion_weight": default_weight,
            "shared_locked_calibrator": str(LOCK_PATH),
            "one_parameter_at_a_time": True,
            "metrics": ["A@1", "A@3", "A@5", "MRR"],
        },
        "parameters": {key: PARAMETERS[key] for key in parameters},
        "rows": rows,
    }
    result = output / "rq4_informative_sensitivity_results.json"
    result.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main():
    args = parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        jobs = {}
        for parameter in args.parameters:
            for value in PARAMETERS[parameter]["values"]:
                for seed in args.seeds:
                    path = sensitivity_run_dir(output, parameter, value, seed)
                    jobs.setdefault(str(path), (parameter, value, seed))
        total = len(jobs)
        for index, (parameter, value, seed) in enumerate(jobs.values(), start=1):
            record = run_one(output, parameter, value, seed, args)
            print(json.dumps({
                "progress": f"{index}/{total}", "parameter": parameter,
                "value": value, "seed": seed, **record,
            }, ensure_ascii=False), flush=True)
    result = build_report(output, args.parameters, args.seeds, args.train_epochs)
    print(json.dumps({"status": "ok", "result": str(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
