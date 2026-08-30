"""Run and report validation-only D1 parameter-sensitivity experiments.

The sweep deliberately never exports test predictions.  Exact configurations
already trained for RQ2 are reused for the default setting, max_lag=0, and
root_symptom_loss_weight=0.  All other configurations are resumable and keep a
command/code manifest beside their validation artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from tools.optimize_reacr_rq1 import build_calibrator, feature_and_label, read_score_matrix
from tools.reacr_rca_final_study_runner import (
    VARIANTS,
    build_pipeline_cmd,
    implementation_fingerprint,
)


PARAMETERS = {
    "max_lag": {
        "flag": "--causal_max_lag",
        "values": (0, 1, 2, 3, 4, 5),
        "default": 3,
        "display": "最大时滞 L",
    },
    "window_size": {
        "flag": "--window_size",
        "values": (6, 8, 10, 12, 14),
        "default": 10,
        "display": "时间窗口 W",
    },
    "topology_weight_gamma": {
        "flag": "--topology_weight_gamma",
        "values": (0.0, 0.1, 0.25, 0.5, 1.0),
        "default": 0.25,
        "display": "拓扑边权系数 γ",
    },
    "root_symptom_loss_weight": {
        "flag": "--root_symptom_loss_weight",
        "values": (0.0, 0.05, 0.1, 0.2, 0.5),
        "default": 0.1,
        "display": "源—响应约束权重 λr",
    },
}

RQ2_ROOT = ROOT / "res" / "rq2_d1_core_rerun_20260825" / "D1_weighted"
DEFAULT_OUTPUT = ROOT / "res" / "rq4_parameter_sensitivity_20260825"
LOCK_PATH = (
    ROOT
    / "res"
    / "reacr_rq2_calibration_lock_20260825"
    / "D1_weighted"
    / "locked_test.json"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", nargs="+", choices=tuple(PARAMETERS), default=list(PARAMETERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def _same_value(left, right):
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


def reusable_rq2_dir(parameter, value, seed):
    spec = PARAMETERS[parameter]
    if _same_value(value, spec["default"]):
        variant = "source_hypothesis_losses"
    elif parameter == "max_lag" and _same_value(value, 0):
        variant = "no_lag_relation"
    elif parameter == "root_symptom_loss_weight" and _same_value(value, 0):
        variant = "no_root_symptom_loss"
    else:
        return None
    path = RQ2_ROOT / variant / f"seed{seed}"
    required = (path / "validation_eval.json", path / "validation_predictions.csv", path / "run_config.json")
    return path if all(item.exists() for item in required) else None


def runner_args(args):
    return SimpleNamespace(
        device=args.device,
        feedback_sample_ratio=None,
        test_start_ratio=0.5,
        normal_pretrain_epochs=3,
        normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0,
        skip_event_chains=True,
    )


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sensitivity_run_dir(output, parameter, value, seed):
    return output / "D1_weighted" / parameter / f"value_{value_slug(value)}" / f"seed{seed}"


def build_sensitivity_cmd(parameter, value, seed, out_dir, args):
    cmd = build_pipeline_cmd(
        "D1_weighted",
        "source_hypothesis_losses",
        VARIANTS["source_hypothesis_losses"],
        seed,
        args.train_epochs,
        out_dir,
        runner_args(args),
    )
    cmd = set_cmd_option(cmd, PARAMETERS[parameter]["flag"], value)
    if "--validation_only" not in cmd:
        cmd.append("--validation_only")
    return cmd


def run_one(output, parameter, value, seed, args):
    reused = reusable_rq2_dir(parameter, value, seed)
    if reused is not None:
        return {"parameter": parameter, "value": value, "seed": seed, "path": str(reused), "reused": True}

    out_dir = sensitivity_run_dir(output, parameter, value, seed)
    cmd = build_sensitivity_cmd(parameter, value, seed, out_dir, args)
    manifest = {
        "research_question": "RQ4 parameter sensitivity",
        "selection_split": "validation_only",
        "test_outputs_exported": False,
        "parameter": parameter,
        "value": value,
        "seed": seed,
        "cmd": cmd,
        "implementation": implementation_fingerprint(),
        "runner_sha256": script_sha256(),
    }
    manifest_path = out_dir / "runner_command.json"
    done = (out_dir / "validation_eval.json").exists() and (out_dir / "validation_predictions.csv").exists()
    if manifest_path.exists() and not args.overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise RuntimeError(f"stale or incompatible RQ4 run: {out_dir}")
    if done and not args.overwrite:
        return {"parameter": parameter, "value": value, "seed": seed, "path": str(out_dir), "reused": False}

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    forbidden = (out_dir / "final_eval.json", out_dir / "eval_predictions.csv")
    if any(path.exists() for path in forbidden):
        raise RuntimeError(f"validation-only RQ4 unexpectedly exported test artifacts: {out_dir}")
    return {"parameter": parameter, "value": value, "seed": seed, "path": str(out_dir), "reused": False}


def aggregate(metrics):
    result = {}
    for key in ("A@1", "A@3", "A@5"):
        values = np.asarray([row[key] for row in metrics], dtype=float)
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        result[key] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": std,
            "display": f"{values.mean():.2f} ± {std:.2f}",
        }
    return result


def build_report(output, parameters, seeds):
    data = load_rq1_data("D1_weighted")
    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))["locked_calibration"]
    calibrator = build_calibrator(lock["calibrator"]).fit(train_x, train_y)
    statistical = calibrator.predict(val_x)
    rows = []
    for parameter in parameters:
        for value in PARAMETERS[parameter]["values"]:
            neural_rows = []
            fused_rows = []
            sources = []
            for seed in seeds:
                path = reusable_rq2_dir(parameter, value, seed)
                reused = path is not None
                if path is None:
                    path = sensitivity_run_dir(output, parameter, value, seed)
                required = (path / "validation_predictions.csv", path / "validation_eval.json", path / "run_config.json")
                missing = [str(item) for item in required if not item.exists()]
                if missing:
                    raise FileNotFoundError("missing RQ4 artifacts: " + ", ".join(missing))
                neural = read_score_matrix(path / "validation_predictions.csv", data.validation, data.node_names)
                neural_rows.append(topk_metrics(torch.as_tensor(neural, dtype=torch.float32), val_y))
                fused = fuse_case_scores(neural, statistical, lock["neural_base_weight"])
                fused_rows.append(topk_metrics(torch.as_tensor(fused, dtype=torch.float32), val_y))
                sources.append({"seed": seed, "path": str(path), "reused_from_rq2": reused})
            rows.append({
                "parameter": parameter,
                "display": PARAMETERS[parameter]["display"],
                "value": value,
                "default": _same_value(value, PARAMETERS[parameter]["default"]),
                "neural": aggregate(neural_rows),
                "complete": aggregate(fused_rows),
                "sources": sources,
            })

    payload = {
        "research_question": "How sensitive is REACR to its core parameters?",
        "protocol": {
            "dataset": "D1_weighted",
            "selection_split": "validation_only",
            "test_artifacts_read": False,
            "new_test_outputs_exported": False,
            "seeds": list(seeds),
            "reporting": "mean_and_sample_std",
            "shared_locked_calibrator": str(LOCK_PATH),
            "neural_base_weight": lock["neural_base_weight"],
            "one_parameter_at_a_time": True,
        },
        "parameters": PARAMETERS,
        "rows": rows,
    }
    result_path = output / "rq4_parameter_sensitivity_results.json"
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RQ4：核心参数敏感性", "",
        "全部参数仅在 D1 验证集上评估；每次只改变一个参数。新运行启用 validation_only，不计算或导出测试集结果。表中为三个随机种子的均值±样本标准差。", "",
        "| 参数 | 取值 | 默认值 | REACR-N A@1 | REACR A@1 | REACR A@3 | REACR A@5 |",
        "|---|---:|:---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['value']} | {'是' if row['default'] else ''} | "
            f"{row['neural']['A@1']['display']} | {row['complete']['A@1']['display']} | "
            f"{row['complete']['A@3']['display']} | {row['complete']['A@5']['display']} |"
        )
    (output / "RQ4_PARAMETER_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result_path


def main():
    args = parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        jobs = [
            (parameter, value, seed)
            for parameter in args.parameters
            for value in PARAMETERS[parameter]["values"]
            for seed in args.seeds
        ]
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, output, parameter, value, seed, args): (parameter, value, seed)
                for parameter, value, seed in jobs
            }
            for future in as_completed(futures):
                parameter, value, seed = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    raise RuntimeError(f"RQ4 failed for {parameter}={value}/seed{seed}") from error
                print(json.dumps(record, ensure_ascii=False), flush=True)
    result = build_report(output, args.parameters, args.seeds)
    print(json.dumps({"status": "ok", "result": str(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
