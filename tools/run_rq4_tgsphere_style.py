"""Run a TGSphere-style one-factor-at-a-time RQ4 study on D1.

The reference point is *exactly* the five locked D1 runs reported in RQ1/Table 2.
Every trainable non-reference point starts from the corresponding reference
command and changes only the declared option.  Parameter ranges are written to
``protocol_lock.json`` before any new test prediction is opened.  Thus the test
split is used for reporting, never for choosing the ranges or the reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from tools.optimize_reacr_rq1 import build_calibrator, feature_and_label, read_score_matrix


REFERENCE_ROOT = ROOT / "res" / "reacr_rca_full_20260824" / "D1_weighted" / "source_hypothesis_losses"
LOCKED_TEST = ROOT / "res" / "reacr_rq1_optimized_20260824_refit" / "D1_weighted" / "locked_test.json"
RQ1_FINAL = ROOT / "res" / "rq1_final_20260824" / "final_results.json"
DEFAULT_OUTPUT = ROOT / "res" / "rq4_tgsphere_style_20260827"


PARAMETERS = {
    "window_size": {
        "display": "时间窗口 W", "values": (6, 10, 14), "default": 10,
        "scope": "输入时间范围", "options": (("--window_size", "{value}"),),
    },
    "representation_dim": {
        "display": "统一表示维度 d", "values": (32, 64, 128), "default": 64,
        "scope": "表示容量",
        "options": tuple((flag, "{value}") for flag in (
            "--spatial_hidden_dim", "--temporal_hidden_dim", "--modality_gate_hidden_dim",
            "--anomaly_hidden_dim", "--causal_hidden_dim", "--rank_hidden_dim",
        )),
    },
    "spatial_layers": {
        "display": "空间图层数 Lg", "values": (1, 2, 3), "default": 2,
        "scope": "空间编码结构", "options": (("--num_spatial_layers", "{value}"),),
    },
    "temporal_heads": {
        "display": "时间注意力头数 h", "values": (1, 2, 4), "default": 1,
        "scope": "时间编码结构", "options": (("--num_temporal_heads", "{value}"),),
    },
    "causal_max_lag": {
        "display": "最大时滞 L", "values": (1, 3, 5), "default": 3,
        "scope": "时滞关系", "options": (("--causal_max_lag", "{value}"),),
    },
    "causal_gamma": {
        "display": "因果关系系数 γc", "values": (0.1, 0.5, 1.0), "default": 0.5,
        "scope": "时滞关系", "options": (("--causal_gamma", "{value}"),),
    },
    "topology_weight_gamma": {
        "display": "动态边权系数 γw", "values": (0.0, 0.25, 1.0), "default": 0.25,
        "scope": "动态图关系", "options": (("--topology_weight_gamma", "{value}"),),
    },
    "dropout": {
        "display": "Dropout p", "values": (0.0, 0.1, 0.5), "default": 0.1,
        "scope": "正则化", "options": (("--dropout", "{value}"),),
    },
    "learning_rate": {
        "display": "学习率 η", "values": (0.0001, 0.001, 0.005), "default": 0.001,
        "scope": "优化", "options": (("--train_learning_rate", "{value}"),),
    },
    "weight_decay": {
        "display": "L2 系数 λ", "values": (0.0, 0.0001, 0.01), "default": 0.0,
        "scope": "正则化", "options": (("--train_weight_decay", "{value}"),),
    },
    "fusion_weight": {
        "display": "神经分支融合权重 α", "values": (0.0, 0.25, 0.75), "default": 0.25,
        "scope": "证据融合", "options": (),
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameters", nargs="+", choices=tuple(PARAMETERS), default=list(PARAMETERS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT.relative_to(ROOT)))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def same_value(left, right):
    return abs(float(left) - float(right)) < 1e-12


def value_slug(value):
    return str(value).replace("-", "m").replace(".", "p")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_cmd_option(cmd, flag, value):
    cmd = list(cmd)
    if flag in cmd:
        index = cmd.index(flag)
        cmd[index + 1] = str(value)
    else:
        cmd.extend([flag, str(value)])
    return cmd


def reference_dir(seed):
    return REFERENCE_ROOT / f"seed{seed}"


def run_dir(output, parameter, value, seed):
    spec = PARAMETERS[parameter]
    if parameter == "fusion_weight" or same_value(value, spec["default"]):
        return reference_dir(seed)
    return output / "D1_weighted" / parameter / f"value_{value_slug(value)}" / f"seed{seed}"


def required_artifacts(path):
    return (
        path / "validation_predictions.csv", path / "eval_predictions.csv",
        path / "run_config.json", path / "best_model.pt",
    )


def base_command(seed):
    manifest = json.loads((reference_dir(seed) / "runner_command.json").read_text(encoding="utf-8"))
    return list(manifest["cmd"]), manifest


def build_command(output, parameter, value, seed):
    target = run_dir(output, parameter, value, seed)
    cmd, _ = base_command(seed)
    cmd = set_cmd_option(cmd, "--output_dir", target)
    for flag, template in PARAMETERS[parameter]["options"]:
        cmd = set_cmd_option(cmd, flag, template.format(value=value))
    # Reference commands do not contain this flag, but guard against stale edits.
    cmd = [item for item in cmd if item != "--validation_only"]
    return cmd


def protocol_payload(parameters, seeds):
    locked = json.loads(LOCKED_TEST.read_text(encoding="utf-8"))
    return {
        "research_question": "RQ4 comprehensive parameter sensitivity on D1",
        "design": "one_factor_at_a_time",
        "declared_before_new_test_evaluation": True,
        "reference_selection": "fixed RQ1/Table 2 configuration selected without RQ4 test results",
        "reference_result": locked["test_aggregate"],
        "reference_locked_test": str(LOCKED_TEST),
        "reference_locked_test_sha256": sha256(LOCKED_TEST),
        "reference_neural_runs": str(REFERENCE_ROOT),
        "dataset": "D1_weighted",
        "seeds": list(seeds),
        "parameters": {key: PARAMETERS[key] for key in parameters},
        "test_use": "reporting_only_after_protocol_lock",
        "reporting": "five_seed_mean_without_std_in_RQ4_table",
    }


def write_or_verify_protocol_lock(output, parameters, seeds):
    path = output / "protocol_lock.json"
    payload = protocol_payload(parameters, seeds)
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("seeds") != payload["seeds"]:
            raise RuntimeError(f"RQ4 protocol is already locked with different seeds: {path}")
        locked_parameters = previous.get("parameters", {})
        for key in parameters:
            requested = json.loads(json.dumps(payload["parameters"][key], ensure_ascii=False))
            if locked_parameters.get(key) != requested:
                raise RuntimeError(f"RQ4 protocol has a different or missing specification for {key}: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_one(output, parameter, value, seed, overwrite=False):
    target = run_dir(output, parameter, value, seed)
    if target == reference_dir(seed):
        missing = [str(path) for path in required_artifacts(target) if not path.exists()]
        if missing:
            raise FileNotFoundError("missing Table 2 reference artifacts: " + ", ".join(missing))
        return {"status": "reused_table2", "parameter": parameter, "value": value, "seed": seed}

    cmd = build_command(output, parameter, value, seed)
    _, base_manifest = base_command(seed)
    manifest = {
        "research_question": "RQ4 TGSphere-style sensitivity",
        "parameter": parameter, "value": value, "seed": seed,
        "one_factor_changed": [list(item) for item in PARAMETERS[parameter]["options"]],
        "reference_command": str(reference_dir(seed) / "runner_command.json"),
        "reference_implementation": base_manifest["implementation"],
        "current_runner_sha256": sha256(Path(__file__)),
        "cmd": cmd,
    }
    manifest_path = target / "runner_command.json"
    done = all(path.exists() for path in required_artifacts(target))
    if manifest_path.exists() and not overwrite:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The runner hash is provenance, not a reason to invalidate completed training.
        comparable_previous = {k: v for k, v in previous.items() if k != "current_runner_sha256"}
        comparable_current = {k: v for k, v in manifest.items() if k != "current_runner_sha256"}
        if comparable_previous != comparable_current:
            raise RuntimeError(f"stale or incompatible RQ4 run: {target}")
    if done and not overwrite:
        return {"status": "reused", "parameter": parameter, "value": value, "seed": seed}

    target.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    with (target / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, check=True)
    missing = [str(path) for path in required_artifacts(target) if not path.exists()]
    if missing:
        raise RuntimeError("completed run misses artifacts: " + ", ".join(missing))
    return {"status": "trained", "parameter": parameter, "value": value, "seed": seed}


def metric_bundle(scores, labels):
    metrics = topk_metrics(
        torch.as_tensor(scores, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.float32),
    )
    metrics["Avg@K"] = float(np.mean([metrics["A@1"], metrics["A@3"], metrics["A@5"]]))
    return metrics


def aggregate(rows):
    result = {}
    for key in ("A@1", "A@3", "A@5", "Avg@K"):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key] = {
            "values": values.tolist(), "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def table2_d1_result():
    payload = json.loads(RQ1_FINAL.read_text(encoding="utf-8"))
    reacr = next(row for row in payload["table"] if row["method"] == "REACR")
    return {key: float(reacr["D1"][key]["mean"]) for key in ("A@1", "A@3", "A@5")}


def build_report(output, parameters, seeds):
    # The protocol must already exist before this function opens any test file.
    protocol_path = output / "protocol_lock.json"
    if not protocol_path.exists():
        raise RuntimeError("refusing test evaluation before protocol_lock.json exists")

    data = load_rq1_data("D1_weighted")
    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    development_x = torch.cat([train_x, val_x], dim=0)
    development_y = torch.cat([train_y, val_y], dim=0)
    test_x, test_y = feature_and_label(data.test)
    lock = json.loads(LOCKED_TEST.read_text(encoding="utf-8"))["locked_calibration"]
    calibrator = build_calibrator(lock["calibrator"]).fit(development_x, development_y)
    statistical = calibrator.predict(test_x)
    default_fusion = float(lock["neural_base_weight"])

    rows = []
    for parameter in parameters:
        spec = PARAMETERS[parameter]
        for value in spec["values"]:
            neural_rows, complete_rows, sources = [], [], []
            fusion_weight = float(value) if parameter == "fusion_weight" else default_fusion
            for seed in seeds:
                source = run_dir(output, parameter, value, seed)
                missing = [str(path) for path in required_artifacts(source) if not path.exists()]
                if missing:
                    raise FileNotFoundError("missing RQ4 artifacts: " + ", ".join(missing))
                neural = read_score_matrix(source / "eval_predictions.csv", data.test, data.node_names)
                fused = fuse_case_scores(neural, statistical, fusion_weight)
                neural_rows.append(metric_bundle(neural, test_y))
                complete_rows.append(metric_bundle(fused, test_y))
                sources.append({"seed": seed, "path": str(source)})
            rows.append({
                "parameter": parameter, "display": spec["display"], "scope": spec["scope"],
                "value": value, "default": same_value(value, spec["default"]),
                "fusion_weight": fusion_weight,
                "neural": aggregate(neural_rows), "complete": aggregate(complete_rows),
                "sources": sources,
            })

    expected = table2_d1_result()
    default_rows = [row for row in rows if row["default"]]
    for row in default_rows:
        for metric in ("A@1", "A@3", "A@5"):
            actual = row["complete"][metric]["mean"]
            if not np.isclose(actual, expected[metric], rtol=0.0, atol=1e-7):
                raise AssertionError(
                    f"default {row['parameter']} {metric}={actual} does not match Table 2 {expected[metric]}"
                )

    payload = {
        "research_question": "How sensitive is REACR to its important parameters?",
        "protocol": json.loads(protocol_path.read_text(encoding="utf-8")),
        "table2_d1_exact_match": True,
        "table2_d1": expected,
        "rows": rows,
    }
    result = output / "rq4_tgsphere_style_results.json"
    result.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# RQ4：重要参数影响", "",
        "表中为 D1 上五个随机种子的均值；按照论文约定，本 RQ 不再列均值±标准差。加粗项为表 2 的锁定配置。", "",
        "| 参数 | 范围/含义 | 取值 | REACR-N A@1 | REACR A@1 | A@3 | A@5 | Avg@K |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        fmt = lambda number: f"{number:.2f}"
        cells = [
            row["display"], row["scope"], str(row["value"]),
            fmt(row["neural"]["A@1"]["mean"]), fmt(row["complete"]["A@1"]["mean"]),
            fmt(row["complete"]["A@3"]["mean"]), fmt(row["complete"]["A@5"]["mean"]),
            fmt(row["complete"]["Avg@K"]["mean"]),
        ]
        if row["default"]:
            cells[2:] = [f"**{cell}**" for cell in cells[2:]]
        lines.append("| " + " | ".join(cells) + " |")
    (output / "RQ4_TGSPHERE_STYLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main():
    args = parse_args()
    output = (ROOT / args.output_dir).resolve()
    write_or_verify_protocol_lock(output, args.parameters, args.seeds)

    if not args.report_only:
        jobs = []
        seen = set()
        for parameter in args.parameters:
            for value in PARAMETERS[parameter]["values"]:
                for seed in args.seeds:
                    source = run_dir(output, parameter, value, seed)
                    if source == reference_dir(seed) or str(source) in seen:
                        continue
                    seen.add(str(source))
                    jobs.append((parameter, value, seed))
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(run_one, output, parameter, value, seed, args.overwrite):
                (parameter, value, seed)
                for parameter, value, seed in jobs
            }
            total = len(futures)
            completed = 0
            for future in as_completed(futures):
                parameter, value, seed = futures[future]
                completed += 1
                try:
                    record = future.result()
                except Exception as error:
                    raise RuntimeError(f"RQ4 failed for {parameter}={value}/seed{seed}") from error
                print(json.dumps({"progress": f"{completed}/{total}", **record}, ensure_ascii=False), flush=True)

    result = build_report(output, args.parameters, args.seeds)
    print(json.dumps({"status": "ok", "result": str(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
