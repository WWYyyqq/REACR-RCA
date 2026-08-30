"""RQ5: paired dynamic-vs-frozen analysis by root-local edge-weight change.

The intensity definition is fixed independently of test performance: for each
case, measure the normalized edge-weight change on edges incident to any
labelled root-cause candidate, average it across adjacent snapshots, and split
cases into equal-count within-dataset terciles. Validation and test results are
exported separately. Existing trained runs are reused; no model is retrained.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_dsbg_dataset import build_case_windows
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from reacr_rca_pipeline import split_cases_for_training
from tools.optimize_reacr_rq1 import build_calibrator, feature_and_label, read_score_matrix
from utils.public_functions import load_init, load_samples


DATASETS = ("TT", "D1_weighted")
DISPLAY = {"TT": "TT", "D1_weighted": "D1"}
SEEDS = tuple(range(5))
RUN_ROOT = ROOT / "res" / "reacr_rca_full_20260824"
LOCK_ROOT = ROOT / "res" / "reacr_rq1_optimized_20260824_refit"
OUTPUT = ROOT / "res" / "rq5_root_local_intensity_20260827"


def raw_windows(dataset):
    config = yaml.load(
        (ROOT / "config" / f"{dataset}.yaml").read_text(encoding="utf-8-sig"),
        Loader=yaml.FullLoader,
    )
    _, test_samples = load_samples(str(ROOT / config["path"]["sample_dir"]))
    cases = pd.read_csv(ROOT / config["path"]["case_dir"])
    node_hash, _, _ = load_init(str(ROOT / config["path"]["graph_dir"]))
    settings = config.get("data_settings", {})
    default_ratio = config.get("feedback", {}).get("sample_num", 0.3)
    if not isinstance(default_ratio, float):
        default_ratio = 0.3
    _, validation_cases, test_cases, _ = split_cases_for_training(
        cases,
        0.3,
        float(default_ratio),
        protocol=settings.get("case_split_protocol", "time_strict"),
        test_start_ratio=0.5,
        group_column=settings.get("case_group_column"),
    )
    kwargs = {
        "window_size": int(settings.get("window_size", 10)),
        "sample_interval_seconds": float(settings.get("sample_interval_seconds", 60.0)),
        "case_pre_seconds": settings.get("case_pre_seconds"),
        "case_post_seconds": settings.get("case_post_seconds"),
        "max_gap_seconds": settings.get("max_gap_seconds"),
        "incomplete_policy": settings.get("case_incomplete_policy", "pad"),
    }
    return (
        build_case_windows(validation_cases, test_samples, node_hash, **kwargs),
        build_case_windows(test_cases, test_samples, node_hash, **kwargs),
    )


def incident_weight_map(graph, roots):
    src, dst = graph.edges()
    if "weight" in graph.edata:
        weight = graph.edata["weight"].detach().cpu().reshape(-1).double().numpy()
    else:
        weight = np.ones(int(graph.num_edges()), dtype=float)
    result = {}
    for left, right, value in zip(src.tolist(), dst.tolist(), weight.tolist()):
        if int(left) not in roots and int(right) not in roots:
            continue
        key = (int(left), int(right))
        result[key] = result.get(key, 0.0) + float(value)
    return result


def root_local_weight_intensity(window):
    roots = set(torch.where(window.label > 0)[0].detach().cpu().tolist())
    changes = []
    for previous, current in zip(window.graphs[:-1], window.graphs[1:]):
        left = incident_weight_map(previous, roots)
        right = incident_weight_map(current, roots)
        keys = set(left) | set(right)
        numerator = sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)
        denominator = sum(abs(left.get(key, 0.0)) + abs(right.get(key, 0.0)) for key in keys)
        changes.append(float(numerator / denominator) if denominator else 0.0)
    return float(np.mean(changes)) if changes else 0.0


def rank_terciles(values):
    order = np.argsort(np.asarray(values), kind="mergesort")
    labels = np.empty(len(order), dtype=object)
    names = ("低", "中", "高")
    for rank, index in enumerate(order):
        labels[index] = names[min(2, int(rank * 3 / len(order)))]
    return labels


def fused_scores(dataset, split, cases, node_names, statistical, weight):
    result = {"dynamic": [], "frozen": []}
    prediction_name = "validation_predictions.csv" if split == "validation" else "eval_predictions.csv"
    for variant, key in (("source_hypothesis_losses", "dynamic"), ("frozen_window_topology", "frozen")):
        for seed in SEEDS:
            neural = read_score_matrix(
                RUN_ROOT / dataset / variant / f"seed{seed}" / prediction_name,
                cases,
                node_names,
            )
            result[key].append(fuse_case_scores(neural, statistical, weight))
    return result


def grouped_metrics(scores, labels, groups):
    rows = []
    for group in ("低", "中", "高"):
        indices = torch.as_tensor(np.where(groups == group)[0], dtype=torch.long)
        item = {"group": group, "case_count": int(len(indices))}
        for name in ("dynamic", "frozen"):
            per_seed = [
                topk_metrics(
                    torch.as_tensor(score, dtype=torch.float32).index_select(0, indices),
                    labels.index_select(0, indices),
                )
                for score in scores[name]
            ]
            item[name] = {
                metric: float(np.mean([value[metric] for value in per_seed]))
                for metric in ("A@1", "A@3", "A@5")
            }
        item["delta"] = {
            metric: item["dynamic"][metric] - item["frozen"][metric]
            for metric in ("A@1", "A@3", "A@5")
        }
        rows.append(item)
    return rows


def analyze(dataset):
    data = load_rq1_data(dataset)
    validation_windows, test_windows = raw_windows(dataset)
    assert [int(item.case_id) for item in validation_windows] == [int(item.case_id) for item in data.validation]
    assert [int(item.case_id) for item in test_windows] == [int(item.case_id) for item in data.test]

    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    test_x, test_y = feature_and_label(data.test)
    lock = json.loads((LOCK_ROOT / dataset / "locked_test.json").read_text(encoding="utf-8"))["locked_calibration"]

    validation_calibrator = build_calibrator(lock["calibrator"]).fit(train_x, train_y)
    test_calibrator = build_calibrator(lock["calibrator"]).fit(
        torch.cat([train_x, val_x]), torch.cat([train_y, val_y])
    )
    split_records = []
    for split, windows, cases, labels, statistical in (
        ("validation", validation_windows, data.validation, val_y, validation_calibrator.predict(val_x)),
        ("test", test_windows, data.test, test_y, test_calibrator.predict(test_x)),
    ):
        intensity = np.asarray([root_local_weight_intensity(window) for window in windows])
        groups = rank_terciles(intensity)
        scores = fused_scores(
            dataset,
            split,
            cases,
            data.node_names,
            statistical,
            float(lock["neural_base_weight"]),
        )
        split_records.append({
            "split": split,
            "intensity_min": float(intensity.min()),
            "intensity_mean": float(intensity.mean()),
            "intensity_max": float(intensity.max()),
            "rows": grouped_metrics(scores, labels, groups),
            "case_records": [
                {"case_id": int(window.case_id), "intensity": float(value), "group": str(group)}
                for window, value, group in zip(windows, intensity, groups)
            ],
        })
    return {"dataset": DISPLAY[dataset], "dataset_key": dataset, "splits": split_records}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = [analyze(dataset) for dataset in DATASETS]
    payload = {
        "protocol": {
            "intensity": "mean normalized edge-weight change on edges incident to labelled root-cause candidates",
            "groups": "equal-count within-dataset rank terciles, computed separately for validation and test",
            "comparison": "observed dynamic graph versus first-snapshot frozen graph",
            "calibration": "same validation-locked calibrator and fusion weight within each paired comparison",
            "seeds": list(SEEDS),
            "model_retraining": False,
            "test_used_for_intensity_definition": False,
        },
        "datasets": datasets,
    }
    (OUTPUT / "rq5_root_local_intensity_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# RQ5：根因邻域边权变化强度与动态图收益",
        "",
        "| 数据集 | 划分 | 强度 | 案例数 | 动态 A@1 | 冻结 A@1 | ΔA@1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for dataset in datasets:
        for split in dataset["splits"]:
            for row in split["rows"]:
                lines.append(
                    f"| {dataset['dataset']} | {split['split']} | {row['group']} | {row['case_count']} | "
                    f"{row['dynamic']['A@1']:.2f} | {row['frozen']['A@1']:.2f} | {row['delta']['A@1']:+.2f} |"
                )
    (OUTPUT / "RQ5_ROOT_LOCAL_INTENSITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
