"""Analyze whether dynamic-graph gains grow with observed topology intensity."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import spearmanr

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_dsbg_dataset import build_case_windows
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from models.reacr_rca_topology_dynamics import canonical_edge_multiset
from reacr_rca_pipeline import split_cases_for_training
from tools.optimize_reacr_rq1 import build_calibrator, feature_and_label, read_score_matrix
from utils.public_functions import load_init, load_samples


DATASETS = ("SN", "TT", "D1_weighted")
DISPLAY = {"SN": "SN", "TT": "TT", "D1_weighted": "D1"}
SEEDS = tuple(range(5))
RUN_ROOT = ROOT / "res" / "reacr_rca_full_20260824"
LOCK_ROOT = ROOT / "res" / "reacr_rq1_optimized_20260824_refit"
OUTPUT = ROOT / "res" / "rq5_topology_intensity_20260825"
METRICS = ("A@1", "A@3", "A@5")


def edge_counter(graph):
    return Counter(canonical_edge_multiset(graph))


def edge_weight_map(graph):
    src, dst = graph.edges()
    if "weight" in graph.edata:
        weight = graph.edata["weight"].detach().cpu().reshape(-1).double().numpy()
    else:
        weight = np.ones(int(graph.num_edges()), dtype=float)
    result = defaultdict(float)
    for source, target, value in zip(src.tolist(), dst.tolist(), weight.tolist()):
        result[(int(source), int(target))] += float(value)
    return result


def transition_intensity(previous, current):
    left = edge_counter(previous)
    right = edge_counter(current)
    symmetric = sum((left - right).values()) + sum((right - left).values())
    union = sum((left | right).values())
    jaccard = float(symmetric / union) if union else 0.0

    left_weight = edge_weight_map(previous)
    right_weight = edge_weight_map(current)
    keys = set(left_weight) | set(right_weight)
    numerator = sum(abs(left_weight[key] - right_weight[key]) for key in keys)
    denominator = sum(
        abs(left_weight[key]) + abs(right_weight[key]) for key in keys
    )
    weight_change = float(numerator / denominator) if denominator else 0.0
    return jaccard, weight_change


def window_intensity(window):
    values = [
        transition_intensity(window.graphs[index - 1], window.graphs[index])
        for index in range(1, len(window.graphs))
    ]
    if not values:
        return {"edge_jaccard": 0.0, "edge_weight": 0.0, "combined": 0.0}
    jaccard = float(np.mean([value[0] for value in values]))
    weight = float(np.mean([value[1] for value in values]))
    return {"edge_jaccard": jaccard, "edge_weight": weight, "combined": (jaccard + weight) / 2.0}


def raw_test_windows(dataset):
    config = yaml.load(
        (ROOT / "config" / f"{dataset}.yaml").read_text(encoding="utf-8-sig"),
        Loader=yaml.FullLoader,
    )
    train_samples, test_samples = load_samples(str(ROOT / config["path"]["sample_dir"]))
    del train_samples
    cases = pd.read_csv(ROOT / config["path"]["case_dir"])
    node_hash, _, _ = load_init(str(ROOT / config["path"]["graph_dir"]))
    settings = config.get("data_settings", {})
    ratio = config.get("feedback", {}).get("sample_num", 0.3)
    ratio = float(ratio) if isinstance(ratio, (int, float)) else 0.3
    _, _, test_cases, _ = split_cases_for_training(
        cases,
        0.3,
        ratio,
        protocol=settings.get("case_split_protocol", "time_strict"),
        test_start_ratio=0.5,
        group_column=settings.get("case_group_column"),
    )
    return build_case_windows(
        test_cases,
        test_samples,
        node_hash,
        window_size=int(settings.get("window_size", 10)),
        sample_interval_seconds=float(settings.get("sample_interval_seconds", 60.0)),
        case_pre_seconds=settings.get("case_pre_seconds"),
        case_post_seconds=settings.get("case_post_seconds"),
        max_gap_seconds=settings.get("max_gap_seconds"),
        incomplete_policy=settings.get("case_incomplete_policy", "pad"),
    )


def rank_bins(intensities):
    order = np.argsort(np.asarray(intensities), kind="mergesort")
    labels = np.empty(len(order), dtype=object)
    names = ("低", "中", "高")
    for rank, index in enumerate(order):
        labels[index] = names[min(2, int(rank * 3 / len(order)))]
    return labels


def metric_rows(scores, labels, indices):
    index = torch.as_tensor(indices, dtype=torch.long)
    return [
        topk_metrics(
            torch.as_tensor(score, dtype=torch.float32).index_select(0, index),
            labels.index_select(0, index),
        )
        for score in scores
    ]


def aggregate(rows):
    result = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = {
            "values": values.tolist(),
            "mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)),
            "display": f"{values.mean():.2f} ± {values.std(ddof=1):.2f}",
        }
    return result


def fused_scores(dataset, variant, data, statistical, weight):
    values = []
    for seed in SEEDS:
        neural = read_score_matrix(
            RUN_ROOT / dataset / variant / f"seed{seed}" / "eval_predictions.csv",
            data.test,
            data.node_names,
        )
        values.append(fuse_case_scores(neural, statistical, weight))
    return values


def hit_vector(score, labels, k):
    score_tensor = torch.as_tensor(score, dtype=torch.float32)
    top = score_tensor.topk(min(k, score_tensor.shape[1]), dim=1).indices
    return torch.gather(labels, 1, top).gt(0).any(dim=1).float().numpy()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = []
    correlations = []
    audit = []
    for dataset in DATASETS:
        data = load_rq1_data(dataset)
        windows = raw_test_windows(dataset)
        if [int(window.case_id) for window in windows] != [int(case.case_id) for case in data.test]:
            raise ValueError(f"{dataset}: raw windows do not align with locked test cases")
        intensity_records = [window_intensity(window) for window in windows]
        intensities = np.asarray([value["combined"] for value in intensity_records])
        bins = rank_bins(intensities)

        lock = json.loads(
            (LOCK_ROOT / dataset / "locked_test.json").read_text(encoding="utf-8")
        )["locked_calibration"]
        train_x, train_y = feature_and_label(data.train)
        val_x, val_y = feature_and_label(data.validation)
        test_x, test_y = feature_and_label(data.test)
        calibrator = build_calibrator(lock["calibrator"]).fit(
            torch.cat([train_x, val_x]), torch.cat([train_y, val_y])
        )
        statistical = calibrator.predict(test_x)
        dynamic = fused_scores(
            dataset, "source_hypothesis_losses", data, statistical, lock["neural_base_weight"]
        )
        frozen = fused_scores(
            dataset, "frozen_window_topology", data, statistical, lock["neural_base_weight"]
        )

        for name in ("低", "中", "高"):
            indices = np.where(bins == name)[0]
            dynamic_summary = aggregate(metric_rows(dynamic, test_y, indices))
            frozen_summary = aggregate(metric_rows(frozen, test_y, indices))
            table.append({
                "dataset": DISPLAY[dataset],
                "dataset_key": dataset,
                "intensity_group": name,
                "case_count": int(len(indices)),
                "combined_intensity_mean": float(intensities[indices].mean()),
                "dynamic": dynamic_summary,
                "frozen": frozen_summary,
                "dynamic_minus_frozen": {
                    metric: float(dynamic_summary[metric]["mean"] - frozen_summary[metric]["mean"])
                    for metric in METRICS
                },
            })

        per_case_delta = np.mean(
            np.stack([hit_vector(score, test_y, 1) for score in dynamic]), axis=0
            ) - np.mean(
            np.stack([hit_vector(score, test_y, 1) for score in frozen]), axis=0
        )
        rho, pvalue = spearmanr(intensities, per_case_delta)
        correlations.append({
            "dataset": DISPLAY[dataset],
            "spearman_rho": None if not np.isfinite(rho) else float(rho),
            "p_value": None if not np.isfinite(pvalue) else float(pvalue),
        })
        audit.append({
            "dataset": dataset,
            "case_count": len(windows),
            "intensity_min": float(intensities.min()),
            "intensity_max": float(intensities.max()),
            "intensity_mean": float(intensities.mean()),
            "case_records": [
                {"case_id": int(window.case_id), "group": str(group), **record}
                for window, group, record in zip(windows, bins, intensity_records)
            ],
        })

    payload = {
        "protocol": {
            "intensity": "mean of edge-multiset Jaccard distance and normalized edge-weight L1 change",
            "groups": "equal-count within-dataset rank terciles",
            "models": "observed dynamic graph versus frozen first graph",
            "calibration": "same validation-locked RQ1 calibrator within each dataset",
            "test_used_for_selection": False,
            "seeds": list(SEEDS),
        },
        "table": table,
        "correlations": correlations,
        "audit": audit,
    }
    (OUTPUT / "rq5_topology_intensity_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# RQ5：拓扑变化强度与动态图收益", "",
        "| 数据集 | 强度 | 案例数 | 动态 A@1 | 冻结 A@1 | ΔA@1 | 动态 A@3 | 冻结 A@3 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['dataset']} | {row['intensity_group']} | {row['case_count']} | "
            f"{row['dynamic']['A@1']['display']} | {row['frozen']['A@1']['display']} | "
            f"{row['dynamic_minus_frozen']['A@1']:+.2f} | "
            f"{row['dynamic']['A@3']['display']} | {row['frozen']['A@3']['display']} |"
        )
    lines += ["", "## 强度—收益相关性", ""]
    for row in correlations:
        lines.append(f"- {row['dataset']}: Spearman ρ={row['spearman_rho']}, p={row['p_value']}")
    (OUTPUT / "RQ5_TOPOLOGY_INTENSITY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
