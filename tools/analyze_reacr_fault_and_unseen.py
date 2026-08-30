"""Stratify locked REACR test results by fault type, label level and seen root identity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from reacr_rca_pipeline import split_cases_for_training
from tools.optimize_reacr_rq1 import (
    build_calibrator,
    feature_and_label,
    read_score_matrix,
)
from models.reacr_rca_statistical_calibrator import fuse_case_scores


DATASETS = ("SN", "TT", "D1_weighted")
DISPLAY = {"SN": "SN", "TT": "TT", "D1_weighted": "D1"}
SEEDS = tuple(range(5))
RUN_ROOT = ROOT / "res" / "reacr_rca_full_20260824"
LOCK_ROOT = ROOT / "res" / "reacr_rq1_optimized_20260824_refit"
OUTPUT = ROOT / "res" / "reacr_rq3_rq4_generalization_20260825"
VARIANT = "source_hypothesis_losses"
METRICS = ("A@1", "A@3", "A@5")


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


def d1_fault_group(value: str) -> str:
    value = str(value).lower()
    if "network" in value:
        return "网络"
    if "memory" in value:
        return "内存"
    if "cpu" in value:
        return "CPU"
    if "disk" in value or "i/o" in value:
        return "存储 I/O"
    if "process" in value:
        return "进程终止"
    raise ValueError(f"unmapped D1 fault type: {value}")


def display_fault(dataset: str, value: str) -> str:
    if dataset == "D1_weighted":
        return d1_fault_group(value)
    mapping = {
        "cpu_load": "CPU 负载",
        "network_delay": "网络延迟",
        "network_loss": "网络丢包",
    }
    return mapping.get(str(value), str(value))


def metadata_splits(dataset: str):
    config = yaml.load(
        (ROOT / "config" / f"{dataset}.yaml").read_text(encoding="utf-8-sig"),
        Loader=yaml.FullLoader,
    )
    cases = pd.read_csv(ROOT / config["path"]["case_dir"])
    settings = config.get("data_settings", {})
    default_ratio = config.get("feedback", {}).get("sample_num", 0.3)
    if not isinstance(default_ratio, float):
        default_ratio = 0.3
    train, validation, test, split_meta = split_cases_for_training(
        cases,
        0.3,
        float(default_ratio),
        protocol=settings.get("case_split_protocol", "time_strict"),
        test_start_ratio=0.5,
        group_column=settings.get("case_group_column"),
    )
    return train, validation, test, split_meta


def locked_scores(dataset: str):
    data = load_rq1_data(dataset)
    lock_artifact = json.loads(
        (LOCK_ROOT / dataset / "locked_test.json").read_text(encoding="utf-8")
    )
    lock = lock_artifact["locked_calibration"]
    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    development_x = torch.cat([train_x, val_x], dim=0)
    development_y = torch.cat([train_y, val_y], dim=0)
    calibrator = build_calibrator(lock["calibrator"]).fit(development_x, development_y)
    test_x, test_y = feature_and_label(data.test)
    statistical = calibrator.predict(test_x)
    scores = []
    for seed in SEEDS:
        neural = read_score_matrix(
            RUN_ROOT / dataset / VARIANT / f"seed{seed}" / "eval_predictions.csv",
            data.test,
            data.node_names,
        )
        scores.append(fuse_case_scores(neural, statistical, lock["neural_base_weight"]))
    reconstructed = [
        topk_metrics(torch.as_tensor(score, dtype=torch.float32), test_y)
        for score in scores
    ]
    expected = lock_artifact["test_per_seed"]
    for seed, (actual, recorded) in enumerate(zip(reconstructed, expected)):
        for metric in METRICS:
            if not np.isclose(actual[metric], recorded[metric], atol=1e-6):
                raise ValueError(
                    f"{dataset}/seed{seed}/{metric}: reconstructed={actual[metric]} "
                    f"does not match locked={recorded[metric]}"
                )
    return data, test_y, scores


def subset_metrics(scores, labels, indices):
    selected = torch.as_tensor(indices, dtype=torch.long)
    return [
        topk_metrics(
            torch.as_tensor(score, dtype=torch.float32).index_select(0, selected),
            labels.index_select(0, selected),
        )
        for score in scores
    ]


def grouped_rows(dataset, test_meta, test_ids, scores, labels, column, dimension):
    meta = test_meta.set_index("case_id").loc[test_ids].reset_index()
    if column == "fault_group":
        meta[column] = meta["failure_type"].map(
            lambda value: display_fault(dataset, value)
        )
    rows = []
    for value, frame in meta.groupby(column, sort=True):
        indices = frame.index.to_numpy(dtype=int)
        rows.append({
            "dataset": DISPLAY[dataset],
            "dimension": dimension,
            "group": str(value),
            "case_count": int(len(indices)),
            "metrics": aggregate(subset_metrics(scores, labels, indices)),
            "case_ids": [int(value) for value in frame["case_id"].tolist()],
        })
    return rows


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fault_rows = []
    granularity_rows = []
    unseen_rows = []
    audits = []

    for dataset in DATASETS:
        train_meta, validation_meta, test_meta, split_meta = metadata_splits(dataset)
        data, labels, scores = locked_scores(dataset)
        test_ids = [int(case.case_id) for case in data.test]
        actual_ids = [int(value) for value in test_meta["case_id"].tolist()]
        if test_ids != actual_ids:
            raise ValueError(f"{dataset}: case order does not match locked evaluation")

        fault_rows.extend(
            grouped_rows(dataset, test_meta, test_ids, scores, labels, "fault_group", "故障类型")
        )
        if dataset == "D1_weighted":
            granularity_rows.extend(
                grouped_rows(dataset, test_meta, test_ids, scores, labels, "level", "标签粒度")
            )

        development_roots = set(train_meta["cmdb_id"].astype(str)) | set(
            validation_meta["cmdb_id"].astype(str)
        )
        aligned = test_meta.set_index("case_id").loc[test_ids].reset_index()
        aligned["root_identity_status"] = np.where(
            aligned["cmdb_id"].astype(str).isin(development_roots),
            "开发集已见",
            "开发集未见",
        )
        unseen_rows.extend(
            grouped_rows(
                dataset,
                aligned,
                test_ids,
                scores,
                labels,
                "root_identity_status",
                "根因身份",
            )
        )
        audits.append({
            "dataset": dataset,
            "split_meta": split_meta,
            "test_case_count": len(test_ids),
            "development_root_count": len(development_roots),
            "test_root_count": int(test_meta["cmdb_id"].nunique()),
            "test_case_ids": test_ids,
            "locked_test_path": str(LOCK_ROOT / dataset / "locked_test.json"),
        })

    payload = {
        "protocol": {
            "source": "locked five-seed RQ1 predictions reconstructed with the validation-selected calibrator",
            "test_used_for_selection": False,
            "metrics": list(METRICS),
            "reporting": "mean_and_sample_std",
            "fault_grouping": {
                "SN_TT": "original controlled injection types",
                "D1": "CPU, memory, network, storage I/O, and process termination",
            },
            "unseen_definition": "cmdb_id absent from both training and validation metadata",
        },
        "fault_type": fault_rows,
        "label_granularity": granularity_rows,
        "root_identity_generalization": unseen_rows,
        "audit": audits,
    }
    (OUTPUT / "generalization_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# REACR 分层性能：故障类型、标签粒度与未见根因身份", "",
        "以下结果直接重建自 RQ1 已锁定的五随机种子测试预测；分层标签未参与超参数选择。", "",
        "## 故障类型", "",
        "| 数据集 | 故障类型 | 案例数 | A@1 (%) | A@3 (%) | A@5 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in fault_rows:
        lines.append(
            f"| {row['dataset']} | {row['group']} | {row['case_count']} | "
            f"{row['metrics']['A@1']['display']} | {row['metrics']['A@3']['display']} | "
            f"{row['metrics']['A@5']['display']} |"
        )
    lines += ["", "## D1 标签粒度", "",
        "| 标签粒度 | 案例数 | A@1 (%) | A@3 (%) | A@5 (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    level_names = {"service": "服务级候选集合", "pod": "Pod", "node": "节点"}
    for row in granularity_rows:
        lines.append(
            f"| {level_names.get(row['group'], row['group'])} | {row['case_count']} | "
            f"{row['metrics']['A@1']['display']} | {row['metrics']['A@3']['display']} | "
            f"{row['metrics']['A@5']['display']} |"
        )
    lines += ["", "## 未见根因身份", "",
        "| 数据集 | 根因身份 | 案例数 | A@1 (%) | A@3 (%) | A@5 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in unseen_rows:
        lines.append(
            f"| {row['dataset']} | {row['group']} | {row['case_count']} | "
            f"{row['metrics']['A@1']['display']} | {row['metrics']['A@3']['display']} | "
            f"{row['metrics']['A@5']['display']} |"
        )
    (OUTPUT / "GENERALIZATION_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
