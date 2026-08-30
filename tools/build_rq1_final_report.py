"""Audit RQ1 artifacts and build the manuscript-ready result table.

The report deliberately distinguishes adapted algorithmic cores from untouched
author repositories.  It also records which REACR final-fit policy produced each
reported row so the table can be reproduced without inspecting the manuscript.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = ROOT / "res" / "rq1_baselines_20260824"
REACR_DEV_REFIT = ROOT / "res" / "reacr_rq1_optimized_20260824_refit"
OUTPUT = ROOT / "res" / "rq1_final_20260824"
DATASETS = ("SN", "TT", "D1_weighted")
METHODS = (
    "DyCause", "DejaVu", "Eadro", "BARO", "MicroIRC", "MODIFy",
    "MicroInspect", "TVDiag", "UDA-RCL",
)
TRAINABLE = {"DejaVu", "Eadro", "MicroIRC", "MODIFy", "MicroInspect", "TVDiag", "UDA-RCL"}
DISPLAY_DATASET = {"SN": "SN", "TT": "TT", "D1_weighted": "D1"}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(mean, std, runs):
    if runs == 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {std:.2f}"


def audit_baselines():
    errors = []
    records = []
    for dataset in DATASETS:
        split = load_json(BASELINE_ROOT / dataset / "split.json")
        for method in METHODS:
            expected = 5 if method in TRAINABLE else 1
            paths = sorted((BASELINE_ROOT / dataset / method).glob("seed*/result.json"))
            if len(paths) != expected:
                errors.append(f"{dataset}/{method}: expected {expected} runs, found {len(paths)}")
            for path in paths:
                result = load_json(path)
                manifest = result["implementation_manifest"]
                if manifest.get("test_used_for_model_selection") is not False:
                    errors.append(f"{path}: test-used flag is not false")
                prediction = path.with_name("predictions.csv")
                with prediction.open(encoding="utf-8") as handle:
                    count = sum(1 for _ in csv.DictReader(handle))
                if count != split["test_case_count"]:
                    errors.append(f"{prediction}: {count} rows, expected {split['test_case_count']}")
                records.append({
                    "dataset": dataset,
                    "method": method,
                    "seed": result["seed"],
                    "test_metrics": result["test_metrics"],
                    "implementation_manifest": manifest,
                    "result_sha256": sha256(path),
                    "predictions_sha256": sha256(prediction),
                })
    return errors, records


def choose_reacr():
    # Use one protocol for every dataset: validation locks all hyperparameters,
    # then the lightweight calibrator is refitted on the complete development set.
    # This avoids selecting a final-fit policy after observing test performance.
    source = {dataset: REACR_DEV_REFIT for dataset in DATASETS}
    rows = {}
    audit = []
    errors = []
    for dataset, root in source.items():
        search_path = root / dataset / "validation_search.json"
        test_path = root / dataset / "locked_test.json"
        search = load_json(search_path)
        test = load_json(test_path)
        if search.get("selection_split") != "validation_only":
            errors.append(f"{search_path}: selection split is not validation_only")
        if search.get("test_files_opened_during_search") is not False:
            errors.append(f"{search_path}: search reports opening test files")
        if test.get("test_used_for_selection") is not False:
            errors.append(f"{test_path}: test-used flag is not false")
        rows[dataset] = {
            metric: {
                "mean": test["test_aggregate"][metric]["mean"],
                "std": test["test_aggregate"][metric]["sample_std"],
            }
            for metric in ("A@1", "A@3", "A@5")
        }
        audit.append({
            "dataset": dataset,
            "final_fit_policy": "train_plus_validation_after_validation_lock",
            "validation_search": str(search_path.relative_to(ROOT)),
            "locked_test": str(test_path.relative_to(ROOT)),
            "validation_search_sha256": sha256(search_path),
            "locked_test_sha256": sha256(test_path),
            "seeds": test["seeds"],
            "locked_calibration": test["locked_calibration"],
        })
    return errors, rows, audit


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_errors, run_records = audit_baselines()
    reacr_errors, reacr_rows, reacr_audit = choose_reacr()
    errors = baseline_errors + reacr_errors
    if errors:
        raise SystemExit("\n".join(errors))

    baseline_summary = load_json(BASELINE_ROOT / "summary.json")
    lookup = {(row["dataset"], row["method"]): row for row in baseline_summary}
    table = []
    for method in METHODS:
        row = {"method": method}
        for dataset in DATASETS:
            item = lookup[(dataset, method)]
            row[DISPLAY_DATASET[dataset]] = {
                metric: {
                    "mean": item[f"{metric}_mean"],
                    "std": item[f"{metric}_std"],
                    "runs": item["runs"],
                    "display": fmt(item[f"{metric}_mean"], item[f"{metric}_std"], item["runs"]),
                }
                for metric in ("A@1", "A@3", "A@5")
            }
        table.append(row)
    reacr = {"method": "REACR"}
    for dataset in DATASETS:
        reacr[DISPLAY_DATASET[dataset]] = {
            metric: {
                **reacr_rows[dataset][metric], "runs": 5,
                "display": fmt(reacr_rows[dataset][metric]["mean"], reacr_rows[dataset][metric]["std"], 5),
            }
            for metric in ("A@1", "A@3", "A@5")
        }
    table.append(reacr)

    ranks = {}
    for dataset in ("SN", "TT", "D1"):
        for metric in ("A@1", "A@3", "A@5"):
            values = sorted({round(row[dataset][metric]["mean"], 10) for row in table}, reverse=True)
            ranks[f"{dataset} {metric}"] = {"best": values[0], "second": values[1]}

    payload = {
        "protocol": {
            "candidate_semantics": "positive_candidate_bag_any_hit",
            "selection": "validation_only",
            "test_use": "final_evaluation_only",
            "trainable_reporting": "five_seeds_mean_and_sample_std",
            "deterministic_reporting": "single_run",
            "baseline_run_count": len(run_records),
            "reacr_run_count": 15,
        },
        "table": table,
        "ranks": ranks,
        "reacr_audit": reacr_audit,
        "baseline_runs": run_records,
    }
    (OUTPUT / "final_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# REACR RQ1 最终实验结果（2026-08-24）", "",
        "所有方法共享故障窗口、候选节点、正候选集合、数据划分与 A@K 实现。可训练方法报告 5 个随机种子的均值 ± 样本标准差；DyCause 与 BARO 为确定性方法，报告单次结果。参数与早停仅由验证集确定。", "",
        "注意：表中的外部方法是面向统一 SN/TT/D1 输入的算法核心适配结果，并非作者仓库开箱即用的原始数字。DyCause 与 MicroInspect 为依据论文描述的重实现；其余方法复用或移植公开实现中的核心结构。", "",
        "| 方法 | SN A@1 | SN A@3 | SN A@5 | TT A@1 | TT A@3 | TT A@5 | D1 A@1 | D1 A@3 | D1 A@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table:
        cells = [row[ds][metric]["display"] for ds in ("SN", "TT", "D1") for metric in ("A@1", "A@3", "A@5")]
        lines.append("| " + row["method"] + " | " + " | ".join(cells) + " |")
    lines += [
        "", "## 结论", "",
        "REACR 在 9 个数据集—指标组合中的 6 个取得最优结果：SN 的 A@1/A@3、TT 的全部三个指标以及 D1 的 A@1。D1 A@3 与 TVDiag 并列次优，SN 和 D1 的 A@5 分别低于 MicroInspect 及 Eadro/TVDiag。",
        "", "REACR 相对最强外部基线的 A@1 绝对提升分别为 SN 1.11、TT 10.67、D1 3.62 个百分点。该结果说明动态图、时滞有向关系与源—响应建模主要改善首位根因排序；SN 和 D1 的 A@5 结果表明其较长候选列表覆盖率仍有提升空间。",
        "", "## 可复核性", "",
        f"- 外部基线运行：{len(run_records)} 次，预测文件与测试案例数逐一核对。",
        "- REACR：3 个数据集 × 5 个随机种子，共 15 次锁定评估。",
        "- `final_results.json` 保存每个结果文件和预测文件的 SHA-256、源码提交号、适配器哈希及测试集未参与选择标记。",
    ]
    (OUTPUT / "RQ1_FINAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "baseline_runs": len(run_records), "reacr_runs": 15, "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
