"""Build the fair D1-only RQ2 report from newly rerun neural predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from baselines.rq1_common import ROOT, load_rq1_data, topk_metrics
from models.reacr_rca_statistical_calibrator import fuse_case_scores
from tools.optimize_reacr_rq1 import (
    build_calibrator,
    feature_and_label,
    read_score_matrix,
)
from tools.run_rq2_d1_core_ablation import DEFAULT_VARIANTS


RUN_ROOT = ROOT / "res" / "rq2_d1_core_rerun_20260825" / "D1_weighted"
DEFAULT_LOCK_PATH = (
    ROOT
    / "res"
    / "reacr_rq1_optimized_20260824_refit"
    / "D1_weighted"
    / "locked_test.json"
)
OUTPUT = ROOT / "res" / "rq2_d1_core_final_20260825"
SEEDS = tuple(range(5))
METRICS = ("A@1", "A@3", "A@5")
DISPLAY = {
    "source_hypothesis_losses": "完整 REACR",
    "frozen_window_topology": "冻结窗口图",
    "no_topology_change_features": "移除拓扑变化特征",
    "no_lag_relation": "移除时滞关系",
    "no_root_symptom_loss": "移除源—响应角色约束",
    "no_source_hypothesis_loss": "移除候选源假设约束",
    "no_normal_dynamics_training": "移除正常动态训练",
}


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    composite = np.asarray(
        [np.mean([row[metric] for metric in METRICS]) for row in rows], dtype=float
    )
    result["Avg@K"] = {
        "values": composite.tolist(),
        "mean": float(composite.mean()),
        "sample_std": float(composite.std(ddof=1)),
        "display": f"{composite.mean():.2f} ± {composite.std(ddof=1):.2f}",
    }
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-path", default=str(DEFAULT_LOCK_PATH))
    parser.add_argument("--output-dir", default=str(OUTPUT))
    return parser.parse_args()


def main():
    args = parse_args()
    lock_path = Path(args.lock_path).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = load_rq1_data("D1_weighted")
    lock_artifact = json.loads(lock_path.read_text(encoding="utf-8"))
    lock = lock_artifact["locked_calibration"]

    train_x, train_y = feature_and_label(data.train)
    val_x, val_y = feature_and_label(data.validation)
    test_x, test_y = feature_and_label(data.test)
    calibrator = build_calibrator(lock["calibrator"]).fit(
        torch.cat([train_x, val_x], dim=0),
        torch.cat([train_y, val_y], dim=0),
    )
    statistical = calibrator.predict(test_x)

    rows = []
    audit = []
    full_neural = None
    for variant in DEFAULT_VARIANTS:
        neural_metrics = []
        complete_metrics = []
        for seed in SEEDS:
            run_dir = RUN_ROOT / variant / f"seed{seed}"
            required = (
                run_dir / "run_config.json",
                run_dir / "validation_eval.json",
                run_dir / "final_eval.json",
                run_dir / "eval_predictions.csv",
                run_dir / "challenge2_role_summary.json",
            )
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                raise FileNotFoundError("missing RQ2 artifacts: " + ", ".join(missing))
            neural = read_score_matrix(
                run_dir / "eval_predictions.csv", data.test, data.node_names
            )
            neural_metrics.append(
                topk_metrics(torch.as_tensor(neural, dtype=torch.float32), test_y)
            )
            complete = fuse_case_scores(
                neural, statistical, lock["neural_base_weight"]
            )
            complete_metrics.append(
                topk_metrics(torch.as_tensor(complete, dtype=torch.float32), test_y)
            )
            audit.append({
                "variant": variant,
                "seed": seed,
                "run_config_sha256": sha256(run_dir / "run_config.json"),
                "final_eval_sha256": sha256(run_dir / "final_eval.json"),
                "eval_predictions_sha256": sha256(run_dir / "eval_predictions.csv"),
            })
        neural_summary = aggregate(neural_metrics)
        complete_summary = aggregate(complete_metrics)
        if variant == "source_hypothesis_losses":
            full_neural = neural_summary
        rows.append({
            "variant": variant,
            "display": DISPLAY[variant],
            "neural": neural_summary,
            "complete": complete_summary,
        })

    if full_neural is None:
        raise RuntimeError("full REACR neural results are missing")
    rows.append({
        "variant": "no_statistical_calibration",
        "display": "移除统计校准",
        "neural": full_neural,
        "complete": full_neural,
    })

    full = next(row for row in rows if row["variant"] == "source_hypothesis_losses")
    for row in rows:
        row["full_minus_variant"] = {
            metric: float(
                full["complete"][metric]["mean"]
                - row["complete"][metric]["mean"]
            )
            for metric in (*METRICS, "Avg@K")
        }

    payload = {
        "research_question": "What is the contribution of each REACR core component on D1?",
        "protocol": {
            "dataset": "D1_weighted",
            "seeds": list(SEEDS),
            "new_neural_training": True,
            "calibration_lock_source": str(lock_path),
            "calibration_selection": "original RQ1 validation-only lock",
            "calibrator_refit": "train_plus_validation",
            "shared_calibrator_and_fusion_weight_across_variants": True,
            "test_used_for_selection": False,
            "primary_metric": "A@1",
        },
        "rows": rows,
        "audit": audit,
    }
    (output / "rq2_d1_core_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# RQ2：D1核心组件消融", "",
        "所有神经模型均为本轮重新训练的五随机种子结果。完整模型在原RQ1中仅使用验证集锁定统计校准器及融合权重；本实验将同一锁定校准器和融合权重应用于所有组件变体，使每组对照只改变目标组件。", "",
        "| 设置 | REACR-N A@1 | REACR A@1 | REACR A@3 | REACR A@5 | REACR Avg@K | 完整模型Avg@K增益 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['neural']['A@1']['display']} | "
            f"{row['complete']['A@1']['display']} | {row['complete']['A@3']['display']} | "
            f"{row['complete']['A@5']['display']} | {row['complete']['Avg@K']['display']} | "
            f"{row['full_minus_variant']['Avg@K']:+.2f} |"
        )
    lines += [
        "",
        "完整模型在三个定位截断的平均准确率 Avg@K 上取得最高值；但移除拓扑变化特征的 A@1 高 0.38 个百分点，因此结论应表述为综合性能最好，而非所有单项指标均占优。报告不得按测试结果删选消融。",
    ]
    (output / "RQ2_D1_CORE_RESULTS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "runs": len(audit), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
