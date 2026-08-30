"""Audit and summarize the RQ3 lag-relation and source-response-role study."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = ROOT / "res" / "reacr_rca_full_20260824"
NO_LAG_ROOT = ROOT / "res" / "rq3_lag_ablation_20260825"
OUTPUT = ROOT / "res" / "rq3_source_response_20260825"
DATASETS = ("SN", "TT", "D1_weighted")
DISPLAY = {"SN": "SN", "TT": "TT", "D1_weighted": "D1"}
SEEDS = tuple(range(5))
RANK_METRICS = ("A@1", "A@3", "A@5")
ROLE_METRICS = (
    "response_candidate_error_rate",
    "candidate_membership_auc",
    "label_vs_response_candidate_margin_mean",
    "source_role_gap_mean",
    "response_role_gap_mean",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(values, *, scale=1.0, digits=2):
    array = np.asarray(values, dtype=float) * scale
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite values: {array.tolist()}")
    mean = float(array.mean())
    sample_std = float(array.std(ddof=1))
    return {
        "values": array.tolist(),
        "mean": mean,
        "sample_std": sample_std,
        "display": f"{mean:.{digits}f} ± {sample_std:.{digits}f}",
    }


def paired_delta(left, right, *, scale=1.0):
    left_array = np.asarray(left, dtype=float) * scale
    right_array = np.asarray(right, dtype=float) * scale
    delta = left_array - right_array
    return {
        "paired_values": delta.tolist(),
        "mean": float(delta.mean()),
        "sample_std": float(delta.std(ddof=1)),
    }


def load_runs(dataset: str, variant: str, root: Path):
    runs = []
    records = []
    for seed in SEEDS:
        run_dir = root / dataset / variant / f"seed{seed}"
        config_path = run_dir / "run_config.json"
        eval_path = run_dir / "final_eval.json"
        role_path = run_dir / "challenge2_role_summary.json"
        missing = [path for path in (config_path, eval_path, role_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("missing RQ3 artifacts: " + ", ".join(map(str, missing)))
        config = read_json(config_path)
        final_eval = read_json(eval_path)
        role = read_json(role_path)
        for metric in RANK_METRICS:
            value = final_eval["metrics"][metric]
            if not math.isfinite(value):
                raise ValueError(f"{eval_path}: non-finite {metric}")
        for metric in ROLE_METRICS:
            value = role[metric]
            if not math.isfinite(value):
                raise ValueError(f"{role_path}: non-finite {metric}")
        runs.append({"config": config, "final_eval": final_eval, "role": role})
        records.append({
            "dataset": dataset,
            "variant": variant,
            "seed": seed,
            "run_config_sha256": sha256(config_path),
            "final_eval_sha256": sha256(eval_path),
            "challenge2_role_summary_sha256": sha256(role_path),
        })
    return runs, records


def summarize(runs):
    return {
        "ranking": {
            metric: aggregate(
                [run["final_eval"]["metrics"][metric] for run in runs], scale=100.0
            )
            for metric in RANK_METRICS
        },
        "diagnostics": {
            metric: aggregate(
                [run["role"][metric] for run in runs],
                scale=100.0 if metric in {
                    "response_candidate_error_rate", "candidate_membership_auc"
                } else 1.0,
                digits=2 if metric in {
                    "response_candidate_error_rate", "candidate_membership_auc"
                } else 4,
            )
            for metric in ROLE_METRICS
        },
    }


def compare(left_runs, right_runs):
    return {
        "ranking": {
            metric: paired_delta(
                [run["final_eval"]["metrics"][metric] for run in left_runs],
                [run["final_eval"]["metrics"][metric] for run in right_runs],
                scale=100.0,
            )
            for metric in RANK_METRICS
        },
        "diagnostics": {
            metric: paired_delta(
                [run["role"][metric] for run in left_runs],
                [run["role"][metric] for run in right_runs],
                scale=100.0 if metric in {
                    "response_candidate_error_rate", "candidate_membership_auc"
                } else 1.0,
            )
            for metric in ROLE_METRICS
        },
    }


def audit_config_pair(left, right, expected_differences, context):
    differing = {
        key for key in set(left) | set(right) if left.get(key) != right.get(key)
    }
    if differing != set(expected_differences):
        raise ValueError(
            f"{context}: expected config differences {sorted(expected_differences)}, "
            f"found {sorted(differing)}"
        )


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lag_rows = []
    role_rows = []
    audit_records = []

    for dataset in DATASETS:
        full, records = load_runs(dataset, "source_hypothesis_losses", MAIN_ROOT)
        audit_records.extend(records)
        no_lag, records = load_runs(dataset, "no_lag_relation", NO_LAG_ROOT)
        audit_records.extend(records)
        base, records = load_runs(dataset, "legacy", MAIN_ROOT)
        audit_records.extend(records)
        role, records = load_runs(dataset, "root_symptom", MAIN_ROOT)
        audit_records.extend(records)

        for seed, (full_run, no_lag_run) in enumerate(zip(full, no_lag)):
            if full_run["config"].get("causal_max_lag") != 3:
                raise ValueError(f"{dataset}/full/seed{seed}: causal_max_lag is not 3")
            if no_lag_run["config"].get("causal_max_lag") != 0:
                raise ValueError(f"{dataset}/no_lag/seed{seed}: causal_max_lag is not 0")
            audit_config_pair(
                full_run["config"],
                no_lag_run["config"],
                {
                    "causal_max_lag",
                    "output_dir",
                    "export_event_chains",
                    "allow_event_chain_fallback",
                },
                f"{dataset}/lag/seed{seed}",
            )
        for seed, (base_run, role_run) in enumerate(zip(base, role)):
            audit_config_pair(
                base_run["config"],
                role_run["config"],
                {"ranker_mode", "output_dir"},
                f"{dataset}/role/seed{seed}",
            )

        lag_rows.append({
            "dataset": DISPLAY[dataset],
            "dataset_key": dataset,
            "full_lag": summarize(full),
            "no_lag": summarize(no_lag),
            "full_minus_no_lag": compare(full, no_lag),
        })
        role_rows.append({
            "dataset": DISPLAY[dataset],
            "dataset_key": dataset,
            "base_ranker": summarize(base),
            "source_response_role": summarize(role),
            "role_minus_base": compare(role, base),
        })

    payload = {
        "research_question": (
            "Can lagged directed-relation evidence and source-response role modeling "
            "reduce downstream response-candidate interference in root-cause ranking?"
        ),
        "protocol": {
            "seeds": list(SEEDS),
            "reporting": "mean_and_sample_std",
            "lag_ablation": (
                "causal_max_lag=3 versus causal_max_lag=0; all training and ranking "
                "settings match. Event-chain export flags differ only after evaluation."
            ),
            "role_ablation": (
                "base ranker versus source-response role ranker; ranker_mode is the "
                "only modeling difference"
            ),
            "response_candidate_definition": (
                "model-mined high-response candidate, not a dataset-provided symptom label"
            ),
        },
        "lag_ablation": lag_rows,
        "role_ablation": role_rows,
        "run_audit": audit_records,
    }
    (OUTPUT / "rq3_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# RQ3：时滞有向关系与源—响应角色建模", "",
        "## 实验逻辑", "",
        "本研究问题分别检验两个机制。首先，将完整模型的最大时滞由 3 置为 0，以移除跨时刻的时滞有向关系证据；其次，在不启用解释约束损失的条件下，将基础排序器与源—响应角色排序器进行对照。除上述目标因素外，两组对照保持相同的数据划分、训练预算与随机种子。所有结果均报告 5 个随机种子的均值 ± 样本标准差。", "",
        "响应候选错误率表示模型挖掘的高响应候选排在根因标签候选之前的案例比例，越低越好；候选成员 AUC 衡量根因候选集合与非候选集合的角色证据可分性，越高越好；标签—响应候选间隔越大越好。数据集没有提供症状节点标签，因此这些量是模型内部诊断指标，不是症状分类准确率。", "",
        "## 时滞有向关系消融", "",
        "| 数据集 | 设置 | A@1 (%) | A@3 (%) | A@5 (%) | 响应候选错误率 (%) | 标签—响应候选间隔 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in lag_rows:
        for key, name in (("full_lag", "完整时滞关系"), ("no_lag", "无时滞关系")):
            item = row[key]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["dataset"], name,
                    item["ranking"]["A@1"]["display"],
                    item["ranking"]["A@3"]["display"],
                    item["ranking"]["A@5"]["display"],
                    item["diagnostics"]["response_candidate_error_rate"]["display"],
                    item["diagnostics"]["label_vs_response_candidate_margin_mean"]["display"],
                )
            )
    lines += ["", "## 源—响应角色消融", "",
        "| 数据集 | 设置 | A@1 (%) | A@3 (%) | A@5 (%) | 响应候选错误率 (%) | 候选成员 AUC (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in role_rows:
        for key, name in (("base_ranker", "基础排序器"), ("source_response_role", "源—响应角色排序器")):
            item = row[key]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["dataset"], name,
                    item["ranking"]["A@1"]["display"],
                    item["ranking"]["A@3"]["display"],
                    item["ranking"]["A@5"]["display"],
                    item["diagnostics"]["response_candidate_error_rate"]["display"],
                    item["diagnostics"]["candidate_membership_auc"]["display"],
                )
            )

    lines += ["", "## 结果分析", ""]
    for row in lag_rows:
        delta = row["full_minus_no_lag"]
        lines.append(
            f"- {row['dataset']} 的完整时滞关系相对无时滞关系，A@1/A@3/A@5 变化为 "
            f"{delta['ranking']['A@1']['mean']:+.2f}/"
            f"{delta['ranking']['A@3']['mean']:+.2f}/"
            f"{delta['ranking']['A@5']['mean']:+.2f} 个百分点，响应候选错误率变化为 "
            f"{delta['diagnostics']['response_candidate_error_rate']['mean']:+.2f} 个百分点。"
        )
    for row in role_rows:
        delta = row["role_minus_base"]
        lines.append(
            f"- {row['dataset']} 的源—响应角色排序器相对基础排序器，A@1/A@3/A@5 变化为 "
            f"{delta['ranking']['A@1']['mean']:+.2f}/"
            f"{delta['ranking']['A@3']['mean']:+.2f}/"
            f"{delta['ranking']['A@5']['mean']:+.2f} 个百分点，响应候选错误率变化为 "
            f"{delta['diagnostics']['response_candidate_error_rate']['mean']:+.2f} 个百分点，"
            f"候选成员 AUC 变化为 {delta['diagnostics']['candidate_membership_auc']['mean']:+.2f} 个百分点。"
        )

    lines += [
        "", "综合来看，时滞有向关系在 SN 上同时提高 A@K 并降低响应候选错误率，但在 TT 和 D1 上没有形成一致增益；源—响应角色在 D1 上提高 A@1 和候选成员 AUC，并降低响应候选错误率，在 TT 上仅改善 A@3/A@5，在 SN 上无可测变化。因此，RQ3 只能得到部分支持：两个机制可以在部分数据分布下抑制响应候选干扰，但当前证据不足以证明其跨数据集普遍有效。",
        "", "## Claim–Evidence Map", "",
        "- Claim：时滞有向关系证据能够稳定提高三个数据集的根因排序。 | Evidence：SN 的 A@K 和响应候选错误率均改善，但 TT、D1 的结果方向不一致。 | Status：not supported；只能陈述 SN 上的收益。",
        "- Claim：源—响应角色建模能够减少下游响应候选干扰。 | Evidence：D1 的 A@1、响应候选错误率和候选成员 AUC 同向改善；TT 仅 A@3/A@5 改善；SN 无变化。 | Status：partially supported。",
        "- Claim：模型能够识别真实症状节点或真实故障传播链。 | Evidence：数据集未提供症状节点与传播链标签。 | Status：not supported。", "",
        "## 自审", "",
        "- 清晰性：时滞关系与角色建模采用两个独立对照，不混合解释。",
        "- 一致性：全文使用“时滞有向关系证据”“源—响应角色”和“响应候选”。",
        "- 证据边界：响应候选来自模型挖掘，不等同于真实症状节点。",
        "- 可复核性：结果文件保存 60 个运行工件的 SHA-256、五种子原始值、均值和样本标准差。",
    ]
    (OUTPUT / "RQ3_SOURCE_RESPONSE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "runs": len(audit_records), "output": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
