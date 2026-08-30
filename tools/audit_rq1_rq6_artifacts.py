"""Audit the formal RQ1--RQ6 artifacts before manuscript generation."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "res"
OUTPUT = RES / "rq_completion_20260825" / "final_audit.json"
PATHS = {
    "RQ1": RES / "rq1_final_20260824" / "final_results.json",
    "RQ2": RES / "rq2_d1_core_final_20260825" / "rq2_d1_core_results.json",
    "RQ3": RES / "reacr_rq3_rq4_generalization_20260825" / "generalization_results.json",
    "RQ4": RES / "rq4_parameter_sensitivity_20260825" / "rq4_parameter_sensitivity_results.json",
    "RQ5": RES / "rq5_topology_intensity_20260825" / "rq5_topology_intensity_results.json",
    "RQ6": RES / "rq6_controlled_roles_20260825" / "rq6_controlled_results.json",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read_all():
    missing = [str(path) for path in PATHS.values() if not path.exists()]
    require(not missing, "missing formal artifacts: " + "; ".join(missing))
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in PATHS.items()}


def audit_rq1(data):
    require(data["protocol"]["selection"] == "validation_only", "RQ1 selection is not validation-only")
    require(data["protocol"]["test_use"] == "final_evaluation_only", "RQ1 test-use protocol mismatch")
    methods = [row["method"] for row in data["table"]]
    require("REACR" in methods and len(methods) >= 10, "RQ1 comparison table is incomplete")
    for row in data["table"]:
        require(all(dataset in row for dataset in ("SN", "TT", "D1")), f"RQ1 missing dataset for {row['method']}")
    return {"methods": methods, "baseline_runs": data["protocol"]["baseline_run_count"]}


def audit_rq2(data):
    require(data["protocol"]["test_used_for_selection"] is False, "RQ2 used test data for selection")
    require(data["protocol"]["seeds"] == [0, 1, 2, 3, 4], "RQ2 does not use five formal seeds")
    require(len(data["rows"]) == 8, "RQ2 must contain the full model and seven ablations")
    for row in data["rows"]:
        require(len(row["complete"]["A@1"]["values"]) == 5, f"RQ2 incomplete seeds: {row['variant']}")
    return {"variants": [row["variant"] for row in data["rows"]]}


def audit_rq3(data):
    require(data["protocol"]["test_used_for_selection"] is False, "RQ3 used test data for selection")
    require(len(data["fault_type"]) == 11, "RQ3 fault-type table is incomplete")
    require(len(data["root_identity_generalization"]) == 4, "RQ3 identity-generalization table is incomplete")
    return {
        "fault_type_rows": len(data["fault_type"]),
        "root_identity_rows": len(data["root_identity_generalization"]),
    }


def audit_rq4(data):
    protocol = data["protocol"]
    require(protocol["selection_split"] == "validation_only", "RQ4 is not validation-only")
    require(protocol["test_artifacts_read"] is False, "RQ4 read test artifacts")
    require(protocol["new_test_outputs_exported"] is False, "RQ4 exported test artifacts")
    require(protocol["seeds"] == [0, 1, 2], "RQ4 must use three seeds")
    require(len(data["rows"]) == 21, "RQ4 parameter table must contain 21 settings")
    checked_new_runs = 0
    for row in data["rows"]:
        require(len(row["sources"]) == 3, f"RQ4 incomplete seeds: {row['parameter']}={row['value']}")
        for source in row["sources"]:
            path = Path(source["path"])
            require((path / "validation_eval.json").exists(), f"RQ4 missing validation result: {path}")
            require((path / "validation_predictions.csv").exists(), f"RQ4 missing validation predictions: {path}")
            if not source["reused_from_rq2"]:
                checked_new_runs += 1
                require(not (path / "final_eval.json").exists(), f"RQ4 contains forbidden test evaluation: {path}")
                require(not (path / "eval_predictions.csv").exists(), f"RQ4 contains forbidden test predictions: {path}")
    require(checked_new_runs == 45, f"RQ4 expected 45 new-run references, found {checked_new_runs}")
    return {"settings": len(data["rows"]), "new_runs_checked": checked_new_runs}


def audit_rq5(data):
    require(data["protocol"]["test_used_for_selection"] is False, "RQ5 used test data for selection")
    require(data["protocol"]["seeds"] == [0, 1, 2, 3, 4], "RQ5 does not use five formal seeds")
    require(len(data["table"]) == 9, "RQ5 must contain three intensity groups for three datasets")
    return {"intensity_rows": len(data["table"]), "correlations": len(data["correlations"])}


def audit_rq6(data):
    protocol = data["protocol"]
    require(protocol["seeds"] == [0, 1, 2, 3, 4], "RQ6 does not use five formal seeds")
    require(protocol["train_cases"] == 600 and protocol["test_cases"] == 240, "RQ6 formal case counts mismatch")
    require(protocol["epochs"] == 35, "RQ6 formal epoch count mismatch")
    require(set(data["summary"]) == {"full", "no_lag", "no_role"}, "RQ6 variants mismatch")
    require(len(data["records"]) == 15, "RQ6 must contain 15 variant-seed records")
    pairs = {(record["variant"], record["seed"]) for record in data["records"]}
    require(len(pairs) == 15, "RQ6 contains duplicate variant-seed records")
    for record in data["records"]:
        config = record["run_config"]
        require(config["train_cases"] == 600 and config["test_cases"] == 240, "RQ6 smoke configuration leaked into formal results")
        require(config["epochs"] == 35, "RQ6 non-formal epoch count found")
        for metric in ("source_role_hit_at_1", "source_role_margin", "response_role_auprc"):
            require(metric in record["metrics"], f"RQ6 missing direct role metric: {metric}")
    return {"records": len(data["records"]), "variant_seed_pairs": len(pairs)}


def main():
    data = read_all()
    audit = {
        "status": "passed",
        "artifacts": {name: str(path) for name, path in PATHS.items()},
        "checks": {
            "RQ1": audit_rq1(data["RQ1"]),
            "RQ2": audit_rq2(data["RQ2"]),
            "RQ3": audit_rq3(data["RQ3"]),
            "RQ4": audit_rq4(data["RQ4"]),
            "RQ5": audit_rq5(data["RQ5"]),
            "RQ6": audit_rq6(data["RQ6"]),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
