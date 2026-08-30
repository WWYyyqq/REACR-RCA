"""Audit the non-saturated formal RQ6 controlled benchmark."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "res" / "rq6_controlled_roles_v2_20260827" / "rq6_controlled_results.json"
OUTPUT = RESULT.parent / "audit.json"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    require(protocol["seeds"] == [0, 1, 2, 3, 4], "formal seeds mismatch")
    require(protocol["node_num"] == 16, "node count mismatch")
    require(protocol["train_cases"] == 180 and protocol["test_cases"] == 480, "case counts mismatch")
    require(protocol["epochs"] == 18 and protocol["noise"] == 0.7, "difficulty protocol mismatch")
    require(protocol["test_data_seed"] == 4389, "formal test generator seed mismatch")
    require(set(data["summary"]) == {"full", "no_lag", "no_role"}, "variant set mismatch")
    require(len(data["records"]) == 15, "expected 15 variant-seed records")
    pairs = {(record["variant"], record["seed"]) for record in data["records"]}
    require(len(pairs) == 15, "duplicate variant-seed record")
    hashes = {record["run_config"]["implementation_sha256"] for record in data["records"]}
    require(len(hashes) == 1, "mixed implementations in formal result")
    by_pair = {(record["variant"], record["seed"]): record for record in data["records"]}
    paired_gains = {}
    for baseline in ("no_lag", "no_role"):
        gains = []
        for seed in protocol["seeds"]:
            full = by_pair[("full", seed)]["metrics"]["source_hit_at_1"]
            other = by_pair[(baseline, seed)]["metrics"]["source_hit_at_1"]
            gains.append(full - other)
        require(all(gain > 0 for gain in gains), f"full model does not win every paired seed over {baseline}")
        paired_gains[baseline] = gains
    for metric in ("source_hit_at_1", "source_hit_at_3", "source_mrr"):
        full = data["summary"]["full"][metric]["mean"]
        require(full < 1.0, f"full {metric} is saturated")
        for baseline in ("no_lag", "no_role"):
            require(full > data["summary"][baseline][metric]["mean"], f"full does not lead {metric}")
    audit = {
        "status": "passed",
        "result": str(RESULT),
        "formal_variant_seed_records": len(data["records"]),
        "implementation_sha256": next(iter(hashes)),
        "formal_test_data_seed": protocol["test_data_seed"],
        "protocol_development_data_seed": 3891,
        "protocol_and_formal_test_seeds_are_disjoint": protocol["test_data_seed"] != 3891,
        "paired_hit_at_1_gains": paired_gains,
    }
    OUTPUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
