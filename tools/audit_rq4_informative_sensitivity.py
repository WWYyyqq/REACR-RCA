"""Audit formal RQ4 sensitivity artifacts without reading any test output."""

import json
from pathlib import Path

from tools.run_rq4_informative_sensitivity import PARAMETERS, required_artifacts, sensitivity_run_dir


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "res" / "rq4_informative_sensitivity_20260826"
SEEDS = (0, 1, 2)


def main():
    unique_paths = {
        sensitivity_run_dir(OUTPUT, parameter, value, seed)
        for parameter, spec in PARAMETERS.items()
        for value in spec["values"]
        for seed in SEEDS
    }
    assert len(unique_paths) == 39, len(unique_paths)
    for path in sorted(unique_paths):
        missing = [str(item) for item in required_artifacts(path) if not item.exists()]
        assert not missing, missing
        manifest = json.loads((path / "runner_command.json").read_text(encoding="utf-8"))
        config = json.loads((path / "run_config.json").read_text(encoding="utf-8"))
        assert manifest["selection_split"] == "validation_only"
        assert manifest["test_outputs_exported"] is False
        assert config["validation_only"] is True
        assert not (path / "final_eval.json").exists()
        assert not (path / "eval_predictions.csv").exists()

    result_path = OUTPUT / "rq4_informative_sensitivity_results.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["protocol"]["selection_split"] == "validation_only"
    assert result["protocol"]["test_artifacts_read"] is False
    assert result["protocol"]["new_test_outputs_exported"] is False
    assert len(result["rows"]) == sum(len(spec["values"]) for spec in PARAMETERS.values()) == 21
    audit = {
        "status": "passed",
        "unique_training_runs": len(unique_paths),
        "reported_parameter_settings": len(result["rows"]),
        "seeds": list(SEEDS),
        "test_outputs_found": False,
        "result": str(result_path),
    }
    target = OUTPUT / "audit.json"
    target.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
