"""Serially finish the remaining RQ experiments after an active RQ2 queue.

This process is intentionally conservative: it waits for the existing RQ2
runner, verifies all artifacts, then runs RQ4 and RQ6 without overlapping the
heavy stages.  Every underlying runner is resumable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS_PATH = ROOT / "res" / "rq_completion_20260825" / "status.json"
RQ2_LOCK_PATH = ROOT / "res" / "reacr_rq2_calibration_lock_20260825" / "D1_weighted" / "locked_test.json"
RQ2_ROOT = ROOT / "res" / "rq2_d1_core_rerun_20260825" / "D1_weighted"
RQ2_VARIANTS = (
    "source_hypothesis_losses",
    "frozen_window_topology",
    "no_topology_change_features",
    "no_lag_relation",
    "no_root_symptom_loss",
    "no_source_hypothesis_loss",
    "no_normal_dynamics_training",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def process_exists(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def write_status(stage, state, **extra):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "state": state,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        **extra,
    }
    STATUS_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run(command, stage):
    write_status(stage, "running", command=command)
    subprocess.run(command, cwd=str(ROOT), check=True)
    write_status(stage, "completed")


def verify_existing():
    required = {
        "RQ1": ROOT / "res" / "rq1_final_20260824" / "final_results.json",
        "RQ3": ROOT / "res" / "reacr_rq3_rq4_generalization_20260825" / "generalization_results.json",
        "RQ5": ROOT / "res" / "rq5_topology_intensity_20260825" / "rq5_topology_intensity_results.json",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing completed RQ artifacts: " + "; ".join(missing))
    return {name: str(path) for name, path in required.items()}


def verify_rq2():
    missing = []
    for variant in RQ2_VARIANTS:
        for seed in range(5):
            path = RQ2_ROOT / variant / f"seed{seed}" / "final_eval.json"
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"RQ2 queue ended with {len(missing)} missing runs: " + "; ".join(missing[:5]))
    return 35


def main():
    args = parse_args()
    existing = verify_existing()
    write_status("existing_results", "completed", artifacts=existing)

    if process_exists(args.wait_pid):
        write_status("RQ2_training", "waiting", pid=args.wait_pid)
    while process_exists(args.wait_pid):
        time.sleep(max(5, min(args.poll_seconds, 60)))
    run_count = verify_rq2()
    write_status("RQ2_training", "completed", completed_runs=run_count)
    run(
        [
            sys.executable, "-u", "-m", "tools.build_rq2_d1_core_report",
            "--lock-path", str(RQ2_LOCK_PATH),
        ],
        "RQ2_report",
    )

    run(
        [
            sys.executable, "-u", "-m", "tools.run_rq4_parameter_sensitivity",
            "--device", args.device, "--workers", str(max(1, min(args.workers, 2))),
        ],
        "RQ4_parameter_sensitivity",
    )
    run(
        [
            sys.executable, "-u", "-m", "tools.run_rq6_controlled_propagation",
            "--device", args.device,
        ],
        "RQ6_controlled_propagation",
    )
    write_status(
        "RQ1-RQ6",
        "completed",
        outputs={
            "RQ1": str(ROOT / "res" / "rq1_final_20260824" / "final_results.json"),
            "RQ2": str(ROOT / "res" / "rq2_d1_core_final_20260825" / "rq2_d1_core_results.json"),
            "RQ3": str(ROOT / "res" / "reacr_rq3_rq4_generalization_20260825" / "generalization_results.json"),
            "RQ4": str(ROOT / "res" / "rq4_parameter_sensitivity_20260825" / "rq4_parameter_sensitivity_results.json"),
            "RQ5": str(ROOT / "res" / "rq5_topology_intensity_20260825" / "rq5_topology_intensity_results.json"),
            "RQ6": str(ROOT / "res" / "rq6_controlled_propagation_20260825" / "rq6_controlled_results.json"),
        },
    )


if __name__ == "__main__":
    main()
