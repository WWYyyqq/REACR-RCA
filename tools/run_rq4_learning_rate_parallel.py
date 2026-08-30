"""Run only non-default learning-rate jobs for the formal RQ4 output."""

import argparse
import json
from pathlib import Path

from baselines.rq1_common import ROOT
from tools.run_rq4_informative_sensitivity import PARAMETERS, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--output-dir", default="res/rq4_informative_sensitivity_20260826"
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = PARAMETERS["train_learning_rate"]
    values = [value for value in spec["values"] if float(value) != float(spec["default"])]
    jobs = [(value, seed) for value in values for seed in args.seeds]
    for index, (value, seed) in enumerate(jobs, start=1):
        record = run_one(output, "train_learning_rate", value, seed, args)
        print(json.dumps({
            "progress": f"{index}/{len(jobs)}", "parameter": "train_learning_rate",
            "value": value, "seed": seed, **record,
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
