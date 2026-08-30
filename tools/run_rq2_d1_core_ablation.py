"""Run the D1-only five-seed RQ2 core-component study."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from tools.reacr_rca_final_study_runner import ROOT, run_pipeline


DEFAULT_VARIANTS = (
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
    parser.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--base-dir", default="res/rq2_d1_core_rerun_20260825")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def runner_args(args):
    return SimpleNamespace(
        device=args.device,
        feedback_sample_ratio=None,
        test_start_ratio=0.5,
        normal_pretrain_epochs=3,
        normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0,
        skip_event_chains=True,
        overwrite=args.overwrite,
    )


def run_one(base_dir, variant, seed, args):
    path = run_pipeline(
        "D1_weighted",
        variant,
        seed,
        args.train_epochs,
        base_dir,
        runner_args(args),
    )
    return variant, seed, path


def main():
    args = parse_args()
    base_dir = (ROOT / args.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(variant, seed) for variant in args.variants for seed in args.seeds]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_one, base_dir, variant, seed, args): (variant, seed)
            for variant, seed in jobs
        }
        for future in as_completed(futures):
            variant, seed = futures[future]
            try:
                _, _, path = future.result()
            except Exception as error:
                raise RuntimeError(f"RQ2 failed for {variant}/seed{seed}") from error
            print(f"completed {variant} seed{seed}: {path}", flush=True)
    print(base_dir)


if __name__ == "__main__":
    main()
