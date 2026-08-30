"""Run the missing five-seed no-lag ablation required by RQ3."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from tools.reacr_rca_final_study_runner import (
    ROOT,
    resolved_train_epochs,
    run_pipeline,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['SN', 'TT', 'D1_weighted'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--base-dir', default='res/rq3_lag_ablation_20260825')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    return parser.parse_args()


def main():
    cli = parse_args()
    base_dir = (ROOT / cli.base_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    runner_args = SimpleNamespace(
        device=cli.device,
        feedback_sample_ratio=None,
        test_start_ratio=0.5,
        normal_pretrain_epochs=3,
        normal_auxiliary_loss_weight=0.1,
        normal_topology_change_oversample_factor=4.0,
        skip_event_chains=True,
        overwrite=cli.overwrite,
    )
    tasks = [
        (dataset, seed, resolved_train_epochs(dataset, None))
        for dataset in cli.datasets
        for seed in cli.seeds
    ]
    with ThreadPoolExecutor(max_workers=max(1, int(cli.workers))) as executor:
        futures = {
            executor.submit(
                run_pipeline,
                dataset,
                'no_lag_relation',
                seed,
                epochs,
                base_dir,
                runner_args,
            ): (dataset, seed)
            for dataset, seed, epochs in tasks
        }
        for future in as_completed(futures):
            dataset, seed = futures[future]
            future.result()
            print(f'completed {dataset} seed{seed}', flush=True)
    print(base_dir)


if __name__ == '__main__':
    main()
