# REACT-RCA

REACT-RCA is a microservice root cause localization prototype built on the DeepHunt data pipeline. It jointly models node-level anomaly evolution, window-conditioned temporal propagation, decomposable root-cause ranking, FAMOS-style modality reliability, BARO-style change-point signals, and CoE-style strict event-chain auditing.

## V1 Code Layout

The V1 implementation uses a unified `REACT_RCA_*_V1` naming scheme. See `REACT_RCA_CODE_MANIFEST_V1.md` for the full file map.

Main files:

- `REACT_RCA_Pipeline_V1.py`
- `REACT_RCA_ExperimentRunner_V1.py`
- `tools/REACT_RCA_RegularizedSearch_V1.py`
- `models/REACT_RCA_*_V1.py`

## Environment

The experiments were run in the local `DeepHunt-env` Conda environment with CUDA:

```bash
source /home/wyq111/anaconda3/etc/profile.d/conda.sh
conda activate /home/wyq111/shiyan/DeepHunt/DeepHunt-env
```

## Example Runs

D1 best-performance configuration:

```bash
python REACT_RCA_Pipeline_V1.py --dataset D1 --seed 1 --device cuda --batch_size 8 --feedback_sample_ratio 0.3 --case_split_protocol fair --train_epochs 20 --enable_change_point_branch --change_point_weight 1.0 --export_event_chains --output_dir res/react_rca_v1/D1_best
```

D2 regularized best trade-off configuration:

```bash
python REACT_RCA_Pipeline_V1.py --dataset D2 --seed 3 --device cuda --batch_size 4 --feedback_sample_ratio 0.3 --case_split_protocol fair --train_epochs 20 --train_patience 8 --enable_change_point_branch --change_point_weight 0.05 --causal_hidden_dim 16 --causal_gamma 0.2 --causal_kernel_size 3 --enable_modality_gate --modality_gate_fusion_mode raw_residual --modality_gate_residual_alpha 0.1 --dropout 0.15 --branch_init_scale 0.075 --train_weight_decay 0.00001 --export_event_chains --output_dir res/react_rca_v1/D2_regularized
```

The D2 fair path now applies the regularized REACR-RCA defaults automatically,
so the compact command below is equivalent unless a flag is explicitly
overridden:

```bash
python REACT_RCA_Pipeline_V1.py --dataset D2 --seed 3 --device cuda --batch_size 4 --feedback_sample_ratio 0.3 --case_split_protocol fair --export_event_chains --output_dir res/react_rca_v1/D2_regularized
```

Full fair rerun:

```bash
python REACT_RCA_ExperimentRunner_V1.py --datasets D1 D2 --mode all --device cuda --base_dir res/reacr_rca_full_rerun --overwrite
```

50/20/30 chronological split rerun:

```bash
python REACT_RCA_ExperimentRunner_V1.py --datasets D1 D2 --mode all --device cuda --base_dir res/reacr_rca_split_50_20_30 --train_ratio 0.5 --test_start_ratio 0.7 --overwrite
```

## Versioning and Rollback

This repository uses git tags to distinguish experiment-code versions:

- `v1.0.0-baseline`: initial REACT-RCA V1 release.
- `v1.1.0-reacr-stable`: stable REACR-RCA update with D2 regularized defaults,
  softplus anomaly stabilization, score-variance checkpoint protection, and
  configurable chronological splits.

To inspect what changed between the baseline and the stable REACR-RCA version:

```bash
git diff v1.0.0-baseline..v1.1.0-reacr-stable
git log --oneline v1.0.0-baseline..v1.1.0-reacr-stable
```

To temporarily run the old baseline code:

```bash
git checkout v1.0.0-baseline
```

To return to the latest main branch after inspecting the old version:

```bash
git checkout main
```

To permanently undo the stable update on `main`, create a revert commit:

```bash
git revert v1.0.0-baseline..v1.1.0-reacr-stable
```

## Notes

- `res/` contains local experiment artifacts and is not required for code review.
- Large local environments such as `DeepHunt-env/` should not be committed.
- Fair evaluation uses chronological train/validation/test case splits and
  validation-based checkpoint selection.
