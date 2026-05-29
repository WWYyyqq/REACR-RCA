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

## Notes

- `res/` contains local experiment artifacts and is not required for code review.
- Large local environments such as `DeepHunt-env/` should not be committed.
- Fair evaluation uses train/validation/test case splits and validation-based checkpoint selection.
