# REACT-RCA V1 Code Manifest

This manifest records the current REACT-RCA V1 code boundary after the naming cleanup.

## Main Entrypoints

- `REACT_RCA_Pipeline_V1.py`: single-run training, validation, testing, prediction export, and event-chain export.
- `REACT_RCA_ExperimentRunner_V1.py`: fair protocol experiment runner for DeepHunt, Baro-CoE-DyCauRCA, and FAMOS-Baro-CoE-DyCauRCA variants.
- `tools/REACT_RCA_RegularizedSearch_V1.py`: regularization and complexity search runner for overfitting analysis.

## Core Components

- `models/REACT_RCA_DSBGDataset_V1.py`: DSBG window construction, case windows, and DGL dataloader logic.
- `models/REACT_RCA_DynamicBackbone_V1.py`: dynamic spatial-temporal encoder.
- `models/REACT_RCA_AnomalyHead_V1.py`: multi-head node-level anomaly evolution representation.
- `models/REACT_RCA_CausalHead_V1.py`: RUN-inspired window-conditioned temporal propagation matrix.
- `models/REACT_RCA_CausalRanker_V1.py`: decomposable causal propagation ranking head.
- `models/REACT_RCA_ModalityGate_V1.py`: FAMOS-inspired raw residual modality reliability gate.
- `models/REACT_RCA_EventChainExplainer_V1.py`: CoE-inspired strict event-chain audit exporter.
- `models/REACT_RCA_Trainer_V1.py`: ranking loss, training loop, validation checkpoint selection, evaluation, and prediction table export.

## Supporting Code Kept From DeepHunt

- `utils/public_functions.py`: data loading helpers.
- `config/D1.yaml`, `config/D2.yaml`: dataset paths and feature dimensions.
- `data/`: D1/D2 local experiment data.

## Backward-Compatible Legacy Files

The following files are retained for old experiment reproducibility, but the V1 entrypoints above should be used for new REACT-RCA runs:

- `main_dycau_rca.py`
- `run_fbc_dycaurca.py`
- `models/dsbg_dataset.py`
- `models/dynamic_backbone.py`
- `models/anomaly_heads.py`
- `models/causal_head.py`
- `models/causal_ranker.py`
- `models/modality_gate.py`
- `models/event_chain_explainer.py`
- `models/dycau_trainer.py`
- `tools/run_react_regularized_search.py`

## Recommended V1 Commands

D1 best-performance run:

```bash
python REACT_RCA_Pipeline_V1.py --dataset D1 --seed 1 --device cuda --batch_size 8 --feedback_sample_ratio 0.3 --case_split_protocol fair --train_epochs 20 --enable_change_point_branch --change_point_weight 1.0 --export_event_chains --output_dir res/react_rca_v1/D1_best
```

D2 regularized best trade-off run:

```bash
python REACT_RCA_Pipeline_V1.py --dataset D2 --seed 3 --device cuda --batch_size 4 --feedback_sample_ratio 0.3 --case_split_protocol fair --train_epochs 20 --train_patience 8 --enable_change_point_branch --change_point_weight 0.05 --causal_hidden_dim 16 --causal_gamma 0.2 --causal_kernel_size 3 --enable_modality_gate --modality_gate_fusion_mode raw_residual --modality_gate_residual_alpha 0.1 --dropout 0.15 --branch_init_scale 0.075 --train_weight_decay 0.00001 --export_event_chains --output_dir res/react_rca_v1/D2_regularized
```
