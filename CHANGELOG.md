# Changelog

## v2.1.0-reacr-rca

### Changed

- Renamed V2 entrypoints and core modules to the stable `reacr_rca_*` naming
  convention without changing model behaviour.
- Updated imports, final-study subprocess entrypoints, and Chinese README
  examples to use the new names.

## v2.0.0-reacr-rca

### Changed

- Added root-symptom disentangled ranking with source, victim, and incoming
  explainedness evidence.
- Added counterfactual propagation coverage and residual correction using the
  lag-aware causal matrix.
- Added role diagnostics for root-versus-high-anomaly candidate ranking.
- Added time-strict case splitting and case-weighted Top-k aggregation.

### Removed

- Removed the hard-symptom contrastive loss and its hyperparameter search.
  Complete paired experiments did not demonstrate a stable overall benefit.

## v1.1.0-reacr-stable

This release promotes the optimized FAMOS-Baro-CoE-DyCauRCA path as the
stable REACR-RCA experiment path.

### Changed

- Added dataset-aware D2 fair defaults in `REACT_RCA_Pipeline_V1.py`:
  - `train_epochs=20`
  - `train_patience=8`
  - `change_point_weight=0.05`
  - `causal_hidden_dim=16`
  - `causal_gamma=0.2`
  - `causal_kernel_size=3`
  - `dropout=0.15`
  - `branch_init_scale=0.075`
  - `train_weight_decay=1e-5`
  - raw-residual modality gate with `modality_gate_residual_alpha=0.1`
- Added explicit `--disable_modality_gate` and `--disable_change_point_branch`
  flags so dataset defaults can be overridden safely.
- Added configurable chronological split controls:
  - `--train_ratio`
  - `--test_start_ratio`
- Updated the D2 fair runner path to use the regularized recipe by default.
- Added score-variance tracking (`score_std`) to validation and test
  evaluation.
- Updated checkpoint selection from `A@1,A@3,A@5,-loss` to
  `A@1,A@3,A@5,score_std,-loss`.

### Fixed

- Replaced the anomaly head's hard `ReLU` output activation with `softplus` to
  avoid all-zero anomaly scores under unlucky D2 initializations.
- Prevented D2 fair reruns from selecting constant-score checkpoints when
  validation `A@k` values tie.
- Restored the DeepHunt compatibility modules required by the runner and event
  chain exporter (`evaluation.py`, `rc_scorer.py`, `train.py`,
  `dycau_trainer.py`, `layers.py`, and `noise_engineering.py`).

### Validation Summary

Local reruns showed that the D2 constant-score collapse was removed:

- D2 BARO-CoE-DyCauRCA collapsed runs: `2/5 -> 0/5`
- D2 REACR-RCA collapsed runs: `16/30 -> 0/30`
- D2 REACR-RCA validation-selected 30/20/50 test mean:
  `A@1/A@3/A@5 = 0.6294 / 0.8451 / 0.9049`
- D2 REACR-RCA validation-selected 50/20/30 test mean:
  `A@1/A@3/A@5 = 0.7800 / 0.8700 / 0.9450`

## v1.0.0-baseline

Initial REACT-RCA V1 release.
