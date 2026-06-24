# REACR-RCA V2 Code Manifest

## Main Entry Point

- `reacr_rca_pipeline.py`: trains, validates, tests, exports predictions,
  and writes Challenge-2 role diagnostics.

## Core V2 Method

- `models/reacr_rca_dsbg_dataset.py`: dynamic case-window construction.
- `models/reacr_rca_dynamic_backbone.py`: dynamic spatial-temporal encoder.
- `models/reacr_rca_anomaly_head.py`: node-level anomaly evolution scores.
- `models/reacr_rca_causal_head.py`: lag-aware causal propagation matrix.
- `models/reacr_rca_causal_ranker.py`: legacy, root-symptom, and
  counterfactual ranking modes.
- `models/reacr_rca_trainer.py`: ranking, root-symptom, counterfactual,
  event-chain-consistency, and propagation-invariance objectives.
- `models/reacr_rca_modality_gate.py`: optional modality reliability gate.
- `models/reacr_rca_event_chain_explainer.py`: post-hoc event-chain export.

## V2 Scope

The V2 contribution is dynamic anomaly evolution modelling, root-symptom
disentangled ranking, and counterfactual propagation evidence correction.
`hard_symptom_contrastive_loss` is intentionally excluded: complete paired
experiments did not show a stable joint gain in Top-k localization and role
diagnostics.

## Experiment Support

- `tools/reacr_rca_final_study_runner.py`: multi-seed main-method and ablation
  runner without hard-symptom-loss variants.
- `config/D1.yaml`, `config/D2.yaml`: public code-side dataset schemas.
- `utils/public_functions.py`: sample and graph initialization loaders.
