# REACR-RCA V2 Code Manifest

## Main Entry Point

- `REACT_RCA_Pipeline_V1.py`: trains, validates, tests, exports predictions,
  and writes Challenge-2 role diagnostics.

## Core V2 Method

- `models/REACT_RCA_DSBGDataset_V1.py`: dynamic case-window construction.
- `models/REACT_RCA_DynamicBackbone_V1.py`: dynamic spatial-temporal encoder.
- `models/REACT_RCA_AnomalyHead_V1.py`: node-level anomaly evolution scores.
- `models/REACT_RCA_CausalHead_V1.py`: lag-aware causal propagation matrix.
- `models/REACT_RCA_CausalRanker_V1.py`: legacy, root-symptom, and
  counterfactual ranking modes.
- `models/REACT_RCA_Trainer_V1.py`: ranking, root-symptom, counterfactual,
  event-chain-consistency, and propagation-invariance objectives.
- `models/REACT_RCA_ModalityGate_V1.py`: optional modality reliability gate.
- `models/REACT_RCA_EventChainExplainer_V1.py`: post-hoc event-chain export.

## V2 Scope

The V2 contribution is dynamic anomaly evolution modelling, root-symptom
disentangled ranking, and counterfactual propagation evidence correction.
`hard_symptom_contrastive_loss` is intentionally excluded: complete paired
experiments did not show a stable joint gain in Top-k localization and role
diagnostics.

## Experiment Support

- `tools/REACR_RCA_FinalStudyRunner.py`: multi-seed main-method and ablation
  runner without hard-symptom-loss variants.
- `config/D1.yaml`, `config/D2.yaml`: public code-side dataset schemas.
- `utils/public_functions.py`: sample and graph initialization loaders.
