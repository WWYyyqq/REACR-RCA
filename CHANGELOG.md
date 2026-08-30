# Changelog

## v3.1.0-reacr-rca (2026-08-30)

### Added

- Added `D1_weighted`, reconstructed from the original AIOps Jaeger spans as
  minute-level instance call graphs with observed call mass, frequency,
  mean/P95 latency, error rate, structural edges, and self-loops.
- Added the D1 weighted-graph converter, audit documentation, configuration,
  and regression tests without overwriting the historical unweighted D1 data.
- Added reproducible RQ1--RQ6 experiment entry points for baseline comparison,
  D1 core ablation, fault-type analysis, parameter sensitivity, dynamic-change
  stratification, and controlled propagation evaluation.
- Added source-response and zero-lag controls for separately evaluating lagged
  relation evidence and role-aware ranking.
- Added public baseline adapters and result-audit/report builders used by the
  locked experiment protocol.

### Changed

- Extended implementation fingerprints to cover the D1 weighted converter.
- Updated topology audits to distinguish edge-set, weight, and edge-attribute
  changes and to report the observed D1 weighted graph statistics.
- Updated the release documentation to use SN, TT, and D1_weighted as the
  current dynamic datasets; D2 remains available only as a historical fixed-
  topology compatibility dataset.
- Expanded the regression suite from 61 tests in V3.0 to 79 tests in V3.1.

### Compatibility

- V3.0 and all earlier branches/tags remain unchanged.
- Data, checkpoints, and large experiment outputs are not included in the
  release.

## v3.0.0-reacr-rca (2026-07-27)

### Added

- Added SN/TT telemetry conversion from Jaeger `CHILD_OF` spans into session-aware, time-bucketed graph sequences with real empty graphs, dynamic edges, call counts, and five edge attributes.
- Added edge-mass and edge-context encoding for observed weighted/attributed graph snapshots.
- Added an identity-free statistical calibrator (ExtraTrees or linear SVC) and case-wise neural/statistical score fusion for cross-service zero-shot evaluation.
- Added SN/TT configurations, conversion/audit tools, and regression coverage; the V3 release passes 61/61 unit tests.


### Corrected

- Replaced unsafe unbounded prefix matching with exact or
  `label + "-"` matching, explicit zero-match handling, and multi-positive
  labels for service-level roots.
- Aligned both REACR-RCA and DeepHunt training with any-hit evaluation: a
  service-level replica set is a positive candidate bag and its maximum score,
  rather than every replica independently, is compared with negatives. Run
  manifests/configs record generic `positive_candidate_bag_any_hit` semantics
  plus D1/D2-specific mapping granularity.
- Replaced DeepHunt's unstable per-sample min-max ranking normalization with a
  bounded candidate-bag pairwise-margin loss. Its default margin is
  `1/rank_range`; negative rank weights are normalized by their sum. Constant
  and near-tied score gradients are finite with maximum absolute value at most
  one.
- Corrected D1 modality indexing so its two non-contiguous metric spans do not
  absorb trace or log features and raw-residual fusion preserves feature order.
- Added future-masked temporal attention and aligned observation-space
  next-step prediction.
- Added normal-window reconstruction/forecast pretraining and an auxiliary
  normal-dynamics objective during supervised localization.
- Added topology auditing and node-level degree/edge-change features computed
  only from serialized DGL snapshots.
- Added `graph_sequence_mode=freeze_first`, which freezes every graph consumer
  to the first observed graph in each window. This is the fixed-topology
  ablation; `topology_feature_mode=static_only` only disables explicit change
  channels and does not freeze spatial message passing or graph constraints.
- Added changed/stable topology-stratified A@K and sample counts, plus a
  training-only factor-4 weighted sampler for sparse changed normal windows.
- Reworked the relation output as pair- and case-conditioned lagged directed
  relation evidence. A 5D `[B,L,W,target,source]` alignment mask now filters
  each `(t-l,t)` product before affinity aggregation; the 4D
  `lag_topology_mask` remains the time-union support, with an optional null
  source.
- Enforced a positive odd integer moving-average kernel; non-positive, even,
  and non-integer values fail before temporal decomposition.
- Changed graph-prior ranking from a final-snapshot calculation to per-step
  graph/anomaly aggregation over the complete window.
- Replaced coupled root/symptom scores with independent source/response role
  evidence; mined high-response nodes remain pseudo candidates, not symptom
  labels.
- Removed the absolute BCE target for mined hard pseudo responses, retained
  only relative role constraints, and excluded zero-padded invalid lag-prefix
  positions from source-hypothesis coverage/residual calculations.
- Reframed the former counterfactual branch as source-hypothesis coverage and
  symmetric residual evidence.
- Corrected the weighted graph prior to divide by effective weight mass and
  prevented zero-strength event-chain edges from passing a zero threshold.
- Added pair-specific argmax/expected lag and confidence for each exported
  event-chain edge, together with onset-lag agreement and absolute error.
- Renamed relation-only event-chain edges to `model_relation_only` and made
  `source_role_score`, `response_role_score`, and `source_hypothesis_*` the
  primary CSV fields. `causal_inferred` and old role/counterfactual fields are
  deprecated compatibility aliases.
- Renamed primary diagnostic fields to `candidate_membership_auc`,
  `source_role_gap`, and `response_role_gap`; final-study reports prefer them.
  `role_auc` and `root_role_gap` remain deprecated compatibility aliases.
- Added candidate-oriented primary fields across artifacts, event chains,
  validation, and diagnostics (`is_label_candidate`, `candidate_node`,
  `label_candidate_nodes`, `pred_candidate`, `label_candidates`,
  `label_candidate_node`, and `response_candidate_node`). Root/true-root and
  hard-symptom names remain deprecated aliases.
- Renamed primary diagnostic outcomes to `response_candidate_error_rate` and
  `label_vs_response_candidate_margin_mean`, and renamed the runner section to
  “Source-Response Candidate Diagnostic”.
- Added changed/stable sample counts and A@K to per-run CSV, multi-seed
  summaries, and the final Markdown report. Pipeline fixed/change topology
  interpretation fields are now computed from the actual audit.
- Added an implementation identity containing `VERSION` plus a SHA-256
  fingerprint over the pipeline, REACR-RCA/baseline code, and dataset configs;
  stale results are not silently reused when either identity component changes.

### Dataset Audit

- D1: 7,200 snapshots, 46 nodes, 8 unique directed edge sets, 15 adjacent
  topology switches, 200--268 edges per snapshot, and no edge attributes or
  edge weights.
- D2: 14,400 snapshots, 18 nodes, one fixed 66-edge set, zero topology
  switches, and no edge attributes or edge weights.
- D1 has changed topology in 6/210 case windows and 27/3,705 overlapping normal
  windows; those 27 windows arise from only three underlying training-sequence
  switches. D2 has no changed windows.
- D1 label mapping has 136/210 single-node cases and 74/210 three-replica
  candidate bags. A bag means any member may satisfy localization; it does not
  establish three independently verified root-cause instances.
- Run metadata records D1 as `mixed_service_to_instance_candidate_bag` and D2
  as `exact_single_instance_candidate`.
- D2 must not be cited as evidence for topology-evolution modelling. D1 is the
  primary dataset, but its sparse changes require both changed/stable reporting
  and the true `frozen_window_topology` control.
- A real D1 window with an edge-count sequence of 268--256--268 produced 56
  non-zero dynamic features and a dynamic-projection gradient norm of
  0.0004352666. This is an implementation-path check, not evidence of an A@K
  improvement.

### Evaluation Status

- No formal performance numbers are released for this development version.
- All 61 unit tests pass, including bounded-gradient, non-integer/even kernel,
  external `base_dir`, and topology-stratified report regressions. Small
  real-data D1/D2 runs completed training,
  evaluation, explanation, and artifact export with exit code 0. D1 exported a
  `[8,3,10,46,46]` alignment mask and a dynamic-projection gradient norm of
  0.0001990379096; D2 exported `[4,3,10,18,18]` and zero dynamic gradient, as
  expected for fixed topology. A real D2 E2E also verified the new candidate
  fields. These are implementation checks, not performance results.
- All main and ablation experiments must be rerun because label, modality,
  temporal, relation, ranking, topology, and training semantics changed.
- The formal runner defaults both the main model and DeepHunt baseline to five
  seeds (0--4), while ablations default to three (0--2). A full five-seed
  ablation study requires `--ablation_seeds 0 1 2 3 4`; those results have not
  yet been rerun.
- Required D1 topology comparisons include `no_topology_change_features`,
  `topology_dynamic_features_only`, `frozen_window_topology`,
  `no_explicit_topology_encoder`, `no_relation_topology_constraint`, and
  `no_topology_change_oversampling`, with identical splits, seeds, and budgets.
- Report changed/stable A@K with sample counts. The formal runner's 0.4 feedback
  protocol yields 3 changed of 84 training cases, 0 of 21 validation cases, and
  3 of 105 test cases; the zero-change validation stratum cannot validate
  dynamic-window generalization.
- D1 results must be described as mixed-granularity, multi-positive Top-k
  localization where applicable, not uniformly as unique instance-level
  ground truth.

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
