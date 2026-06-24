# REACR-RCA V2

REACR-RCA V2 is a microservice root-cause localization implementation built on
dynamic anomaly evolution, root-symptom disentangled ranking, and
counterfactual propagation evidence correction.

## Core Method

- Dynamic spatial-temporal windows represent anomaly evolution.
- The ranker separates source-like evidence from victim-like evidence instead
  of treating the largest anomaly as the root cause.
- Counterfactual coverage and residual evaluate whether a candidate can
  explain other observed anomalies.

## Run

```bash
python REACT_RCA_Pipeline_V1.py \
  --dataset D1 --device cuda --seed 0 \
  --case_split_protocol time_strict \
  --ranker_mode counterfactual \
  --enable_explanation_losses \
  --root_symptom_loss_weight 0.1 \
  --counterfactual_loss_weight 0.1 \
  --event_chain_loss_weight 0.05 \
  --invariance_loss_weight 0.05 \
  --train_epochs 20
```

The pipeline exports `final_eval.json`, `eval_predictions.csv`, and
Challenge-2 role diagnostics. Event-chain exports are interpretable ranking
evidence, not claims of real propagation-chain localization.

## Version Boundary

`v1.1.0-reacr-stable` remains intact for reproducibility. V2 excludes the
hard-symptom contrastive loss because complete paired experiments did not show
a stable joint improvement in Top-k root-cause localization and role
diagnostics.

See `REACR_RCA_CODE_MANIFEST_V2.md` and `REACR_RCA_V2_RELEASE_NOTES.md` for
the code boundary and release details.
