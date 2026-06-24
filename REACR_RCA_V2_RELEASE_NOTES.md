# REACR-RCA V2

REACR-RCA V2 extends `v1.1.0-reacr-stable` with three connected mechanisms:

1. Dynamic window-level anomaly evolution modelling.
2. Root-symptom disentangled ranking using source, victim, and incoming
   explainedness evidence.
3. Counterfactual propagation evidence correction using coverage and residual.

The old ranker remains available through `--ranker_mode legacy` for
reproducibility. New experiments should use `--ranker_mode counterfactual`.

V2 does not claim real propagation-chain localization. Event-chain output is
reported as interpretable ranking evidence. The hard-symptom contrastive loss
and its search runner are not included because their complete paired
experiments did not provide a stable overall benefit.
