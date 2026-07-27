# SN/TT optimization V2

All hyperparameters were selected with `--validation_only`; the search runs contain no test metrics or test predictions. After configuration lock, seeds 0/1/2 were evaluated once on the test split.

## Locked results

| Dataset | Variant | A@1 | A@3 | A@5 |
|---|---|---:|---:|---:|
| SN | Neural dynamic model | 46.30 +/- 6.42 | 70.37 +/- 13.98 | 79.63 +/- 12.83 |
| SN | Neural + statistical evidence | 62.96 +/- 3.21 | 77.78 +/- 9.62 | 90.74 +/- 8.49 |
| SN | Paired gain (percentage points) | 16.67 | 7.41 | 11.11 |
| TT | Neural dynamic model | 49.63 +/- 5.59 | 68.15 +/- 7.80 | 76.30 +/- 3.39 |
| TT | Neural + statistical evidence | 66.67 +/- 0.00 | 90.37 +/- 2.57 | 94.07 +/- 1.28 |
| TT | Paired gain (percentage points) | 17.04 | 22.22 | 17.78 |

## Validation-only candidate selection

| Candidate | Validation A@1 | A@3 | A@5 | Audit |
|---|---:|---:|---:|---|
| SN_extra_trees | 88.89 | 100.00 | 100.00 | PASS |
| TT_linear_svc | 72.22 | 83.33 | 83.33 | PASS |
| TT_extra_trees | 77.78 | 83.33 | 83.33 | PASS |

## Dynamic graph ablation (three-seed means)

| Dataset | Graph mode | Split/variant | A@1 | A@3 | A@5 |
|---|---|---|---:|---:|---:|
| SN | observed | Validation fused | 70.37 | 96.30 | 100.00 |
| SN | observed | Test fused | 62.96 | 77.78 | 90.74 |
| SN | observed | Test neural | 46.30 | 70.37 | 79.63 |
| SN | freeze_first | Validation fused | 62.96 | 88.89 | 96.30 |
| SN | freeze_first | Test fused | 62.96 | 75.93 | 90.74 |
| SN | freeze_first | Test neural | 42.59 | 72.22 | 77.78 |
| TT | observed | Validation fused | 75.93 | 79.63 | 83.33 |
| TT | observed | Test fused | 66.67 | 90.37 | 94.07 |
| TT | observed | Test neural | 49.63 | 68.15 | 76.30 |
| TT | freeze_first | Validation fused | 70.37 | 77.78 | 83.33 |
| TT | freeze_first | Test fused | 67.41 | 90.37 | 94.81 |
| TT | freeze_first | Test neural | 46.67 | 70.37 | 77.04 |

## Topology evidence

| Dataset | Split | Unique edge sets | Topology switches | Fixed topology |
|---|---|---:|---:|---|
| SN | train | 9 | 58 | False |
| SN | test | 64 | 290 | False |
| TT | train | 6 | 10 | False |
| TT | test | 40 | 280 | False |

## Interpretation

- The session-group split creates zero-shot service localization: root-service identities do not overlap across train, validation, and test sessions.
- The statistical branch uses shared node-wise temporal statistics (mean, standard deviation, maximum, minimum, range, and end-to-start delta) and never uses node identity.
- Observed dynamic graphs are preferred by validation metrics, but their test-set contribution is small and mixed. The large final improvement is primarily attributable to the identity-independent statistical evidence fused with the dynamic neural model.
- Therefore, results should not be described as proving that dynamic topology alone causes the full performance gain.

Machine-readable artifacts are in `res/optimization_v2/summary/`.
