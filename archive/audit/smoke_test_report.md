# Smoke Test Report

Status: passed for code-path smoke testing; full-scale experiments were not started.

## Scope

Smoke tests used a small revision subset and one training epoch where applicable. These results are for interface validation only and must not be used as manuscript performance results.

## Checks

| Check | Status | Evidence |
|---|---|---|
| `onset_flag` is monotonic | Passed | `outputs/revision/target_definition_audit.md`: 0 one-to-zero transitions |
| Corrected reference `Pf(t)` is monotonic | Passed | `outputs/revision/target_definition_audit.md`: 0 decreasing steps |
| Final cumulative label equals cumulative max of threshold exceedance | Passed | `outputs/revision/target_definition_audit.md`: true |
| `C_th`/Ccrit present in all model inputs | Passed | `outputs/revision/feature_audit.md` |
| Forbidden feature leakage blocked | Passed | `scripts/revision_data.py` asserts against `chloride_rebar`, target fields, onset-derived fields, and Pf outputs |
| Series splits are disjoint | Passed | `outputs/revision/split_audit.md`: train/validation/test overlaps all 0 |
| All four benchmark models run | Passed | `outputs/revision/tables/model_comparison.csv` includes Logistic Regression, MLP, GRU, and TFT |
| Evaluation time ranges match | Passed | Shared evaluator restricts all benchmark models to the common post-encoder range |
| MC Dropout produces nonzero predictive variability | Passed | `outputs/revision/tables/mc_dropout_metrics.csv`: mean predictive std > 0 |
| Sobol script runs on small N | Passed with caveat | `outputs/revision/tables/sobol_indices.csv`; 20-year binary response was marked inconclusive rather than forced |
| Runtime benchmark produces measured values | Passed | `outputs/revision/tables/runtime_scaling.csv`; missing/incompatible TFT point-harness timing is explicitly marked |

## Smoke Benchmark Summary

Path: `outputs/revision/tables/model_comparison.csv`

The smoke table contains four models and confirms that all use the corrected cumulative target and revision split. Values are not manuscript results because training used a tiny subset and one epoch.

## MC Dropout Summary

Path: `outputs/revision/tables/mc_dropout_metrics.csv`

The smoke run used a small held-out subset and reduced stochastic passes. It verified dropout reactivation and nonzero predictive variability.

## Sobol Summary

Path: `outputs/revision/tables/sobol_indices.csv`

The Sobol script runs in diagnostic bounds-only mode. The 20-year response produced an inconclusive SALib result for the binary event response; this is retained in the table and report rather than hidden.

## Runtime Summary

Path: `outputs/revision/tables/runtime_scaling.csv`

The smoke runtime harness records measured simulator, MLP, and GRU timings for 20 scenarios. Deterministic TFT timing is not measured in the point-harness path and is explicitly marked as not measured.

## Stop Condition

Full-scale data generation and final reviewer experiments must not begin until the advisor confirms the missing distribution parameters listed in `outputs/revision/PARAMETER_CONFIRMATION_REQUIRED.md`.

Missing values:

- `Cs`: lognormal mean and standard deviation; clarify natural-space vs log-space moments.
- `D28`: lognormal mean and standard deviation; clarify natural-space vs log-space moments.
- `m_aging`: normal mean and standard deviation.
- `C_th`: lognormal mean and standard deviation; clarify natural-space vs log-space moments.

## Status

Smoke test status: passed with documented caveats.

Training status: smoke-only. Full-scale training intentionally not started.

Unresolved issue requiring advisor confirmation: non-uniform parameter distribution shape/central parameters.
