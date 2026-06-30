# Revision Experiment Report

Status: not complete; full-scale reviewer experiments are blocked pending advisor confirmation of parameter distribution shape/central values.

## Exact Code Changes

- Added revision configuration and shared utilities:
  - `scripts/revision_config.py`
  - `scripts/revision_data.py`
  - `scripts/revision_metrics.py`
  - `scripts/scripts_compat.py`
- Added revision experiment scripts:
  - `scripts/07_train_benchmarks.py`
  - `scripts/08_sobol_sensitivity.py`
  - `scripts/09_mc_dropout_uq.py`
  - `scripts/10_bootstrap_reference_uq.py`
  - `scripts/11_runtime_scaling.py`
- Updated active scripts to use cumulative `onset_flag` and include `C_th` where applicable:
  - `scripts/03_train_model.py`
  - `scripts/04_infer.py`
  - `scripts/05_make_figures.py`
  - `scripts/06_baseline_comparison.py`
- Added `SALib` to `requirements.txt`.

## Corrected Target Definition

Canonical revision target: `onset_flag = I(Ti <= t)`.

Target audit path: `outputs/revision/target_definition_audit.md`

Confirmed:

- `onset_flag` 1-to-0 transitions: 0.
- Corrected reference `Pf(t)` decreasing steps: 0.
- Final cumulative label equals cumulative max of threshold exceedance: true.

## Parameter Distributions Actually Used

No final regenerated dataset was produced.

Reason: the manuscript-specified bounded lognormal/normal distributions require means and standard deviations that are not documented in the repository or available Git history.

Advisor confirmation required: `outputs/revision/PARAMETER_CONFIRMATION_REQUIRED.md`

## Data Split Counts

Split file: `data/processed/revision/series_split.csv`

- Train: 700 series.
- Validation: 150 series.
- Test: 150 series.
- Train/validation/test overlaps: 0.

Split audit path: `outputs/revision/split_audit.md`

## Model Architectures

Implemented for revision workflow:

- Logistic Regression: five physical variables plus time.
- MLP: compact feed-forward binary classifier with ReLU/dropout.
- GRU: sequence classifier using the same encoder length and prediction horizon where practical.
- TFT: current architecture retained with corrected target, `C_th` static input, and series-level split.

## Model Comparison Results

Smoke-only path: `outputs/revision/tables/model_comparison.csv`

These values are not manuscript results because smoke training used a small subset and one epoch.

## Sobol Sensitivity Results

Smoke/diagnostic path: `outputs/revision/tables/sobol_indices.csv`

Final Sobol analysis is blocked until parameter distributions are confirmed. The diagnostic run marks inconclusive binary-response cases explicitly.

## MC Dropout Uncertainty Results

Smoke-only paths:

- `outputs/revision/predictions/mc_dropout_population_predictions.csv`
- `outputs/revision/tables/mc_dropout_metrics.csv`
- `outputs/revision/tables/mc_dropout_convergence.csv`
- `outputs/revision/figures/mc_dropout_uncertainty_band.png`
- `outputs/revision/mc_dropout_report.md`

Terminology used: MC Dropout-based approximate Bayesian inference, approximate posterior predictive distribution, and epistemic uncertainty estimate.

## Runtime-Scaling Results

Smoke-only paths:

- `outputs/revision/tables/runtime_scaling.csv`
- `outputs/revision/figures/runtime_scaling.png`
- `outputs/revision/runtime_report.md`

No computational-advantage conclusion is made from smoke results.

## Failed or Inconclusive Experiments

- Full-scale data generation: not run; blocked by missing distribution parameters.
- Full-scale model training: not run.
- Final Sobol sensitivity: not run; blocked by missing distribution parameters.
- Sobol diagnostic at 20 years: marked inconclusive for binary response.
- Deterministic TFT runtime in the point-inference harness: not measured and explicitly marked.

## Exact Figure and Table Paths

Generated revision smoke/audit artifacts include:

- `outputs/revision/tables/model_comparison.csv`
- `outputs/revision/tables/model_comparison_by_seed.csv`
- `outputs/revision/tables/sobol_indices.csv`
- `outputs/revision/tables/mc_dropout_metrics.csv`
- `outputs/revision/tables/mc_dropout_convergence.csv`
- `outputs/revision/tables/runtime_scaling.csv`
- `outputs/revision/figures/model_comparison.png`
- `outputs/revision/figures/sobol_total_effect_20_40_60.png`
- `outputs/revision/figures/sobol_time_evolution.png`
- `outputs/revision/figures/mc_dropout_uncertainty_band.png`
- `outputs/revision/figures/runtime_scaling.png`

## Manuscript-Update Checklist

- Update target definition/code description to cumulative `onset_flag`.
- State `C_th`/Ccrit is included as a model input.
- State train/validation/test are disjoint by `series_id`.
- Report final benchmark means and standard deviations only after full-scale training.
- Report Sobol results only after distribution parameters are confirmed.
- Label MC Dropout as approximate Bayesian uncertainty, not exact Bayesian inference.
- Report runtime measurements without forcing a computational-advantage claim.

## Reviewer-Response Checklist

- Corrected cumulative target: implemented and smoke-tested.
- Ccrit in all predictive models: implemented and audited.
- Independent series-level split: implemented and audited.
- Logistic Regression, MLP, GRU, TFT benchmarks: implemented and smoke-tested.
- Five-parameter Sobol: implemented with confirmation gate; final run blocked.
- MC Dropout UQ: implemented and smoke-tested.
- Runtime scaling: implemented and smoke-tested.

## Unresolved Issue

Advisor confirmation is required before final full-scale experiments:

- `Cs`: bounded lognormal mean/std and moment convention.
- `D28`: bounded lognormal mean/std and moment convention.
- `m_aging`: bounded normal mean/std.
- `C_th`: bounded lognormal mean/std and moment convention.
