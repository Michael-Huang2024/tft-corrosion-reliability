# Phase 1-4 Status

Status: completed and locked. Full model training has not been started.

## Physical Simulator Audit

Physical simulator implementation valid: yes.

No unit, one-time cover conversion, equation-directionality, or boundary-condition error was found. The implementation uses the apparent-diffusivity form `D(t) * t`, not integrated exposure `integral_0^t D(tau) d tau`; this matches the active repository formulation and no conflicting manuscript file was present.

Report:

- `outputs/revision/physical_simulator_audit.md`

## Candidate Screening

Simulator-only screening was completed without inspecting machine-learning performance.

| Candidate | Pf(20) | Pf(40) | Pf(60) | Acceptable? |
|---|---:|---:|---:|---|
| A | 0.003 | 0.041 | 0.093 | No |
| B | 0.019 | 0.125 | 0.211 | No |
| C | 0.048 | 0.193 | 0.310 | Yes |

Selection: Candidate C.

Reason: A and B remained below the prespecified preferred ranges; C is the first candidate satisfying all criteria.

Reports and outputs:

- `outputs/revision/tables/parameter_candidate_screening.csv`
- `outputs/revision/figures/parameter_candidate_pf_curves.png`
- `outputs/revision/figures/parameter_candidate_cover_groups.png`
- `outputs/revision/parameter_candidate_screening_report.md`

## Final Locked Configuration

`scripts/revision_config.py` now records:

- `PARAMETER_LOCK_STATUS = "FINAL_LOCKED_BEFORE_MODEL_TRAINING"`
- `SELECTED_PARAMETER_CANDIDATE = "C"`

Final locked parameters:

- `Cs`: truncated lognormal, mean 4.6, SD 0.60, bounds 2.0-6.0 kg/m^3.
- `D28`: truncated lognormal, mean 4.0e-12, SD 0.45e-12, bounds 1.0e-12-5.0e-12 m^2/s.
- `m_aging`: truncated normal, mean 0.30, SD 0.050, bounds 0.20-0.60.
- `cover_mm`: uniform, bounds 40-110 mm.
- `C_th`: truncated lognormal, mean 0.75, SD 0.090, bounds 0.60-1.20 kg/m^3.

Reports:

- `outputs/revision/final_locked_parameter_configuration.md`
- `outputs/revision/final_locked_data_sufficiency_report.md`

## Final Locked Dataset

Generated revision-only data:

- `data/processed/revision/final_chloride_labeled.parquet`
- `data/processed/revision/final_onset_summary.csv`

Dimensions:

- 1,000 independently sampled parameterized trajectories.
- 783 time steps per trajectory.
- 783,000 rows.

Final sufficiency classification: `ACCEPTABLE`.

Final cumulative initiation probabilities:

| Year | Pf |
|---:|---:|
| 20 | 0.048 |
| 40 | 0.193 |
| 60 | 0.310 |

Cover-depth ordering status: preserved in the candidate screening and final locked data diagnostics.

## Final Validations

Target validation passed:

- `onset_flag` has zero 1-to-0 transitions.
- Reference cumulative `Pf(t)` has zero decreasing steps.
- `onset_flag` equals cumulative max of instantaneous exceedance.
- All five physical inputs remain constant within each `series_id`.
- `C_th` is included in predictive feature lists.
- `chloride_rebar` is excluded from predictive feature lists.

Split validation passed:

- Train: 700 series.
- Validation: 150 series.
- Test: 150 series.
- Overlaps: 0.

Validation reports:

- `outputs/revision/final_target_validation_report.md`
- `outputs/revision/final_split_validation_report.md`

## Decision

Ready for full training: yes, pending user approval.
