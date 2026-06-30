# Final Data Generation Report

Dataset: 1,000 independently sampled parameterized deterioration trajectories. This is not a nested scenario/realization design.

Rows: 783000
Series: 1000
Time steps per series: 783
Onset-observed series by 60 years: 310

Sufficiency classification: `ACCEPTABLE`

Reasons/caveats:
- none

One-time adjustment used: `False`
Adjustment reason: none

Target validation:
- onset_flag 1-to-0 transitions: 0
- reference Pf(t) decreasing steps: 0
- cumulative label equals cumulative max: True
- static parameters constant by series: {'Cs': True, 'D28': True, 'm_aging': True, 'cover_mm': True, 'C_th': True}

Year diagnostics:

| Requested year | Nearest year | Pf | Initiated | Non-initiated | Response variance |
|---:|---:|---:|---:|---:|---:|
| 20 | 20.008214 | 0.048000 | 48 | 952 | 0.045696 |
| 40 | 40.016427 | 0.193000 | 193 | 807 | 0.155751 |
| 60 | 59.947981 | 0.310000 | 310 | 690 | 0.213900 |
