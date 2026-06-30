# Final Target Validation Report

Lock status: `FINAL_LOCKED_BEFORE_MODEL_TRAINING`
Selected candidate: `C`

Rows: 783000
Series: 1000
`onset_flag` 1-to-0 transitions: 0
Reference cumulative `Pf(t)` decreasing steps: 0
Final cumulative label equals cumulative max of instantaneous exceedance: True

Static parameters constant within each `series_id`:

| Parameter | Constant within series? |
|---|---|
| `Cs` | True |
| `D28` | True |
| `m_aging` | True |
| `cover_mm` | True |
| `C_th` | True |

`C_th` is present in all configured predictive feature lists.
`chloride_rebar` is excluded from every configured predictive feature list.
