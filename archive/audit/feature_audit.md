# Feature Audit

All predictive revision models include the five physical inputs required by the manuscript. `C_th` is the code column corresponding to `Ccrit`.

Forbidden leakage predictors are rejected by `scripts/revision_data.py::assert_no_forbidden_predictors` and include `chloride_rebar`, target columns, onset-derived fields, and Pf outputs.

| Model/component | Input features |
|---|---|
| Logistic Regression | `Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year` |
| MLP | `Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year` |
| GRU | `Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year` |
| TFT static_reals | `Cs, D28, m_aging, cover_mm, C_th` |
| TFT known time-varying reals | `time_idx, t_year` |
| TFT unknown time-varying reals | `(none)` |
