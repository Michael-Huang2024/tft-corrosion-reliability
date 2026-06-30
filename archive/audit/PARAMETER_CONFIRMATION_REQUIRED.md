# Parameter Confirmation Required

The June 25 manuscript specifies distribution families and bounds, but the repository, README, active scripts, legacy scripts, and available Git history do not document the central/shape parameters needed for the bounded lognormal and normal distributions.

Do not run final full-scale data generation until the following values are confirmed by the advisor.

| Parameter | Distribution | Bounds | Missing values | Units |
|---|---|---|---|---|
| Cs | lognormal | 2 to 6 | mean and standard deviation, with clarification whether lognormal moments are natural-space or log-space | kg/m^3 |
| D28 | lognormal | 1e-12 to 5e-12 | mean and standard deviation, with clarification whether lognormal moments are natural-space or log-space | m^2/s |
| m_aging | normal | 0.2 to 0.6 | mean and standard deviation, with clarification whether lognormal moments are natural-space or log-space | dimensionless |
| cover_mm | uniform | 40 to 110 | none | mm |
| C_th | lognormal | 0.6 to 1.2 | mean and standard deviation, with clarification whether lognormal moments are natural-space or log-space | kg/m^3 |

The revision config in `scripts/revision_config.py` stores these bounds and exposes a configurable interface, but non-uniform distributions remain intentionally incomplete until confirmed.
