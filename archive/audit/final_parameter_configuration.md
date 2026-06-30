# Final Parameter Configuration

Random seed: `20250111`

Sampling method: reproducible rejection sampling for truncated lognormal and truncated normal variables; direct uniform sampling for cover depth. Values are not clipped after generation.

Lognormal conversion equations:

`sigma_log = sqrt(log(1 + (sd_physical / mean_physical)^2))`

`mu_log = log(mean_physical) - 0.5 * sigma_log^2`

One-time adjustment used: `False`

| Parameter | Distribution | Physical mean | Physical SD | Internal mu | Internal sigma | Lower | Upper | Acceptance rate | Units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cs | truncated_lognormal | 4.6 | 0.6 | 1.51762 | 0.129885 | 2 | 6 | 0.244141 | kg/m^3 |
| D28 | truncated_lognormal | 4e-12 | 4.5e-13 | -26.251 | 0.112146 | 1e-12 | 5e-12 | 0.244141 | m^2/s |
| m_aging | truncated_normal | 0.3 | 0.05 | 0.3 | 0.05 | 0.2 | 0.6 | 0.244141 | dimensionless |
| cover_mm | uniform | nan | nan | nan | nan | 40 | 110 | 1.000000 | mm |
| C_th | truncated_lognormal | 0.75 | 0.09 | -0.294831 | 0.119571 | 0.6 | 1.2 | 0.244141 | kg/m^3 |
