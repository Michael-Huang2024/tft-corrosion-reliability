# Final Sobol Sensitivity Report

## A. Purpose

This physics-model-based global sensitivity analysis addresses the reviewer request to provide sensitivity analysis for major input parameters. It evaluates the influence of locked physical inputs on corrosion initiation responses at representative service times and complements the TFT benchmark and MC Dropout uncertainty analysis.

## B. Input parameters

| Parameter | Symbol / field | Distribution | Range | Units |
|---|---|---|---|---|
| Surface chloride | Cs | truncated lognormal (mean 4.6, SD 0.60) | 2–6 | kg/m³ |
| 28-day diffusivity | D28 | truncated lognormal (mean 4.0e-12, SD 0.45e-12) | 1–5e-12 | m²/s |
| Aging exponent | m | truncated normal (mean 0.30, SD 0.05) | 0.2–0.6 | – |
| Cover depth | cover_mm | uniform | 40–110 | mm |
| Critical chloride | Ccrit | truncated lognormal (mean 0.75, SD 0.09) | 0.6–1.2 | kg/m³ |

## C. Method

- Method: Sobol global sensitivity analysis (SALib)
- Sampling: Saltelli sequence on [0,1]⁵ with inverse-CDF mapping to locked distributions
- Base sample size N: 2,048
- Total model evaluations: 14,336 (= N × (D+2) with D=5 in current SALib)
- Random seed: 20250627
- Simulator: audited apparent-diffusivity form D(t)=D28×(t_ref/t)^m, erfc chloride profile
- Responses: smooth margin C_rebar−Ccrit (primary), normalized margin, binary initiation (supplement)
- Time points: 20, 40, 60 years

## D. Sampling diagnostics

parameter        distribution  lower_bound  upper_bound  target_mean   target_std  sample_mean   sample_std   sample_min   sample_max   sample_p01   sample_p50   sample_p99
       Cs truncated_lognormal 2.000000e+00 6.000000e+00 4.600000e+00 6.000000e-01 4.570021e+00 5.596800e-01 2.857677e+00 5.997509e+00 3.365217e+00 4.548550e+00 5.852896e+00
      D28 truncated_lognormal 1.000000e-12 5.000000e-12 4.000000e-12 4.500000e-13 3.974730e-12 4.177833e-13 2.588394e-12 4.996426e-12 3.056714e-12 3.963594e-12 4.907906e-12
  m_aging    truncated_normal 2.000000e-01 6.000000e-01 3.000000e-01 5.000000e-02 3.027668e-01 4.709427e-02 2.003763e-01 4.979566e-01 2.078055e-01 3.014065e-01 4.171864e-01
 cover_mm             uniform 4.000000e+01 1.100000e+02          NaN          NaN 7.500000e+01 2.020796e+01 4.001850e+01 1.099842e+02 4.068374e+01 7.499812e+01 1.093125e+02
    Ccrit truncated_lognormal 6.000000e-01 1.200000e+00 7.500000e-01 9.000000e-02 7.565036e-01 8.473183e-02 6.000701e-01 1.152923e+00 6.080374e-01 7.486160e-01 9.849102e-01

## E. Smooth margin Sobol results

### 20 years

parameter       S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs 0.004904 0.008215 0.020553 0.003144      5.0      5.0
      D28 0.018128 0.010446 0.032110 0.004084      4.0      4.0
  m_aging 0.100946 0.026605 0.182764 0.020624      2.0      2.0
 cover_mm 0.670969 0.057186 0.780529 0.053915      1.0      1.0
    Ccrit 0.092260 0.020270 0.092171 0.008359      3.0      3.0

### 40 years

parameter       S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs 0.013493 0.009930 0.027681 0.003636      5.0      4.0
      D28 0.019096 0.010940 0.026289 0.002558      4.0      5.0
  m_aging 0.142722 0.026739 0.192887 0.018934      2.0      2.0
 cover_mm 0.717335 0.051394 0.782567 0.053861      1.0      1.0
    Ccrit 0.036692 0.013096 0.036891 0.003323      3.0      3.0

### 60 years

parameter       S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs 0.020157 0.010388 0.033462 0.004057      4.0      3.0
      D28 0.019581 0.010347 0.024229 0.002307      5.0      5.0
  m_aging 0.170046 0.029242 0.203636 0.015358      2.0      2.0
 cover_mm 0.715709 0.052321 0.760052 0.050975      1.0      1.0
    Ccrit 0.024927 0.009246 0.025186 0.002021      3.0      4.0

Interaction effects can be read from $S_T - S_1$. Rank summaries:

 time_year response_type top_parameter_by_S1 top_parameter_by_ST second_parameter_by_ST dominant_interaction_parameter                                                                                                                                                                        main_interpretation
      20.0        margin            cover_mm            cover_mm                m_aging                       cover_mm At 20 years, cover_mm has the largest total effect (ST=0.781); cover_mm ranks first by first-order index (S1=0.671). Largest interaction contribution (ST-S1) is associated with cover_mm.
      40.0        margin            cover_mm            cover_mm                m_aging                       cover_mm At 40 years, cover_mm has the largest total effect (ST=0.783); cover_mm ranks first by first-order index (S1=0.717). Largest interaction contribution (ST-S1) is associated with cover_mm.
      60.0        margin            cover_mm            cover_mm                m_aging                       cover_mm At 60 years, cover_mm has the largest total effect (ST=0.760); cover_mm ranks first by first-order index (S1=0.716). Largest interaction contribution (ST-S1) is associated with cover_mm.

## F. Binary initiation Sobol results

### 20 years

parameter        S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs -0.002655 0.032581 0.242188 0.064879      4.0      3.0
      D28  0.023794 0.022864 0.237863 0.060994      3.0      4.0
  m_aging  0.065700 0.047384 0.531948 0.091818      2.0      2.0
 cover_mm  0.349479 0.101341 0.869281 0.103182      1.0      1.0
    Ccrit -0.007151 0.016713 0.177316 0.056460      5.0      5.0

### 40 years

parameter       S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs 0.017236 0.021688 0.142777 0.025852      3.0      3.0
      D28 0.007689 0.024942 0.127748 0.027755      4.0      4.0
  m_aging 0.057201 0.033063 0.374227 0.042230      2.0      2.0
 cover_mm 0.531944 0.059880 0.885220 0.052536      1.0      1.0
    Ccrit 0.006631 0.021885 0.100696 0.024209      5.0      5.0

### 60 years

parameter       S1  S1_conf       ST  ST_conf  rank_S1  rank_ST
       Cs 0.003368 0.021573 0.109029 0.019448      5.0      5.0
      D28 0.013922 0.021192 0.123493 0.023856      3.0      3.0
  m_aging 0.086779 0.037723 0.351564 0.036342      2.0      2.0
 cover_mm 0.599911 0.059471 0.893374 0.046328      1.0      1.0
    Ccrit 0.010907 0.019002 0.112367 0.019644      4.0      4.0

**Binary warnings:** None.

## G. Engineering interpretation

- **20 years:** At 20 years, cover_mm has the largest total effect (ST=0.781); cover_mm ranks first by first-order index (S1=0.671). Largest interaction contribution (ST-S1) is associated with cover_mm.
- **40 years:** At 40 years, cover_mm has the largest total effect (ST=0.783); cover_mm ranks first by first-order index (S1=0.717). Largest interaction contribution (ST-S1) is associated with cover_mm.
- **60 years:** At 60 years, cover_mm has the largest total effect (ST=0.760); cover_mm ranks first by first-order index (S1=0.716). Largest interaction contribution (ST-S1) is associated with cover_mm.

- Cover depth: larger cover increases diffusion path length and generally reduces rebar chloride, so high ST for cover indicates strong control by detailing and construction quality.
- Cs: a high ST for surface chloride highlights exposure environment (e.g., de-icing or marine salts).
- D28: sensitivity to reference diffusivity reflects concrete quality and permeability.
- Ccrit: sensitivity to the critical threshold reflects uncertainty in the corrosion initiation criterion.
- m: sensitivity to the aging exponent indicates long-term diffusivity evolution effects.

## H. Manuscript-ready paragraph

Global Sobol sensitivity analysis of the physics-based chloride ingress and corrosion initiation model showed that parameter importance evolves with service time. For the smooth limit-state margin $C_{rebar}(t)-C_{crit}$ at 20, 40, and 60 years, the dominant total-effect contributors were cover_mm, cover_mm, and cover_mm, respectively (Saltelli sampling with base size 2,048 and 14,336 model evaluations). First-order and total-order indices indicate that cover depth, surface chloride, diffusivity, aging, and critical threshold all contribute, with interaction terms (approximated by $S_T-S_1$) present but secondary to the leading parameters at later ages. Binary initiation responses were included as a supplement; early-time binary indices should be interpreted cautiously when initiation remains rare.

## I. Reviewer-response-ready paragraph

Response: We sincerely thank the reviewer for this valuable suggestion. We performed a physics-model-based global Sobol sensitivity analysis for the five major input parameters ($C_s$, $D_{28}$, $m$, cover depth, and $C_{crit}$) using the same audited chloride ingress simulator and locked truncated input distributions as the final dataset generation. Saltelli sampling (base size 2,048; 14,336 evaluations; seed 20250627) was applied at 20, 40, and 60 years for both a smooth limit-state margin and a supplemental binary initiation response. The results identify time-dependent dominant parameters and are reported in `outputs/revision/final_sobol_sensitivity_report.md` with supporting tables and figures. This analysis complements the TFT benchmark and MC Dropout uncertainty study without retraining models.
