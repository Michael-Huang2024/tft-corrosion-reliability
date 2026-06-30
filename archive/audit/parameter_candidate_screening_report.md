# Parameter Candidate Screening Report

Physical simulator audit conclusion: implementation valid for the active apparent-diffusivity formulation; no correction was made.

Selected candidate: `C`

Objective selection rule: A is selected if acceptable; otherwise B; otherwise C only if A and B remain too low. Selection is simulator-only and independent of any machine-learning performance.

Candidate summary:

candidate  acceptable                                                                                               reasons  Pf10  Pf20  Pf30  Pf40  Pf50  Pf60  initiated20  initiated40  initiated60  variance20  variance40  variance60  never_initiated_pct  t_init_min  t_init_median  t_init_p95  cover_ordering_violations  max_cover_crossing_magnitude  Cs_acceptance_rate  D28_acceptance_rate  m_aging_acceptance_rate  cover_mm_acceptance_rate  C_th_acceptance_rate  Cs_near_lower_2pct  Cs_near_upper_2pct  D28_near_lower_2pct  D28_near_upper_2pct  m_aging_near_lower_2pct  m_aging_near_upper_2pct  cover_mm_near_lower_2pct  cover_mm_near_upper_2pct  C_th_near_lower_2pct  C_th_near_upper_2pct
        A       False Pf(20)=0.003 outside [0.02, 0.2]; Pf(40)=0.041 outside [0.15, 0.55]; Pf(60)=0.093 outside [0.3, 0.75] 0.000 0.003 0.015 0.041 0.068 0.093            3           41           93    0.002991    0.039319    0.084351                 90.7   13.568789      42.469541   57.387543                          0                           0.0            0.244141             0.244141                 0.244141                       1.0              0.244141                 0.0               0.002                  0.0                0.002                    0.003                      0.0                      0.02                      0.02                 0.006                 0.001
        B       False Pf(20)=0.019 outside [0.02, 0.2]; Pf(40)=0.125 outside [0.15, 0.55]; Pf(60)=0.211 outside [0.3, 0.75] 0.000 0.019 0.068 0.125 0.174 0.211           19          125          211    0.018639    0.109375    0.166479                 78.9   11.575633      37.103354   55.655031                          0                           0.0            0.244141             0.244141                 0.244141                       1.0              0.244141                 0.0               0.005                  0.0                0.005                    0.004                      0.0                      0.02                      0.02                 0.007                 0.001
        C        True                                                                                                  none 0.002 0.048 0.121 0.193 0.256 0.310           48          193          310    0.045696    0.155751    0.213900                 69.0    8.969199      35.416838   57.307050                          0                           0.0            0.244141             0.244141                 0.244141                       1.0              0.244141                 0.0               0.002                  0.0                0.005                    0.010                      0.0                      0.02                      0.02                 0.011                 0.000

Selection reason:
- Candidate C is the first candidate satisfying the prespecified criteria.
- Pf(20/40/60) = 0.048 / 0.193 / 0.310.
- Suitable for benchmark model training: yes.
- Suitable for Sobol analysis: yes for years with nonzero response variance; sparse years must still be flagged if encountered.
- Suitable for MC Dropout UQ: yes, because both initiated and non-initiated test cases should remain available.
