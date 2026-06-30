# Physical Simulator Audit

Conclusion: implementation is internally consistent with the active repository/manuscript-style Fickian formulation. No unit, one-time cover conversion, or directionality error was found. No simulator correction was made before candidate screening.

Important formulation note: the implementation uses the apparent-diffusivity form `D(t) * t` in `x / (2 sqrt(D(t) t))`, not the integrated exposure `integral_0^t D(tau) d tau`. This matches the active code and legacy code comments. No June 25 manuscript file with a conflicting integrated-exposure equation was present in the repository.

Checks:

- `D28` is interpreted in `m^2/s`.
- `cover_mm` is converted to meters exactly once as `cover_mm / 1000`.
- Service time uses seconds internally.
- The 28-day reference is converted to seconds as `28 * 24 * 3600`.
- Time-dependent diffusivity is `D28 * (t_ref / t)^m` for `t >= t_ref`, otherwise `D28`.
- Boundary condition is `C(x,t) = Cs * erfc(x / (2 sqrt(D(t)t)))` with initial/bulk chloride `Cb = 0`.

Directionality check at 60 years:

parameter_increased  baseline_chloride_60  changed_chloride_60  chloride_change                                           expected_direction
                 Cs              0.030521             0.032047         0.001526                       increase chloride / earlier initiation
                D28              0.030521             0.043855         0.013335                       increase chloride / earlier initiation
           cover_mm              0.030521             0.013350        -0.017171                         decrease chloride / delay initiation
            m_aging              0.030521             0.009207        -0.021314                         decrease chloride / delay initiation
               C_th              0.030521             0.030521         0.000000 higher threshold delays initiation without changing chloride

Corner cases:

                case  requested_year  nearest_t_year  chloride_rebar  C_th  instantaneous_exceedance  cumulative_initiated_by_year
     most_aggressive            20.0       20.008214    2.279179e+00   0.6                      True                          True
     most_aggressive            40.0       40.016427    3.034338e+00   0.6                      True                          True
     most_aggressive            60.0       59.947981    3.427698e+00   0.6                      True                          True
median_configuration            20.0       20.008214    8.351073e-04   0.9                     False                         False
median_configuration            40.0       40.016427    1.038281e-02   0.9                     False                         False
median_configuration            60.0       59.947981    3.052064e-02   0.9                     False                         False
        most_durable            20.0       20.008214    2.224931e-60   1.2                     False                         False
        most_durable            40.0       40.016427    4.034680e-46   1.2                     False                         False
        most_durable            60.0       59.947981    1.881872e-39   1.2                     False                         False
