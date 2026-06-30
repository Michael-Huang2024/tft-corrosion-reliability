# MC Dropout Uncertainty Quantification Audit Report

**Date:** 2026-06-28  
**Scope:** Read-only inspection of scripts, outputs, and in-repo text. No code changes or reruns were performed.

**Bottom line:** MC Dropout UQ is **implemented and has full-scale outputs** (50-pass formal run + 100-pass convergence study). The repository does **not** contain the manuscript `.docx`, a dedicated Response-to-Comment-6 draft, or a file explicitly named “Figure 6.” Implementation language is generally cautious, but **PICP ≈ 11%** should be reported honestly and intervals must not be called “validated confidence intervals.”

---

## 1. TFT Checkpoint Used for MC Dropout

| Item | Value |
|---|---|
| **Checkpoint path** | `outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt` |
| **Seed** | **20250111** |
| **Selection criterion** | Best validation loss (not test MAE) |
| **Best validation loss** | **0.007161** (reported); **0.007161** in `final_tft_three_seed_results.csv` |
| **Epochs trained** | **10** completed; **best epoch = 8** |
| **Representative checkpoint?** | **Yes** — same seed/checkpoint used for Table 5 representative TFT and deterministic Pf(t) (`tft_20250111_10epoch.csv`) |

**Sources:** `scripts/09_mc_dropout_uq.py`, `scripts/18_mc_dropout_convergence.py`, `outputs/revision/mc_dropout_report.md`, `outputs/revision/final_mc_dropout_50_vs_100_convergence_report.md`

---

## 2. MC Dropout Implementation

| Question | Answer |
|---|---|
| **Model in eval mode?** | **Yes** — `model.eval()` globally |
| **Dropout active during inference?** | **Yes** — only `nn.Dropout*` modules set to `.train()` |
| **Other layers in eval mode?** | **Yes** — BatchNorm/LayerNorm etc. stay in eval |
| **Stochastic forward passes** | **50** (formal UQ, `09_mc_dropout_uq.py`); **100** (convergence, `18_mc_dropout_convergence.py`) |
| **Main reported result** | **50 passes** (`mc_dropout_population_predictions.csv`, `mc_dropout_metrics.csv`, `mc_dropout_uncertainty_band.png`) |
| **50 vs 100 convergence check?** | **Yes** — full nested 20/50/100 analysis in `18_mc_dropout_convergence.py` |

### Dropout activation (script 18)

- **24 active dropout modules** listed in `final_mc_dropout_50_vs_100_convergence_report.md` (GRN gates, attention dropout, etc.)

### Seeding difference (important)

| Script | Per-pass seeding |
|---|---|
| `09_mc_dropout_uq.py` | **No explicit per-pass seed** |
| `18_mc_dropout_convergence.py` | **Yes** — `base_seed=20250626 + pass_id` |

The convergence report notes the original 50-pass run lacked pass-level trajectories/reproducible seeds, so **all 100 passes were re-run** for fair nested comparison. The two 50-pass result sets differ slightly (MAE 0.004608 vs 0.004602).

### Runtime

| Run | Total time | Per pass |
|---|---|---|
| 50-pass (`09`) | 5964.6 s (~99 min) | 119.29 s |
| 100-pass (`18`) | 12237.8 s (~204 min) | 122.38 s |

### Key implementation code

From `scripts/09_mc_dropout_uq.py`:

```python
def enable_dropout_only(model: nn.Module) -> None:
    """Keep normalization in eval mode; retain dropout during stochastic prediction."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
```

---

## 3. Input Data for MC Dropout Inference

| Item | Value |
|---|---|
| **Split** | **Test only** (150 series) |
| **Training data role** | Used only to build `TimeSeriesDataSet` schema from train split |
| **Evaluation time range** | `time_idx ≥ 52` → **731 points**, **3.99–59.95 years** |
| **Encoder window** | **52** steps |
| **Prediction horizon** | **13** steps |
| **Stride** | **1** (implicit via `TimeSeriesDataSet`, `stop_randomization=True`) |
| **Target** | `onset_flag` (cumulative P(Tᵢ ≤ t)) |
| **Covariates** | Static: `Cs, D28, m_aging, cover_mm, C_th`; known time: `time_idx, t_year` |
| **`chloride_rebar` excluded?** | **Yes** — `TFT_TIME_VARYING_UNKNOWN_REALS = []` |
| **Data file** | `data/processed/revision/final_chloride_labeled.parquet` |
| **Split file** | `data/processed/revision/series_split.csv` |

Same evaluation domain as deterministic TFT benchmark.

---

## 4. Stochastic Prediction Aggregation

| Step | Implementation |
|---|---|
| **Within-pass window overlap** | **Yes** — average per `(series_id, time_idx)` before population aggregation (same as deterministic TFT in `17_tft_three_seed_benchmark.py`) |
| **Population Pf(t)** | **Yes** — test-set mean of scenario-level `p_onset_pred` at each `t_year` |
| **Clipping to [0,1]** | **No explicit clip**; values come from softmax class-1 probabilities, so naturally ∈ [0,1] |
| **Same rules as deterministic TFT?** | **Yes** — same test loader, overlap averaging, and population mean |

**Pipeline per pass:** stochastic forward → point-level probs → overlap average → population Pf(t) curve → aggregate across passes for mean/std/quantiles.

---

## 5. Uncertainty Statistics — Available Values

### 5.1 50-pass formal run (`mc_dropout_metrics.csv`)

| Statistic | Value |
|---|---|
| Predictive mean MAE | **0.004608** |
| Predictive mean RMSE | **0.006499** |
| Mean predictive std | **0.000461** |
| Max predictive std | **0.001682** |
| PICP (95%) | **0.1149** (11.5%) |
| MPIW (95%) | **0.001668** |
| Max interval width | **0.006213** |
| Evaluation points | **731** |
| Test series | **150** |
| Time range | **3.99–59.95 years** |

### 5.2 100-pass convergence run (`final_mc_dropout_metrics_20_50_100.csv`)

| Passes | MAE | RMSE | Mean std | Max std | PICP | MPIW | Final-year abs error |
|---|---|---|---|---|---|---|---|
| 20 | 0.004610 | 0.006497 | 0.000435 | 0.001226 | 0.0944 | 0.001452 | 0.001288 |
| 50 | 0.004602 | 0.006489 | 0.000451 | 0.001562 | 0.1053 | 0.001643 | 0.001071 |
| 100 | 0.004616 | 0.006503 | 0.000455 | 0.001803 | 0.1135 | 0.001715 | 0.000824 |

### 5.3 Final-year values (~59.95 y, reference Pf = 0.3867)

| Passes | Predictive mean | Std | q025 (lower) | q975 (upper) |
|---|---|---|---|---|
| 50 | 0.38564 | 0.00168 | 0.38309 | 0.38930 |
| 100 | 0.38584 | 0.00180 | 0.38366 | 0.39069 |

### 5.4 50 vs 100 convergence (`final_mc_dropout_convergence_20_50_100.csv`)

| Metric | Value |
|---|---|
| Mean abs change in predictive mean | **3.77×10⁻⁵** |
| Max abs change in predictive mean | **2.47×10⁻⁴** |
| Mean abs change in predictive std | **2.73×10⁻⁵** |
| Max abs change in predictive std | **2.40×10⁻⁴** |
| Relative MPIW change | **4.37%** |
| PICP change | **+0.0082** |

**20 vs 50 convergence (from same file):**

| Metric | Value |
|---|---|
| Mean abs change in predictive mean | 7.11×10⁻⁵ |
| Max abs change in predictive mean | 4.75×10⁻⁴ |
| Mean abs change in predictive std | 4.78×10⁻⁵ |
| Relative MPIW change | 13.17% |

### 5.5 Deterministic TFT comparison (same checkpoint)

| Comparison | Value |
|---|---|
| Deterministic TFT MAE | **0.004543** |
| MC Dropout 50-pass predictive mean MAE | **0.004608** |
| Mean abs diff (MC mean vs deterministic Pf_pred) | **0.001301** |

### 5.6 Per-time-point series

Available in CSV for all 731 points:

- `predictive_std(t)`, `q025(t)`, `q975(t)` in:
  - `outputs/revision/predictions/mc_dropout_population_predictions.csv` (50-pass)
  - `outputs/revision/predictions/mc_dropout_seed20250111_summary_20_50_100.csv` (20/50/100)

### 5.7 Metric checklist vs audit request

| Requested statistic | Computed? | Source |
|---|---|---|
| Predictive mean MAE | Yes | `mc_dropout_metrics.csv`, `final_mc_dropout_metrics_20_50_100.csv` |
| Predictive mean RMSE | Yes | Same |
| Predictive std over time | Yes | Per time point in prediction CSVs; plotted in std convergence figure |
| Average predictive std | Yes | `mean_predictive_std` |
| Maximum predictive std | Yes | `max_predictive_std` |
| Approximate 95% predictive interval | Yes | `q025`, `q975` |
| Lower / upper predictive bounds | Yes | Same |
| Average interval width | Yes | `MPIW_95` |
| Maximum interval width | Yes | Computed: 0.00621 (50-pass), 0.00703 (100-pass) |
| Final-year predictive mean | Yes | Last row of prediction CSVs |
| Final-year lower / upper bound | Yes | Same |
| PICP | Yes | ~11% (low; see Section 10) |
| 50→100 mean change (avg / max) | Yes | `final_mc_dropout_convergence_20_50_100.csv` |
| 50→100 std change (avg / max) | Yes | Same |

---

## 6. Figures and Tables Generated

### 6.1 Figures

| File | Content |
|---|---|
| `outputs/revision/figures/mc_dropout_uncertainty_band.png` | **50-pass**: reference Pf, MC predictive mean, 95% band |
| `outputs/revision/figures/mc_dropout_uncertainty_band_100passes.png` | **100-pass**: reference Pf, MC mean, 95% band |
| `outputs/revision/figures/mc_dropout_mean_convergence_20_50_100.png` | Mean convergence across 20/50/100 passes |
| `outputs/revision/figures/mc_dropout_std_convergence_20_50_100.png` | Std convergence across 20/50/100 passes |
| `outputs/revision/figures/mc_dropout_convergence_difference_50_vs_100.png` | \|mean_100−mean_50\|, \|std_100−std_50\| vs time |

### 6.2 Tables and reports

| File | Role |
|---|---|
| `outputs/revision/tables/mc_dropout_metrics.csv` | 50-pass summary metrics |
| `outputs/revision/tables/mc_dropout_convergence.csv` | 20 vs 50 (from script 09) |
| `outputs/revision/tables/final_mc_dropout_metrics_20_50_100.csv` | 20/50/100 metrics |
| `outputs/revision/tables/final_mc_dropout_convergence_20_50_100.csv` | 20 vs 50 and 50 vs 100 |
| `outputs/revision/predictions/mc_dropout_population_predictions.csv` | 50-pass time series |
| `outputs/revision/predictions/mc_dropout_seed20250111_100pass_level_population.csv` | Pass-level trajectories (100 passes) |
| `outputs/revision/predictions/mc_dropout_seed20250111_100pass_state.json` | Resume state (100 passes complete) |
| `outputs/revision/predictions/mc_dropout_seed20250111_summary_20_50_100.csv` | Nested summaries at 20/50/100 |
| `outputs/revision/mc_dropout_report.md` | Formal 50-pass report |
| `outputs/revision/final_mc_dropout_50_vs_100_convergence_report.md` | Convergence report |

### 6.3 Figure 6 mapping

- **No file named “Figure 6”** exists in the repository.
- Likely manuscript Figure 6 content maps to:
  - **`mc_dropout_uncertainty_band.png`** (50-pass main result), or
  - **`mc_dropout_uncertainty_band_100passes.png`** (100-pass convergence figure)
- Both show: **simulator/reference Pf(t)**, **MC Dropout predictive mean**, **approximate 95% predictive interval**.
- **Deterministic TFT line is NOT plotted** in either figure (only computable from saved CSVs).

### 6.4 Related but separate UQ

| Script | Purpose | Outputs |
|---|---|---|
| `scripts/10_bootstrap_reference_uq.py` | Bootstrap **reference sampling uncertainty** (distinct from MC Dropout epistemic UQ) | **Not generated** — no `bootstrap_reference_intervals.csv` or `reference_bootstrap_band.png` in `outputs/revision/` |

---

## 7. Missing Values vs Guidance Document

**`The_Section_of_Uncertainty_Quantification_by_Using_MC_Dropout.docx` is not in the repository** — a direct placeholder-by-placeholder comparison could not be performed.

### Computed vs potentially missing

| Item | Status |
|---|---|
| All core metrics in Section 5 | **Computed** |
| Deterministic TFT on UQ figure | **Missing from figure** (data available in `tft_20250111_10epoch.csv`) |
| Manuscript “XX” placeholders | **Unknown** — source doc not in repo |
| Bootstrap reference band | **Not generated** (optional supplement) |
| Dedicated Comment 6 response text | **Not in repo** |

---

## 8. Manuscript Text (Current State in Repo)

**No manuscript `.tex`/`.docx` draft is stored in the repository.**

### 8.1 Methods-style language (from `mc_dropout_report.md` / script 09)

> MC Dropout-based approximate Bayesian inference on the locked TFT benchmark. The trained checkpoint is loaded once with fixed weights. The model is set globally to evaluation mode (`model.eval()`), while dropout layers remain active during stochastic forward passes. Fifty independent passes approximate the approximate posterior predictive distribution and provide an epistemic uncertainty estimate.
>
> This analysis does **not** claim exact Bayesian inference, a full Bayesian posterior, or a rigorous Bayesian TFT.

### 8.2 Results-style content (from saved metrics)

- 50-pass MAE = 0.004608, mean predictive std = 0.000461, PICP = 11.5%, MPIW = 0.00167
- 100-pass: MAE = 0.004616, mean std = 0.000455, PICP = 11.4%, MPIW = 0.00172
- 50 passes deemed sufficient vs 100 (mean/std changes < 1×10⁻⁴; MPIW relative change 4.37%)

### 8.3 Discussion / limitations (from reports)

- Not exact Bayesian inference / not fully Bayesian TFT
- Intervals reflect **approximate epistemic uncertainty of the surrogate**, not total physical uncertainty
- MC Dropout is **computationally expensive** (~100 min for 50 passes on RTX 4060)
- Low PICP implies intervals are **not well-calibrated** for covering reference Pf(t)

### 8.4 Manuscript-update checklist (`revision_experiment_report.md`)

- Label MC Dropout as approximate Bayesian uncertainty, not exact Bayesian inference

---

## 9. Response to Reviewers — Comment 6

**No dedicated “Reviewer Comment 6” response text was found in the repository.**

### Closest existing draft language

| Source | Content |
|---|---|
| `revision_experiment_report.md` | “MC Dropout UQ: implemented and smoke-tested” (status outdated — full runs now exist) |
| `final_computational_efficiency_report.md` | Mentions 50-pass MC Dropout **runtime** in Editor computational-advantage response, not UQ methodology |
| `final_sobol_sensitivity_report.md` | Notes MC Dropout as complementary analysis to Sobol sensitivity |

**A Comment 6 response still needs to be written** using the final metrics in Section 5.

---

## 10. Scientific Accuracy of Current Wording

| Claim | Repo handling | Accurate? |
|---|---|---|
| Exact Bayesian inference | Explicitly disclaimed | **Good** |
| Fully Bayesian TFT | Explicitly disclaimed | **Good** |
| Field-validated confidence interval | Not claimed; PICP ≈ 11% shows miscalibration | **Must not overclaim** |
| Approximate surrogate epistemic uncertainty | Correct framing | **Good** |
| Total physical/environmental uncertainty | Not claimed; should state exclusion of aleatoric/parameter uncertainty | **Needs explicit limitation sentence** |

### Caution flags

1. **PICP ≈ 11%** — 95% intervals cover reference Pf only ~11% of the time; do not call them “95% confidence intervals” without an “approximate / uncalibrated” qualifier.
2. Intervals are **across MC Dropout masks of one checkpoint**, not across training-data or parameter uncertainty.
3. **50-pass and 100-pass runs used different seeding protocols** — prefer citing the **100-pass reproducible run** (`base_seed=20250626`) for convergence claims.
4. Title phrase “approximate posterior predictive distribution” is acceptable if paired with MC Dropout limitations.

---

## 11. Completeness for Reviewer Response

### Substantively complete

- Implementation (dropout-only stochastic inference)
- Representative checkpoint selection documented
- Full test-set evaluation (731 points, 150 series)
- 50-pass main results + 100-pass convergence study
- Figures and tables for mean, std, 95% band, convergence
- Appropriate “not exact Bayesian” disclaimers in code reports

### Minimal gaps before submission

| Gap | Minimal fix |
|---|---|
| **No Comment 6 response draft** | Write response paragraph citing 50-pass main + 100-pass convergence |
| **No manuscript UQ section text in repo** | Add Methods/Results/Limitations paragraphs with numeric values from Section 5 |
| **Low PICP not prominently discussed** | Add limitation: intervals are approximate epistemic bands, not calibrated coverage |
| **Figure 6 not explicitly linked** | Map to `mc_dropout_uncertainty_band_100passes.png` (or 50-pass) and label per journal style |
| **Deterministic TFT not on UQ figure** | Optional: add dashed deterministic line for comparison |
| **Guidance docx placeholders** | Manually compare docx “XX” fields against Section 5 values |
| **Bootstrap reference UQ** | Optional supplement only; not required if MC Dropout is primary UQ |

### Verdict

**Implementation is complete enough for a reviewer response**, provided the manuscript/response:

1. Uses the **100-pass convergence study** to justify 50 passes.
2. Reports **PICP honestly** (~11%) and frames intervals as **approximate epistemic uncertainty**, not validated confidence intervals.
3. States scope limits (single checkpoint, no parameter/aleatoric uncertainty, no field validation).

---

## Key Scripts Inspected

| Script | Purpose |
|---|---|
| `scripts/09_mc_dropout_uq.py` | Formal 50-pass MC Dropout UQ |
| `scripts/18_mc_dropout_convergence.py` | 100-pass nested 20/50/100 convergence |
| `scripts/10_bootstrap_reference_uq.py` | Optional reference bootstrap (not run) |
| `scripts/07_train_benchmarks.py` | Benchmark context (deterministic TFT evaluation pipeline) |
| `scripts/17_tft_three_seed_benchmark.py` | Representative TFT checkpoint training/eval |
| `scripts/revision_config.py` | Shared constants (52/13 window, features, split) |
| `scripts/revision_metrics.py` | Pf aggregation and MAE/RMSE (deterministic benchmark) |
| `scripts/20_final_computational_efficiency.py` | MC Dropout runtime in efficiency comparison |

---

## Explicit Answers to Audit Questions

1. **Checkpoint:** `outputs/revision/checkpoints/tft/20250111_10epoch/best.ckpt`; seed 20250111; 10 epochs (best epoch 8); representative by validation loss.

2. **Implementation:** `model.eval()` + dropout-only `.train()`; 50 passes main; 100-pass 20/50/100 convergence check performed.

3. **Input data:** Test only; 731 points; 52/13 window; stride 1; `onset_flag`; leakage-free covariates; no `chloride_rebar`.

4. **Aggregation:** Overlap-averaged per series/time; population mean; no clip; same as deterministic TFT.

5. **Statistics:** Full set in Section 5; PICP ~11%; 50→100 convergence metrics available.

6. **Figures/tables:** Listed in Section 6; Figure 6 likely maps to uncertainty band PNGs; deterministic TFT not on figure.

7. **Guidance docx:** Not in repo; core metrics computed; deterministic-on-figure and Comment 6 text missing.

8. **Manuscript:** No draft in repo; in-repo language summarized in Section 8.

9. **Response Comment 6:** Not drafted in repo.

10. **Accuracy:** Disclaimers good; must not overclaim calibrated intervals or total uncertainty.

11. **Completeness:** Sufficient for response with honest PICP and scope limitations; minimal manuscript/response edits needed.

---

*End of audit report.*
