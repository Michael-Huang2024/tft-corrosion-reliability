# Windowed Multi-Output Linear/Sigmoid Baseline — Feasibility Audit

**Date:** 2026-06-28  
**Scope:** Read-only code inspection of `scripts/07_train_benchmarks.py` and shared evaluation utilities. No manuscript changes, no training runs performed.

**Verdict:** **Yes — straightforward to add with low implementation risk.** Expected runtime is **minutes, not hours** using sklearn on the existing `WindowDataset`. Recommended model: **`StandardScaler` + `MultiOutputClassifier(LogisticRegression)`**.

---

## Executive Summary

| Question | Answer |
|---|---|
| Reuse exact Windowed MLP 52×7 → 13 dataset? | **Yes** |
| Quick to implement? | **Yes** (~80–120 lines, one small script) |
| Safest model choice? | **MultiOutput LogisticRegression** (not logit-transform targets) |
| Low-cost runtime? | **Yes** — smoke <1 min; full run ~6–25 min |
| Worth adding? | **Yes**, for a window-matched linear reference alongside pointwise Logistic and Windowed MLP |

---

## 1. Can We Reuse the Exact Windowed MLP 52×7 → 13 Dataset?

**Yes, exactly.**

`WindowDataset` in `scripts/07_train_benchmarks.py` already defines the shared benchmark windows:

```python
for start in range(max_start + 1):
    enc = values[start : start + encoder_length]                    # (52, 7)
    dec_target = target[start + encoder_length : start + encoder_length + prediction_length]  # (13,)
    self.windows.append((enc, dec_target, series_id, dec_time_idx, dec_t_year))
```

Windowed MLP uses:

| Component | Shared? |
|---|---|
| `prepare_data()` / series split | Yes |
| `fit_point_scaler()` on train `POINT_FEATURES` | Yes |
| `WindowDataset(..., encoder=52, horizon=13)` | Yes |
| Train / validation / test splits | Yes |
| Stride = 1 | Yes |

**Sklearn input:** flatten encoder `(52, 7) → 364` dimensions — same as `WindowedMLPClassifier.forward()`.

No new dataset logic is required; only export windows from `WindowDataset` (loop or `DataLoader` + reshape).

---

## 2. Can a Deterministic Windowed Multi-Output Linear/Sigmoid Baseline Be Implemented Quickly?

**Yes.**

### Minimal new code (~80–120 lines)

1. Build scaled train/test `WindowDataset` (copy from `train_windowed_mlp`, lines 243–254).
2. Stack `X_train` shape `(n_windows, 364)`, `y_train` shape `(n_windows, 13)`.
3. Fit sklearn pipeline (see Section 3).
4. Reuse **identical test inference + overlap averaging** (lines 311–339 in `train_windowed_mlp`).
5. Call `evaluate_point_predictions()` from `scripts/revision_metrics.py`.
6. Append one row to benchmark CSVs via `merge_existing_by_seed` / `summarize_results`.

### Not required

- PyTorch training loop
- GPU
- Multi-seed loop (deterministic fit)

---

## 3. Which Option Is Safer?

| Option | Pros | Cons | Recommendation |
|---|---|---|---|
| **A. MultiOutput `LogisticRegression` + `StandardScaler`** | True linear-in-features + sigmoid; matches “windowed logistic”; same loss family as pointwise LR | 13 independent fits; slower on ~503k windows | **Best semantic match — recommended** |
| **B. Multi-output `Ridge` + clip to [0,1]** | Very fast (seconds); stable closed-form | Not a sigmoid model; clipping is ad hoc for probabilities | Good speed fallback; weaker naming |
| **C. Ridge on logit-transformed targets → sigmoid back** | Links to log-odds in theory | `onset_flag ∈ {0,1}` → logit undefined at 0/1; needs epsilon hacks | **Avoid** |
| **D. 13 separate horizon-specific ridge/logit models** | Clear per-horizon models | Equivalent to `MultiOutputClassifier`; more boilerplate | Use **A** instead |

### Recommended implementation

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

model = make_pipeline(
    StandardScaler(),
    MultiOutputClassifier(
        LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    ),
)
model.fit(X_train, y_train.astype(int))
```

Optional: select `C` from `{0.1, 1, 10}` on validation BCE once — still cheap and deterministic.

---

## 4. Estimated Runtime

### Window counts (approximate)

| Item | Value |
|---|---|
| Series length | ~783 steps |
| Windows per series | `783 - 52 - 13 + 1 = 719` |
| Train windows | `700 × 719 ≈ 503,000` |
| Test windows | `150 × 719 ≈ 108,000` |

### Expected timing (CPU, full data)

| Stage | MultiOutput Logistic | Ridge + clip |
|---|---|---|
| Build windows + scale | ~30–90 s | ~30–90 s |
| Fit | **~5–20 min** (13 × lbfgs on ~503k×364) | **~5–30 s** |
| Test inference + overlap average | ~10–60 s | ~10–60 s |
| Pf aggregation + CSV write | <5 s | <5 s |
| **Total** | **~6–25 min** | **~1–3 min** |

### Comparison to existing benchmarks

| Model | Approximate cost |
|---|---|
| Windowed MLP (1 seed) | ~4 min training + inference |
| Windowed MLP (3 seeds) | ~12+ min |
| GRU / TFT | Much longer |
| **Windowed Logistic (proposed)** | ~6–25 min, **no seeds** |

**Smoke test** (`--smoke-series 40`): **<1 min** — recommended before full run.

**Fallback if lbfgs is slow:** use `solver="saga"` or `SGDClassifier(loss="log_loss")`.

---

## 5. Expected Output Files

| File | Purpose |
|---|---|
| `outputs/revision/predictions/final_pf_windowed_logistic_regression.csv` | Population Pf(t) curve |
| `outputs/revision/checkpoints/final_windowed_logistic_regression.joblib` | Fitted pipeline (optional) |
| Updated `outputs/revision/tables/final_model_comparison_by_seed.csv` | One new row |
| Updated `outputs/revision/tables/final_model_comparison.csv` | Summary row |
| Optional: `outputs/revision/windowed_logistic_baseline_report.md` | Short run report |

Regenerating comparison plots (`final_model_error_comparison.png`) is optional.

---

## 6. Manuscript Table Row (If Successful)

Add a **new row**, distinct from pointwise Logistic:

| Model | MAE | RMSE | Notes |
|---|---|---|---|
| **Windowed Logistic Regression** | TBD | TBD | 52-step encoder, 13-step horizon, stride 1; deterministic |
| Logistic Regression (pointwise) | 0.02065 | 0.02427 | Existing row — keep unchanged |

### Suggested Methods wording

> A windowed linear baseline was also evaluated by flattening the 52×7 encoder window and fitting independent logistic regressions for each of the 13 future initiation indicators, using the same overlap-averaged inference and population Pf(t) evaluation as the sequence models.

### Naming convention

| Name | Meaning |
|---|---|
| **Pointwise Logistic Regression** | Current baseline — one row of 7 features → one probability |
| **Windowed Logistic Regression** | Proposed — 52×7 flattened → 13 horizon logits |

Always use distinct labels to avoid confusion.

---

## 7. Risks and Terminology Issues

| Risk | Severity | Mitigation |
|---|---|---|
| Name collision with pointwise Logistic | **High** | Label **“Windowed Logistic Regression”** vs **“Pointwise Logistic Regression”** |
| Not a recurrent model | Low | Describe as **linear feedforward window baseline** |
| 13 independent horizon outputs | Low | Standard sklearn multi-output; state explicitly |
| No random seeds / no ± SD | Low | Single deterministic fit; report without seed dispersion |
| No monotonicity constraint within window | Low | Same as MLP/GRU; cumulative target is per-step label |
| lbfgs slow on ~503k windows | Medium | Smoke test first; fallback to `saga` or SGD |
| Logit-transform of 0/1 targets | **High** | **Do not use** |
| Calling it a “sequence model” | Medium | Fair on **input window and horizon**; architecture is linear, not sequential |
| Benchmark CSV merge | Low | Use distinct `model` string in `merge_existing_by_seed` |
| May beat or underperform Windowed MLP | Low | Report honestly |

---

## 8. Reuse Map of Existing Pipeline Components

| Component | Location | Reusable? |
|---|---|---|
| `prepare_data()` | `07_train_benchmarks.py` | Yes |
| `fit_point_scaler()` | `07_train_benchmarks.py` | Yes |
| `WindowDataset` | `07_train_benchmarks.py` | Yes |
| Overlap averaging loop | `train_windowed_mlp` lines 311–328 | Yes — replace torch forward with sklearn `predict_proba` |
| `evaluate_point_predictions()` | `revision_metrics.py` | Yes |
| `merge_existing_by_seed()` | `07_train_benchmarks.py` | Yes |
| `summarize_results()` | `07_train_benchmarks.py` | Yes |
| `load_available_pf_curves()` | `07_train_benchmarks.py` | Needs one new glob if plots regenerated |

### Inference pipeline (unchanged logic)

```
For each test window:
  X_flat (364,) → model → probs (13,)
  For each horizon step j:
    accumulate probs[b,j] at (series_id, time_idx[j])
Overlap-average per (series_id, time_idx)
→ point-level p_onset_pred
→ aggregate_population_pf (test-set mean by t_year)
→ MAE / RMSE on Pf curve
```

Same as Windowed MLP, GRU, and TFT.

---

## 9. Minimal Script Plan (Not Yet Implemented)

### Recommended: standalone script (lowest risk)

**New file:** `scripts/23_windowed_logistic_baseline.py`

```
1. parse_args: --smoke-test, --smoke-series
2. df = prepare_data(args)
3. scaler = fit_point_scaler(df); scale splits
4. Build WindowDataset for train / test
5. windows_to_arrays(ds) → X (n, 364), y (n, 13)
6. Fit MultiOutput LogisticRegression pipeline on train
7. (Optional) validation BCE for reporting
8. Test inference:
     probs = predict_proba per window (batch, 13)
     overlap-average per (series_id, time_idx)  # copy MLP loop
9. evaluate_point_predictions(..., model_name="Windowed Logistic Regression")
10. Save final_pf_windowed_logistic_regression.csv
11. Merge into final_model_comparison*.csv
12. Print MAE/RMSE + runtime
```

### Alternative: extend `07_train_benchmarks.py`

Add `train_windowed_logistic()` + `--models windowed_logistic`.

**Recommendation:** standalone script first — no risk to existing MLP/GRU/TFT reruns.

### Suggested commands (when ready)

```bash
# Smoke test first
python scripts/23_windowed_logistic_baseline.py --smoke-test --smoke-series 40

# Full run
python scripts/23_windowed_logistic_baseline.py
```

---

## 10. Parameter Count Estimate

| Component | Count |
|---|---|
| Shared `StandardScaler` | 7 means + 7 scales (not always counted as model params) |
| Per-horizon LogisticRegression | 364 coefficients + 1 intercept = **365** |
| 13 horizons | **13 × 365 = 4,745** parameters |

Compare: pointwise Logistic = **8**; Windowed MLP = **135,437**.

---

## 11. Expected Benchmark Position

Based on model complexity ordering:

```
Pointwise Logistic (weakest, no temporal window)
    ↓
Windowed Logistic (linear, has window)     ← proposed
    ↓
Windowed MLP (nonlinear, has window)
    ↓
GRU (sequential)
    ↓
TFT (attention-based sequential)
```

Exact MAE/RMSE unknown until run. Windowed Logistic will likely sit **between pointwise Logistic and Windowed MLP**, but this is not guaranteed.

---

## 12. Explicit Answers to Audit Questions

| # | Question | Answer |
|---|---|---|
| 1 | Reuse exact 52×7→13 dataset? | **Yes** — same `WindowDataset`, scaler, splits |
| 2 | Quick deterministic implementation? | **Yes** — ~80–120 lines, sklearn only |
| 3 | Safest option? | **MultiOutput LogisticRegression**; avoid logit-transform targets |
| 4 | Estimated runtime? | Smoke <1 min; full ~6–25 min |
| 5 | Expected outputs? | Pf CSV + updated comparison tables (Section 5) |
| 6 | Manuscript table row? | **Windowed Logistic Regression** — distinct from pointwise (Section 6) |
| 7 | Risks / terminology? | Naming collision, deterministic/no seeds, training time (Section 7) |

---

## 13. Recommendation

**Proceed with implementation** using standalone script `scripts/23_windowed_logistic_baseline.py`:

1. Run smoke test first (`--smoke-series 40`).
2. If smoke passes, run full fit (~6–25 min).
3. Merge results into `final_model_comparison.csv`.
4. Add manuscript row only after verifying MAE/RMSE.
5. Do **not** replace or relabel the existing pointwise Logistic row.

---

*End of report.*
