# Benchmark Model Comparison Document Audit

**Date:** 2026-06-28  
**Purpose:** Collect final benchmark information needed to complete `The_Section_of_Benchmark_Model_Comparison.docx`  
**Scope:** Read-only inspection of scripts, CSV outputs, and reports. No files modified, no training rerun.

**Note:** `The_Section_of_Benchmark_Model_Comparison.docx` is **not in the repository**. Section 9 lists likely placeholder fields and replacement text inferred from saved outputs.

---

## Executive Summary

| Item | Status |
|---|---|
| Fair sequence benchmark (MLP / GRU / TFT) | **Complete** — 52→13 window, stride 1, shared split, target, metrics |
| Windowed MLP correction | **Applied** — replaces prior pointwise MLP |
| Pointwise Logistic baseline | **By design** — not window-matched |
| GPR | **Not implemented** — omission acceptable with justification |
| Table 10 values | **Available** in `final_model_comparison.csv` |
| Runtime in comparison CSV | **Partial** — MLP complete; GRU/TFT/Logistic gaps |
| Ranking (MAE) | GRU > TFT > Windowed MLP > Logistic |

---

## 1. Seven Covariates (52 × 7 Input Window)

Used by **Windowed MLP**, **GRU**, and **TFT**. TFT splits the same seven quantities into static vs time-varying roles; GRU and Windowed MLP consume all seven at each encoder time step.

| # | Code name | Manuscript-friendly name | TFT role |
|---|---|---|---|
| 1 | `Cs` | Surface chloride concentration | Static |
| 2 | `D28` | Reference chloride diffusivity at 28 days | Static |
| 3 | `m_aging` | Diffusion aging exponent | Static |
| 4 | `cover_mm` | Concrete cover depth (mm) | Static |
| 5 | `C_th` | Critical chloride threshold (C_crit) | Static |
| 6 | `time_idx` | Discrete time index | Known time-varying |
| 7 | `t_year` | Service time (years) | Known time-varying |

**Code definitions** (`scripts/revision_config.py`):

```python
PHYSICAL_FEATURES = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
POINT_FEATURES = ["Cs", "D28", "m_aging", "cover_mm", "C_th", "time_idx", "t_year"]
TFT_STATIC_REALS = PHYSICAL_FEATURES
TFT_TIME_VARYING_KNOWN_REALS = ["time_idx", "t_year"]
TFT_TIME_VARYING_UNKNOWN_REALS = []
```

**Sources:** `scripts/revision_config.py`, `outputs/revision/feature_audit.md`

---

## 2. Common Forecasting Setup

| Item | Value |
|---|---|
| Input window length | **52** steps (`MAX_ENCODER_LENGTH = 52`) |
| Prediction horizon | **13** steps (`MAX_PREDICTION_LENGTH = 13`) |
| Sliding stride | **1** |
| Target variable | `onset_flag` — cumulative P(Tᵢ ≤ t) |
| Split method | Series-level **70% / 15% / 15%** |
| Split random seed | **20250111** (`SPLIT_SEED`) |
| Train / validation / test series | **700 / 150 / 150** |
| Total series | **1000** |
| Evaluation split | **Test only** |
| Evaluation time filter | `time_idx ≥ 52` |
| Population evaluation points | **731** |
| Evaluation time range | **~3.99–59.95 years** |
| Data file | `data/processed/revision/final_chloride_labeled.parquet` |
| Split file | `data/processed/revision/series_split.csv` |
| Overlap handling | Average predictions per `(series_id, time_idx)` on test |
| Population Pf(t) | Test-set mean of scenario probabilities at each `t_year` |
| Metrics | MAE and RMSE on population Pf(t) curve |

### Leakage exclusion

**Confirmed:** `chloride_rebar` and all target-derived fields are excluded.

Forbidden predictors (`FORBIDDEN_PREDICTORS`):

- `chloride_rebar`
- `target_onset`, `onset_raw`, `onset_flag`
- `time_to_onset`, `t_init_year`, `t_init_idx`
- `target_cont`, `Pf`, `Pf_true`, `Pf_pred`

Enforced by `scripts/revision_data.py::assert_no_forbidden_predictors`.

### Task alignment by model

| Model | 52-step encoder | 13-step horizon | Stride 1 | Notes |
|---|---|---|---|---|
| Windowed MLP | Yes | Yes | Yes | 52×7 flattened → 364-dim input |
| GRU | Yes | Yes | Yes | Sequential encoder |
| TFT | Yes | Yes | Yes | TemporalFusionTransformer |
| Logistic Regression | **No** | **No** | N/A | **Pointwise** tabular baseline |

---

## 3. Benchmark Results

### 3.1 Primary sources

| File | Content |
|---|---|
| `outputs/revision/tables/final_model_comparison.csv` | Table 10 summary (mean ± SD) |
| `outputs/revision/tables/final_model_comparison_by_seed.csv` | Per-seed MAE/RMSE |
| `outputs/revision/final_benchmark_report.md` | Narrative ranking and comparison answers |
| `outputs/revision/predictions/final_pf_*.csv` | Population Pf(t) curves per model |

### 3.2 Summary table (test Pf(t) MAE / RMSE)

| Model | MAE (mean ± std) | RMSE (mean ± std) | Seed count |
|---|---|---|---|
| **GRU** | **0.001934 ± 0.000169** | **0.002931 ± 0.000207** | 3 |
| **TFT** | **0.004542 ± 0.000261** | **0.006373 ± 0.000649** | 3 |
| **Windowed MLP** | **0.006975 ± 0.000843** | **0.009964 ± 0.001328** | 3 |
| **Logistic Regression** | **0.020652 ± 0.000000** | **0.024271 ± 0.000000** | 1 (deterministic) |

### 3.3 Per-seed results

| Model | Seed | MAE | RMSE | Epochs | Best epoch |
|---|---|---|---|---|---|
| GRU | 20250111 | 0.001839 | 0.002876 | 18 | — |
| GRU | 20250112 | 0.002172 | 0.003208 | 15 | — |
| GRU | 20250113 | 0.001790 | 0.002710 | 22 | — |
| TFT | 20250111 | 0.004543 | 0.006382 | 10 | 8 |
| TFT | 20250112 | 0.004861 | 0.007164 | 7 | 3 |
| TFT | 20250113 | 0.004222 | 0.005575 | 8 | 4 |
| MLP | 20250111 | 0.007104 | 0.009440 | 11 | 4 |
| MLP | 20250112 | 0.007935 | 0.011787 | 10 | 3 |
| MLP | 20250113 | 0.005884 | 0.008663 | 15 | 8 |
| Logistic | — | 0.020652 | 0.024271 | — | — |

### 3.4 MAE ranking

| Rank | Model | MAE mean |
|---|---|---|
| 1 | GRU | 0.001934 |
| 2 | TFT | 0.004542 |
| 3 | Windowed MLP | 0.006975 |
| 4 | Logistic Regression | 0.020652 |

### 3.5 Head-to-head comparisons

| Comparison | Result |
|---|---|
| TFT vs Logistic Regression | **TFT better** (MAE 0.00454 vs 0.02065) |
| TFT vs Windowed MLP | **TFT better** (MAE 0.00454 vs 0.00697) |
| TFT vs GRU | **GRU better** (MAE 0.00193 vs 0.00454) |
| GRU vs Windowed MLP | **GRU better** |
| All sequence models vs Logistic | **Sequence models substantially better** |

### 3.6 Additional error metrics (from summary CSV)

| Model | Max abs error (mean ± std) | Final-year abs error (mean ± std) |
|---|---|---|
| GRU | 0.012885 ± 0.001165 | 0.001986 ± 0.001361 |
| TFT | 0.023737 ± 0.004354 | 0.002666 ± 0.000948 |
| MLP | 0.034792 ± 0.003773 | 0.004863 ± 0.002175 |
| Logistic | 0.052977 ± 0.000000 | 0.051192 ± 0.000000 |

---

## 4. Random Seeds

| Model | Seeds | Notes |
|---|---|---|
| Windowed MLP | **20250111, 20250112, 20250113** | `torch.manual_seed(seed)` per run |
| GRU | **20250111, 20250112, 20250113** | Same |
| TFT | **20250111, 20250112, 20250113** | `pl.seed_everything(seed)` |
| Logistic Regression | **None** | Single deterministic sklearn fit |

**Configuration:** `REVISION_SEEDS = (20250111, 20250112, 20250113)` in `scripts/revision_config.py`  
**Data split seed:** `SPLIT_SEED = 20250111` (independent of model training seeds)

---

## 5. Model Configurations

### 5.1 Windowed MLP

**Script:** `scripts/07_train_benchmarks.py` — `WindowedMLPClassifier`, `train_windowed_mlp`

| Setting | Value |
|---|---|
| Input | Flatten 52 × 7 = **364** dimensions |
| Output | **13** logits |
| Architecture | Linear(364→256) → ReLU → Dropout(0.1) → Linear(256→128) → ReLU → Dropout(0.1) → Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→13) |
| Activation | ReLU |
| Dropout | 0.1 |
| Loss | `BCEWithLogitsLoss` over 13-step output |
| Optimizer | Adam, learning rate **1×10⁻³** |
| Batch size | **64** |
| Max epochs | **30** |
| Early stopping | Validation BCE, patience **6** |
| Input scaling | `StandardScaler` fit on train split only |
| Epochs completed | 11, 10, 15 (seeds 20250111–20250113) |
| Parameters | **135,437** |
| Checkpoints | `outputs/revision/checkpoints/final_mlp_seed{seed}.pt` |

### 5.2 GRU

**Script:** `scripts/07_train_benchmarks.py` — `GRUClassifier`, `train_gru`

| Setting | Value |
|---|---|
| Input | 52 × 7 per-step sequence |
| Output | **13** logits |
| Architecture | GRU(input_size=7, hidden_size=**32**, num_layers=**1**) → Dropout(0.1) → Linear(32→13) |
| Recurrent dropout | 0 (single layer) |
| Loss | `BCEWithLogitsLoss` |
| Optimizer | Adam, learning rate **1×10⁻³** |
| Batch size | **64** |
| Max epochs | **30** |
| Early stopping | Validation BCE, patience **6** |
| Input scaling | `StandardScaler` on point features, applied per time step |
| Epochs completed | 18, 15, 22 |
| Parameters | **Not logged** in final benchmark CSVs |
| Checkpoints | `outputs/revision/checkpoints/final_gru_seed{seed}.pt` |

### 5.3 TFT

**Script:** `scripts/17_tft_three_seed_benchmark.py` (final Table 10 path; not `07_train_benchmarks.py` 40-epoch default)

| Setting | Value |
|---|---|
| Model | `TemporalFusionTransformer` (pytorch-forecasting) |
| Encoder / decoder | 52 / 13 |
| Hidden size | **32** |
| Attention heads | **4** |
| Dropout | **0.1** |
| Hidden continuous size | **16** |
| Output | 2-class (`CrossEntropy`) |
| Learning rate | **3×10⁻⁴** |
| Train batch size | **32** |
| Inference batch size | **64** |
| Max epochs | **10** |
| Min epochs | **5** |
| Early stopping | `val_loss`, patience **3** |
| Extra TFT features | `add_relative_time_idx=True`, `add_encoder_length=True` |
| Epochs completed | 10, 7, 8 |
| Best epochs | 8, 3, 4 |
| Parameters | **75,992** |
| Checkpoints | `outputs/revision/checkpoints/tft/{seed}_10epoch/best.ckpt` |

**Important asymmetry:** TFT final benchmark uses a **10-epoch cap**; Windowed MLP and GRU use **30-epoch cap** with patience 6. Document explicitly in Methods.

### 5.4 Logistic Regression

**Script:** `scripts/07_train_benchmarks.py` — `train_logistic`

| Setting | Value |
|---|---|
| Task | **Pointwise** (one feature row → one probability) |
| Pipeline | `StandardScaler` + `LogisticRegression(max_iter=1000, solver="lbfgs")` |
| Regularization | Default sklearn **L2**, **C = 1.0** (not overridden in code) |
| Features | Same 7 `POINT_FEATURES` as above |
| Parameters | **8** |
| Random seed | None (deterministic given fixed data) |

---

## 6. Runtime Availability

### 6.1 In `final_model_comparison.csv`

| Model | Training time (mean ± std) | Inference time (mean ± std) |
|---|---|---|
| Logistic Regression | — | — |
| Windowed MLP | **247.1 ± 42.5 s** | **2.66 ± 0.17 s** |
| GRU | — | — |
| TFT | — | — |

### 6.2 Per-seed Windowed MLP (from `final_model_comparison_by_seed.csv`)

| Seed | Training (s) | Inference (s) |
|---|---|---|
| 20250111 | 233.6 | 2.90 |
| 20250112 | 203.1 | 2.53 |
| 20250113 | 304.7 | 2.54 |

### 6.3 TFT (from `final_tft_three_seed_results.csv`)

| Seed | Training (s) | Inference (s) |
|---|---|---|
| 20250111 | 17020.5 | 90.9 |
| 20250112 | 14573.4 | 86.1 |
| 20250113 | 14425.2 | 92.5 |
| **Mean ± std** | **15339.7 ± 1190.0 s** | **89.9 ± 2.7 s** |

### 6.4 Remeasured inference harness (`final_computational_efficiency_summary.csv`, seed 20250111)

| Method | End-to-end (s) | Inference-only (s) |
|---|---|---|
| Logistic Regression | 0.46 | 0.010 |
| MLP | 0.136 | 0.0013* |
| GRU | 8.37 | 1.63 |
| TFT (deterministic) | 47.7 | 44.6 |
| TFT (50-pass MC Dropout) | 5964.6 | 5964.6 |

\*MLP harness note: simplified forward-pass timing; full windowed sliding-window test evaluation is ~2.5–2.9 s per seed in benchmark CSV.

### 6.5 Runtime recommendation for manuscript

| Recommendation | Detail |
|---|---|
| **Do not** claim uniform runtime table from `final_model_comparison.csv` alone | GRU/TFT/Logistic training times missing there |
| **Use** `final_computational_efficiency_summary.csv` for inference comparisons | Remeasured under locked 731-point test domain |
| **Weaken** any universal TFT speed-advantage claim | GRU best accuracy; MLP fastest surrogate inference in harness |
| **Do not use** `final_training_runtime_summary.csv` for MLP MAE | Contains **outdated pointwise MLP** values (MAE 0.003017) |

---

## 7. Windowed Logistic Regression

| Question | Answer |
|---|---|
| Implemented? | **No** |
| Current Logistic baseline | **Pointwise** only (`train_logistic`) |
| Manuscript description | **Pointwise linear reference baseline** using the same seven covariates and cumulative target at each evaluation time point, not window-matched to sequence models |

---

## 8. GPR / Sparse GPR

| Question | Answer |
|---|---|
| Implemented? | **No** — no GPR / `GaussianProcess` code in repository |
| Evidence | `outputs/revision_experiment_audit_v2.md`: "Gaussian Process — Not found" |
| Acceptable to omit? | **Yes**, under reviewer wording: *"MLP, GRU, Gaussian process regression, or any other suitable options"* |
| Suggested justification | Scalability on 1000 scenarios × 783 time steps; reported baselines span linear (Logistic), feedforward windowed (MLP), recurrent (GRU), and attention-based (TFT) surrogates |

---

## 9. Likely Document Placeholders and Replacement Text

`The_Section_of_Benchmark_Model_Comparison.docx` was not found in the repository. The table below maps common benchmark-section fields to locked values.

| Likely placeholder / field | Replacement text or value |
|---|---|
| Number of input covariates | **7** |
| Encoder window length | **52 time steps** |
| Prediction horizon | **13 time steps** |
| Sliding stride | **1** |
| Target variable | Cumulative initiation indicator **`onset_flag` ≡ P(Tᵢ ≤ t)** |
| Train / validation / test series | **700 / 150 / 150** |
| Test evaluation time points | **731** |
| Evaluation time range | **~3.99–59.95 years** |
| Neural model seeds | **20250111, 20250112, 20250113** |
| GRU MAE ± std | **0.00193 ± 0.00017** |
| GRU RMSE ± std | **0.00293 ± 0.00021** |
| TFT MAE ± std | **0.00454 ± 0.00026** |
| TFT RMSE ± std | **0.00637 ± 0.00065** |
| Windowed MLP MAE ± std | **0.00697 ± 0.00084** |
| Windowed MLP RMSE ± std | **0.00996 ± 0.00133** |
| Logistic MAE | **0.02065** |
| Logistic RMSE | **0.02427** |
| Best model (MAE) | **GRU** |
| TFT beats Logistic? | **Yes** |
| TFT beats MLP? | **Yes** |
| TFT beats GRU? | **No** |
| MLP parameters | **135,437** |
| TFT parameters | **75,992** |
| Logistic parameters | **8** |
| GRU parameters | **Not logged** — leave blank or compute separately |
| GPR row | **Remove** or state "Not implemented" |
| Windowed Logistic row | **Remove** — not implemented |
| TFT max epochs | **10** (best epochs: 8, 3, 4) |
| MLP / GRU max epochs | **30** with early stopping |

### Suggested Methods paragraph (benchmark fairness)

> Sequence baselines (windowed MLP, GRU, and TFT) were trained with an identical 52-week encoder and 13-week prediction horizon (stride = 1) on the cumulative initiation label `onset_flag`. The windowed MLP flattens the 52 × 7 covariate window into a 364-dimensional vector and outputs 13 logits trained with binary cross-entropy. GRU and TFT consume the same per-step covariates in sequential form. Test-set predictions from overlapping windows were averaged at each time index before population-level Pf(t) aggregation. Logistic regression is reported separately as a pointwise linear baseline using the same covariates and target at each evaluation time. All models were evaluated on 150 held-out series over 731 common population time points (time index ≥ 52). Neural models used seeds 20250111, 20250112, and 20250113; Table 10 reports mean ± standard deviation across seeds.

---

## 10. Final Tables for Document Insertion

### 10.1 Benchmark design summary

| Item | Value |
|---|---|
| Models compared | Logistic Regression, Windowed MLP, GRU, TFT |
| Target | `onset_flag` (cumulative P(Tᵢ ≤ t)) |
| Covariates | 7 (see Section 1) |
| Window / horizon / stride | 52 / 13 / 1 (sequence models) |
| Split | 700 / 150 / 150 series; seed 20250111 |
| Test metrics domain | 731 population time points |
| Leakage controls | No `chloride_rebar`; no target-derived inputs |
| Fair sequence comparison | MLP, GRU, TFT aligned |
| Simple baseline | Pointwise Logistic only |

### 10.2 Final benchmark results (Table 10)

| Model | MAE | RMSE | Seeds |
|---|---|---|---|
| GRU | 0.00193 ± 0.00017 | 0.00293 ± 0.00021 | 3 |
| TFT | 0.00454 ± 0.00026 | 0.00637 ± 0.00065 | 3 |
| Windowed MLP | 0.00697 ± 0.00084 | 0.00996 ± 0.00133 | 3 |
| Logistic Regression | 0.02065 | 0.02427 | 1 |

### 10.3 Model configuration table

| Model | Input task | Architecture | Optimizer / LR | Batch | Max epochs | Early stop | Parameters |
|---|---|---|---|---|---|---|---|
| Logistic | Pointwise (7 features) | StandardScaler + L2 logistic (lbfgs) | — | — | — | — | 8 |
| Windowed MLP | 52×7 → 13 | 256-128-64-13, ReLU, dropout 0.1 | Adam, 1e-3 | 64 | 30 | patience 6 | 135,437 |
| GRU | 52×7 → 13 | GRU-32, 1 layer, linear head | Adam, 1e-3 | 64 | 30 | patience 6 | not logged |
| TFT | 52→13 seq2seq | TFT h=32, heads=4, dropout 0.1 | Adam, 3e-4 | 32 | 10 | patience 3 | 75,992 |

---

## 11. Remaining Gaps

| Gap | Severity | Recommended action |
|---|---|---|
| Benchmark docx not in repo | Low | Copy values from this report into docx manually |
| GRU parameter count missing | Low | Optional one-time count from checkpoint |
| GRU training time not in `final_model_comparison.csv` | Medium | Omit from Table 10 or re-log from training run |
| Logistic runtime not in comparison CSV | Low | Use computational efficiency table (0.46 s end-to-end) |
| TFT 10-epoch vs MLP/GRU 30-epoch cap | Medium | Explicit Methods caveat |
| `final_training_runtime_summary.csv` outdated MLP MAE | **High if misused** | **Do not use** — superseded by windowed MLP results |
| Inference harness vs full benchmark inference differ | Low | Cite benchmark CSV for MLP (~2.6 s); harness for relative ranking only |

---

## 12. Key Source Files

### Scripts

```
scripts/07_train_benchmarks.py          # Windowed MLP, GRU, Logistic; can train TFT
scripts/17_tft_three_seed_benchmark.py  # Final TFT 10-epoch benchmark
scripts/revision_config.py            # Shared constants
scripts/revision_data.py              # Split, leakage checks
scripts/revision_metrics.py           # Pf aggregation, MAE/RMSE
scripts/20_final_computational_efficiency.py  # Runtime remeasurement
```

### Tables and reports

```
outputs/revision/tables/final_model_comparison.csv
outputs/revision/tables/final_model_comparison_by_seed.csv
outputs/revision/tables/final_training_summary.csv
outputs/revision/tables/final_tft_three_seed_results.csv
outputs/revision/tables/final_tft_three_seed_summary.csv
outputs/revision/tables/final_computational_efficiency_summary.csv
outputs/revision/final_benchmark_report.md
outputs/revision/mlp_windowed_benchmark_correction_report.md
outputs/revision/benchmark_fairness_audit_reviewer_comment5.md
```

### Data and predictions

```
data/processed/revision/series_split.csv
data/processed/revision/final_chloride_labeled.parquet
outputs/revision/predictions/final_pf_logistic_regression.csv
outputs/revision/predictions/final_pf_mlp_seed*.csv
outputs/revision/predictions/final_pf_gru_seed*.csv
outputs/revision/predictions/final_pf_tft_seed*.csv
```

### Checkpoints

```
outputs/revision/checkpoints/final_mlp_seed{seed}.pt
outputs/revision/checkpoints/final_gru_seed{seed}.pt
outputs/revision/checkpoints/tft/{seed}_10epoch/best.ckpt
```

---

## 13. Explicit Answers to Audit Questions

1. **Seven covariates:** `Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year` — see Section 1.
2. **Common setup:** 52 / 13 / stride 1 / `onset_flag` / 700-150-150 split / no leakage — see Section 2.
3. **Benchmark results:** See Section 3; sources in Section 3.1.
4. **Seeds:** 20250111, 20250112, 20250113 for MLP/GRU/TFT; none for Logistic — see Section 4.
5. **Model configs:** See Section 5.
6. **Runtime:** Partial in comparison CSV; fuller data in Sections 6.3–6.4; see recommendations in 6.5.
7. **Windowed Logistic:** Not implemented — pointwise only — see Section 7.
8. **GPR:** Not implemented — omission acceptable — see Section 8.
9. **XX placeholders:** Docx not in repo — inferred replacements in Section 9.
10. **Audit summary:** Sections 10–11.

---

*End of report.*
