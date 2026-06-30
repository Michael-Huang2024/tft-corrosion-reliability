# Benchmark Fairness Audit — Reviewer Comment 5

**Date:** 2026-06-28  
**Scope:** Read-only inspection of code, saved artifacts, and reports. No code changes or training runs were performed.  
**Goal:** Determine whether the current comparison among Logistic Regression, MLP, GRU, and TFT is fair for Reviewer Comment 5.

---

## Executive Summary

Table 10 is driven by the **revision benchmark** (`scripts/07_train_benchmarks.py` + `scripts/17_tft_three_seed_benchmark.py`), **not** by `run_pipeline.py` or `src/legacy/*`.

**Evaluation protocol** (split, target, metrics, Pf aggregation, time range) is aligned across models, but **MLP and Logistic Regression use a different learning task** than GRU/TFT. The benchmark is **not fully fair** for a like-for-like sequence-model comparison.

| Verdict | Detail |
|---|---|
| Shared and defensible | Cumulative target, leakage-free features, series-level split, identical test Pf(t) metrics, 731-point horizon |
| Not fully fair | MLP/Logistic are pointwise single-step models; GRU/TFT are 52→13 windowed sequence models; TFT used shorter training in final Table 10 artifacts |
| GPR | Not implemented; omission is reasonable if stated |
| Practical ranking | GRU best, MLP second, TFT third — not obviously TFT-favorable |

---

## What Actually Powers Table 10

| Artifact | Role |
|---|---|
| `outputs/revision/tables/final_model_comparison.csv` | Table 10 summary (mean ± SD over seeds) |
| `outputs/revision/tables/final_model_comparison_by_seed.csv` | Per-seed MAE/RMSE |
| `outputs/revision/predictions/final_pf_*.csv` | Population Pf(t) curves used to compute metrics |
| `scripts/07_train_benchmarks.py` | Trains/evaluates Logistic, MLP, GRU; can train TFT |
| `scripts/17_tft_three_seed_benchmark.py` | **Final TFT** training (10 epochs), evaluation, and benchmark rebuild |

Legacy files listed in the audit request (`src/legacy/03_train_TFT_onset.py`, `05_cross_validation_vs_paper.py`, etc.) are **stubs or superseded**; they are not used for Table 10.

---

## 1. TFT — Current Implementation

**Canonical code:** `scripts/07_train_benchmarks.py::train_tft`, finalized via `scripts/17_tft_three_seed_benchmark.py`.

| Item | Value |
|---|---|
| **Supervised task** | Binary classification of cumulative initiation indicator `onset_flag` (= P(Tᵢ ≤ t) per scenario) |
| **Encoder window length** | 52 steps (`MAX_ENCODER_LENGTH = 52` in `revision_config.py`) |
| **Prediction horizon** | 13 steps (`MAX_PREDICTION_LENGTH = 13`) |
| **Sliding stride** | 1 (default `--tft-window-stride 1`; training index subsampled only if stride > 1) |
| **Target variable** | `onset_flag` |
| **Input covariates** | Static: `Cs, D28, m_aging, cover_mm, C_th`; known time: `time_idx, t_year`; **no** `chloride_rebar` |
| **Train / validation / test split** | Series-level 70/15/15 (`data/processed/revision/series_split.csv`, seed 20250111) |
| **Training data** | `data/processed/revision/final_chloride_labeled.parquet` |
| **Inference** | All sliding windows on **test series only**; overlapping predictions averaged per `(series_id, time_idx)` |
| **Population-level Pf(t) aggregation** | `revision_metrics.aggregate_population_pf`: test-set mean of `onset_flag` (true) and mean of `p_onset_pred` (pred) at each `t_year` |
| **Metrics** | MAE/RMSE on the population Pf curve via `evaluate_pf_curve` |
| **Evaluation range** | `time_idx ≥ 52` → **731** time points (~3.99–59.95 y) |
| **Table 10 TFT training** | 10-epoch runs in `outputs/revision/checkpoints/tft/{seed}_10epoch/` (not the 40-epoch path in `07_train_benchmarks`) |

### Key code references

- Configuration: `scripts/revision_config.py` — `MAX_ENCODER_LENGTH = 52`, `MAX_PREDICTION_LENGTH = 13`, `TARGET_COLUMN = "onset_flag"`
- Training/inference: `scripts/07_train_benchmarks.py::train_tft`
- Final TFT artifacts: `scripts/17_tft_three_seed_benchmark.py`

---

## 2. MLP — Current Implementation

**Code:** `scripts/07_train_benchmarks.py::train_torch_point_model`, class `MLPClassifier`.

| Question | Answer |
|---|---|
| **Uses 52-step historical input window?** | **No** |
| **Is the 52-step window flattened into a fixed vector?** | **No** |
| **Predicts 13 future steps or only one step/current point?** | **One point per row** (single logit) |
| **Same covariates as TFT?** | **Yes** — `POINT_FEATURES = [Cs, D28, m_aging, cover_mm, C_th, time_idx, t_year]` |
| **Same target?** | **Yes** — `onset_flag` |
| **Same train/validation/test split?** | **Yes** — same `series_split.csv` |
| **Training** | Pointwise BCE on train rows; early stopping on validation rows (max 30 epochs; seeds ran 14–30 epochs) |
| **Test evaluation** | Pointwise inference on test rows with `time_idx ≥ 52` (`restrict_common_evaluation_range`) |
| **Pf / metrics** | Same `evaluate_point_predictions` pipeline as other models |

MLP is a **static + time tabular classifier**, not a windowed sequence model.

### Key code references

- `MLPClassifier`: 7-dim input → hidden 64 → 32 → 1 logit
- `PointDataset` + `train_torch_point_model` in `scripts/07_train_benchmarks.py`

---

## 3. GRU — Current Implementation

**Code:** `scripts/07_train_benchmarks.py::train_gru`, `WindowDataset`, `GRUClassifier`.

| Question | Answer |
|---|---|
| **Same 52-step input window?** | **Yes** |
| **Same 13-step horizon?** | **Yes** — head outputs 13 logits |
| **Stride = 1?** | **Yes** — `for start in range(max_start + 1)` |
| **Same covariates?** | **Yes** — `POINT_FEATURES` at each encoder timestep |
| **Same target?** | **Yes** — 13-step `onset_flag` vector per window |
| **Same split?** | **Yes** |
| **Inference** | Sliding windows on test split only; overlap-averaged like TFT |
| **Pf / metrics** | Same shared evaluation functions |

GRU and TFT are aligned on windowing, horizon, stride, split, target, and evaluation.

### Key code references

- `WindowDataset` — encoder 52, decoder target 13, stride 1
- `GRUClassifier` — GRU + linear head → `MAX_PREDICTION_LENGTH` outputs
- Overlap averaging at inference: `train_gru` lines 403–431

---

## 4. Logistic Regression — Current Implementation

**Code:** `scripts/07_train_benchmarks.py::train_logistic`.

| Item | Value |
|---|---|
| **Pointwise or sliding-window based?** | **Pointwise** |
| **Inputs** | Same 7 `POINT_FEATURES` (standardized via `StandardScaler`) |
| **Target** | `onset_flag` |
| **Split** | Same series-level split |
| **Seeds** | None (deterministic; single row in Table 10) |
| **Test evaluation** | Test rows, `time_idx ≥ 52` |
| **Pf / metrics** | Same aggregation and MAE/RMSE as other models |

**Note:** `scripts/06_baseline_comparison.py` is an older logistic-only script with a **time-cutoff** split; it is **not** used for Table 10.

---

## 5. Gaussian Process Regression

| Status | Detail |
|---|---|
| **Implemented?** | **No** — no GPR / `GaussianProcess` code anywhere in the repository |
| **Acceptable?** | **Yes, with justification.** Reviewer text is *"MLP, GRU, Gaussian process regression, or any other suitable options"* — a subset is fine if you report what was run and why (e.g., scalability on 1,000 × 783 points, windowed multi-horizon setup). Logistic + MLP + GRU already span linear, feedforward, and recurrent baselines. |

Confirmed in `outputs/revision_experiment_audit_v2.md`: Gaussian Process — not found.

---

## 6. Is Table 10 Generated from a Fair Benchmark?

### Cross-model comparison matrix

| Criterion | Logistic | MLP | GRU | TFT | Fair? |
|---|---|---|---|---|---|
| Same input window (52-step history) | No (point) | No (point) | Yes | Yes | **No** |
| Same prediction horizon (13-step) | No (1-step) | No (1-step) | Yes | Yes | **No** |
| Same sliding stride | N/A | N/A | Yes (1) | Yes (1) | Partial |
| Same scenario-level split | Yes | Yes | Yes | Yes | **Yes** |
| Same covariates (no leakage) | Yes | Yes | Yes | Yes | **Yes** |
| Same target (`onset_flag`) | Yes | Yes | Yes | Yes | **Yes** |
| Same MAE/RMSE calculation | Yes | Yes | Yes | Yes | **Yes** |
| Same population-level aggregation | Yes | Yes | Yes | Yes | **Yes** |
| Same evaluation time points | 731 | 731 | 731 | 731 | **Yes** |
| Same training budget | — | ≤30 epochs | ≤30 epochs | **10 epochs** (final TFT) | **No** |

### Saved Table 10 values

Source: `outputs/revision/tables/final_model_comparison.csv`

| Model | MAE (mean ± std) | RMSE (mean ± std) |
|---|---|---|
| GRU | 0.001934 ± 0.000169 | 0.002931 ± 0.000207 |
| MLP | 0.003017 ± 0.000182 | 0.004371 ± 0.000224 |
| TFT | 0.004542 ± 0.000261 | 0.006373 ± 0.000649 |
| Logistic Regression | 0.020652 ± 0.000000 | 0.024271 ± 0.000000 |

Source: `outputs/revision/tables/final_model_comparison_by_seed.csv` — all models report `evaluation_time_points = 731`.

### Interpretation

Evaluation **outcomes** are comparable (same test scenarios, same Pf(t), same metrics). Learning **tasks** are not: GRU/TFT are windowed multi-horizon sequence models; MLP/Logistic are pointwise tabular models. That is the main fairness gap for Reviewer Comment 5.

---

## 7. Manuscript Paragraph (If Current Setup Is Presented as Fair)

Use this **only** if you explicitly frame MLP/Logistic as **non-sequential tabular baselines** (not window-matched to GRU/TFT):

> All surrogate models were evaluated on the same held-out test scenarios (150 independent series, 15% series-level split), the same cumulative initiation target `onset_flag` ≡ P(Tᵢ ≤ t), and the same leakage-free predictors (Cs, D₂₈, m_aging, cover depth, C_th, and calendar time). Sequence models (GRU and TFT) used a 52-step encoder and 13-step prediction horizon with unit stride; test-set predictions from overlapping windows were averaged at each time index before population aggregation. Tabular baselines (logistic regression and MLP) used the same predictors and target at each evaluation time point on the identical test horizon (time index ≥ 52, 731 population evaluation points). Population-level Pf(t) was computed as the test-set mean of scenario-level predicted initiation probabilities at each year, and model accuracy was reported as MAE and RMSE of the Pf(t) curve. Neural sequence models were trained with three random seeds (20250111, 20250112, 20250113); Table 10 reports mean ± standard deviation across seeds.

---

## 8. Unfair Elements and Minimal Correction

### What is unfair

1. **MLP does not use a 52-step window** — contrary to a strict “same information” benchmark.
2. **MLP predicts one step, not 13** — different supervision geometry from GRU/TFT.
3. **Logistic is pointwise** — acceptable as a weak/simple baseline if labeled clearly, but not window-matched.
4. **TFT final runs used 10 epochs** (`17_tft_three_seed_benchmark.py`) while MLP/GRU used up to **30 epochs** (`07_train_benchmarks.py`).

### Minimal correction (recommended)

**Goal:** Give MLP the same 52→13 windowed task as GRU (simplest fix; largest gap).

| Action | Detail |
|---|---|
| **Scripts to modify** | `scripts/07_train_benchmarks.py` — replace `train_torch_point_model` with a windowed MLP: flatten `52 × 7 = 364` inputs **or** use the existing `WindowDataset` + linear head on flattened encoder; output **13** logits; same overlap averaging at inference as GRU |
| **Retrain MLP?** | **Yes** — all 3 seeds |
| **Retrain GRU?** | **No** (already aligned) |
| **Retrain TFT?** | **Optional** — only if you also harmonize epoch budget (e.g., 30 epochs for all); not required for window fairness |
| **Retrain Logistic?** | **Optional** — only if you want a windowed linear baseline; can keep pointwise as “simple baseline” with explicit labeling |
| **Outputs to regenerate** | `final_pf_mlp_seed*.csv`, `final_model_comparison.csv`, `final_model_comparison_by_seed.csv`, `final_benchmark_report.md`, `final_model_error_comparison.png`, `final_population_trajectories_by_model.png` |
| **Manuscript updates** | **Table 10**; methods subsection on benchmark parity; any text claiming MLP uses the same temporal window |

### Alternative (stronger fairness on training budget)

- Retrain **TFT** via `07_train_benchmarks.py` with `--tft-max-epochs 30` (same cap as MLP/GRU), then rebuild Table 10 — fixes epoch-budget mismatch without changing architecture.

---

## Legacy vs Active Pipeline

| Component | Status for Table 10 |
|---|---|
| `run_pipeline.py` → `scripts/03_train_model.py` | **Not used** — temporal-cutoff validation, no MLP/GRU benchmark |
| `src/legacy/03_train_TFT_onset.py` | **Stub** — `print("TODO")` |
| `src/legacy/04_train_TFT_continuous.py` | **Stub** — `print("TODO")` |
| `src/legacy/04b_train_tft_onset_flag.py` | **Superseded** — used `target_onset`, included `chloride_rebar` |
| `src/legacy/05_cross_validation_vs_paper.py` | **Stub** |
| `src/legacy/05b_infer_onset_flag_and_pf.py` | **Superseded** — old checkpoint path, `target_onset`, `chloride_rebar` |
| `src/legacy/05c_rolling_pf_fullcurve_stream.py` | **Superseded** — same old setup |
| `src/legacy/07_plot_pf_compare.py` | **Legacy** — reads `outputs/predictions/pf_true_vs_pred.csv` |
| `src/legacy/08_make_paper_figures.py` | **Legacy** — original manuscript figures |
| `scripts/07_train_benchmarks.py` | **Active** — MLP, GRU, Logistic; defines shared evaluation |
| `scripts/17_tft_three_seed_benchmark.py` | **Active** — final TFT checkpoints and Table 10 rebuild |

---

## Shared Configuration Reference

From `scripts/revision_config.py`:

```python
TARGET_COLUMN = "onset_flag"
POINT_FEATURES = ["Cs", "D28", "m_aging", "cover_mm", "C_th", "time_idx", "t_year"]
TFT_STATIC_REALS = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
TFT_TIME_VARYING_KNOWN_REALS = ["time_idx", "t_year"]
TFT_TIME_VARYING_UNKNOWN_REALS = []
MAX_ENCODER_LENGTH = 52
MAX_PREDICTION_LENGTH = 13
SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REVISION_SEEDS = (20250111, 20250112, 20250113)
```

Evaluation utilities in `scripts/revision_metrics.py`:

- `restrict_common_evaluation_range(df, min_time_idx=52, split_name="test")` — point models
- `aggregate_population_pf` — mean of true/predicted probabilities by `t_year`
- `evaluate_pf_curve` — MAE, RMSE, max error, final-year error

---

## Training Artifact Summary

### MLP / GRU (from `final_training_summary.csv`)

| Model | Seed | Epochs | Best val loss |
|---|---|---|---|
| MLP | 20250111 | 30 | 0.00400 |
| MLP | 20250112 | 14 | 0.00429 |
| MLP | 20250113 | 15 | 0.00415 |
| GRU | 20250111 | 18 | 0.00270 |
| GRU | 20250112 | 15 | 0.00322 |
| GRU | 20250113 | 22 | 0.00259 |

### TFT (from `final_tft_three_seed_results.csv`)

| Seed | Epochs | Best epoch | MAE | RMSE | Checkpoint dir |
|---|---|---|---|---|---|
| 20250111 | 10 | 8 | 0.004543 | 0.006382 | `checkpoints/tft/20250111_10epoch` |
| 20250112 | 7 | 3 | 0.004861 | 0.007164 | `checkpoints/tft/20250112_10epoch` |
| 20250113 | 8 | 4 | 0.004222 | 0.005575 | `checkpoints/tft/20250113_10epoch_test` |

---

## Explicit Answers to Reviewer Comment 5 Questions

1. **TFT task** — 52-step encoder, 13-step horizon, stride 1, target `onset_flag`, leakage-free covariates, series-level 70/15/15 split, overlap-averaged test inference, population Pf(t) = test-set mean by year.

2. **MLP task** — **Pointwise**; no 52-step window; no flattening; **single-step** prediction; same covariates/target/split; same Pf metrics.

3. **GRU task** — **Same** 52-step window, 13-step horizon, stride 1, covariates, target, split as TFT; same evaluation pipeline.

4. **Logistic task** — **Pointwise**; same inputs/target/split/evaluation aggregation.

5. **GPR** — **Not implemented**; acceptable given reviewer’s “or any other suitable options” wording.

6. **Table 10 fairness** — **Partially fair** on evaluation; **not fair** on input window, prediction horizon, or TFT training budget.

7. **Manuscript paragraph** — See Section 7 (use only with explicit tabular-vs-sequence framing).

8. **Minimal correction** — See Section 8; primary fix is windowed 13-step MLP in `07_train_benchmarks.py` + MLP retrain + Table 10 regeneration.

---

## Files Inspected

### Scripts (primary)

- `run_pipeline.py`
- `scripts/07_train_benchmarks.py`
- `scripts/17_tft_three_seed_benchmark.py`
- `scripts/revision_config.py`
- `scripts/revision_data.py`
- `scripts/revision_metrics.py`
- `scripts/06_baseline_comparison.py`
- `scripts/21_representative_tft_table5.py`
- `scripts/03_train_model.py`
- `scripts/04_infer.py`

### Legacy (requested)

- `src/legacy/03_train_TFT_onset.py`
- `src/legacy/04_train_TFT_continuous.py`
- `src/legacy/04b_train_tft_onset_flag.py`
- `src/legacy/05_cross_validation_vs_paper.py`
- `src/legacy/05b_infer_onset_flag_and_pf.py`
- `src/legacy/05c_rolling_pf_fullcurve_stream.py`
- `src/legacy/07_plot_pf_compare.py`
- `src/legacy/08_make_paper_figures.py`

### Outputs (Table 10 / benchmark)

- `outputs/revision/tables/final_model_comparison.csv`
- `outputs/revision/tables/final_model_comparison_by_seed.csv`
- `outputs/revision/tables/final_training_summary.csv`
- `outputs/revision/tables/final_tft_three_seed_results.csv`
- `outputs/revision/tables/final_tft_three_seed_summary.csv`
- `outputs/revision/tables/benchmark_checkpoints.csv`
- `outputs/revision/predictions/final_pf_*.csv`
- `outputs/revision/final_benchmark_report.md`
- `outputs/revision/final_representative_tft_error_report.md`
- `data/processed/revision/series_split.csv`

---

*End of audit report.*
