# Revision Experiment Audit V2

Manuscript: "Temporal Fusion Transformer Surrogate Modeling of Chloride-Induced Corrosion Initiation Probability in Reinforced Concrete Bridge Infrastructure."

Audit constraints followed: no training, no data generation, no script edits, no output overwrites. The only new repository artifact created by this audit is this report.

## 1. Executive Summary

This checkout is not ready for reviewer-requested revision experiments without correction. The active workflow trains and evaluates both TFT and Logistic Regression on `target_onset`, an instantaneous chloride-threshold exceedance label, not the cumulative initiation event required by `Pf(t) = P(Ti <= t)`. A cumulative label, `onset_flag`, is computed in `scripts/02_label_onset.py` but is not used by active TFT training, active TFT inference, Logistic Regression baseline code, or active figure/table generation.

The existing data are a single-level Monte Carlo sample: 1,000 independent parameter trajectories with 783 time steps each. There is no nested hierarchy of 1,000 realizations per scenario. There are no `scenario_id` or `realization_id` columns. The phrase "1000 Monte Carlo realizations per series/scenario" is not supported by the active code or data.

The repository contains no current `outputs/` artifacts before this report: no checkpoint, no prediction table, no manuscript figures/tables, no runtime table, and no training logs. `run_pipeline.py` can invoke the workflow, but it regenerates data and trains by default. `--skip-training` cannot work in this checkout unless a checkpoint is supplied externally.

Go/no-go: no-go for retraining or reviewer experiments until the target definition, split logic, Ccrit feature policy, and evaluation protocol are corrected.

## 2. Repository Identity and Version

Confirmed repository metadata:

- Git branch: `main`
- Git HEAD: `83b8931d835f7e34580205630a83b31a4e7f28ea`
- Remote: `https://github.com/Michael-Huang2024/tft-corrosion-reliability.git`
- Working tree at audit time: clean before this report was created.
- Recent commits:
  - `83b8931 remove large simulation data`, author/commit date 2026-05-02 00:11:13 -0400
  - `e73033e clean repository`, author/commit date 2026-05-01 23:56:16 -0400

Version interpretation:

- This appears to be a cleaned repository snapshot with active scripts moved to `scripts/`, while exploratory scripts remain in `src/legacy/`.
- I cannot prove whether it is the same, earlier, later, or materially different than a previous package because no previous package artifact was available in this audit. However, the specific concern checklist in Section 14 is largely confirmed, so it is not materially corrected with respect to the main scientific blockers.

Local artifact state:

- `data/processed/chloride_labeled.parquet` and `data/processed/onset_summary.csv` are tracked.
- `data/sim/chloride_long.csv` and `data/sim/chloride_long.parquet` exist locally but `data/sim/` is ignored and the latest commit removed the Parquet file from version control.
- No `outputs/` directory existed before this report. Therefore no current checkpoints, predictions, figures, tables, logs, or runtime outputs were available.
- `.gitignore` ignores `.venv/`, `.idea/`, `__pycache__/`, `lightning_logs/`, `logs/`, `outputs/checkpoints/`, `outputs/predictions/*.parquet`, `outputs/predictions/*.npy`, and `data/sim/`.

## 3. Active Pipeline Map

Active publication workflow:

1. `run_pipeline.py`
   - `run_step(script, extra_args)`
   - Calls `scripts/01_generate_data.py`, `scripts/02_label_onset.py`, optionally `scripts/03_train_model.py`, then `scripts/04_infer.py`, `scripts/05_make_figures.py`.
   - Does not call `scripts/06_baseline_comparison.py`.

2. `scripts/01_generate_data.py`
   - Entry point: `main()`
   - Configuration: `SimConfig`
   - Transport functions:
     - `D_time_dependent(D_ref, t_s, t_ref_s, m)`
     - `chloride_at_depth_erfc(x_m, t_s, Cs, Cb, D_eff)`
     - `simulate_one_series(cfg, rng, series_id)`
   - Outputs:
     - `data/sim/chloride_long.parquet`
     - `data/sim/chloride_long.csv`

3. `scripts/02_label_onset.py`
   - Entry point: `main()`
   - Functions:
     - `read_sim_data(sim_path)`
     - `ensure_required_columns(df)`
     - `ensure_cover_mm(df)`
     - `compute_event_labels(df)`
     - `build_onset_summary(labeled)`
     - `add_time_to_onset(labeled, summary)`
   - Outputs:
     - `data/processed/chloride_labeled.parquet`
     - `data/processed/onset_summary.csv`

4. `scripts/03_train_model.py`
   - Entry point: `main()`
   - Target: `target_onset`
   - Dataset: `TimeSeriesDataSet`
   - Model: `TemporalFusionTransformer.from_dataset(...)`
   - Loss: `CrossEntropy()`
   - Outputs if run:
     - `outputs/checkpoints/tft_onset_flag-*.ckpt`
     - `outputs/checkpoints/best_checkpoint.txt`

5. `scripts/04_infer.py`
   - Entry point: `main()`
   - Checkpoint resolver: `resolve_checkpoint(checkpoint, checkpoint_dir)`
   - Aggregates duplicate rolling-window predictions into `p_onset1_pred`.
   - Calculates `Pf_true` as `df.groupby("t_year")["target_onset"].mean()`.
   - Outputs if run:
     - `outputs/predictions/pf_full_true_vs_pred.csv`
     - `outputs/predictions/onset_flag_pred_point.parquet`
     - `outputs/predictions/series_static.csv`

6. `scripts/05_make_figures.py`
   - Entry point: `main()`
   - Figure functions:
     - `make_fig1(pf, fig_dir)`
     - `make_fig2(pf, fig_dir, table_dir)`
     - `make_fig3(point, static_path, fig_dir)`
     - `make_fig4(point, fig_dir, table_dir)`
   - Outputs if run:
     - `outputs/figures/Fig1_pf_true_vs_pred.png/.pdf`
     - `outputs/figures/Fig2_pf_abs_error_vs_time.png/.pdf`
     - `outputs/figures/Fig3_pf_by_cover_depth.png/.pdf`
     - `outputs/figures/Fig4_efficiency_comparison.png/.pdf`
     - `outputs/tables/Fig2_pf_error_table.csv`
     - `outputs/tables/Fig4_efficiency_timing.csv`

7. `scripts/06_baseline_comparison.py`
   - Not called by `run_pipeline.py`.
   - Implements Logistic Regression only.
   - Writes reviewer-response comparison outputs if run.

Code classification:

- Active code: `run_pipeline.py`, `scripts/01_generate_data.py`, `scripts/02_label_onset.py`, `scripts/03_train_model.py`, `scripts/04_infer.py`, `scripts/05_make_figures.py`.
- Active but outside one-command publication pipeline: `scripts/06_baseline_comparison.py`.
- Legacy code: all files under `src/legacy/`; no active script imports them.
- Experimental/conflicting code: `src/legacy/04_train_tft_time_to_onset.py`, `src/legacy/05_infer_time_to_onset.py`, and related time-to-onset extraction scripts use a different regression target and sometimes include `chloride_rebar` as an input.
- Publication outputs: not present in this checkout before this report.

README consistency:

- README correctly identifies `scripts/` as the canonical workflow and `src/legacy/` as retained for auditability.
- README says `python run_pipeline.py --skip-training --checkpoint outputs/checkpoints/<checkpoint-file>.ckpt` can use an existing checkpoint, but no checkpoint exists in this checkout.
- README says the pipeline fully reproduces reported results, but no generated outputs or checkpoint are included to verify this, and active labels do not match cumulative `Pf(t)`.

## 4. End-to-End Pipeline Reconstruction

Data generation:

- Entry point: `scripts/01_generate_data.py:main()`
- Configuration variables:
  - `SimConfig.n_series = 1000`
  - `SimConfig.years = 60`
  - `SimConfig.dt_weeks = 4`
  - `SimConfig.seed = 20250111`
  - `SimConfig.t_ref_days = 28`
  - `SimConfig.noise_std = 0.01`
  - `SimConfig.Cs_range = (0.21, 1.63)`
  - `SimConfig.Cth_range = (0.09, 0.51)`
  - `SimConfig.cover_range_m = (0.04, 0.11)`
  - `SimConfig.D28_range = (3.5e-12, 9.0e-12)`
  - `SimConfig.m_range = (0.20, 0.45)`

Chloride transport equation:

- Function: `chloride_at_depth_erfc(x_m, t_s, Cs, Cb, D_eff)`
- Equation in code:
  - `Cb + (Cs - Cb) * math.erfc(x_m / (2.0 * sqrt(D_eff * t_s)))`

Time-dependent diffusion:

- Function: `D_time_dependent(D_ref, t_s, t_ref_s, m)`
- Equation in code:
  - If `t_s <= 0` or `t_s < t_ref_s`: `D_ref`
  - Else: `D_ref * (t_ref_s / t_s) ** m`

Threshold and labels:

- Threshold variable: `C_th` in code, corresponding to manuscript `Ccrit`.
- Instantaneous exceedance in generator:
  - `target_onset = (chloride_rebar >= C_th).astype(int)`
- Label script:
  - `onset_raw = (chloride_rebar >= C_th).astype(int)`
  - `onset_flag = groupby("series_id")["onset_raw"].cummax()`
  - `time_to_onset = t_init_year - t_year`, clamped at 0 after observed onset.

Population probability:

- Active inference: `scripts/04_infer.py`
  - `Pf_true = df.groupby("t_year")["target_onset"].mean()`
  - `Pf_pred = pred_point.groupby("t_year")["p_onset1_pred"].mean()`
- This is a population average of instantaneous exceedance, not cumulative initiation, because it uses `target_onset`.

Model training:

- Active TFT: `scripts/03_train_model.py`
- Target: `target_onset`
- Static inputs: `Cs`, `cover_mm`, `D28`, `m_aging`
- Known time-varying inputs: `time_idx`, `t_year`
- Unknown time-varying inputs: none.

Model inference:

- Active TFT inference: `scripts/04_infer.py`
- Uses rolling `TimeSeriesDataSet.from_dataset(base_ds, df, stop_randomization=True)`.
- Loads checkpoint via `TemporalFusionTransformer.load_from_checkpoint(...)`.
- No checkpoint exists in this checkout.

Baseline comparison:

- Only `scripts/06_baseline_comparison.py` exists.
- It trains Logistic Regression on `target_onset` with `FEATURE_COLUMNS = ["Cs", "cover_mm", "D28", "m_aging", "time_idx", "t_year"]`.
- It is not part of `run_pipeline.py`.

Figure/table generation:

- `scripts/05_make_figures.py` generates Fig1-Fig4 and only Fig2/Fig4 tables.
- `scripts/06_baseline_comparison.py` separately writes `Table_model_comparison.csv`, `Table8_computational_efficiency.csv`, and a caption text file.

One-command reproducibility:

- `python run_pipeline.py` can invoke data generation, labeling, training, inference, and figures.
- It will overwrite generated data and outputs if run.
- It requires training unless `--skip-training` is used.
- `--skip-training` is not usable from repository contents alone because no checkpoint exists.
- `scripts/03_train_model.py` hard-codes `accelerator="gpu", devices=1`, so CPU-only reproduction would fail.
- The one-command pipeline does not run `scripts/06_baseline_comparison.py`, so reviewer comparison tables are not included.

## 5. Critical Target-Definition Findings

Search terms found:

- `target_onset`: generator, TFT train, TFT infer, figure aggregation, Logistic Regression baseline, legacy scripts.
- `onset_raw`: label script and labeled data.
- `onset_flag`: label script and labeled data.
- `time_to_onset`: label script and legacy time-to-onset TFT scripts.
- `Pf_true`: inference, figure scripts, baseline script.
- `t_init_year`, `t_init_idx`: label summary.
- Not found in active code: `binary_label`, `failure_flag`, `initiated`, `corrosion_state`, `t_init` as a standalone variable.

Confirmed from `data/processed/chloride_labeled.parquet`:

- Rows: 783,000
- `target_onset == onset_raw`: true for all rows.
- `target_onset == onset_flag`: false.
- `target_onset` 1-to-0 transitions: 3,262.
- Series with at least one `target_onset` 1-to-0 transition: 269.
- `target_onset` 0-to-1 transitions: 3,505.
- Recomputed cumulative `onset_flag` is monotonic per series.

Labels by model/output:

- TFT training label: `target_onset` in `scripts/03_train_model.py`.
- TFT inference reference `Pf_true`: `target_onset` in `scripts/04_infer.py`.
- Logistic Regression label: `target_onset` in `scripts/06_baseline_comparison.py`.
- Manuscript reference `Pf(t)` from active scripts: `target_onset` mean by `t_year`.

Instantaneous vs cumulative:

- `target_onset` and `onset_raw` are instantaneous threshold exceedance labels.
- `onset_flag` is cumulative initiation by time `t`.
- Active `Pf_true` uses instantaneous exceedance, not cumulative initiation.

Monotonicity:

- Instantaneous `Pf_target = mean(target_onset)` has 282 decreasing time steps.
- Minimum one-step decrease in instantaneous `Pf_target`: -0.012.
- Maximum instantaneous `Pf_target`: 0.246.
- Final instantaneous `Pf_target`: 0.243.
- Recomputed cumulative `Pf_flag = mean(onset_flag)` has 0 decreasing time steps.
- Final cumulative `Pf_flag`: 0.269.
- `onset_summary.csv` observed initiation count: 269 series; censored count: 731 series.

Selected `Pf` values from current data:

| Time index | Time years | Instantaneous `Pf_target` | Cumulative `Pf_flag` |
|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000 | 0.000 |
| 260 | 19.931554 | 0.079 | 0.090 |
| 521 | 39.939767 | 0.174 | 0.195 |
| 782 | 59.947981 | 0.243 | 0.269 |

Mathematical consistency:

- The active code is not mathematically consistent with `Pf(t) = P(Ti <= t)` because it uses instantaneous threshold exceedance.
- The repository computes a cumulative label (`onset_flag`) that would be consistent with `P(Ti <= t)`, but active TFT, Logistic Regression, inference, and figures do not use it.
- Added chloride noise in `scripts/01_generate_data.py` can make `chloride_rebar >= C_th` non-monotonic, and current data confirm non-monotonic labels and `Pf`.

## 6. Monte Carlo Hierarchy Findings

Confirmed schema:

- Present: `series_id`
- Not found: `scenario_id`, `realization_id`
- Static per series: `Cs`, `C_th`, `cover_m`, `cover_mm`, `D28`, `m_aging`
- Time-varying per series: `time_idx`, `t_year`, `chloride_rebar`, targets.

Counts from current data:

| Quantity | Count |
|---|---:|
| Unique scenarios | Not present |
| Unique realizations per scenario | Not present |
| Unique `series_id` values | 1,000 |
| Time steps per series | 783 |
| Rows | 783,000 |
| Unique parameter combinations including `C_th` | 1,000 |
| Probability trajectories | One global `Pf(t)` curve in active inference; stratified cover curves in Fig3 |
| Active TFT training sequences | 706,000 |
| Active TFT base sequences | 719,000 |
| Active validation prediction sequences | 1,000 |
| Active rolling inference windows | 719,000 |

Conclusion:

- The actual design is one trajectory per sampled parameter set.
- It is not a nested Monte Carlo design with repeated stochastic realizations under fixed scenario parameters.
- The manuscript claim "1000 Monte Carlo realizations per series/scenario" is unsupported if interpreted as nested realizations per scenario.

## 7. Feature Findings

Parameter existence and model usage:

| Variable | Exists in simulator/data? | Used in simulator? | Used by active TFT? | Used by Logistic Regression? | Used in probability aggregation? |
|---|---|---|---|---|---|
| `Cs` | Yes | Yes, chloride equation | Yes, static real | Yes | Indirect through labels/predictions |
| `D28` | Yes | Yes, `D_time_dependent` | Yes, static real | Yes | Indirect |
| `m` / `m_aging` | Yes as `m_aging` | Yes, aging exponent | Yes, static real | Yes | Indirect |
| cover depth `x` | Yes as `cover_m`, `cover_mm` | Yes, `x_m` in erfc equation | Yes as `cover_mm` | Yes as `cover_mm` | Fig3 stratification |
| `Ccrit` | Yes as `C_th` | Yes, target threshold | No | No | Indirect through labels only |
| time | Yes as `time_idx`, `t_year` | Yes | Yes, known real | Yes | Aggregation key |
| chloride at reinforcement | Yes as `chloride_rebar` | Output of simulator | No in active scripts | No | Used to construct labels; not aggregation input |
| derived transport variable | `target_cont` exists | Derived after simulation | No | No | No |

`Ccrit` findings:

- `C_th` is sampled uniformly in `scripts/01_generate_data.py`.
- `C_th` directly determines `target_onset`, `onset_raw`, `onset_flag`, and `time_to_onset`.
- `C_th` is explicitly forbidden from active TFT inputs by `FORBIDDEN_INPUTS` and absent from `MODEL_STATIC_REALS`.
- `C_th` is excluded from Logistic Regression `FEATURE_COLUMNS`.
- Excluding `C_th` makes the learning problem incomplete for scenario-level prediction because two otherwise identical chloride/transport parameter sets with different thresholds can have different labels.
- All five reviewer-requested physical parameters cannot currently be analyzed in the trained TFT because `C_th` is not a TFT input.

Leakage note:

- Excluding `chloride_rebar` is defensible if the surrogate is intended to avoid using simulator intermediate outputs.
- Excluding `C_th` is not equivalent: `C_th` is a sampled physical threshold parameter and one of the requested sensitivity variables.

## 8. Split and Leakage Findings

Active TFT split:

- File: `scripts/03_train_model.py`
- Logic:
  - `training_cutoff = df["time_idx"].max() - args.max_prediction_length`
  - Training rows: `df[df.time_idx <= training_cutoff]`
  - Validation source rows: `df[df.time_idx > training_cutoff - max_encoder_length - max_prediction_length]`
  - Validation dataset: `TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)`

Default split counts:

- `max(time_idx) = 782`
- `max_prediction_length = 13`
- `max_encoder_length = 52`
- `training_cutoff = 769`
- Training rows: 770,000
- Validation source rows: 78,000, time indices 705-782
- Training `series_id` count: 1,000
- Validation `series_id` count: 1,000
- Train/validation `series_id` intersection: 1,000
- Train/validation parameter-combination overlap: all 1,000 combinations.

Baseline split:

- File: `scripts/06_baseline_comparison.py`
- Logic:
  - Train Logistic Regression on `df[df["time_idx"] <= training_cutoff]`.
  - Predict on all rows of `df`, including training-period rows.
  - Aggregate over all time steps 0-782.

Evaluation range mismatch:

- TFT rolling inference predicts only time indices 52-782 because decoder windows require encoder history.
- TFT `Pf` table time range: about 3.9863 to 59.9480 years.
- Logistic Regression baseline `Pf` time range: 0.0 to 59.9480 years.
- `scripts/06_baseline_comparison.py` computes baseline MAE over all 783 time steps but TFT MAE from the TFT table, which would have 731 time steps. This is not a fair same-range comparison.

Leakage classification:

- Split type: time-based within the same `series_id` and same parameter combinations.
- Same `series_id` appears in train and validation: yes.
- Same parameter combination appears across splits: yes.
- Independent test set: no.
- Predictions evaluated on training-period rows: yes for Logistic Regression; active TFT inference also reconstructs many post-encoder time points from the same series used in training.
- Static features from validation/test scenarios already seen in training: yes.
- Intermediate simulator output leakage into active model inputs: not in active TFT/Logistic inputs because `chloride_rebar` is excluded. Legacy scripts do include `chloride_rebar`.
- Normalization fitted only on training data: TFT `TimeSeriesDataSet.from_dataset(...)` reuses training encoders/scalers; Logistic Regression `StandardScaler` is fitted on training rows only.
- Target/scenario leakage: scenario leakage is present because train/validation share all series and static parameter combinations. Target leakage via `chloride_rebar` is absent in active scripts but present in legacy classifiers.
- Current evaluation tests interpolation/extrapolation in time for seen scenarios, not generalization to unseen scenarios.

Recommended corrected split logic:

- Split by `series_id` or, better, by unique static parameter combinations before creating time windows.
- Use disjoint train/validation/test scenario sets.
- Fit all preprocessing only on train scenarios.
- Evaluate TFT and baselines on the same held-out scenarios, same cumulative target, same time range, same aggregation rule, and same metrics.
- Keep a final untouched test set for manuscript comparison.

## 9. TFT Implementation Findings

Active TFT file: `scripts/03_train_model.py`

| Item | Active value |
|---|---|
| Target | `target_onset` |
| Loss | `CrossEntropy()` |
| Output type | 2-class logits, `output_size=2`; inference applies `softmax(...)[..., 1]` |
| Static variables | `["Cs", "cover_mm", "D28", "m_aging"]` |
| Known time-varying variables | `["time_idx", "t_year"]` |
| Unknown time-varying variables | `[]` |
| Encoder length | `max_encoder_length=52` |
| Prediction horizon | `max_prediction_length=13` |
| Hidden size | 32 |
| Attention heads | 4 |
| Dropout | 0.1 |
| Hidden continuous size | 16 |
| Batch size | 64 training, 128 inference |
| Learning rate | `3e-4` |
| Early stopping | `EarlyStopping(monitor="val_loss", patience=6, mode="min")` |
| Epochs | CLI default `max_epochs=40` |
| Accelerator | `gpu`, `devices=1` |
| Checkpoint monitor | `val_loss`, save top 1 |
| Saved checkpoint pattern | `outputs/checkpoints/tft_onset_flag-{epoch}-{val_loss:.4f}.ckpt` |

Checkpoint status:

- No `outputs/checkpoints/` directory or `.ckpt` files existed in this checkout.
- `best_checkpoint.txt` was not present.
- Existing checkpoint cannot be matched to current scripts/data because no checkpoint exists.
- Checkpoint loading cannot work without retraining or supplying an external checkpoint.

Best validation metrics:

- Not found. There are no logs or checkpoints in this checkout.

Interpretability outputs:

- Variable selection weights: no extraction code found.
- Feature importance: no active extraction code found.
- Attention weights: no active extraction code found.
- Encoder/decoder importance: no active extraction code found.
- PyTorch Forecasting can expose interpretation utilities, but this repository does not call them.

Inference windowing:

- File: `scripts/04_infer.py`
- Uses rolling overlapping windows from `TimeSeriesDataSet.from_dataset(base_ds, df, stop_randomization=True)`.
- It is multi-horizon prediction with `max_prediction_length=13`.
- It creates repeated votes for the same `(series_id, time_idx)` and averages them.

Measured dataset-window counts without training:

- Base sequence count: 719,000.
- Rolling inference window count: 719,000.
- Total decoder votes: 9,347,000.
- Unique predicted `(series_id, time_idx)` keys: 731,000.
- Vote counts:
  - 1-12 votes: 2,000 keys at each count.
  - 13 votes: 707,000 keys.
- Per-key vote count min/max/mean: 1 / 13 / 12.7866.

## 10. Baseline-Model Findings

Existing baseline implementations:

| Model | Found? | File path | Status |
|---|---|---|---|
| Logistic Regression | Yes | `scripts/06_baseline_comparison.py` | Active but not in `run_pipeline.py` |
| MLP | No | Not found | Missing |
| GRU | No | Not found | Missing |
| LSTM | No active implementation | Search found no implementation | Missing |
| XGBoost | No | Not found | Missing |
| Gaussian Process | No | Not found | Missing |
| Random Forest | No | Not found | Missing |
| TFT time-to-onset regression | Yes | `src/legacy/04_train_tft_time_to_onset.py` | Legacy/experimental, not active |

Logistic Regression details:

- File: `scripts/06_baseline_comparison.py`
- Function: `train_logistic_baseline(df, max_prediction_length, max_iter)`
- Inputs: `FEATURE_COLUMNS = ["Cs", "cover_mm", "D28", "m_aging", "time_idx", "t_year"]`
- Target: `target_onset`
- Split: time-based, same `series_id` in train and evaluation.
- Prediction horizon: not a sequence horizon; pointwise probabilities for all rows.
- Evaluation time range: all time indices 0-782.
- Preprocessing: `StandardScaler()` fitted inside `make_pipeline(...)` on training rows.
- Metrics: MAE and RMSE after population aggregation.
- Runtime: no measured model-fit runtime recorded; script uses hard-coded defaults for simulation/training/inference times in Table 8.
- Fairness versus TFT: not fair because target is wrong for cumulative initiation, split shares scenarios, `C_th` is excluded, time range differs, and Logistic Regression predicts all rows pointwise while TFT uses overlapping multi-horizon windows.

No existing MLP/GRU benchmark code is available. New benchmark scripts are required, but they should not be written until target and split corrections are made.

## 11. Sensitivity Readiness

Existing sensitivity implementations:

- Sobol analysis: not found.
- Morris screening: not found.
- Permutation importance: not found.
- Partial dependence: not found.
- SHAP: not found.
- TFT variable importance extraction: not found.
- Parameter sweep plots: only cover-depth stratification in Fig3, not a formal sensitivity analysis.

Dependencies:

- `requirements.txt` includes `scikit-learn` but not `SALib` and not `shap`.
- `sklearn.inspection` tools are available through scikit-learn in principle, but no code uses them.

Parameter bounds in active simulator:

| Parameter | Code variable | Distribution | Bounds |
|---|---|---|---|
| Surface chloride | `Cs` | independent uniform | 0.21 to 1.63 |
| Diffusion at 28 days | `D28` | independent uniform | 3.5e-12 to 9.0e-12 |
| Aging exponent | `m_aging` | independent uniform | 0.20 to 0.45 |
| Cover depth | `cover_m` | independent uniform | 0.04 to 0.11 m |
| Critical chloride | `C_th` | independent uniform | 0.09 to 0.51 |

Sobol readiness:

- Current random samples are not a Sobol/Saltelli design.
- New samples are required for defensible Sobol indices.
- The simulator can support direct five-parameter analysis conceptually because `simulate_one_series(...)` accepts sampled values internally, but the current function samples internally from RNG rather than accepting an explicit parameter vector. A later modification should expose a deterministic evaluator that accepts `Cs`, `D28`, `m_aging`, `cover_m`, and `C_th`.
- Outputs can be evaluated near 20, 40, and 60 years on the existing 4-week grid; exact 20/40/60-year evaluation would require direct calculation at those times or interpolation.

Estimated Saltelli/Sobol model evaluations for `D = 5`:

| Base N | First-order/total only, `N * (D + 2)` | With second-order, `N * (2D + 2)` |
|---:|---:|---:|
| 512 | 3,584 | 6,144 |
| 1,024 | 7,168 | 12,288 |
| 2,048 | 14,336 | 24,576 |

Recommended sensitivity target:

- Primary: cumulative initiation indicator `I(Ti <= t)` evaluated at 20, 40, and 60 years, then Sobol indices for the probability/mean response.
- Secondary: initiation time `Ti` with censoring handled explicitly.
- Optional: full trajectory sensitivity using time-indexed Sobol curves.
- Perform simulator-side Sobol first because it is physically interpretable and includes `C_th`.
- Perform TFT-side sensitivity only after retraining a corrected TFT that includes `C_th` and uses the cumulative target.

## 12. UQ Readiness

Existing UQ implementations:

- Repeated Monte Carlo runs: no repeated independent runs under fixed scenarios.
- Bootstrap: not found.
- Multiple seeds: not found.
- Deep ensembles: not found.
- Quantile regression: not found.
- MC dropout: not implemented.
- Bayesian layers: not found.
- Prediction intervals/confidence intervals: not found.
- Calibration curves: not found.

Current TFT uncertainty character:

- The active TFT is a deterministic classifier after training.
- It outputs softmax class probabilities, not predictive uncertainty intervals.
- Target is binary (`target_onset`), but the active target is instantaneous and scientifically wrong for cumulative initiation.
- Quantile loss is not used.
- Quantile TFT is not scientifically meaningful for the current binary instantaneous target and one-trajectory-per-parameter hierarchy.

Available repeated reference trajectories:

- There are 1,000 sampled parameter trajectories.
- There are no repeated realizations per fixed scenario, so scenario-level aleatory distributions cannot be separated from between-parameter variability.

Potential UQ options:

- Bootstrap over held-out `series_id` can quantify sampling uncertainty of global `Pf(t)` under the sampled population.
- Multiple model seeds can quantify model/optimization uncertainty after corrected train/validation/test splits.
- MC dropout would require deliberate inference-mode dropout handling and a checkpoint; it cannot be extracted from this checkout because no checkpoint exists.
- PICP and MPIW are not valid for current outputs because there are no prediction intervals and the target/aggregation structure does not define interval targets.

Recommended UQ design:

1. Correct target to cumulative `onset_flag` and use held-out scenarios.
2. Report nonparametric bootstrap confidence bands for simulator reference `Pf(t)` by resampling series.
3. Train a small ensemble of corrected TFT/MLP/GRU seeds and separate model uncertainty from Monte Carlo sampling uncertainty.
4. Use calibration metrics for probabilistic binary predictions on held-out scenarios.
5. Avoid quantile TFT unless the response is redefined as continuous `Ti` with appropriate censoring treatment.

## 13. Runtime and Computational-Advantage Readiness

Runtime values found:

- `scripts/01_generate_data.py`: prints `[Timing] Simulation: ...` using `time.perf_counter()`.
- `scripts/03_train_model.py`: prints `[Timing] Model training: ...` using `time.perf_counter()`.
- `scripts/04_infer.py`: prints `[Timing] Inference: ...` using `time.perf_counter()`.
- `scripts/05_make_figures.py`: measures only aggregation of `target_onset` vs `p_onset1_pred` with `perf_counter()` and writes `Fig4_efficiency_timing.csv`.
- `scripts/06_baseline_comparison.py`: hard-coded CLI defaults:
  - `--simulation-seconds = 5.3482`
  - `--training-seconds = 49161.4134`
  - `--inference-seconds = 612.0890`

Timing audit:

- Measured vs hard-coded: Table 8 uses hard-coded defaults for simulation, training, and inference unless user overrides arguments.
- Included stages:
  - Simulation print includes simulation plus file writing.
  - Training print includes data loading, dataset creation, training, checkpoint writing.
  - Inference print includes data loading, checkpoint loading, rolling inference, aggregation, file writing.
  - Fig4 only times two Pandas groupby aggregations.
- CPU/GPU mixing: yes. Training is GPU-only in active code; inference selects CUDA if available; figure aggregation is CPU.
- `torch.cuda.synchronize()`: not found.
- Warm-up runs: not found.
- Repeated timings: not found.
- Mean/std timings: not found.
- Hardware metadata: not recorded.
- Peak memory: not recorded.
- Excessive overlapping inference windows: yes, 9,347,000 decoder votes for 731,000 unique predicted points.
- Current hard-coded values imply simulation generation (5.3482 s) is much faster than TFT inference (612.0890 s), so they do not support a computational advantage claim.

Runtime scaling readiness:

- Current scripts are not designed for controlled scaling at 1, 10, 100, 1,000, 10,000, or 100,000 scenarios.
- `scripts/01_generate_data.py` can change `--n-series`, but downstream training/inference/plotting overwrite fixed paths and no benchmark harness records repeated measurements.
- No protocol exists to separate simulation, data loading, model loading, inference, aggregation, and plotting.

Recommended fair benchmark protocol:

- Use corrected cumulative target and fixed held-out scenario sets.
- Benchmark simulator-only evaluation and surrogate-only inference for identical scenario counts.
- Include 1, 10, 100, 1,000, 10,000, and 100,000 scenarios where memory permits.
- Separate timing stages: parameter generation, chloride simulation, data loading, model loading, inference, aggregation, plotting, file writing.
- Use CPU-only and GPU-only modes separately.
- Add warm-up runs, repeated runs, mean/std, hardware metadata, and peak memory.
- Use `torch.cuda.synchronize()` around GPU timing.
- Avoid overlapping-window inference for pointwise `Pf(t)` if a more direct decoder design is sufficient.

## 14. Manuscript-Code Consistency Table

| Manuscript/README claim | Actual code behavior | Consistent? | Evidence path | Required action |
|---|---|---|---|---|
| Target represents corrosion initiation by time `t` | Active target is instantaneous `chloride_rebar >= C_th`; cumulative `onset_flag` is computed but not used | No | `scripts/01_generate_data.py`, `scripts/02_label_onset.py`, `scripts/03_train_model.py` | Train/evaluate on `onset_flag` or explicit `I(Ti <= t)` |
| `Pf(t) = P(Ti <= t)` | Active `Pf_true` is `mean(target_onset)` | No | `scripts/04_infer.py` | Aggregate `onset_flag` |
| 1,000 scenarios/series | Data contain 1,000 `series_id` values | Partially | `data/processed/chloride_labeled.parquet` | State as 1,000 sampled trajectories |
| 1,000 realizations per scenario | No scenario/realization hierarchy exists | No | all scripts/data schema | Do not claim nested MC; implement if needed |
| Input distributions documented | Code uses independent uniform ranges | Partially | `scripts/01_generate_data.py` | Align manuscript with code or implement stated distributions |
| Parameter ranges | Code ranges confirmed | Yes if manuscript matches these exact ranges | `SimConfig` | Verify manuscript text |
| Dataset size | Current data: 783,000 rows | Yes if manuscript says 1,000 x 783 | data inspection | State exact time grid |
| Split method | Time-based split within same series | No if claiming independent validation/test | `scripts/03_train_model.py` | Split by held-out series/scenarios |
| Independent test set | None | No | all active scripts | Add test split |
| `Ccrit` inclusion | `C_th` sampled and labels depend on it, but models exclude it | No for five-parameter study | `scripts/03_train_model.py`, `scripts/06_baseline_comparison.py` | Include `C_th` for corrected models/sensitivity |
| TFT loss | CrossEntropy classification | Yes if claiming binary classifier | `scripts/03_train_model.py` | Keep after target correction |
| TFT output | 2-class logits converted to softmax probability | Yes | `scripts/04_infer.py` | Clarify |
| Prediction horizon | 13 time steps, about one year | Yes if manuscript states this | `scripts/03_train_model.py` | Clarify time-step duration |
| Logistic Regression comparison | Logistic Regression exists but outside `run_pipeline.py`; wrong target and range mismatch | No | `scripts/06_baseline_comparison.py` | Rebuild fair benchmarks |
| MLP/GRU comparison | Not implemented | No | repository search | Add after corrections |
| Runtime values | Some printed, Table 8 defaults hard-coded | No for measured benchmark claims | `scripts/06_baseline_comparison.py` | Implement benchmark harness |
| Computational advantage | Current hard-coded numbers imply simulator faster than TFT inference | No | `scripts/06_baseline_comparison.py` | Re-benchmark fairly |
| Uncertainty quantification | No UQ implementation | No | repository search | Add bootstrap/ensembles/calibration after target fix |

## 15. Comparison With Previous Concern Checklist

This section does not assume a prior audit is correct; it independently checks each concern in the current repository.

| Concern | Status | Evidence |
|---|---|---|
| Instantaneous `target_onset` used instead of cumulative `onset_flag` | Confirmed | `scripts/03_train_model.py` target is `target_onset`; Parquet shows `target_onset == onset_raw` and not `onset_flag` |
| Non-monotonic `Pf(t)` | Confirmed | `mean(target_onset)` has 282 decreasing time steps; cumulative recomputation has 0 |
| No nested Monte Carlo realizations per scenario | Confirmed | No `scenario_id` or `realization_id`; 1,000 unique parameter combinations for 1,000 series |
| Same series used in train and validation | Confirmed | Train/validation `series_id` intersection is 1,000 of 1,000 |
| No independent test set | Confirmed | No active test split or test artifact |
| `Ccrit` excluded from TFT | Confirmed | `C_th` not in `MODEL_STATIC_REALS`; included in `FORBIDDEN_INPUTS` |
| Logistic and TFT evaluated over different time ranges | Confirmed | Logistic 0-782; TFT rolling predictions 52-782 |
| Hard-coded runtime values | Confirmed | `scripts/06_baseline_comparison.py` defaults 5.3482, 49161.4134, 612.0890 |
| Excessive overlapping TFT inference windows | Confirmed | 719,000 rolling windows, 9,347,000 decoder votes, most keys have 13 votes |
| Quantile UQ not supported by current target structure | Confirmed | No quantile loss/code; binary instantaneous target and no repeated scenario hierarchy |

## 16. Critical Blockers

1. Active target definition is wrong for cumulative corrosion initiation probability.
2. Reference `Pf_true` is non-monotonic and not `P(Ti <= t)`.
3. Monte Carlo hierarchy does not support "realizations per scenario" claims.
4. `C_th`/`Ccrit` is excluded from model inputs despite defining the label and being required for sensitivity.
5. Train/validation split shares every series and parameter combination.
6. No independent test set exists.
7. No checkpoint or manuscript outputs exist in this checkout.
8. Baseline comparison is incomplete and unfair.
9. Runtime evidence is insufficient and partly hard-coded.
10. UQ and sensitivity implementations are absent.

## 17. Recommended Correction Order

1. Define the scientific target explicitly as cumulative initiation: use `onset_flag` or `I(Ti <= t)` consistently.
2. Regenerate labels and all reference `Pf_true` outputs from the cumulative target.
3. Decide feature policy and include `C_th` for all five-parameter revision experiments.
4. Replace time-only split with scenario/series-level train/validation/test splits.
5. Rebuild TFT inference to avoid unnecessary duplicate votes or document and justify the voting scheme.
6. Implement fair Logistic Regression, MLP, GRU, and TFT benchmarks using the same target, inputs, held-out scenarios, time range, aggregation, and metrics.
7. Add simulator-side Sobol sensitivity using explicit parameter vectors.
8. Add UQ bands via bootstrap and model-seed ensembles.
9. Add a runtime benchmark harness with repeated CPU/GPU timings, synchronization, hardware metadata, and memory.
10. Only then retrain and regenerate manuscript figures/tables.

## 18. Go/No-Go Recommendation for Revision Experiments

| Experiment | Recommendation | Reason |
|---|---|---|
| MLP/GRU comparison | No-go | Current target/split/evaluation are invalid; MLP/GRU code is absent |
| Five-parameter sensitivity | No-go for current TFT; conditional go for simulator after adding explicit evaluator | `C_th` excluded from TFT; no Sobol design/code |
| Uncertainty quantification | No-go | No UQ code, no checkpoint, no repeated hierarchy, wrong target |
| Runtime scaling | No-go | No benchmark harness; runtime values hard-coded/mixed; overlapping inference is inefficient |

First correction required before any retraining: change the active model target and all `Pf_true` calculations from instantaneous `target_onset` to cumulative initiation (`onset_flag` or an equivalent `I(Ti <= t)`), then rebuild splits by held-out series/scenarios.

## 19. Exact Files That Would Need Modification Later

Do not modify these until the correction phase is approved.

- `scripts/01_generate_data.py`
  - Optional: expose deterministic simulation from explicit parameter vectors for Sobol and scaling.
  - Optional: separate chloride noise policy from cumulative initiation target.

- `scripts/02_label_onset.py`
  - Ensure canonical target naming cannot be confused.
  - Optionally write `target_cumulative` or replace `target_onset` with cumulative semantics.

- `scripts/03_train_model.py`
  - Use cumulative target.
  - Include `C_th` if five-parameter learning/sensitivity is required.
  - Replace time-only split with held-out scenario splits.
  - Make accelerator configurable.
  - Add test-set evaluation.

- `scripts/04_infer.py`
  - Use cumulative reference `Pf_true`.
  - Align inference time range with baselines.
  - Reduce or justify overlapping duplicate votes.
  - Add optional interpretability extraction.

- `scripts/05_make_figures.py`
  - Ensure all figures/tables use cumulative `Pf`.
  - Remove misleading runtime comparison or feed measured benchmark outputs.

- `scripts/06_baseline_comparison.py`
  - Add MLP and GRU benchmarks or split into benchmark modules.
  - Use same target, features, split, time range, aggregation, metrics as TFT.
  - Remove hard-coded runtime defaults from manuscript tables.

- New file needed later, suggested:
  - `scripts/07_sensitivity_analysis.py`
  - `scripts/08_uncertainty_quantification.py`
  - `scripts/09_runtime_scaling.py`

## 20. Concise Terminal Summary

Report path: `outputs/revision_experiment_audit_v2.md`

Repository difference from previous package: unable to prove without the previous package artifact. This checkout is a clean Git snapshot at `83b8931d835f7e34580205630a83b31a4e7f28ea`; the main prior-concern checklist remains confirmed, so it is not materially corrected with respect to the critical blockers.

Five most important findings:

1. Active TFT, Logistic Regression, and `Pf_true` use instantaneous `target_onset`, not cumulative `onset_flag`.
2. Active `Pf_true` is non-monotonic with 282 decreasing time steps; cumulative `Pf` would be monotonic.
3. The data are 1,000 single parameter trajectories, not 1,000 realizations per scenario.
4. `C_th`/`Ccrit` defines the label but is excluded from TFT and Logistic Regression inputs.
5. There is no independent test set, no checkpoint, no existing manuscript outputs, no UQ implementation, and runtime values are not defensible.

Can new experiments safely begin: no.

First correction before retraining: make the cumulative initiation target the active target everywhere and recompute `Pf_true` as `mean(onset_flag)` or equivalent `P(Ti <= t)`.
