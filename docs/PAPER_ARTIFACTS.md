# Paper Artifact Map

**Manuscript:** *Transformer-Based Sequence Learning for Multi-Horizon Corrosion Initiation Probability Prediction in Reinforced Concrete Bridges*

This table maps manuscript figures and tables to repository files and generating scripts. **Curated release copies** live under `outputs/paper/`; scripts write primary outputs to `outputs/revision/` during regeneration.

**Legend — Regeneration:**

| Tag | Meaning |
|---|---|
| **Regenerated** | Produced by running the listed script |
| **Derived** | Computed from regenerated outputs (may need a short manual/scripted aggregation step) |
| **Archived reference** | Shipped from the reference run; full regeneration requires archived inputs or long retraining |
| **Manually drawn** | Not auto-generated in this repository |

---

## Figures

### Figure 1 — Population-level Pf(t) by model

| Field | Value |
|---|---|
| Manuscript artifact | Figure 1: Population-level corrosion initiation probability trajectories |
| Curated output | `outputs/paper/figures/final_population_trajectories_by_model.png` |
| Script source | `scripts/07_train_benchmarks.py` (`write_population_plot`); rebuilt by `scripts/17_tft_three_seed_benchmark.py rebuild-benchmark` |
| Regeneration | **Regenerated** (requires `final_pf_*.csv` per model) |

### Figure 2 — Model error comparison

| Field | Value |
|---|---|
| Manuscript artifact | Figure 2: Absolute error in Pf(t) vs time |
| Curated output | `outputs/paper/figures/final_model_error_comparison.png` |
| Script source | `scripts/07_train_benchmarks.py` (`write_comparison_plot`) |
| Regeneration | **Regenerated** |

### Figure 3 — Cover-depth-stratified simulator vs TFT Pf(t)

| Field | Value |
|---|---|
| Manuscript artifact | Figure 3: Cover-depth-stratified comparison between simulator-derived and TFT-predicted Pf(t) |
| Curated output | `outputs/paper/figures/Fig3_pf_by_cover_depth.pdf`, `.png` |
| Script source | `scripts/generate_fig3_revision.py` (revision pipeline); legacy equivalent in `archive/legacy/06_fig3_pf_by_cover_depth.py` and `scripts/05_make_figures.py` (demo pipeline) |
| Regeneration | **Regenerated** — requires point-level file `outputs/revision/predictions/tft_20250111_10epoch_points.csv` (reference copy in `archive/diagnostics/predictions/`) |

### Figure 4 — Computational efficiency / accuracy–runtime tradeoff

| Field | Value |
|---|---|
| Manuscript artifact | Figure 4: Accuracy vs runtime (computational efficiency) |
| Curated output | `outputs/paper/figures/computational_efficiency_accuracy_vs_runtime.png` |
| Supporting outputs | `computational_efficiency_total_runtime.png`, `computational_efficiency_inference_only.png` |
| Script source | `scripts/20_final_computational_efficiency.py` |
| Regeneration | **Regenerated** (loads existing checkpoints; remeasures runtime) |

### Supplemental figures (paper / supplement)

| Manuscript role | Curated output | Script | Regeneration |
|---|---|---|---|
| TFT three-seed trajectories | `final_tft_three_seed_trajectories.png` | `scripts/17_tft_three_seed_benchmark.py` | Regenerated |
| TFT seed metric variability | `final_tft_seed_metric_variability.png` | `scripts/17_tft_three_seed_benchmark.py build-tft-summary` | Regenerated |
| Representative TFT time error (Table 5 companion) | `final_tft_time_dependent_error_seed20250111.png` | `scripts/21_representative_tft_table5.py` | Regenerated |
| MC Dropout uncertainty band | `mc_dropout_uncertainty_band_revised.png` | `scripts/22_plot_mc_dropout_revised_figure.py` (from `09` CSV) | Regenerated (plot only) |
| MC Dropout convergence | `mc_dropout_mean_convergence_20_50_100.png`, etc. | `scripts/18_mc_dropout_convergence.py` | Regenerated |
| Sobol S1/ST margins | `sobol_s1_margin_20_40_60.png`, `sobol_st_margin_20_40_60.png` | `scripts/19_final_sobol_sensitivity.py` | Regenerated |
| Sobol time evolution | `sobol_time_evolution.png` | `scripts/19_final_sobol_sensitivity.py` | Regenerated |
| Parameter screening (methods) | `parameter_candidate_pf_curves.png`, `parameter_candidate_cover_groups.png` | `archive/scripts/13_screen_parameter_candidates.py` | **Archived reference** (tables in `outputs/paper/tables/`) |

---

## Tables

### Table 1 — Locked input parameter distributions

| Field | Value |
|---|---|
| Manuscript artifact | Table 1: Input parameter distributions (Candidate C, locked) |
| Curated output | `docs/results/final_locked_parameter_configuration.md` |
| Supporting data | `outputs/paper/tables/final_input_distribution_summary.csv`, `parameter_candidate_C_distribution_summary.csv` |
| Script source | `scripts/12_generate_final_revision_data.py` + `scripts/revision_config.py` |
| Regeneration | **Regenerated** (data gen + distribution summary from script 12) |

### Table 2 — Parameter candidate screening

| Field | Value |
|---|---|
| Manuscript artifact | Table 2: Prespecified parameter candidate screening (A/B/C) |
| Curated output | `outputs/paper/tables/parameter_candidate_screening.csv` |
| Script source | `archive/scripts/13_screen_parameter_candidates.py` |
| Regeneration | **Archived reference** (screening completed pre-lock; re-run from archive script if needed) |

### Table 3 — Dataset and split summary

| Field | Value |
|---|---|
| Manuscript artifact | Table 3: Final dataset dimensions and train/val/test split |
| Curated output | `docs/results/final_data_generation_report.md`, `docs/results/final_split_validation_report.md` |
| Supporting data | `data/processed/revision/series_split.csv`, `outputs/paper/tables/final_training_summary.csv` |
| Script source | `scripts/12_generate_final_revision_data.py`, `scripts/revision_data.py` |
| Regeneration | **Regenerated** |

### Table 4 — Full model comparison (test MAE / RMSE)

| Field | Value |
|---|---|
| Manuscript artifact | Table 4: Benchmark model comparison on held-out test split |
| Curated output | `outputs/paper/tables/final_model_comparison.csv`, `final_model_comparison_by_seed.csv` |
| Script source | `scripts/07_train_benchmarks.py`, `scripts/17_tft_three_seed_benchmark.py rebuild-benchmark` |
| Regeneration | **Regenerated** |

### Table 5 — Representative TFT accuracy

| Field | Value |
|---|---|
| Manuscript artifact | Table 5: Representative TFT (seed 20250111) test accuracy |
| Curated output | `outputs/paper/tables/final_representative_tft_accuracy_table5.csv` |
| Script source | `scripts/21_representative_tft_table5.py` |
| Regeneration | **Regenerated** |

### Table 6 — Cover-depth groups at final evaluation year

| Field | Value |
|---|---|
| Manuscript artifact | Table 6: Broad cover-depth reference vs predicted Pf at final year |
| Curated output | *Not a standalone CSV in release* — values documented in `archive/audit/Table6_verification_report.md` |
| Script source | **Derived** from same inputs as Figure 3 (`generate_fig3_revision.py` data + aggregation per audit report) |
| Regeneration | **Derived** — restore `tft_20250111_10epoch_points.csv` and apply Table 6 aggregation logic |

### Table 7 — TFT three-seed variability

| Field | Value |
|---|---|
| Manuscript artifact | Table 7: TFT metrics across seeds 20250111/12/13 |
| Curated output | `outputs/paper/tables/final_tft_three_seed_summary.csv`, `final_tft_three_seed_results.csv` |
| Script source | `scripts/17_tft_three_seed_benchmark.py build-tft-summary` |
| Regeneration | **Regenerated** |

### Table 8 — Computational efficiency

| Field | Value |
|---|---|
| Manuscript artifact | Table 8: Training and inference runtime comparison |
| Curated output | `outputs/paper/tables/final_computational_efficiency_summary.csv`, `final_computational_efficiency_scaling.csv` |
| Script source | `scripts/20_final_computational_efficiency.py` |
| Regeneration | **Regenerated** |

### Table 9 — Sobol global sensitivity indices

| Field | Value |
|---|---|
| Manuscript artifact | Table 9: Sobol S1 and ST indices at 20/40/60 years |
| Curated output | `outputs/paper/tables/final_sobol_indices_margin.csv`, `final_sobol_indices_binary.csv`, `final_sobol_rank_summary.csv` |
| Script source | `scripts/19_final_sobol_sensitivity.py` |
| Regeneration | **Regenerated** |

### Table 10 — MC Dropout uncertainty metrics

| Field | Value |
|---|---|
| Manuscript artifact | Table 10: MC Dropout convergence and uncertainty calibration metrics |
| Curated output | `outputs/paper/tables/final_mc_dropout_metrics_20_50_100.csv`, `final_mc_dropout_convergence_20_50_100.csv` |
| Script source | `scripts/09_mc_dropout_uq.py`, `scripts/18_mc_dropout_convergence.py` |
| Regeneration | **Regenerated** |

---

## Checkpoints and predictions (not tables, required for reproduction)

| Role | Path | Script |
|---|---|---|
| TFT × 3 seeds | `outputs/revision/checkpoints/final_tft_seed202501{11,12,13}-*.ckpt` | `scripts/14_tft_stable.py`, `scripts/17_tft_three_seed_benchmark.py` |
| MLP × 3 | `outputs/revision/checkpoints/final_mlp_seed*.pt` | `scripts/07_train_benchmarks.py` |
| GRU × 3 | `outputs/revision/checkpoints/final_gru_seed*.pt` | `scripts/07_train_benchmarks.py` |
| Windowed LR | `outputs/revision/checkpoints/final_windowed_logistic_regression.joblib` | `scripts/23_windowed_logistic_baseline.py` |
| Population Pf curves | `outputs/revision/predictions/final_pf_*.csv` | Benchmark / TFT evaluation scripts |

---

## Items not auto-generated in this repository

| Item | Status |
|---|---|
| Journal-composed figure panels | **Manually drawn** / composed in manuscript software |
| Table 6 standalone CSV | **Derived** — documented in audit report; no dedicated export script in active pipeline |
| Response-to-reviewers text | **Manually drawn** — see `archive/audit/` and `docs/results/` |

---

## Quick regeneration command

```bash
python run_revision_pipeline.py --skip-training --skip-data-generation
```

Full reproduction (including training):

```bash
python run_revision_pipeline.py
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for details.
