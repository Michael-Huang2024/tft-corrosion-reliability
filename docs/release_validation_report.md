# Release Validation Report

**Mode:** dry-run (no training, no data regeneration)
**Root:** repository root (relative paths only)

## Summary

- Passed checks: **27**
- Missing files: **0**
- Warnings: **1**
- Metrics validation: **PASS** — All expected metrics match within tolerance

## Missing files

None.

## Passed checks

- `[release_docs] README.md`
- `[release_docs] LICENSE`
- `[release_docs] CITATION.cff`
- `[release_docs] run_revision_pipeline.py`
- `[release_docs] docs/REPRODUCIBILITY.md`
- `[release_docs] docs/PAPER_ARTIFACTS.md`
- `[release_docs] docs/CHANGELOG.md`
- `[release_docs] docs/GITHUB_RELEASE_CHECKLIST.md`
- `[required_data_and_paper_artifacts] data/processed/revision/final_chloride_labeled.parquet`
- `[required_data_and_paper_artifacts] data/processed/revision/series_split.csv`
- `[required_data_and_paper_artifacts] data/processed/revision/final_onset_summary.csv`
- `[required_data_and_paper_artifacts] outputs/paper/tables/final_model_comparison.csv`
- `[required_data_and_paper_artifacts] outputs/paper/tables/final_sobol_indices_margin.csv`
- `[required_data_and_paper_artifacts] outputs/paper/tables/final_computational_efficiency_summary.csv`
- `[required_data_and_paper_artifacts] outputs/paper/tables/final_mc_dropout_metrics_20_50_100.csv`
- `[required_data_and_paper_artifacts] outputs/paper/figures/Fig3_pf_by_cover_depth.pdf`
- `[required_data_and_paper_artifacts] outputs/paper/figures/Fig3_pf_by_cover_depth.png`
- `[final_checkpoints] outputs/revision/checkpoints/final_tft_seed20250111-epoch=2-val_loss=0.0082.ckpt`
- `[final_checkpoints] outputs/revision/checkpoints/final_tft_seed20250112-epoch=7-val_loss=0.0064.ckpt`
- `[final_checkpoints] outputs/revision/checkpoints/final_tft_seed20250113-epoch=0-val_loss=0.0173.ckpt`
- `[final_checkpoints] outputs/revision/checkpoints/final_mlp_seed20250111.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_mlp_seed20250112.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_mlp_seed20250113.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_gru_seed20250111.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_gru_seed20250112.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_gru_seed20250113.pt`
- `[final_checkpoints] outputs/revision/checkpoints/final_windowed_logistic_regression.joblib`

## Warnings

- generate_fig3_revision.py expects this path; copy from archive before regenerating Fig 3 (`outputs/revision/predictions/tft_20250111_10epoch_points.csv`)

## Exact next git commands

```bash
git status
git add README.md LICENSE CITATION.cff requirements.txt .gitignore
git add run_pipeline.py run_revision_pipeline.py scripts/
git add docs/ outputs/paper/ data/processed/revision/
git add outputs/revision/predictions/final_pf_*.csv outputs/revision/predictions/mc_dropout_*.csv
git add archive/legacy/ archive/scripts/ archive/audit/
git commit -m "Release v1.0.0: public research reproducibility package."
git tag -a v1.0.0 -m "v1.0.0 public research release"
# git push origin main && git push origin v1.0.0  # when ready
```

Upload `outputs/revision/checkpoints/final_*` to Zenodo or enable Git LFS before users need --skip-training.
