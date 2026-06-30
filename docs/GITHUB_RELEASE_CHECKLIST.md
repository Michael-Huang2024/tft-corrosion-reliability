# GitHub Release Checklist

**Release:** v1.0.0 — Public research release  
**Date:** 2026-06-30

---

## 1. Files to commit

### Required (code + documentation)

- [ ] `README.md`
- [ ] `LICENSE`
- [ ] `CITATION.cff`
- [ ] `requirements.txt`
- [ ] `run_pipeline.py`
- [ ] `run_revision_pipeline.py`
- [ ] `scripts/` (active revision + demo scripts only; not `archive/scripts/`)
- [ ] `docs/` (all markdown except transient manifests)
- [ ] `.gitignore`

### Required (paper data — ~15 MB)

- [ ] `data/processed/revision/final_chloride_labeled.parquet`
- [ ] `data/processed/revision/final_onset_summary.csv`
- [ ] `data/processed/revision/series_split.csv`
- [ ] `data/processed/.gitkeep`, `data/raw/.gitkeep`, `data/sim/.gitkeep`

### Required (curated paper artifacts)

- [ ] `outputs/paper/figures/` (all PNG/PDF)
- [ ] `outputs/paper/tables/` (all CSV, parquet, JSON)

### Required (predictions for inference-only reproduction)

- [ ] `outputs/revision/predictions/final_pf_*.csv`
- [ ] `outputs/revision/predictions/mc_dropout_population_predictions.csv`
- [ ] `outputs/revision/predictions/mc_dropout_seed20250111_summary_20_50_100.csv`

### Optional but recommended

- [ ] `data/processed/chloride_labeled.parquet`, `onset_summary.csv` (demo pipeline)
- [ ] `outputs/revision/environment.json`
- [ ] `archive/legacy/`, `archive/audit/` (development history; exclude `archive/diagnostics/` — gitignored)

### Do NOT commit (gitignored)

- `outputs/revision/checkpoints/*` (unless using Git LFS — see §3)
- `*.ckpt`, `*.pt`, `*.joblib` (unless LFS)
- `.venv/`, `__pycache__/`, `.idea/`
- `data/sim/*.csv`, `data/sim/*.parquet`
- `archive/diagnostics/`
- `outputs/revision/logs/`

---

## 2. Files to keep local (or external hosting)

| Asset | Size (approx.) | Recommendation |
|---|---:|---|
| 10 final checkpoints | ~25 MB | **Zenodo** release asset or **Git LFS** |
| `archive/diagnostics/predictions/tft_*_10epoch_points.csv` | ~16 MB | Zenodo (needed for Figure 3 / Table 6 regeneration) |
| `archive/diagnostics/` (full) | ~50 MB | Local backup or Zenodo supplement |
| Training logs | small | Keep local only |

---

## 3. Checkpoint hosting decision

| Option | Pros | Cons |
|---|---|---|
| **GitHub only (no checkpoints)** | Small repo; users retrain | Long GPU time (~40+ h TFT) |
| **Git LFS** | Integrated with GitHub | LFS bandwidth quotas; 25 MB fits easily |
| **Zenodo (recommended)** | DOI for data; no LFS limits | Separate download step |

**Recommended for v1.0.0:**

1. Commit code + `outputs/paper/` + `data/processed/revision/` + predictions CSVs.
2. Upload checkpoint bundle to **Zenodo**; link DOI in README and `CITATION.cff`.
3. Keep checkpoints **gitignored** in `.gitignore` unless you enable LFS:

```bash
git lfs install
git lfs track "outputs/revision/checkpoints/final_*"
git add .gitattributes
```

---

## 4. Final validation commands (dry-run, no training)

From repository root:

```bash
# File existence check
python scripts/validate_release.py
# writes docs/release_validation_report.md

# Pipeline dry-run (plotting / tables only)
python run_revision_pipeline.py --skip-training --skip-data-generation --skip-uq --skip-sobol

# Verify metrics file
python -c "import pandas as pd; df=pd.read_csv('outputs/paper/tables/final_model_comparison.csv'); print(df[['model','MAE_mean','RMSE_mean']])"
```

Expected MAE (approximate): GRU 0.001934, TFT 0.004542, MLP 0.006975.

---

## 5. Pre-commit review

- [ ] README states revision pipeline is authoritative
- [ ] No secrets in `environment.json` (paths are OK)
- [ ] `archive/diagnostics/checkpoints/benchmark_checkpoints.csv` not committed (absolute paths)
- [ ] Run `git status` — confirm no accidental `.ckpt` / `.pt` unless LFS
- [ ] Review `docs/release_validation_report.md`

---

## 6. Git commands (do not run automatically)

```bash
# Stage release files (adjust after reviewing git status)
git add README.md LICENSE CITATION.cff requirements.txt .gitignore
git add run_pipeline.py run_revision_pipeline.py scripts/
git add docs/ outputs/paper/ data/processed/
git add outputs/revision/predictions/final_pf_*.csv
git add outputs/revision/predictions/mc_dropout_*.csv
git add archive/legacy/ archive/audit/ archive/scripts/

# Commit
git commit -m "$(cat <<'EOF'
Release v1.0.0: public research reproducibility package.

Authoritative revision pipeline, curated paper artifacts, and documentation
for transformer-based Pf(t) prediction in reinforced concrete bridges.
EOF
)"

# Tag
git tag -a v1.0.0 -m "v1.0.0 public research release"

# Push (when ready)
git push origin main
git push origin v1.0.0
```

---

## 7. GitHub release assets

Upload to GitHub Releases (or Zenodo):

1. `checkpoints-v1.0.0.zip` — all `outputs/revision/checkpoints/final_*`
2. `fig3-points-v1.0.0.zip` — `archive/diagnostics/predictions/tft_20250111_10epoch_points.csv`
3. Optional: full `outputs/paper/` tarball

---

## 8. Post-release

- [ ] Update README with Zenodo DOI
- [ ] Add DOI badge to README
- [ ] Verify clone + `pip install -r requirements.txt` on clean machine
- [ ] Open issue template for reproducibility questions
