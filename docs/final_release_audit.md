# Final GitHub Release Audit

**Date:** 2026-06-30 (post-blocker remediation)  
**Repository:** `tft-corrosion-reliability-main`  
**Target release:** v1.0.0

---

## Verdict

# SAFE TO RELEASE

All blockers from the 2026-06-30 initial audit (B1–B4) have been resolved. The repository is ready for `git commit`, tag `v1.0.0`, and public push after a final `git status` review.

---

## Remediation log

| Blocker | Resolution | Status |
|---|---|---|
| **B1** Git tree not staged | Staged `archive/`, `docs/`, `outputs/`, `data/processed/revision/`, release scripts; `src/legacy/` → `archive/legacy/` recorded as renames; `scripts/06_baseline_comparison.py` → `archive/scripts/` | **Fixed** |
| **B2** Absolute paths in `outputs/paper/tables/*.csv` | Stripped `D:\论文1-Pf(t)\tft-corrosion-reliability-main\` prefix; normalized to relative `outputs/revision/...` paths | **Fixed** |
| **B3** Absolute paths in `docs/results/*.md` | Redacted in `final_tft_three_seed_report.md`, `final_representative_tft_error_report.md`, `final_mc_dropout_50_vs_100_convergence_report.md` | **Fixed** |
| **B4** `environment.json` username / local paths | Replaced with redacted schema: `python_version`, `platform`, `cpu`, `gpu`, `packages` only | **Fixed** |

Additional hygiene:

- Deleted stray `git_status.txt`
- `scripts/validate_release.py` no longer writes absolute root path in report
- `git ls-files --cached` contains zero `.ckpt` / `.pt` / `.joblib` files

---

## Validation (2026-06-30)

```
python scripts/validate_release.py
→ passed=27, missing=0, warnings=1, metrics=PASS
```

| Check | Result |
|---|---|
| README-referenced files | PASS |
| Active script imports (no `src/legacy`) | PASS |
| Files > 50 MB | PASS (max ~7.5 MB) |
| Checkpoints in git index | PASS (none) |
| Absolute paths in `outputs/paper/` | PASS |
| Absolute paths in `docs/results/` | PASS |
| Sensitive paths in `environment.json` | PASS |
| Git staging complete | PASS |
| Expected benchmark metrics | PASS |

---

## Non-blocking warning (documented)

| Item | Note |
|---|---|
| Figure 3 point-level CSV | `outputs/revision/predictions/tft_20250111_10epoch_points.csv` not in active tree; archived at `archive/diagnostics/predictions/`. Curated Figure 3 in `outputs/paper/figures/` is present. Copy archive file before re-running `generate_fig3_revision.py`. |

---

## `environment.json` (redacted schema)

```json
{
  "python_version": "3.11.9",
  "platform": { "system", "release", "version", "machine" },
  "cpu": { "processor", "logical_cores" },
  "ram_gb": 16,
  "gpu": { "name", "vram_mb", "cuda_version", "driver" },
  "packages": { ... }
}
```

Removed: `python_executable`, `git_commit`, `git_branch`, absolute paths, username.

---

## Git staging summary

Staged for v1.0.0 commit:

- `archive/` (legacy, scripts, audit, data)
- `docs/` (all release documentation)
- `outputs/paper/`, `outputs/revision/predictions/`, `outputs/revision/environment.json`
- `data/processed/revision/` (locked paper dataset)
- `LICENSE`, `CITATION.cff`, `run_revision_pipeline.py`, revision `scripts/`, updated `README.md`, `.gitignore`

Not staged (gitignored): `outputs/revision/checkpoints/`, `archive/diagnostics/`

---

## Recommended next commands

```bash
git status                    # confirm staged set
git commit -m "Release v1.0.0: public research reproducibility package."
git tag -a v1.0.0 -m "v1.0.0 public research release"
# git push origin main && git push origin v1.0.0
```

Upload `outputs/revision/checkpoints/final_*` to Zenodo and link DOI in README.

---

*Post-remediation audit. No training executed.*
