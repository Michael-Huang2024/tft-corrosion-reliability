from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =======================
# Inputs you already have
# =======================
PRED_PARQUET = Path("outputs/predictions/onset_flag_pred_point.parquet")

# These are the columns you showed in the screenshot
ID_COL = "series_id"
T_COL = "t_year"               # already in years in your file
PRED_COL = "p_onset1_pred"     # predicted probability of onset=1
TRUE_COL = "target_onset"      # true onset flag (0/1)

# =======================
# Cover-depth mapping file
# =======================
# You need a mapping table: series_id -> cover_mm
# Put it at ONE of these paths (CSV or Parquet). The script will auto-pick the first existing.
COVER_MAP_CANDIDATES = [
    Path("outputs/predictions/series_static.csv"),
    Path("outputs/predictions/series_static.parquet"),
    Path("data/processed/series_static.csv"),
    Path("data/processed/series_static.parquet"),
]

COVER_COL = "cover_mm"         # column name inside the mapping file

# =======================
# Plot settings
# =======================
OUT_DIR = Path("outputs/figures")
OUT_PNG = OUT_DIR / "Fig3_pf_by_cover_depth.png"
OUT_PDF = OUT_DIR / "Fig3_pf_by_cover_depth.pdf"

# fixed bins (mm). adjust to your simulation range
COVER_BINS_MM: List[Tuple[float, float]] = [
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
]


def _find_cover_map() -> Optional[Path]:
    for p in COVER_MAP_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_cover_map(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        m = pd.read_csv(path)
    else:
        m = pd.read_parquet(path)

    if ID_COL not in m.columns or COVER_COL not in m.columns:
        raise ValueError(
            f"Cover map file must contain columns: {ID_COL}, {COVER_COL}\n"
            f"Found columns: {list(m.columns)}\n"
            f"File: {path}"
        )

    m = m[[ID_COL, COVER_COL]].copy()
    m[COVER_COL] = pd.to_numeric(m[COVER_COL], errors="coerce")
    m = m.dropna(subset=[COVER_COL])
    m = m.drop_duplicates(subset=[ID_COL], keep="first")
    return m


def _label_bin(lo: float, hi: float) -> str:
    def fmt(v: float) -> str:
        return str(int(v)) if abs(v - round(v)) < 1e-9 else f"{v:.1f}"
    return f"{fmt(lo)}–{fmt(hi)} mm"


def main():
    if not PRED_PARQUET.exists():
        raise FileNotFoundError(f"Missing: {PRED_PARQUET}")

    df = pd.read_parquet(PRED_PARQUET)

    required = [ID_COL, T_COL, PRED_COL, TRUE_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction parquet missing columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )

    df = df[[ID_COL, T_COL, PRED_COL, TRUE_COL]].copy()
    df[T_COL] = pd.to_numeric(df[T_COL], errors="coerce")
    df[PRED_COL] = pd.to_numeric(df[PRED_COL], errors="coerce")
    df[TRUE_COL] = pd.to_numeric(df[TRUE_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, T_COL, PRED_COL])

    # -----------------------
    # Attach cover depth
    # -----------------------
    cover_map_path = _find_cover_map()
    if cover_map_path is None:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        msg = (
            "\n❌ Cannot plot Fig.3 by cover depth because cover depth is NOT present in onset_flag_pred_point.parquet.\n"
            "You must provide a mapping file with columns:\n"
            f"  - {ID_COL}\n"
            f"  - {COVER_COL}\n\n"
            "Put it at one of these paths (CSV or Parquet):\n"
            + "\n".join([f"  - {p}" for p in COVER_MAP_CANDIDATES])
            + "\n\n"
            "Then rerun:\n"
            "  python src/06_fig3_pf_by_cover_depth.py\n"
        )
        raise FileNotFoundError(msg)

    m = _load_cover_map(cover_map_path)
    df = df.merge(m, on=ID_COL, how="left")

    if df[COVER_COL].isna().all():
        raise ValueError(
            f"After merging cover map ({cover_map_path}), all {COVER_COL} are NaN.\n"
            f"Check that {ID_COL} values match between the prediction parquet and cover map."
        )

    # -----------------------
    # Compute grouped Pf(t)
    # -----------------------
    curves = []
    for lo, hi in COVER_BINS_MM:
        g = df[(df[COVER_COL] >= lo) & (df[COVER_COL] < hi)]
        if g.empty:
            print(f"⚠️ No rows for cover bin {lo}-{hi} mm, skipped.")
            continue

        pf_pred = g.groupby(T_COL)[PRED_COL].mean().sort_index()
        pf_true = g.groupby(T_COL)[TRUE_COL].mean().sort_index()
        curves.append((_label_bin(lo, hi), pf_pred, pf_true))

    if not curves:
        raise RuntimeError("No curves generated. Adjust COVER_BINS_MM to match your cover_mm range.")

    # -----------------------
    # Plot (paper style)
    # -----------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7.5, 4.5))

    # Pred solid; True dashed
    for label, pf_pred, pf_true in curves:
        plt.plot(pf_pred.index, pf_pred.values * 100.0, label=f"{label} (Pred)")
        plt.plot(pf_true.index, pf_true.values * 100.0, linestyle="--", label=f"{label} (True)")

    plt.xlabel("Time (years)")
    plt.ylabel("Probability of corrosion initiation, Pf (%)")
    plt.title("Pf(t) stratified by concrete cover depth")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300)
    plt.savefig(OUT_PDF)
    plt.close()

    print("✅ Saved:", OUT_PNG)
    print("✅ Saved:", OUT_PDF)
    print("✅ Using cover map:", cover_map_path)
    print("Cover bins (mm):", COVER_BINS_MM)


if __name__ == "__main__":
    main()
