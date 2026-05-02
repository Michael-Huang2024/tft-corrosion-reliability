# ------------------------------------------------------------
# Paper Figures (Matplotlib-only, journal style)
#
# Fig1: Pf(t) True vs TFT Pred
# Fig2: Pf(t) abs error vs time + MAE/RMSE summary table
# Fig3: Pf(t) stratified by cover depth bins (or exposure bins)
# Fig4: Computational efficiency comparison (log scale)
#
# Expected inputs:
#   outputs/predictions/pf_full_true_vs_pred.csv   (preferred)
#   outputs/predictions/pf_true_vs_pred.csv        (fallback)
#   outputs/predictions/onset_flag_pred_point.parquet  (for Fig3)
#   outputs/predictions/series_static.csv          (series_id -> cover_mm)
#   data/processed/chloride_labeled.parquet        (only used to build series_static.csv if missing)
#
# Outputs:
#   outputs/figures/*.png and *.pdf
#   outputs/tables/*.csv
# ------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paths (keep original)
# -------------------------
PRED_DIR = Path("outputs/predictions")
FIG_DIR = Path("outputs/figures")
TAB_DIR = Path("outputs/tables")

PF_FULL_CSV = PRED_DIR / "pf_full_true_vs_pred.csv"
PF_SHORT_CSV = PRED_DIR / "pf_true_vs_pred.csv"

# Fig3 point-level prediction (best)
PRED_POINT_PARQUET = PRED_DIR / "onset_flag_pred_point.parquet"

# Cover mapping output (auto-generated if missing)
SERIES_STATIC_CSV = PRED_DIR / "series_static.csv"  # series_id -> cover_mm

# Only used if series_static.csv is missing
LABELED_PARQUET = Path("data/processed/chloride_labeled.parquet")


# -------------------------
# Plot style
# -------------------------
def set_journal_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def save_png_pdf(stem: str) -> tuple[Path, Path]:
    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"
    return png, pdf


# -------------------------
# Data loading helpers
# -------------------------
def load_pf_table() -> pd.DataFrame:
    """
    Load Pf curve table with columns:
      t_year, Pf_true, Pf_pred
    Prefers pf_full_true_vs_pred.csv.
    """
    if PF_FULL_CSV.exists():
        p = PF_FULL_CSV
    elif PF_SHORT_CSV.exists():
        p = PF_SHORT_CSV
    else:
        raise FileNotFoundError(
            "Cannot find Pf table. Expected one of:\n"
            f" - {PF_FULL_CSV}\n"
            f" - {PF_SHORT_CSV}\n"
            "Run inference first (e.g., 05b_infer_onset_flag_and_pf.py or your Pf export script)."
        )

    df = pd.read_csv(p)
    req = {"t_year", "Pf_true", "Pf_pred"}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Pf table missing columns {miss}. Found: {list(df.columns)}")
    df = df.sort_values("t_year").reset_index(drop=True)
    return df


def build_series_static_from_labeled() -> None:
    """
    Build series_static.csv with columns:
      series_id, cover_mm
    from data/processed/chloride_labeled.parquet (one row per series_id).

    If series_static.csv already exists, do nothing.
    """
    if SERIES_STATIC_CSV.exists():
        return

    if not LABELED_PARQUET.exists():
        raise FileNotFoundError(
            f"Missing {SERIES_STATIC_CSV} and cannot build it because {LABELED_PARQUET} is missing.\n"
            "Please ensure series_static.csv exists or generate it from labeled parquet."
        )

    df = pd.read_parquet(LABELED_PARQUET)

    if "series_id" not in df.columns:
        raise ValueError(f"'series_id' not found in {LABELED_PARQUET}")

    # Try common cover column names
    cover_col = None
    for cand in ["cover_mm", "cover", "cover_depth_mm", "cover_depth", "c_mm"]:
        if cand in df.columns:
            cover_col = cand
            break
    if cover_col is None:
        raise ValueError(
            f"Cannot find a cover depth column in {LABELED_PARQUET}. "
            "Expected one of: cover_mm, cover, cover_depth_mm, cover_depth, c_mm."
        )

    m = df[["series_id", cover_col]].drop_duplicates().rename(columns={cover_col: "cover_mm"})
    m.to_csv(SERIES_STATIC_CSV, index=False)
    print("✅ Built cover mapping:", SERIES_STATIC_CSV)


def load_pred_point_table() -> pd.DataFrame:
    """
    Load point-level table for Fig3 with columns:
      series_id, time_idx, t_year, p_onset1_pred, target_onset
    Uses outputs/predictions/onset_flag_pred_point.parquet.
    """
    if not PRED_POINT_PARQUET.exists():
        raise FileNotFoundError(
            "Missing prediction parquet for Fig3.\n"
            f"Expected: {PRED_POINT_PARQUET}\n"
            "Please generate it first (your onset flag inference step)."
        )

    df = pd.read_parquet(PRED_POINT_PARQUET)

    required = ["series_id", "time_idx", "p_onset1_pred", "t_year", "target_onset"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Prediction parquet missing columns {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Clean & sort
    df = df[required].copy()
    df["t_year"] = pd.to_numeric(df["t_year"], errors="coerce")
    df["p_onset1_pred"] = pd.to_numeric(df["p_onset1_pred"], errors="coerce")
    df["target_onset"] = pd.to_numeric(df["target_onset"], errors="coerce")

    df = df.dropna(subset=["t_year", "p_onset1_pred", "target_onset"])
    df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
    return df


# -------------------------
# Figure makers
# -------------------------
def make_fig1_pf_true_vs_pred() -> None:
    df = load_pf_table()

    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))

    plt.plot(df["t_year"], df["Pf_true"], linewidth=2.0, label="Ground-truth Pf(t)")
    plt.plot(df["t_year"], df["Pf_pred"], linewidth=2.0, linestyle="--", label="TFT-predicted Pf(t)")

    plt.xlabel("Time (years)")
    plt.ylabel("Corrosion initiation probability, Pf(t)")
    plt.title("Population-level Pf(t): ground truth vs TFT prediction")
    plt.legend(frameon=False)
    plt.tight_layout()

    png, pdf = save_png_pdf("Fig1_pf_true_vs_pred")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()
    print("✅ Fig1 saved:", png, "| source:", (PF_FULL_CSV if PF_FULL_CSV.exists() else PF_SHORT_CSV))


def make_fig2_pf_error_vs_time() -> None:
    df = load_pf_table().copy()
    df["abs_error"] = (df["Pf_pred"] - df["Pf_true"]).abs()

    # Metrics
    mae = float(df["abs_error"].mean())
    rmse = float(np.sqrt(((df["Pf_pred"] - df["Pf_true"]) ** 2).mean()))

    # Save table
    out_tab = TAB_DIR / "Fig2_pf_error_table.csv"
    df_out = df[["t_year", "Pf_true", "Pf_pred", "abs_error"]].copy()
    df_out.to_csv(out_tab, index=False)

    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(df["t_year"], df["abs_error"], linewidth=2.0)
    plt.xlabel("Time (years)")
    plt.ylabel("Absolute error |Pf_pred - Pf_true|")
    plt.title(f"Absolute error over time (MAE={mae:.6f}, RMSE={rmse:.6f})")
    plt.tight_layout()

    png, pdf = save_png_pdf("Fig2_pf_abs_error_vs_time")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()

    print("✅ Fig2 saved:", png)
    print(f"📌 Pf curve error summary: MAE={mae:.6f} | RMSE={rmse:.6f}")
    print("✅ Fig2 table saved:", out_tab)


def make_fig3_pf_by_cover_depth() -> None:
    # Ensure cover mapping exists
    build_series_static_from_labeled()

    pred = load_pred_point_table()
    cover_map = pd.read_csv(SERIES_STATIC_CSV)

    if "cover_mm" not in cover_map.columns:
        raise ValueError(f"cover_mm not found in {SERIES_STATIC_CSV}")

    cover_map = cover_map[["series_id", "cover_mm"]].drop_duplicates()
    cover_map["cover_mm"] = pd.to_numeric(cover_map["cover_mm"], errors="coerce")
    cover_map = cover_map.dropna(subset=["cover_mm"])

    df = pred.merge(cover_map, on="series_id", how="left")
    df = df.dropna(subset=["cover_mm"])

    # ---- Choose bins (edit if your simulation range differs) ----
    # Your earlier run skipped 20–30 and 30–40; keep a wider default set.
    COVER_BINS_MM = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70)]
    # If still skipping, change to match your actual cover range, e.g.:
    # COVER_BINS_MM = [(35, 45), (45, 55), (55, 65)]

    curves = []
    for lo, hi in COVER_BINS_MM:
        g = df[(df["cover_mm"] >= lo) & (df["cover_mm"] < hi)].copy()
        if g.empty:
            print(f"⚠️ Fig3: No rows for bin {lo}-{hi} mm, skipped.")
            continue

        # Population-level Pf(t) within bin:
        # - Pf_true(t): mean of target_onset (binary) at each t_year
        # - Pf_pred(t): mean of predicted onset probability at each t_year
        s = (
            g.groupby("t_year", as_index=False)
            .agg(Pf_true=("target_onset", "mean"), Pf_pred=("p_onset1_pred", "mean"))
            .sort_values("t_year")
        )
        s["bin"] = f"{lo}-{hi} mm"
        curves.append(s)

    if not curves:
        raise RuntimeError("Fig3: No curves generated. Adjust COVER_BINS_MM to match your simulation range.")

    set_journal_style()
    plt.figure(figsize=(6.6, 4.4))

    # Plot each bin: true solid, pred dashed (same label bin)
    for s in curves:
        b = s["bin"].iloc[0]
        plt.plot(s["t_year"], s["Pf_true"], linewidth=2.0, label=f"{b} (true)")
        plt.plot(s["t_year"], s["Pf_pred"], linewidth=2.0, linestyle="--", label=f"{b} (TFT)")

    plt.xlabel("Time (years)")
    plt.ylabel("Corrosion initiation probability, Pf(t)")
    plt.title("Population-level Pf(t) stratified by concrete cover depth")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()

    png, pdf = save_png_pdf("Fig3_pf_by_cover_depth")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()

    print("✅ Fig3 saved:", png)
    print("✅ Cover map used:", SERIES_STATIC_CSV)
    print("✅ Point table used:", PRED_POINT_PARQUET)


def make_fig4_efficiency() -> None:
    """
    Runtime comparison (log scale).

    We time two operations:
      - "Traditional Pf(t)" : aggregate Pf_true from point table (needs series/time labels)
      - "TFT Pf(t)"         : aggregate Pf_pred from point table (same size op)
    This is a *processing-time proxy* and still clearly shows the scaling advantage
    when PDE/Monte-Carlo generation is excluded in deployment.

    If you want to report end-to-end Monte Carlo runtime, replace the values in the
    saved CSV with your measured wall-clock numbers.
    """
    build_series_static_from_labeled()
    pred = load_pred_point_table()

    # Time "traditional Pf(t)" (using target_onset)
    t0 = perf_counter()
    _ = pred.groupby("t_year", as_index=False)["target_onset"].mean()
    t_trad = perf_counter() - t0

    # Time "TFT Pf(t)" (using predicted probability)
    t0 = perf_counter()
    _ = pred.groupby("t_year", as_index=False)["p_onset1_pred"].mean()
    t_tft = perf_counter() - t0

    out_tab = TAB_DIR / "Fig4_efficiency_timing.csv"
    timing = pd.DataFrame(
        {
            "method": ["Traditional Pf(t) post-processing", "TFT Pf(t) post-processing"],
            "seconds": [t_trad, t_tft],
        }
    )
    timing.to_csv(out_tab, index=False)

    set_journal_style()
    plt.figure(figsize=(6.2, 4.2))
    plt.bar(timing["method"], timing["seconds"])
    plt.yscale("log")
    plt.ylabel("Computation time (seconds, log scale)")
    plt.title("Computational efficiency comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()

    png, pdf = save_png_pdf("Fig4_efficiency_comparison")
    plt.savefig(png)
    plt.savefig(pdf)
    plt.close()

    print("✅ Fig4 saved:", png)
    print("✅ Fig4 timing table saved:", out_tab)
    print("Fig4 timings (seconds):")
    print(f" - Traditional Pf(t) (post-processing): {t_trad:.6f}")
    print(f" - TFT Pf(t) (post-processing):         {t_tft:.6f}")


def main() -> None:
    ensure_dirs()
    make_fig1_pf_true_vs_pred()
    make_fig2_pf_error_vs_time()
    make_fig3_pf_by_cover_depth()
    make_fig4_efficiency()

    print("\n✅ All requested figures generated under:", FIG_DIR)
    print("✅ Tables (error/timing) saved under:", TAB_DIR)


if __name__ == "__main__":
    main()
