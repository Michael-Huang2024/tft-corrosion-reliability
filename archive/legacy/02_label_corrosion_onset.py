"""
02_label_corrosion_onset.py

Read simulated chloride long-format data and create:
1) event-style onset flag: once onset happens, it stays 1 afterwards
2) corrosion initiation time per series: t_init_idx, t_init_year (NaN if never)
3) optional survival-style target: time_to_onset (remaining time to onset)

Outputs:
- data/processed/chloride_labeled.parquet  (long-format with event labels)
- data/processed/onset_summary.csv         (one row per series_id with t_i)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _read_sim_data(sim_path: Path) -> pd.DataFrame:
    if not sim_path.exists():
        raise FileNotFoundError(f"Sim data not found: {sim_path}")

    if sim_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(sim_path)
    elif sim_path.suffix.lower() == ".csv":
        df = pd.read_csv(sim_path)
    else:
        raise ValueError(f"Unsupported file type: {sim_path.suffix}")

    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["series_id", "time_idx", "t_year", "chloride_rebar", "C_th"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in sim data: {missing}")

    # Ensure types (best effort)
    df["series_id"] = df["series_id"].astype(int)
    df["time_idx"] = df["time_idx"].astype(int)

    return df


def _ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure cover_mm exists in the dataset.
    Priority:
    - If cover_mm exists -> keep it
    - Else if cover_m exists -> create cover_mm = cover_m * 1000
    - Else -> do nothing (but warn via print)
    """
    if "cover_mm" in df.columns:
        return df

    if "cover_m" in df.columns:
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df

    print("⚠️ Warning: neither 'cover_m' nor 'cover_mm' exists in input. 'cover_mm' will not be created.")
    return df


def _compute_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create:
    - onset_raw: 1 if chloride_rebar >= C_th at that time
    - onset_flag: event-style (once 1, stays 1 afterwards per series)
    """
    df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)

    onset_raw = (df["chloride_rebar"].values >= df["C_th"].values).astype(int)
    df["onset_raw"] = onset_raw

    # event-style: cumulative max per series
    df["onset_flag"] = (
        df.groupby("series_id")["onset_raw"]
        .cummax()
        .astype(int)
    )

    return df


def _build_onset_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    """
    One row per series_id confirmed onset time.
    t_init_idx: first time_idx when onset_raw == 1
    t_init_year: corresponding t_year
    If never onset -> NaN
    """
    # Find first onset per series
    first_onset = (
        labeled.loc[labeled["onset_raw"] == 1, ["series_id", "time_idx", "t_year"]]
        .sort_values(["series_id", "time_idx"])
        .groupby("series_id", as_index=False)
        .first()
        .rename(columns={"time_idx": "t_init_idx", "t_year": "t_init_year"})
    )

    # Build base summary: unique series with static params if present
    keep_cols = ["series_id"]
    for c in ["Cs", "cover_m", "cover_mm", "D28", "m_aging", "C_th"]:
        if c in labeled.columns:
            keep_cols.append(c)

    base = (
        labeled.sort_values(["series_id", "time_idx"])
        .groupby("series_id", as_index=False)[keep_cols]
        .first()
    )

    summary = base.merge(first_onset, on="series_id", how="left")

    # Optional: add boolean whether onset ever observed
    summary["onset_observed"] = summary["t_init_year"].notna()

    return summary


def _add_time_to_onset(labeled: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """
    Add survival-style target:
    time_to_onset (in years): remaining time until initiation time.
    If never onset -> NaN (or could be max horizon; we keep NaN to be explicit)
    """
    labeled = labeled.merge(summary[["series_id", "t_init_year"]], on="series_id", how="left")
    labeled["time_to_onset"] = labeled["t_init_year"] - labeled["t_year"]
    # After onset, remaining time becomes 0 (or negative); clamp at 0 for stability
    labeled["time_to_onset"] = labeled["time_to_onset"].where(labeled["time_to_onset"].notna(), np.nan)
    labeled.loc[labeled["time_to_onset"].notna(), "time_to_onset"] = np.maximum(
        0.0, labeled.loc[labeled["time_to_onset"].notna(), "time_to_onset"].values
    )
    return labeled


def main():
    root = Path(__file__).resolve().parents[1]

    in_parquet = root / "data" / "sim" / "chloride_long.parquet"
    in_csv = root / "data" / "sim" / "chloride_long.csv"

    # Prefer parquet; fallback to csv
    sim_path = in_parquet if in_parquet.exists() else in_csv

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_long = out_dir / "chloride_labeled.parquet"
    out_csv = out_dir / "onset_summary.csv"

    print("📥 Reading simulated data:")
    print(" -", sim_path)

    df = _read_sim_data(sim_path)
    df = _ensure_required_columns(df)

    # ✅ Critical fix: keep / create cover_mm here so it survives into labeled parquet
    df = _ensure_cover_mm(df)

    # Labels
    labeled = _compute_event_labels(df)
    summary = _build_onset_summary(labeled)
    labeled = _add_time_to_onset(labeled, summary)

    # Optional: sanity checks
    if "cover_mm" in labeled.columns:
        # cover_mm should be constant within each series (if generated from cover_m)
        nunique = labeled.groupby("series_id")["cover_mm"].nunique()
        if not (nunique == 1).all():
            print("⚠️ Warning: cover_mm is not constant within some series_id. Please check upstream generation.")

    # Save outputs
    labeled.to_parquet(out_long, index=False)
    summary.to_csv(out_csv, index=False)

    print("✅ Saved labeled long-format:")
    print(" -", out_long, "rows=", len(labeled))
    print("✅ Saved onset summary:")
    print(" -", out_csv, "rows=", len(summary))

    onset_rate = summary["t_init_year"].notna().mean()
    print(f"📊 Onset observed in {onset_rate*100:.1f}% of series")

    # Show a few examples
    print(summary.head(10))


if __name__ == "__main__":
    main()
