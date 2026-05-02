"""
Label corrosion initiation onset for the manuscript pipeline.

This script implements Step 2: corrosion initiation labeling. It reads the
simulated chloride time series and writes the labeled long-format dataset used
for TFT training and population-level corrosion initiation probability Pf(t).

Generated files:
- data/processed/chloride_labeled.parquet
- data/processed/onset_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read_sim_data(sim_path: Path) -> pd.DataFrame:
    if not sim_path.exists():
        raise FileNotFoundError(f"Simulated data not found: {sim_path}")
    if sim_path.suffix.lower() == ".parquet":
        return pd.read_parquet(sim_path)
    if sim_path.suffix.lower() == ".csv":
        return pd.read_csv(sim_path)
    raise ValueError(f"Unsupported file type: {sim_path.suffix}")


def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = ["series_id", "time_idx", "t_year", "chloride_rebar", "C_th"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in simulated data: {missing}")
    df["series_id"] = df["series_id"].astype(int)
    df["time_idx"] = df["time_idx"].astype(int)
    return df


def ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    if "cover_mm" in df.columns:
        return df
    if "cover_m" in df.columns:
        df = df.copy()
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df
    print("Warning: neither 'cover_m' nor 'cover_mm' exists in input data.")
    return df


def compute_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["series_id", "time_idx"]).reset_index(drop=True)
    df["onset_raw"] = (df["chloride_rebar"].values >= df["C_th"].values).astype(int)
    df["onset_flag"] = df.groupby("series_id")["onset_raw"].cummax().astype(int)
    return df


def build_onset_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    first_onset = (
        labeled.loc[labeled["onset_raw"] == 1, ["series_id", "time_idx", "t_year"]]
        .sort_values(["series_id", "time_idx"])
        .groupby("series_id", as_index=False)
        .first()
        .rename(columns={"time_idx": "t_init_idx", "t_year": "t_init_year"})
    )

    keep_cols = ["series_id"]
    for col in ["Cs", "cover_m", "cover_mm", "D28", "m_aging", "C_th"]:
        if col in labeled.columns:
            keep_cols.append(col)

    base = (
        labeled.sort_values(["series_id", "time_idx"])
        .groupby("series_id", as_index=False)[keep_cols]
        .first()
    )
    summary = base.merge(first_onset, on="series_id", how="left")
    summary["onset_observed"] = summary["t_init_year"].notna()
    return summary


def add_time_to_onset(labeled: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    labeled = labeled.merge(summary[["series_id", "t_init_year"]], on="series_id", how="left")
    labeled["time_to_onset"] = labeled["t_init_year"] - labeled["t_year"]
    labeled["time_to_onset"] = labeled["time_to_onset"].where(labeled["time_to_onset"].notna(), np.nan)
    labeled.loc[labeled["time_to_onset"].notna(), "time_to_onset"] = np.maximum(
        0.0,
        labeled.loc[labeled["time_to_onset"].notna(), "time_to_onset"].values,
    )
    return labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label corrosion initiation onset.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_parquet = ROOT / "data" / "sim" / "chloride_long.parquet"
    in_csv = ROOT / "data" / "sim" / "chloride_long.csv"
    sim_path = args.input or (in_parquet if in_parquet.exists() else in_csv)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_sim_data(sim_path)
    df = ensure_required_columns(df)
    df = ensure_cover_mm(df)

    labeled = compute_event_labels(df)
    summary = build_onset_summary(labeled)
    labeled = add_time_to_onset(labeled, summary)

    out_long = out_dir / "chloride_labeled.parquet"
    out_summary = out_dir / "onset_summary.csv"
    labeled.to_parquet(out_long, index=False)
    summary.to_csv(out_summary, index=False)

    onset_rate = summary["t_init_year"].notna().mean()
    print(f"Saved labeled data: {out_long} rows={len(labeled)}")
    print(f"Saved onset summary: {out_summary} rows={len(summary)}")
    print(f"Observed onset rate: {onset_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
