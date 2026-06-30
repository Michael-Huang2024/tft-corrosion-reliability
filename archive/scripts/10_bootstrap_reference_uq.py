"""
Optional bootstrap uncertainty bands for the reference cumulative Pf(t).

This quantifies sampling uncertainty of the held-out reference series and is
distinct from MC Dropout epistemic uncertainty.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_config import GROUP_COLUMN, REVISION_FIGURE_DIR, REVISION_LABELED_DATA, REVISION_TABLE_DIR, TARGET_COLUMN, TIME_INDEX_COLUMN, ensure_revision_dirs
from revision_data import attach_split, load_or_create_series_split, load_source_labeled, write_revision_labeled_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap reference cumulative Pf(t).")
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20250111)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    replicates = min(args.replicates, 20) if args.smoke_test else args.replicates
    df = pd.read_parquet(REVISION_LABELED_DATA) if REVISION_LABELED_DATA.exists() else load_source_labeled()
    if not REVISION_LABELED_DATA.exists():
        write_revision_labeled_data(df, REVISION_LABELED_DATA)
    split = load_or_create_series_split(df)
    df = attach_split(df, split)
    test = df[(df["split"] == "test") & (df[TIME_INDEX_COLUMN] >= 52)].copy()
    series = np.array(sorted(test[GROUP_COLUMN].unique()))
    rng = np.random.default_rng(args.seed)
    curves = []
    for rep in range(replicates):
        sampled = rng.choice(series, size=len(series), replace=True)
        boot = pd.concat([test[test[GROUP_COLUMN] == sid] for sid in sampled], ignore_index=True)
        curve = boot.groupby("t_year", as_index=False)[TARGET_COLUMN].mean().rename(columns={TARGET_COLUMN: f"rep_{rep}"})
        curves.append(curve)
    merged = curves[0]
    for curve in curves[1:]:
        merged = merged.merge(curve, on="t_year", how="inner")
    values = merged.drop(columns=["t_year"]).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "t_year": merged["t_year"],
            "reference_mean": values.mean(axis=1),
            "q025": np.quantile(values, 0.025, axis=1),
            "q975": np.quantile(values, 0.975, axis=1),
        }
    )
    out.to_csv(REVISION_TABLE_DIR / "bootstrap_reference_intervals.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(out["t_year"], out["reference_mean"], label="Bootstrap mean reference Pf(t)")
    plt.fill_between(out["t_year"], out["q025"], out["q975"], alpha=0.25, label="95% bootstrap band")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative corrosion initiation probability")
    plt.title("Reference sampling uncertainty from held-out series bootstrap")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "reference_bootstrap_band.png", dpi=300)
    plt.close()
    print((REVISION_TABLE_DIR / "bootstrap_reference_intervals.csv").as_posix())


if __name__ == "__main__":
    main()
