"""
Generate the final full-scale reviewer-revision dataset.

This script locks the advisor-confirmed parameter distributions before any
model training, uses rejection sampling for truncated non-uniform variables,
and writes only revision-specific final outputs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_config import (
    FINAL_LABELED_DATA,
    FINAL_ONSET_SUMMARY,
    GROUP_COLUMN,
    MANUSCRIPT_SEED,
    PARAMETER_SPECS,
    RAW_ONSET_COLUMN,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)


SECONDS_PER_WEEK = 7.0 * 24.0 * 3600.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final revision chloride dataset.")
    parser.add_argument("--n-series", type=int, default=1000)
    parser.add_argument("--years", type=int, default=60)
    parser.add_argument("--dt-weeks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=MANUSCRIPT_SEED)
    parser.add_argument("--cs-mean-override", type=float, default=None)
    return parser.parse_args()


def sample_truncated(spec, rng: np.random.Generator, n: int) -> tuple[np.ndarray, dict[str, float]]:
    accepted: list[np.ndarray] = []
    draws = 0
    batch = max(4096, n * 2)
    while sum(len(chunk) for chunk in accepted) < n:
        if "lognormal" in spec.distribution:
            mu, sigma = spec.lognormal_mu_sigma
            values = rng.lognormal(mean=mu, sigma=sigma, size=batch)
        elif "normal" in spec.distribution:
            values = rng.normal(loc=spec.mean, scale=spec.std, size=batch)
        elif spec.distribution == "uniform":
            values = rng.uniform(spec.lower, spec.upper, size=n)
            return values, {"draws": float(n), "accepted": float(n), "acceptance_rate": 1.0}
        else:
            raise ValueError(f"Unsupported distribution: {spec.distribution}")
        draws += len(values)
        kept = values[(values >= spec.lower) & (values <= spec.upper)]
        if len(kept):
            accepted.append(kept)
    out = np.concatenate(accepted)[:n]
    return out, {"draws": float(draws), "accepted": float(n), "acceptance_rate": float(n / draws)}


def sample_parameters(n: int, seed: int, cs_mean_override: float | None = None) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    specs = dict(PARAMETER_SPECS)
    if cs_mean_override is not None:
        specs["Cs"] = replace(specs["Cs"], mean=cs_mean_override)
    values = {}
    stats = {}
    for name in ["Cs", "D28", "m_aging", "cover_mm", "C_th"]:
        values[name], stats[name] = sample_truncated(specs[name], rng, n)
    return pd.DataFrame(values), stats


def time_grid(years: int, dt_weeks: int) -> tuple[np.ndarray, np.ndarray]:
    dt_s = dt_weeks * SECONDS_PER_WEEK
    n_steps = int((years * SECONDS_PER_YEAR) // dt_s) + 1
    t_s = np.arange(n_steps, dtype=float) * dt_s
    return t_s, t_s / SECONDS_PER_YEAR


def simulate_dataset(params: pd.DataFrame, years: int, dt_weeks: int) -> pd.DataFrame:
    t_s, t_year = time_grid(years, dt_weeks)
    t_ref_s = 28.0 * SECONDS_PER_DAY
    n_series = len(params)
    n_steps = len(t_s)
    t_eval = np.maximum(t_s, 1.0)[None, :]

    Cs = params["Cs"].to_numpy()[:, None]
    D28 = params["D28"].to_numpy()[:, None]
    m_aging = params["m_aging"].to_numpy()[:, None]
    cover_m = (params["cover_mm"].to_numpy() / 1000.0)[:, None]
    C_th = params["C_th"].to_numpy()[:, None]

    D_eff = np.where(t_eval < t_ref_s, D28, D28 * (t_ref_s / t_eval) ** m_aging)
    denom = 2.0 * np.sqrt(np.maximum(D_eff, 1e-30) * t_eval)
    chloride = Cs * np.vectorize(math.erfc)(cover_m / np.maximum(denom, 1e-30))
    chloride[:, 0] = 0.0

    onset_raw = (chloride >= C_th).astype(np.int8)
    onset_flag = np.maximum.accumulate(onset_raw, axis=1).astype(np.int8)

    repeated_params = params.loc[np.repeat(params.index.to_numpy(), n_steps)].reset_index(drop=True)
    df = repeated_params.copy()
    df.insert(0, GROUP_COLUMN, np.repeat(np.arange(n_series, dtype=int), n_steps))
    df.insert(1, TIME_INDEX_COLUMN, np.tile(np.arange(n_steps, dtype=int), n_series))
    df.insert(2, TIME_COLUMN, np.tile(t_year, n_series))
    df["chloride_rebar"] = chloride.reshape(-1)
    df["target_onset"] = onset_raw.reshape(-1)
    df[RAW_ONSET_COLUMN] = onset_raw.reshape(-1)
    df[TARGET_COLUMN] = onset_flag.reshape(-1)
    return df


def onset_summary(df: pd.DataFrame) -> pd.DataFrame:
    first = (
        df.loc[df[RAW_ONSET_COLUMN] == 1, [GROUP_COLUMN, TIME_INDEX_COLUMN, TIME_COLUMN]]
        .sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN])
        .groupby(GROUP_COLUMN, as_index=False)
        .first()
        .rename(columns={TIME_INDEX_COLUMN: "t_init_idx", TIME_COLUMN: "t_init_year"})
    )
    base = df.groupby(GROUP_COLUMN, as_index=False)[["Cs", "D28", "m_aging", "cover_mm", "C_th"]].first()
    summary = base.merge(first, on=GROUP_COLUMN, how="left")
    summary["onset_observed"] = summary["t_init_year"].notna()
    return summary


def distribution_summary(params: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, spec in PARAMETER_SPECS.items():
        values = params[name].to_numpy()
        rows.append(
            {
                "parameter": name,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "coefficient_of_variation": float(values.std(ddof=1) / values.mean()),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "lower_bound_count": int((values == spec.lower).sum()),
                "upper_bound_count": int((values == spec.upper).sum()),
                "out_of_range_count": int(((values < spec.lower) | (values > spec.upper)).sum()),
            }
        )
    return pd.DataFrame(rows)


def pf_at_year(df: pd.DataFrame, requested_year: float) -> dict[str, float]:
    times = df[[TIME_INDEX_COLUMN, TIME_COLUMN]].drop_duplicates()
    idx = (times[TIME_COLUMN] - requested_year).abs().idxmin()
    t_year = float(times.loc[idx, TIME_COLUMN])
    subset = df[df[TIME_COLUMN] == t_year]
    y = subset[TARGET_COLUMN].to_numpy(dtype=float)
    return {
        "requested_year": requested_year,
        "nearest_t_year": t_year,
        "Pf": float(y.mean()),
        "initiated_series": int(y.sum()),
        "non_initiated_series": int(len(y) - y.sum()),
        "response_variance": float(np.var(y)),
    }


def cover_depth_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    bins = [(40.0, 60.0), (60.0, 80.0), (80.0, 110.0)]
    rows = []
    for year in [20.0, 40.0, 60.0]:
        times = df[[TIME_COLUMN]].drop_duplicates()
        t_year = float(times.iloc[(times[TIME_COLUMN] - year).abs().argsort().iloc[0]][TIME_COLUMN])
        at_time = df[df[TIME_COLUMN] == t_year]
        for low, high in bins:
            if high == 110.0:
                group = at_time[(at_time["cover_mm"] >= low) & (at_time["cover_mm"] <= high)]
            else:
                group = at_time[(at_time["cover_mm"] >= low) & (at_time["cover_mm"] < high)]
            rows.append(
                {
                    "requested_year": year,
                    "nearest_t_year": t_year,
                    "cover_bin": f"{int(low)}-{int(high)} mm",
                    "series_count": int(group[GROUP_COLUMN].nunique()),
                    "Pf": float(group[TARGET_COLUMN].mean()) if not group.empty else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def classify_sufficiency(year_rows: list[dict[str, float]], cover: pd.DataFrame) -> tuple[str, list[str]]:
    by_year = {row["requested_year"]: row for row in year_rows}
    reasons = []
    if by_year[60.0]["Pf"] < 0.15 or (by_year[40.0]["response_variance"] < 1e-12 and by_year[60.0]["response_variance"] < 1e-12):
        reasons.append("low initiation")
        return "DEGENERATE_LOW_INITIATION", reasons
    if by_year[20.0]["Pf"] > 0.50 or by_year[60.0]["Pf"] > 0.90 or by_year[40.0]["Pf"] > 0.90:
        reasons.append("high initiation")
        return "DEGENERATE_HIGH_INITIATION", reasons
    if any(row["response_variance"] < 1e-12 for row in [by_year[40.0], by_year[60.0]]):
        reasons.append("zero variance at key years")
        return "DEGENERATE_ZERO_VARIANCE", reasons

    for year, group in cover.groupby("requested_year"):
        vals = {row.cover_bin: row.Pf for row in group.itertuples(index=False)}
        if not (vals.get("40-60 mm", np.nan) + 1e-12 >= vals.get("60-80 mm", np.nan) >= vals.get("80-110 mm", np.nan) - 1e-12):
            reasons.append(f"cover ordering violation at {year:g} years")
    if reasons:
        return "DEGENERATE_COVER_ORDERING", reasons

    diagnostic_ranges = {
        20.0: (0.02, 0.20),
        40.0: (0.15, 0.55),
        60.0: (0.30, 0.75),
    }
    for year, (low, high) in diagnostic_ranges.items():
        pf = by_year[year]["Pf"]
        if not (low <= pf <= high):
            reasons.append(f"Pf({year:g})={pf:.3f} outside preferred diagnostic range [{low}, {high}]")
    return "ACCEPTABLE", reasons


def validate_final_data(df: pd.DataFrame) -> dict[str, object]:
    ordered = df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN])
    one_to_zero = int((ordered.groupby(GROUP_COLUMN)[TARGET_COLUMN].diff() < 0).sum())
    pf = ordered.groupby(TIME_COLUMN)[TARGET_COLUMN].mean()
    decreasing = int((pf.diff() < -1e-12).sum())
    cummax_matches = bool((ordered[TARGET_COLUMN].to_numpy() == ordered.groupby(GROUP_COLUMN)[RAW_ONSET_COLUMN].cummax().to_numpy()).all())
    static_cols = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
    static_constant = {col: bool((ordered.groupby(GROUP_COLUMN)[col].nunique() == 1).all()) for col in static_cols}
    return {
        "one_to_zero_transitions": one_to_zero,
        "pf_decreasing_steps": decreasing,
        "cumulative_label_matches_cummax": cummax_matches,
        "static_parameters_constant": static_constant,
    }


def write_parameter_configuration(stats: dict[str, dict[str, float]], cs_mean: float, adjustment_used: bool) -> None:
    lines = [
        "# Final Parameter Configuration",
        "",
        f"Random seed: `{MANUSCRIPT_SEED}`",
        "",
        "Sampling method: reproducible rejection sampling for truncated lognormal and truncated normal variables; direct uniform sampling for cover depth. Values are not clipped after generation.",
        "",
        "Lognormal conversion equations:",
        "",
        "`sigma_log = sqrt(log(1 + (sd_physical / mean_physical)^2))`",
        "",
        "`mu_log = log(mean_physical) - 0.5 * sigma_log^2`",
        "",
        f"One-time adjustment used: `{adjustment_used}`",
        "",
        "| Parameter | Distribution | Physical mean | Physical SD | Internal mu | Internal sigma | Lower | Upper | Acceptance rate | Units |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, spec in PARAMETER_SPECS.items():
        active_spec = replace(spec, mean=cs_mean) if name == "Cs" else spec
        if "lognormal" in active_spec.distribution:
            mu, sigma = active_spec.lognormal_mu_sigma
        elif "normal" in active_spec.distribution:
            mu, sigma = active_spec.mean, active_spec.std
        else:
            mu, sigma = np.nan, np.nan
        lines.append(
            f"| {name} | {active_spec.distribution} | {active_spec.mean if active_spec.mean is not None else np.nan:g} | {active_spec.std if active_spec.std is not None else np.nan:g} | {mu:g} | {sigma:g} | {active_spec.lower:g} | {active_spec.upper:g} | {stats[name]['acceptance_rate']:.6f} | {active_spec.units} |"
        )
    Path("outputs/revision/final_parameter_configuration.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reports(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    distribution: pd.DataFrame,
    year_rows: list[dict[str, float]],
    cover: pd.DataFrame,
    classification: str,
    reasons: list[str],
    validation: dict[str, object],
    adjustment_used: bool,
    adjustment_reason: str,
) -> None:
    distribution.to_csv(REVISION_TABLE_DIR / "final_input_distribution_summary.csv", index=False)
    lines = [
        "# Final Data Generation Report",
        "",
        "Dataset: 1,000 independently sampled parameterized deterioration trajectories. This is not a nested scenario/realization design.",
        "",
        f"Rows: {len(df)}",
        f"Series: {df[GROUP_COLUMN].nunique()}",
        f"Time steps per series: {df[TIME_INDEX_COLUMN].nunique()}",
        f"Onset-observed series by 60 years: {int(summary['onset_observed'].sum())}",
        "",
        f"Sufficiency classification: `{classification}`",
        "",
        "Reasons/caveats:",
        *(f"- {reason}" for reason in (reasons or ["none"])),
        "",
        f"One-time adjustment used: `{adjustment_used}`",
        f"Adjustment reason: {adjustment_reason or 'none'}",
        "",
        "Target validation:",
        f"- onset_flag 1-to-0 transitions: {validation['one_to_zero_transitions']}",
        f"- reference Pf(t) decreasing steps: {validation['pf_decreasing_steps']}",
        f"- cumulative label equals cumulative max: {validation['cumulative_label_matches_cummax']}",
        f"- static parameters constant by series: {validation['static_parameters_constant']}",
        "",
        "Year diagnostics:",
        "",
        "| Requested year | Nearest year | Pf | Initiated | Non-initiated | Response variance |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in year_rows:
        lines.append(
            f"| {row['requested_year']:.0f} | {row['nearest_t_year']:.6f} | {row['Pf']:.6f} | {row['initiated_series']} | {row['non_initiated_series']} | {row['response_variance']:.6f} |"
        )
    Path("outputs/revision/final_data_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    suff = [
        "# Final Data Sufficiency Report",
        "",
        f"Classification: `{classification}`",
        "",
        "Diagnostic criteria are internal checks only and are not physical calibration targets.",
        "",
        "Reasons/caveats:",
        *(f"- {reason}" for reason in (reasons or ["none"])),
        "",
        "Cover-depth diagnostics:",
        "",
        cover.to_string(index=False),
    ]
    Path("outputs/revision/final_data_sufficiency_report.md").write_text("\n".join(suff) + "\n", encoding="utf-8")

    adjustment_lines = [
        "# Final Parameter Adjustment Log",
        "",
        f"One-time adjustment used: `{adjustment_used}`",
        f"Reason: {adjustment_reason or 'none'}",
    ]
    Path("outputs/revision/final_parameter_adjustment_log.md").write_text("\n".join(adjustment_lines) + "\n", encoding="utf-8")


def generate_once(args: argparse.Namespace, cs_mean_override: float | None = None):
    params, stats = sample_parameters(args.n_series, args.seed, cs_mean_override)
    df = simulate_dataset(params, args.years, args.dt_weeks)
    summary = onset_summary(df)
    distribution = distribution_summary(params)
    year_rows = [pf_at_year(df, year) for year in [20.0, 40.0, 60.0]]
    cover = cover_depth_diagnostics(df)
    classification, reasons = classify_sufficiency(year_rows, cover)
    validation = validate_final_data(df)
    return params, stats, df, summary, distribution, year_rows, cover, classification, reasons, validation


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    params, stats, df, summary, distribution, year_rows, cover, classification, reasons, validation = generate_once(args)
    adjustment_used = False
    adjustment_reason = ""
    active_cs_mean = PARAMETER_SPECS["Cs"].mean

    if classification == "DEGENERATE_LOW_INITIATION":
        adjustment_used = True
        adjustment_reason = "Prespecified low-initiation rule: Cs physical mean changed from 4.0 to 4.3 kg/m^3."
        active_cs_mean = 4.3
        params, stats, df, summary, distribution, year_rows, cover, classification, reasons, validation = generate_once(args, cs_mean_override=active_cs_mean)
    elif classification == "DEGENERATE_HIGH_INITIATION":
        adjustment_used = True
        adjustment_reason = "Prespecified high-initiation rule: Cs physical mean changed from 4.0 to 3.7 kg/m^3."
        active_cs_mean = 3.7
        params, stats, df, summary, distribution, year_rows, cover, classification, reasons, validation = generate_once(args, cs_mean_override=active_cs_mean)

    if any(distribution["out_of_range_count"] != 0):
        raise RuntimeError("Out-of-range sampled parameter values detected.")
    if validation["one_to_zero_transitions"] != 0 or validation["pf_decreasing_steps"] != 0 or not validation["cumulative_label_matches_cummax"]:
        raise RuntimeError(f"Target validation failed: {validation}")
    if not all(validation["static_parameters_constant"].values()):
        raise RuntimeError(f"Static-parameter validation failed: {validation}")

    df.to_parquet(FINAL_LABELED_DATA, index=False)
    summary.to_csv(FINAL_ONSET_SUMMARY, index=False)
    write_parameter_configuration(stats, active_cs_mean, adjustment_used)
    write_reports(df, summary, distribution, year_rows, cover, classification, reasons, validation, adjustment_used, adjustment_reason)
    print(f"classification={classification}")
    print("final_data=data/processed/revision/final_chloride_labeled.parquet")


if __name__ == "__main__":
    main()
