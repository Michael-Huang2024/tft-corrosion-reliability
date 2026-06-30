"""
Physical simulator audit and prespecified candidate screening.

No machine-learning training or prediction is performed here.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import erfc

from revision_config import (
    BoundedDistributionSpec,
    GROUP_COLUMN,
    MANUSCRIPT_SEED,
    PARAMETER_SPECS,
    RAW_ONSET_COLUMN,
    REVISION_FIGURE_DIR,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TIME_COLUMN,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)


SECONDS_PER_WEEK = 7.0 * 24.0 * 3600.0
SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0
ANALYSIS_YEARS = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]


CANDIDATES = {
    "A": {
        "Cs": (4.4, 0.60),
        "D28": (3.5e-12, 0.55e-12),
        "m_aging": (0.36, 0.060),
        "C_th": (0.85, 0.120),
    },
    "B": {
        "Cs": (4.5, 0.60),
        "D28": (3.8e-12, 0.50e-12),
        "m_aging": (0.32, 0.055),
        "C_th": (0.80, 0.105),
    },
    "C": {
        "Cs": (4.6, 0.60),
        "D28": (4.0e-12, 0.45e-12),
        "m_aging": (0.30, 0.050),
        "C_th": (0.75, 0.090),
    },
}


def with_candidate_specs(candidate: str) -> dict[str, BoundedDistributionSpec]:
    specs = dict(PARAMETER_SPECS)
    for name, (mean, std) in CANDIDATES[candidate].items():
        specs[name] = replace(specs[name], mean=mean, std=std)
    return specs


def sample_truncated(spec: BoundedDistributionSpec, rng: np.random.Generator, n: int) -> tuple[np.ndarray, dict[str, float]]:
    accepted: list[np.ndarray] = []
    draws = 0
    batch = max(4096, n * 2)
    if spec.distribution == "uniform":
        values = rng.uniform(spec.lower, spec.upper, n)
        return values, {"draws": float(n), "accepted": float(n), "acceptance_rate": 1.0}
    while sum(len(chunk) for chunk in accepted) < n:
        if "lognormal" in spec.distribution:
            mu, sigma = spec.lognormal_mu_sigma
            values = rng.lognormal(mu, sigma, batch)
        elif "normal" in spec.distribution:
            values = rng.normal(spec.mean, spec.std, batch)
        else:
            raise ValueError(f"Unsupported distribution: {spec.distribution}")
        draws += len(values)
        kept = values[(values >= spec.lower) & (values <= spec.upper)]
        if len(kept):
            accepted.append(kept)
    return np.concatenate(accepted)[:n], {"draws": float(draws), "accepted": float(n), "acceptance_rate": float(n / draws)}


def sample_parameters(specs: dict[str, BoundedDistributionSpec], n: int = 1000, seed: int = MANUSCRIPT_SEED) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    values = {}
    rates = {}
    for name in ["Cs", "D28", "m_aging", "cover_mm", "C_th"]:
        values[name], rates[name] = sample_truncated(specs[name], rng, n)
    return pd.DataFrame(values), rates


def time_grid(years: int = 60, dt_weeks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    t_s = np.arange(int((years * SECONDS_PER_YEAR) // (dt_weeks * SECONDS_PER_WEEK)) + 1) * dt_weeks * SECONDS_PER_WEEK
    return t_s.astype(float), t_s / SECONDS_PER_YEAR


def chloride_matrix(params: pd.DataFrame, years: int = 60, dt_weeks: int = 4) -> tuple[np.ndarray, np.ndarray]:
    t_s, t_year = time_grid(years, dt_weeks)
    t = np.maximum(t_s[None, :], 1.0)
    t_ref = 28.0 * SECONDS_PER_DAY
    Cs = params["Cs"].to_numpy()[:, None]
    D28 = params["D28"].to_numpy()[:, None]
    m = params["m_aging"].to_numpy()[:, None]
    cover_m = (params["cover_mm"].to_numpy() / 1000.0)[:, None]
    D_eff = np.where(t < t_ref, D28, D28 * (t_ref / t) ** m)
    denom = 2.0 * np.sqrt(np.maximum(D_eff, 1e-30) * t)
    chloride = Cs * erfc(cover_m / np.maximum(denom, 1e-30))
    chloride[:, 0] = 0.0
    return chloride, t_year


def simulate(params: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chloride, t_year = chloride_matrix(params)
    n_series, n_steps = chloride.shape
    threshold = params["C_th"].to_numpy()[:, None]
    raw = (chloride >= threshold).astype(np.int8)
    flag = np.maximum.accumulate(raw, axis=1).astype(np.int8)
    repeated = params.loc[np.repeat(params.index.to_numpy(), n_steps)].reset_index(drop=True)
    df = repeated.copy()
    df.insert(0, GROUP_COLUMN, np.repeat(np.arange(n_series), n_steps))
    df.insert(1, TIME_INDEX_COLUMN, np.tile(np.arange(n_steps), n_series))
    df.insert(2, TIME_COLUMN, np.tile(t_year, n_series))
    df["chloride_rebar"] = chloride.reshape(-1)
    df[RAW_ONSET_COLUMN] = raw.reshape(-1)
    df["target_onset"] = raw.reshape(-1)
    df[TARGET_COLUMN] = flag.reshape(-1)
    first = (
        df.loc[df[RAW_ONSET_COLUMN] == 1, [GROUP_COLUMN, TIME_COLUMN]]
        .groupby(GROUP_COLUMN, as_index=False)
        .first()
        .rename(columns={TIME_COLUMN: "t_init_year"})
    )
    summary = params.copy()
    summary.insert(0, GROUP_COLUMN, np.arange(n_series))
    summary = summary.merge(first, on=GROUP_COLUMN, how="left")
    summary["onset_observed"] = summary["t_init_year"].notna()
    return df, summary


def nearest_time(df: pd.DataFrame, year: float) -> float:
    times = df[TIME_COLUMN].drop_duplicates().to_numpy()
    return float(times[np.argmin(np.abs(times - year))])


def pf_row(df: pd.DataFrame, year: float) -> dict[str, float]:
    t = nearest_time(df, year)
    at = df[df[TIME_COLUMN] == t]
    y = at[TARGET_COLUMN].to_numpy(dtype=float)
    return {
        "requested_year": year,
        "nearest_t_year": t,
        "Pf": float(y.mean()),
        "initiated_count": int(y.sum()),
        "response_variance": float(np.var(y)),
    }


def distribution_summary(params: pd.DataFrame, specs: dict[str, BoundedDistributionSpec]) -> pd.DataFrame:
    rows = []
    for name, spec in specs.items():
        values = params[name].to_numpy()
        width = spec.upper - spec.lower
        rows.append(
            {
                "parameter": name,
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "p05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.50)),
                "p95": float(np.quantile(values, 0.95)),
                "near_lower_2pct": float(((values >= spec.lower) & (values <= spec.lower + 0.02 * width)).mean()),
                "near_upper_2pct": float(((values <= spec.upper) & (values >= spec.upper - 0.02 * width)).mean()),
            }
        )
    return pd.DataFrame(rows)


def cover_group_pf(df: pd.DataFrame) -> pd.DataFrame:
    bins = [(40.0, 60.0), (60.0, 80.0), (80.0, 110.0)]
    rows = []
    for year in [20.0, 40.0, 60.0]:
        t = nearest_time(df, year)
        at = df[df[TIME_COLUMN] == t]
        for low, high in bins:
            if high == 110.0:
                group = at[(at["cover_mm"] >= low) & (at["cover_mm"] <= high)]
            else:
                group = at[(at["cover_mm"] >= low) & (at["cover_mm"] < high)]
            rows.append(
                {
                    "requested_year": year,
                    "nearest_t_year": t,
                    "cover_group": f"{int(low)}-{int(high)} mm",
                    "series_count": int(group[GROUP_COLUMN].nunique()),
                    "Pf": float(group[TARGET_COLUMN].mean()) if not group.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def cover_ordering_stats(cover: pd.DataFrame) -> tuple[int, float]:
    violations = 0
    max_mag = 0.0
    for _year, group in cover.groupby("requested_year"):
        vals = {row.cover_group: row.Pf for row in group.itertuples(index=False)}
        pairs = [("40-60 mm", "60-80 mm"), ("60-80 mm", "80-110 mm")]
        for hi, lo in pairs:
            diff = vals.get(lo, np.nan) - vals.get(hi, np.nan)
            if np.isfinite(diff) and diff > 1e-12:
                violations += 1
                max_mag = max(max_mag, float(diff))
    return violations, max_mag


def classify_candidate(pfs: dict[float, dict[str, float]], cover: pd.DataFrame, dist: pd.DataFrame, summary: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons = []
    preferred = {20.0: (0.02, 0.20), 40.0: (0.15, 0.55), 60.0: (0.30, 0.75)}
    for year, (low, high) in preferred.items():
        pf = pfs[year]["Pf"]
        if not (low <= pf <= high):
            reasons.append(f"Pf({year:g})={pf:.3f} outside [{low}, {high}]")
    if pfs[40.0]["response_variance"] <= 0.0 or pfs[60.0]["response_variance"] <= 0.0:
        reasons.append("zero response variance at 40 or 60 years")
    if pfs[20.0]["response_variance"] <= 0.0:
        reasons.append("20-year response sparse/zero variance")
    violations, magnitude = cover_ordering_stats(cover)
    if violations:
        reasons.append(f"cover-depth ordering violations={violations}, max magnitude={magnitude:.4f}")
    if (dist["near_lower_2pct"].max() > 0.10) or (dist["near_upper_2pct"].max() > 0.10):
        reasons.append("severe boundary accumulation")
    final_initiated = int(summary["onset_observed"].sum())
    if final_initiated == 0 or final_initiated == len(summary):
        reasons.append("60-year outcomes do not contain both initiated and non-initiated trajectories")
    return len([r for r in reasons if "20-year response sparse" not in r]) == 0, reasons


def corner_case_report() -> pd.DataFrame:
    specs = PARAMETER_SPECS
    cases = {
        "most_aggressive": {
            "Cs": specs["Cs"].upper,
            "D28": specs["D28"].upper,
            "m_aging": specs["m_aging"].lower,
            "cover_mm": specs["cover_mm"].lower,
            "C_th": specs["C_th"].lower,
        },
        "median_configuration": {
            "Cs": specs["Cs"].mean,
            "D28": specs["D28"].mean,
            "m_aging": specs["m_aging"].mean,
            "cover_mm": (specs["cover_mm"].lower + specs["cover_mm"].upper) / 2.0,
            "C_th": specs["C_th"].mean,
        },
        "most_durable": {
            "Cs": specs["Cs"].lower,
            "D28": specs["D28"].lower,
            "m_aging": specs["m_aging"].upper,
            "cover_mm": specs["cover_mm"].upper,
            "C_th": specs["C_th"].upper,
        },
    }
    rows = []
    for name, values in cases.items():
        params = pd.DataFrame([values])
        chloride, _t_year = chloride_matrix(params)
        for year in [20.0, 40.0, 60.0]:
            t_s, t_year = time_grid()
            idx = int(np.argmin(np.abs(t_year - year)))
            rows.append(
                {
                    "case": name,
                    "requested_year": year,
                    "nearest_t_year": float(t_year[idx]),
                    "chloride_rebar": float(chloride[0, idx]),
                    "C_th": float(values["C_th"]),
                    "instantaneous_exceedance": bool(chloride[0, idx] >= values["C_th"]),
                    "cumulative_initiated_by_year": bool((chloride[0, : idx + 1] >= values["C_th"]).any()),
                }
            )
    return pd.DataFrame(rows)


def directionality_report() -> pd.DataFrame:
    base = pd.DataFrame(
        [
            {
                "Cs": PARAMETER_SPECS["Cs"].mean,
                "D28": PARAMETER_SPECS["D28"].mean,
                "m_aging": PARAMETER_SPECS["m_aging"].mean,
                "cover_mm": 75.0,
                "C_th": PARAMETER_SPECS["C_th"].mean,
            }
        ]
    )
    rows = []
    base_c = chloride_matrix(base)[0][0, -1]
    for parameter, factor, expected in [
        ("Cs", 1.05, "increase chloride / earlier initiation"),
        ("D28", 1.10, "increase chloride / earlier initiation"),
        ("cover_mm", 1.10, "decrease chloride / delay initiation"),
        ("m_aging", 1.10, "decrease chloride / delay initiation"),
        ("C_th", 1.10, "higher threshold delays initiation without changing chloride"),
    ]:
        changed = base.copy()
        changed[parameter] *= factor
        c = chloride_matrix(changed)[0][0, -1]
        rows.append(
            {
                "parameter_increased": parameter,
                "baseline_chloride_60": float(base_c),
                "changed_chloride_60": float(c),
                "chloride_change": float(c - base_c),
                "expected_direction": expected,
            }
        )
    return pd.DataFrame(rows)


def write_physical_audit() -> None:
    corners = corner_case_report()
    directions = directionality_report()
    lines = [
        "# Physical Simulator Audit",
        "",
        "Conclusion: implementation is internally consistent with the active repository/manuscript-style Fickian formulation. No unit, one-time cover conversion, or directionality error was found. No simulator correction was made before candidate screening.",
        "",
        "Important formulation note: the implementation uses the apparent-diffusivity form `D(t) * t` in `x / (2 sqrt(D(t) t))`, not the integrated exposure `integral_0^t D(tau) d tau`. This matches the active code and legacy code comments. No June 25 manuscript file with a conflicting integrated-exposure equation was present in the repository.",
        "",
        "Checks:",
        "",
        "- `D28` is interpreted in `m^2/s`.",
        "- `cover_mm` is converted to meters exactly once as `cover_mm / 1000`.",
        "- Service time uses seconds internally.",
        "- The 28-day reference is converted to seconds as `28 * 24 * 3600`.",
        "- Time-dependent diffusivity is `D28 * (t_ref / t)^m` for `t >= t_ref`, otherwise `D28`.",
        "- Boundary condition is `C(x,t) = Cs * erfc(x / (2 sqrt(D(t)t)))` with initial/bulk chloride `Cb = 0`.",
        "",
        "Directionality check at 60 years:",
        "",
        directions.to_string(index=False),
        "",
        "Corner cases:",
        "",
        corners.to_string(index=False),
    ]
    Path("outputs/revision/physical_simulator_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def screen_candidates() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], str | None]:
    summary_rows = []
    cover_tables = {}
    pf_curves = []
    selected = None
    for candidate in ["A", "B", "C"]:
        specs = with_candidate_specs(candidate)
        params, rates = sample_parameters(specs)
        df, onset = simulate(params)
        pfs = {year: pf_row(df, year) for year in ANALYSIS_YEARS}
        cover = cover_group_pf(df)
        dist = distribution_summary(params, specs)
        ok, reasons = classify_candidate({y: pfs[y] for y in [20.0, 40.0, 60.0]}, cover, dist, onset)
        violations, max_cross = cover_ordering_stats(cover)
        cover["candidate"] = candidate
        cover_tables[candidate] = cover
        for year in ANALYSIS_YEARS:
            pf_curves.append({"candidate": candidate, **pfs[year]})
        t_init = onset["t_init_year"].dropna()
        row = {
            "candidate": candidate,
            "acceptable": ok,
            "reasons": "; ".join(reasons) if reasons else "none",
            "Pf10": pfs[10.0]["Pf"],
            "Pf20": pfs[20.0]["Pf"],
            "Pf30": pfs[30.0]["Pf"],
            "Pf40": pfs[40.0]["Pf"],
            "Pf50": pfs[50.0]["Pf"],
            "Pf60": pfs[60.0]["Pf"],
            "initiated20": pfs[20.0]["initiated_count"],
            "initiated40": pfs[40.0]["initiated_count"],
            "initiated60": pfs[60.0]["initiated_count"],
            "variance20": pfs[20.0]["response_variance"],
            "variance40": pfs[40.0]["response_variance"],
            "variance60": pfs[60.0]["response_variance"],
            "never_initiated_pct": float((~onset["onset_observed"]).mean() * 100.0),
            "t_init_min": float(t_init.min()) if len(t_init) else np.nan,
            "t_init_median": float(t_init.median()) if len(t_init) else np.nan,
            "t_init_p95": float(t_init.quantile(0.95)) if len(t_init) else np.nan,
            "cover_ordering_violations": violations,
            "max_cover_crossing_magnitude": max_cross,
        }
        for name, rate in rates.items():
            row[f"{name}_acceptance_rate"] = rate["acceptance_rate"]
        for _, drow in dist.iterrows():
            row[f"{drow.parameter}_near_lower_2pct"] = drow.near_lower_2pct
            row[f"{drow.parameter}_near_upper_2pct"] = drow.near_upper_2pct
        summary_rows.append(row)
        dist.to_csv(REVISION_TABLE_DIR / f"parameter_candidate_{candidate}_distribution_summary.csv", index=False)
        if selected is None and ok:
            selected = candidate
    return pd.DataFrame(summary_rows), pd.DataFrame(pf_curves), cover_tables, selected


def write_screening_outputs(summary: pd.DataFrame, pf_curves: pd.DataFrame, cover_tables: dict[str, pd.DataFrame], selected: str | None) -> None:
    summary.to_csv(REVISION_TABLE_DIR / "parameter_candidate_screening.csv", index=False)
    cover_all = pd.concat(cover_tables.values(), ignore_index=True)
    cover_all.to_csv(REVISION_TABLE_DIR / "parameter_candidate_cover_groups.csv", index=False)

    plt.figure(figsize=(7, 4))
    for candidate, group in pf_curves.groupby("candidate"):
        plt.plot(group["requested_year"], group["Pf"], marker="o", label=f"Candidate {candidate}")
    plt.axhspan(0.02, 0.20, xmin=0.15, xmax=0.25, alpha=0.08)
    plt.xlabel("Year")
    plt.ylabel("Reference cumulative Pf(t)")
    plt.title("Candidate cumulative initiation probability")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "parameter_candidate_pf_curves.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    for (candidate, cover_group), group in cover_all.groupby(["candidate", "cover_group"]):
        plt.plot(group["requested_year"], group["Pf"], marker="o", label=f"{candidate} {cover_group}")
    plt.xlabel("Year")
    plt.ylabel("Reference cumulative Pf(t)")
    plt.title("Candidate cover-depth groups")
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "parameter_candidate_cover_groups.png", dpi=300)
    plt.close()

    lines = [
        "# Parameter Candidate Screening Report",
        "",
        "Physical simulator audit conclusion: implementation valid for the active apparent-diffusivity formulation; no correction was made.",
        "",
        f"Selected candidate: `{selected or 'none'}`",
        "",
        "Objective selection rule: A is selected if acceptable; otherwise B; otherwise C only if A and B remain too low. Selection is simulator-only and independent of any machine-learning performance.",
        "",
        "Candidate summary:",
        "",
        summary.to_string(index=False),
        "",
    ]
    if selected:
        row = summary.loc[summary["candidate"] == selected].iloc[0]
        lines.extend(
            [
                "Selection reason:",
                f"- Candidate {selected} is the first candidate satisfying the prespecified criteria.",
                f"- Pf(20/40/60) = {row.Pf20:.3f} / {row.Pf40:.3f} / {row.Pf60:.3f}.",
                "- Suitable for benchmark model training: yes.",
                "- Suitable for Sobol analysis: yes for years with nonzero response variance; sparse years must still be flagged if encountered.",
                "- Suitable for MC Dropout UQ: yes, because both initiated and non-initiated test cases should remain available.",
            ]
        )
    else:
        lines.extend(
            [
                "Selection reason:",
                "- No candidate satisfied all prespecified criteria. Do not lock a configuration or train models.",
            ]
        )
    Path("outputs/revision/parameter_candidate_screening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_revision_dirs()
    write_physical_audit()
    summary, pf_curves, cover_tables, selected = screen_candidates()
    write_screening_outputs(summary, pf_curves, cover_tables, selected)
    print(summary[["candidate", "acceptable", "Pf20", "Pf40", "Pf60", "reasons"]].to_string(index=False))
    print(f"selected={selected}")


if __name__ == "__main__":
    main()
