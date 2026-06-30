"""
Five-parameter Sobol sensitivity for cumulative corrosion initiation.

Final manuscript-scale analysis requires confirmed distribution parameters.
When they are missing, this script writes a blocked sensitivity report unless
``--smoke-test`` or ``--allow-bounds-only`` is supplied for diagnostic runs.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from revision_config import PARAMETER_SPECS, REVISION_FIGURE_DIR, REVISION_TABLE_DIR, missing_distribution_parameters, ensure_revision_dirs


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run revision Sobol sensitivity analysis.")
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-bounds-only", action="store_true")
    return parser.parse_args()


def d_time_dependent(D28: float, t_s: float, m_aging: float, t_ref_s: float = 28.0 * SECONDS_PER_DAY) -> float:
    if t_s <= 0.0 or t_s < t_ref_s:
        return D28
    return D28 * (t_ref_s / t_s) ** m_aging


def chloride_at_cover(Cs: float, D28: float, m_aging: float, cover_mm: float, t_year: float) -> float:
    t_s = t_year * SECONDS_PER_YEAR
    if t_s <= 0.0:
        return 0.0
    cover_m = cover_mm / 1000.0
    D_eff = d_time_dependent(D28, t_s, m_aging)
    denom = 2.0 * math.sqrt(max(D_eff, 1e-30) * t_s)
    return Cs * math.erfc(cover_m / max(denom, 1e-30))


def cumulative_initiation_indicator(row: np.ndarray, t_year: float) -> float:
    Cs, D28, m_aging, cover_mm, C_th = row
    return float(chloride_at_cover(Cs, D28, m_aging, cover_mm, t_year) >= C_th)


def write_blocked_report(missing: list[object]) -> None:
    lines = [
        "# Sensitivity Report",
        "",
        "Final Sobol sensitivity analysis was not run.",
        "",
        "Reason: the June 25 manuscript specifies bounded lognormal/normal distribution families, but this repository does not document the means/standard deviations or log-space parameters needed to sample those distributions.",
        "",
        "Missing confirmations:",
        "",
    ]
    for spec in missing:
        lines.append(f"- `{spec.name}`: {spec.distribution}, bounds {spec.lower:g} to {spec.upper:g}, units {spec.units}")
    lines.extend(
        [
            "",
            "A bounds-only diagnostic can be run with `--smoke-test` or `--allow-bounds-only`, but it must not be reported as the final manuscript Sobol analysis.",
        ]
    )
    Path("outputs/revision/sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sobol(N: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from SALib.analyze import sobol
        from SALib.sample import saltelli
    except Exception as exc:
        raise RuntimeError("SALib is required. Install with `pip install SALib` or `pip install -r requirements.txt`.") from exc

    names = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
    bounds = [[PARAMETER_SPECS[name].lower, PARAMETER_SPECS[name].upper] for name in names]
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    X = saltelli.sample(problem, N, calc_second_order=False)

    rows = []
    trajectories = []
    for year in [20.0, 40.0, 60.0]:
        y = np.array([cumulative_initiation_indicator(row, year) for row in X], dtype=float)
        status = "measured"
        error_message = ""
        try:
            if np.var(y) == 0.0:
                raise ValueError("constant binary response")
            result = sobol.analyze(problem, y, calc_second_order=False, print_to_console=False)
            s1 = result["S1"]
            st = result["ST"]
            s1_conf = result.get("S1_conf", np.full(len(names), np.nan))
            st_conf = result.get("ST_conf", np.full(len(names), np.nan))
        except Exception as exc:
            s1 = np.full(len(names), np.nan)
            st = np.full(len(names), np.nan)
            s1_conf = np.full(len(names), np.nan)
            st_conf = np.full(len(names), np.nan)
            status = "inconclusive_binary_response"
            error_message = str(exc)
        for i, name in enumerate(names):
            rows.append(
                {
                    "year": year,
                    "parameter": name,
                    "S1": float(s1[i]),
                    "S1_conf": float(s1_conf[i]),
                    "ST": float(st[i]),
                    "ST_conf": float(st_conf[i]),
                    "response_mean": float(y.mean()),
                    "response_variance": float(np.var(y)),
                    "status": status,
                    "error_message": error_message,
                }
            )
        trajectories.append({"year": year, "response_mean": float(y.mean()), "response_variance": float(np.var(y))})
    return pd.DataFrame(rows), pd.DataFrame(trajectories)


def write_figures(indices: pd.DataFrame, trajectory: pd.DataFrame) -> None:
    pivot = indices.pivot(index="parameter", columns="year", values="ST")
    pivot.plot(kind="bar", figsize=(7, 4))
    plt.ylabel("Total-order Sobol index (ST)")
    plt.title("Bounds-only Sobol diagnostic at 20, 40, and 60 years")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "sobol_total_effect_20_40_60.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(trajectory["year"], trajectory["response_mean"], marker="o")
    plt.xlabel("Year")
    plt.ylabel("Mean cumulative initiation indicator")
    plt.title("Sobol response mean over time")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "sobol_time_evolution.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    missing = missing_distribution_parameters()
    if missing and not (args.smoke_test or args.allow_bounds_only):
        write_blocked_report(missing)
        print("Sobol analysis blocked pending parameter confirmation. See outputs/revision/sensitivity_report.md")
        return

    N = min(args.N, 64) if args.smoke_test else args.N
    indices, trajectory = run_sobol(N)
    indices.to_csv(REVISION_TABLE_DIR / "sobol_indices.csv", index=False)
    write_figures(indices, trajectory)

    ranked = (
        indices.sort_values(["year", "ST"], ascending=[True, False])
        .groupby("year", as_index=False)
        .first()[["year", "parameter", "ST"]]
    )
    lines = [
        "# Sensitivity Report",
        "",
        "This run used a bounds-only independent uniform diagnostic because confirmed non-uniform distribution shape parameters were unavailable."
        if missing
        else "This run used the configured manuscript distributions.",
        "",
        "The response is the deterministic cumulative initiation indicator `I(C(x,t) >= Ccrit)` at each analysis year. Sobol indices decompose variance of this binary event over the input parameter space.",
        "",
        "Top total-effect parameter by year:",
        "",
        "| Year | Parameter | ST |",
        "|---:|---|---:|",
    ]
    for row in ranked.itertuples(index=False):
        lines.append(f"| {row.year:.0f} | {row.parameter} | {row.ST:.6f} |")
    Path("outputs/revision/sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(indices.to_string(index=False))


if __name__ == "__main__":
    main()
