"""
01_generate_chloride_data.py

Generate chloride ingress time series at rebar depth using a Fick's 2nd-law based model.

Goal:
- Create a long-format dataset for TFT:
  series_id, time_idx, chloride_rebar, (static params...)
- Later we will label corrosion initiation time (ti) in 02_label_corrosion_onset.py

Model (1D, semi-infinite slab approximation):
C(x,t) = Cb + (Cs - Cb) * erfc( x / (2 * sqrt(D_eff(t) * t)) )

Where:
- x = cover depth (m)
- t = time (s)
- Cs = surface chloride concentration (unit-consistent with threshold)
- Cb = initial/bulk chloride concentration
- D_eff(t) = time-dependent diffusion coefficient (aging)
  A simple aging law:
  D(t) = D_ref * (t_ref / t)^m   for t >= t_ref
  and D(t) = D_ref              for t < t_ref

We use an "effective" D over each time step by evaluating D(t_mid).
This is a pragmatic approach to generate realistic trajectories.

IMPORTANT:
- Units must be consistent: if Cs and C_th are in "% by mass of cement",
  then chloride_rebar will also be in that unit (relative scale).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass
class SimConfig:
    n_series: int = 1000                 # number of simulated sequences
    years: int = 60                      # total duration (years)
    dt_weeks: int = 4                    # sampling interval (weeks). 4 weeks ~ monthly
    seed: int = 20250111

    # Reference time for D_ref
    t_ref_days: int = 28                 # D28 in paper
    m_aging: float = 0.284               # example aging exponent (can randomize later)

    # Initial chloride (Cb)
    Cb: float = 0.0

    # Add small measurement/model noise to chloride time series
    noise_std: float = 0.01              # in same unit as Cs (tune later)

    # Parameter sampling ranges (placeholder; later we map to paper distributions)
    Cs_range: Tuple[float, float] = (0.21, 1.63)     # surface chloride (e.g., %)
    Cth_range: Tuple[float, float] = (0.09, 0.51)    # threshold (e.g., %)
    cover_range_m: Tuple[float, float] = (0.04, 0.11)  # cover in meters
    D28_range: Tuple[float, float] = (3.5e-12, 9.0e-12) # m^2/s (placeholder)

    # Optional: allow random aging exponent
    m_range: Tuple[float, float] = (0.20, 0.45)


def weeks_to_seconds(weeks: float) -> float:
    return float(weeks) * 7.0 * 24.0 * 3600.0


def years_to_seconds(years: float) -> float:
    # Use 365.25 for leap-year average
    return float(years) * 365.25 * 24.0 * 3600.0


def days_to_seconds(days: float) -> float:
    return float(days) * 24.0 * 3600.0


def D_time_dependent(D_ref: float, t_s: float, t_ref_s: float, m: float) -> float:
    """
    Aging law:
    D(t) = D_ref * (t_ref / t)^m, for t >= t_ref
    D(t) = D_ref, for t < t_ref
    """
    if t_s <= 0:
        return D_ref
    if t_s < t_ref_s:
        return D_ref
    return D_ref * (t_ref_s / t_s) ** m


def chloride_at_depth_erfc(x_m: float, t_s: float, Cs: float, Cb: float, D_eff: float) -> float:
    """
    1D semi-infinite Fick solution using complementary error function.
    """
    if t_s <= 0:
        return Cb
    # Avoid divide-by-zero
    denom = 2.0 * math.sqrt(max(D_eff, 1e-30) * t_s)
    arg = x_m / max(denom, 1e-30)
    # erfc(arg) in [0,1]
    return Cb + (Cs - Cb) * math.erfc(arg)


def simulate_one_series(cfg: SimConfig, rng: np.random.Generator, series_id: int) -> Dict[str, np.ndarray]:
    """
    Simulate one chloride time series at rebar depth with random parameters.
    """
    # Sample parameters (simple ranges now; later replace with paper distributions)
    Cs = rng.uniform(*cfg.Cs_range)
    C_th = rng.uniform(*cfg.Cth_range)
    cover = rng.uniform(*cfg.cover_range_m)
    D28 = rng.uniform(*cfg.D28_range)
    m = rng.uniform(*cfg.m_range)

    # Time grid
    dt_s = weeks_to_seconds(cfg.dt_weeks)
    n_steps = int(years_to_seconds(cfg.years) // dt_s) + 1
    t_s = np.arange(n_steps, dtype=float) * dt_s

    t_ref_s = days_to_seconds(cfg.t_ref_days)

    chloride = np.zeros(n_steps, dtype=float)
    for i in range(n_steps):
        # Use mid-point time for D(t) (more stable than left-point)
        t_mid = max(t_s[i], 1.0)
        D_eff = D_time_dependent(D28, t_mid, t_ref_s, m)
        chloride[i] = chloride_at_depth_erfc(cover, t_s[i], Cs, cfg.Cb, D_eff)

    # Add small noise (optional)
    if cfg.noise_std > 0:
        chloride = np.clip(chloride + rng.normal(0.0, cfg.noise_std, size=n_steps), 0.0, None)

    return {
        "series_id": np.full(n_steps, series_id, dtype=int),
        "time_idx": np.arange(n_steps, dtype=int),
        "t_year": t_s / years_to_seconds(1.0),
        "chloride_rebar": chloride,
        # static parameters repeated (TFT can take as static covariates)
        "Cs": np.full(n_steps, Cs, dtype=float),
        "C_th": np.full(n_steps, C_th, dtype=float),
        "cover_m": np.full(n_steps, cover, dtype=float),
        "D28": np.full(n_steps, D28, dtype=float),
        "m_aging": np.full(n_steps, m, dtype=float),
    }


def main():
    cfg = SimConfig()

    out_dir = Path("data/sim")
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)

    all_frames = []
    for sid in range(cfg.n_series):
        s = simulate_one_series(cfg, rng, sid)
        all_frames.append(pd.DataFrame(s))

    df = pd.concat(all_frames, ignore_index=True)

    # Basic sanity columns
    # (Later we will create target labels in 02_label_corrosion_onset.py)
    df["target_cont"] = np.maximum(0.0, df["chloride_rebar"] - df["C_th"])
    # placeholder onset label; real label in next script
    df["target_onset"] = (df["chloride_rebar"] >= df["C_th"]).astype(int)

    # Save
    pq_path = out_dir / "chloride_long.parquet"
    csv_path = out_dir / "chloride_long.csv"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)

    print("✅ Saved:")
    print(" -", pq_path, "rows=", len(df))
    print(" -", csv_path)

    # Quick check
    print(df.head(5))


if __name__ == "__main__":
    main()
