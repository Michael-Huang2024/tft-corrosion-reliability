"""
Generate chloride ingress time series for the manuscript pipeline.

This script implements Step 1: chloride diffusion simulation. It preserves the
original Fick's-law-based sampling logic and writes the simulated long-format
dataset used by the downstream labeling step.

Generated files:
- data/sim/chloride_long.parquet
- data/sim/chloride_long.csv
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class SimConfig:
    n_series: int = 1000
    years: int = 60
    dt_weeks: int = 4
    seed: int = 20250111
    t_ref_days: int = 28
    m_aging: float = 0.284
    Cb: float = 0.0
    noise_std: float = 0.01
    Cs_range: Tuple[float, float] = (0.21, 1.63)
    Cth_range: Tuple[float, float] = (0.09, 0.51)
    cover_range_m: Tuple[float, float] = (0.04, 0.11)
    D28_range: Tuple[float, float] = (3.5e-12, 9.0e-12)
    m_range: Tuple[float, float] = (0.20, 0.45)


def weeks_to_seconds(weeks: float) -> float:
    return float(weeks) * 7.0 * 24.0 * 3600.0


def years_to_seconds(years: float) -> float:
    return float(years) * 365.25 * 24.0 * 3600.0


def days_to_seconds(days: float) -> float:
    return float(days) * 24.0 * 3600.0


def D_time_dependent(D_ref: float, t_s: float, t_ref_s: float, m: float) -> float:
    if t_s <= 0 or t_s < t_ref_s:
        return D_ref
    return D_ref * (t_ref_s / t_s) ** m


def chloride_at_depth_erfc(x_m: float, t_s: float, Cs: float, Cb: float, D_eff: float) -> float:
    if t_s <= 0:
        return Cb
    denom = 2.0 * math.sqrt(max(D_eff, 1e-30) * t_s)
    arg = x_m / max(denom, 1e-30)
    return Cb + (Cs - Cb) * math.erfc(arg)


def simulate_one_series(cfg: SimConfig, rng: np.random.Generator, series_id: int) -> Dict[str, np.ndarray]:
    Cs = rng.uniform(*cfg.Cs_range)
    C_th = rng.uniform(*cfg.Cth_range)
    cover = rng.uniform(*cfg.cover_range_m)
    D28 = rng.uniform(*cfg.D28_range)
    m = rng.uniform(*cfg.m_range)

    dt_s = weeks_to_seconds(cfg.dt_weeks)
    n_steps = int(years_to_seconds(cfg.years) // dt_s) + 1
    t_s = np.arange(n_steps, dtype=float) * dt_s
    t_ref_s = days_to_seconds(cfg.t_ref_days)

    chloride = np.zeros(n_steps, dtype=float)
    for i in range(n_steps):
        t_mid = max(t_s[i], 1.0)
        D_eff = D_time_dependent(D28, t_mid, t_ref_s, m)
        chloride[i] = chloride_at_depth_erfc(cover, t_s[i], Cs, cfg.Cb, D_eff)

    if cfg.noise_std > 0:
        chloride = np.clip(chloride + rng.normal(0.0, cfg.noise_std, size=n_steps), 0.0, None)

    return {
        "series_id": np.full(n_steps, series_id, dtype=int),
        "time_idx": np.arange(n_steps, dtype=int),
        "t_year": t_s / years_to_seconds(1.0),
        "chloride_rebar": chloride,
        "Cs": np.full(n_steps, Cs, dtype=float),
        "C_th": np.full(n_steps, C_th, dtype=float),
        "cover_m": np.full(n_steps, cover, dtype=float),
        "D28": np.full(n_steps, D28, dtype=float),
        "m_aging": np.full(n_steps, m, dtype=float),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated chloride ingress data.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "sim")
    parser.add_argument("--n-series", type=int, default=1000)
    parser.add_argument("--years", type=int, default=60)
    parser.add_argument("--dt-weeks", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20250111)
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    cfg = SimConfig(
        n_series=args.n_series,
        years=args.years,
        dt_weeks=args.dt_weeks,
        seed=args.seed,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(cfg.seed)
    df = pd.concat(
        [pd.DataFrame(simulate_one_series(cfg, rng, sid)) for sid in range(cfg.n_series)],
        ignore_index=True,
    )

    df["target_cont"] = np.maximum(0.0, df["chloride_rebar"] - df["C_th"])
    df["target_onset"] = (df["chloride_rebar"] >= df["C_th"]).astype(int)

    pq_path = out_dir / "chloride_long.parquet"
    csv_path = out_dir / "chloride_long.csv"
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"Saved simulated chloride data: {pq_path} rows={len(df)}")
    print(f"Saved CSV copy: {csv_path}")
    print(f"[Timing] Simulation: {time.perf_counter() - start_time:.4f} s")


if __name__ == "__main__":
    main()
