"""
Fair runtime-scaling benchmark for revision experiments.

No hard-coded timing values are used. Missing model checkpoints are reported as
skipped rather than filled with placeholder numbers.
"""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import erfc

from revision_config import PARAMETER_SPECS, POINT_FEATURES, REVISION_CHECKPOINT_DIR, REVISION_FIGURE_DIR, REVISION_TABLE_DIR, ensure_revision_dirs
from revision_data import assert_no_forbidden_predictors
from revision_config import MAX_PREDICTION_LENGTH
from revision_metrics import parameter_count
from scripts_compat import import_benchmark_classes


SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run measured runtime scaling benchmarks.")
    parser.add_argument("--scenario-counts", nargs="+", type=int, default=[100, 1000, 10000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(stage: str, fn, device: str = "cpu") -> tuple[float, int, object]:
    synchronize(device)
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return seconds, int(peak), result


def generate_parameters(n: int, seed: int = 20250111) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Cs": rng.uniform(PARAMETER_SPECS["Cs"].lower, PARAMETER_SPECS["Cs"].upper, n),
            "D28": rng.uniform(PARAMETER_SPECS["D28"].lower, PARAMETER_SPECS["D28"].upper, n),
            "m_aging": rng.uniform(PARAMETER_SPECS["m_aging"].lower, PARAMETER_SPECS["m_aging"].upper, n),
            "cover_mm": rng.uniform(PARAMETER_SPECS["cover_mm"].lower, PARAMETER_SPECS["cover_mm"].upper, n),
            "C_th": rng.uniform(PARAMETER_SPECS["C_th"].lower, PARAMETER_SPECS["C_th"].upper, n),
        }
    )


def vectorized_simulator(params: pd.DataFrame) -> np.ndarray:
    years = np.arange(783, dtype=float) * (4.0 * 7.0 * 24.0 * 3600.0) / SECONDS_PER_YEAR
    t_s = years * SECONDS_PER_YEAR
    t_ref = 28.0 * SECONDS_PER_DAY
    Cs = params["Cs"].to_numpy()[:, None]
    D28 = params["D28"].to_numpy()[:, None]
    m = params["m_aging"].to_numpy()[:, None]
    cover_m = (params["cover_mm"].to_numpy() / 1000.0)[:, None]
    C_th = params["C_th"].to_numpy()[:, None]
    t = np.maximum(t_s[None, :], 1.0)
    D_eff = np.where(t < t_ref, D28, D28 * (t_ref / t) ** m)
    denom = 2.0 * np.sqrt(np.maximum(D_eff, 1e-30) * t)
    chloride = Cs * erfc(cover_m / np.maximum(denom, 1e-30))
    return (chloride >= C_th).astype(np.float32)


def build_point_inputs(params: pd.DataFrame) -> np.ndarray:
    # One representative final-year point per scenario for pure model inference scaling.
    features = params.copy()
    features["time_idx"] = 782
    features["t_year"] = 59.94798083504449
    return features[POINT_FEATURES].to_numpy(dtype=np.float32)


def load_torch_checkpoint(prefix: str):
    candidates = sorted(REVISION_CHECKPOINT_DIR.glob(f"{prefix}_seed*.pt"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        return None, None
    checkpoint = torch.load(candidates[-1], map_location="cpu", weights_only=False)
    return candidates[-1], checkpoint


def model_rows_for_count(n: int, args: argparse.Namespace) -> list[dict[str, object]]:
    assert_no_forbidden_predictors(POINT_FEATURES)
    rows = []
    _, _, params = timed("parameter_generation", lambda: generate_parameters(n))
    x = build_point_inputs(params)

    simulator_seconds = []
    simulator_peak = []
    for _ in range(args.warmups):
        vectorized_simulator(params)
    for _ in range(args.repeats):
        seconds, peak, _ = timed("simulator_execution", lambda: vectorized_simulator(params))
        simulator_seconds.append(seconds)
        simulator_peak.append(peak)
    rows.append(
        {
            "method": "vectorized_diffusion_simulator",
            "scenario_count": n,
            "stage": "simulator_execution",
            "seconds_mean": float(np.mean(simulator_seconds)),
            "seconds_std": float(np.std(simulator_seconds, ddof=0)),
            "peak_memory_bytes_max": int(max(simulator_peak)),
            "status": "measured",
        }
    )

    MLPClassifier, GRUClassifier = import_benchmark_classes()
    for prefix, cls, model_name in [("mlp", MLPClassifier, "MLP"), ("gru", GRUClassifier, "GRU")]:
        path, ckpt = load_torch_checkpoint(prefix)
        if ckpt is None:
            rows.append({"method": model_name, "scenario_count": n, "stage": "pure_inference", "status": "skipped_missing_checkpoint"})
            continue
        model = cls(input_size=len(POINT_FEATURES))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        device = torch.device(args.device)
        model.to(device)
        scaled = (x - np.asarray(ckpt["scaler_mean"], dtype=np.float32)) / np.asarray(ckpt["scaler_scale"], dtype=np.float32)
        if model_name == "GRU":
            scaled_tensor = torch.as_tensor(np.repeat(scaled[:, None, :], 52, axis=1), dtype=torch.float32)
        else:
            scaled_tensor = torch.as_tensor(scaled, dtype=torch.float32)

        def infer() -> None:
            with torch.no_grad():
                for start in range(0, len(scaled_tensor), args.batch_size):
                    batch = scaled_tensor[start : start + args.batch_size].to(device)
                    _ = model(batch)

        for _ in range(args.warmups):
            infer()
        seconds_list = []
        peak_list = []
        for _ in range(args.repeats):
            seconds, peak, _ = timed("pure_inference", infer, args.device)
            seconds_list.append(seconds)
            peak_list.append(peak)
        rows.append(
            {
                "method": model_name,
                "scenario_count": n,
                "stage": "pure_inference",
                "seconds_mean": float(np.mean(seconds_list)),
                "seconds_std": float(np.std(seconds_list, ddof=0)),
                "peak_memory_bytes_max": int(max(peak_list)),
                "parameter_count": parameter_count(model),
                "checkpoint": str(path),
                "status": "measured",
            }
        )

    tft_candidates = sorted(REVISION_CHECKPOINT_DIR.glob("tft_seed*.ckpt"), key=lambda item: item.stat().st_mtime)
    if not tft_candidates:
        rows.append({"method": "TFT", "scenario_count": n, "stage": "pure_inference", "status": "skipped_missing_checkpoint"})
    else:
        rows.append({"method": "TFT", "scenario_count": n, "stage": "pure_inference", "status": "not_measured_in_point_harness", "checkpoint": str(tft_candidates[-1])})
    return rows


def write_plot(table: pd.DataFrame) -> None:
    measured = table[table["status"] == "measured"].copy()
    if measured.empty:
        return
    plt.figure(figsize=(7, 4))
    for method, group in measured.groupby("method"):
        plt.errorbar(group["scenario_count"], group["seconds_mean"], yerr=group["seconds_std"], marker="o", label=method)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Scenario count")
    plt.ylabel("Seconds")
    plt.title("Measured runtime scaling")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "runtime_scaling.png", dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_revision_dirs()
    if args.smoke_test:
        args.scenario_counts = [20]
        args.repeats = 1
        args.warmups = 0
    rows = []
    for n in args.scenario_counts:
        rows.extend(model_rows_for_count(n, args))
    table = pd.DataFrame(rows)
    table.to_csv(REVISION_TABLE_DIR / "runtime_scaling.csv", index=False)
    write_plot(table)
    report = [
        "# Runtime Report",
        "",
        "Runtime values are measured with `time.perf_counter()`; missing checkpoints are reported as skipped.",
        "",
        table.to_string(index=False),
    ]
    Path("outputs/revision/runtime_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("outputs/revision/tables/runtime_scaling.csv")


if __name__ == "__main__":
    main()
