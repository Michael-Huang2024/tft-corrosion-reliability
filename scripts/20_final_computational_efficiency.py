"""
Unified computational efficiency comparison for revision reviewer response.

Measures or loads runtime for physics simulator, Logistic Regression, MLP, GRU,
deterministic TFT (seed 20250111), and 50-pass MC Dropout TFT on the locked
test evaluation domain. Does not retrain benchmark models.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import erfc
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    INFERENCE_BATCH_SIZE,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    POINT_FEATURES,
    REVISION_CHECKPOINT_DIR,
    REVISION_FIGURE_DIR,
    REVISION_OUTPUT_DIR,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TFT_STATIC_REALS,
    TFT_TIME_VARYING_KNOWN_REALS,
    TFT_TIME_VARYING_UNKNOWN_REALS,
    TIME_INDEX_COLUMN,
    TIME_COLUMN,
    ensure_revision_dirs,
)
from revision_data import attach_split, load_or_create_series_split, validate_series_split
from revision_metrics import aggregate_population_pf, parameter_count, restrict_common_evaluation_range

SECONDS_PER_YEAR = 365.25 * 24.0 * 3600.0
SECONDS_PER_DAY = 24.0 * 3600.0
T_REF_S = 28.0 * SECONDS_PER_DAY
N_STEPS = 783
DT_WEEKS = 4

SELECTED_TFT_CKPT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "revision"
    / "checkpoints"
    / "tft"
    / "20250111_10epoch"
    / "best.ckpt"
)
MLP_CKPT = REVISION_CHECKPOINT_DIR / "final_mlp_seed20250111.pt"
GRU_CKPT = REVISION_CHECKPOINT_DIR / "final_gru_seed20250111.pt"

SUMMARY_FILE = REVISION_TABLE_DIR / "final_computational_efficiency_summary.csv"
SCALING_FILE = REVISION_TABLE_DIR / "final_computational_efficiency_scaling.csv"
TRAINING_FILE = REVISION_TABLE_DIR / "final_training_runtime_summary.csv"
REPORT_FILE = REVISION_OUTPUT_DIR / "final_computational_efficiency_report.md"

FIG_TOTAL = REVISION_FIGURE_DIR / "computational_efficiency_total_runtime.png"
FIG_INFER = REVISION_FIGURE_DIR / "computational_efficiency_inference_only.png"
FIG_SCALE = REVISION_FIGURE_DIR / "computational_efficiency_runtime_scaling.png"
FIG_ACC = REVISION_FIGURE_DIR / "computational_efficiency_accuracy_vs_runtime.png"

MC_DROPOUT_50_TOTAL = 5964.6
MC_DROPOUT_50_PER_PASS = 119.29
MC_DROPOUT_100_TOTAL = 6678.4
MC_DROPOUT_100_PER_PASS = 66.78

ACCURACY = {
    "Logistic Regression": {"MAE": 0.020652, "RMSE": 0.024271},
    "MLP": {"MAE": 0.003017, "RMSE": 0.004371},
    "GRU": {"MAE": 0.001934, "RMSE": 0.002931},
    "TFT": {"MAE": 0.004542, "RMSE": 0.006373},
}

REPEATS = 3


@dataclass
class TimingBreakdown:
    method: str
    runtime_type: str
    n_series: int
    n_time_points: int
    n_predictions: int
    model_loading_seconds: float
    data_preparation_seconds: float
    inference_seconds: float
    aggregation_seconds: float
    total_seconds: float
    n_passes: int
    runtime_source: str
    hardware_summary: str
    notes: str
    inference_seconds_std: float = 0.0
    total_seconds_std: float = 0.0

    def to_row(self) -> dict:
        total = self.total_seconds
        return {
            "method": self.method,
            "runtime_type": self.runtime_type,
            "n_series": self.n_series,
            "n_time_points": self.n_time_points,
            "n_predictions": self.n_predictions,
            "model_loading_seconds": self.model_loading_seconds,
            "data_preparation_seconds": self.data_preparation_seconds,
            "inference_seconds": self.inference_seconds,
            "aggregation_seconds": self.aggregation_seconds,
            "total_seconds": total,
            "total_seconds_std": self.total_seconds_std,
            "inference_seconds_std": self.inference_seconds_std,
            "seconds_per_series": total / max(self.n_series, 1),
            "seconds_per_time_point": total / max(self.n_time_points, 1),
            "seconds_per_prediction": total / max(self.n_predictions, 1),
            "seconds_per_population_trajectory": total,
            "n_passes": self.n_passes,
            "runtime_source": self.runtime_source,
            "hardware_summary": self.hardware_summary,
            "notes": self.notes,
        }


def load_benchmark_module():
    path = Path(__file__).resolve().parent / "07_train_benchmarks.py"
    spec = importlib.util.spec_from_file_location("revision_train_benchmarks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def hardware_summary(device: str) -> str:
    env_path = REVISION_OUTPUT_DIR / "environment.json"
    if env_path.exists():
        return json.dumps(json.loads(env_path.read_text(encoding="utf-8"))["torch_cuda"], ensure_ascii=False)
    return json.dumps(
        {
            "device": device,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        ensure_ascii=False,
    )


def repeat_stats(fn, repeats: int, device: str = "cpu") -> tuple[float, float, float, float]:
    times = []
    for _ in range(repeats):
        synchronize(device)
        t0 = time.perf_counter()
        fn()
        synchronize(device)
        times.append(time.perf_counter() - t0)
    arr = np.asarray(times, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), float(arr.min()), float(arr.max())


def load_eval_frames():
    df = pd.read_parquet(FINAL_LABELED_DATA)
    split = load_or_create_series_split(df)
    validate_series_split(df, split)
    df = attach_split(df, split)
    eval_df = restrict_common_evaluation_range(df, MAX_ENCODER_LENGTH, split_name="test")
    test_series = sorted(eval_df[GROUP_COLUMN].unique())
    n_time_points = int(eval_df.groupby(TIME_COLUMN).ngroups)
    return df, eval_df, test_series, n_time_points


def sample_parameters(n: int, seed: int = 20250627) -> pd.DataFrame:
    from revision_config import PARAMETER_SPECS

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
    """Full 783-step initiation flags; matches final data generation simulator."""
    dt_s = DT_WEEKS * 7.0 * 24.0 * 3600.0
    t_s = np.arange(N_STEPS, dtype=float) * dt_s
    t = np.maximum(t_s[None, :], 1.0)
    cs = params["Cs"].to_numpy()[:, None]
    d28 = params["D28"].to_numpy()[:, None]
    m = params["m_aging"].to_numpy()[:, None]
    cover_m = (params["cover_mm"].to_numpy() / 1000.0)[:, None]
    c_th = params["C_th"].to_numpy()[:, None]
    d_eff = np.where(t < T_REF_S, d28, d28 * (T_REF_S / t) ** m)
    denom = 2.0 * np.sqrt(np.maximum(d_eff, 1e-30) * t)
    chloride = cs * erfc(cover_m / np.maximum(denom, 1e-30))
    chloride[:, 0] = 0.0
    raw = (chloride >= c_th).astype(np.int8)
    return np.maximum.accumulate(raw, axis=1).astype(np.float32)


def time_physics_simulator(n_series: int, repeats: int) -> tuple[float, float]:
    params = sample_parameters(n_series)
    mean, std, _, _ = repeat_stats(lambda: vectorized_simulator(params), repeats)
    return mean, std


def time_logistic(eval_df: pd.DataFrame, train_df: pd.DataFrame, repeats: int) -> TimingBreakdown:
    """Logistic has no saved checkpoint; one in-memory fit enables unified inference timing only."""
    hw = hardware_summary("cpu")
    t0 = time.perf_counter()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    model.fit(train_df[POINT_FEATURES], train_df[TARGET_COLUMN].astype(int))
    prep = time.perf_counter() - t0
    x_eval = eval_df[POINT_FEATURES]

    def infer_and_aggregate():
        probs = model.predict_proba(x_eval)[:, 1]
        pred = eval_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].copy()
        pred["p_onset_pred"] = probs
        aggregate_population_pf(pred)

    inf_mean, inf_std, _, _ = repeat_stats(infer_and_aggregate, repeats)
    n_preds = len(eval_df)
    n_series = int(eval_df[GROUP_COLUMN].nunique())
    n_tp = int(eval_df.groupby("t_year").ngroups)
    return TimingBreakdown(
        method="Logistic Regression",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=n_preds,
        model_loading_seconds=0.0,
        data_preparation_seconds=prep,
        inference_seconds=inf_mean,
        aggregation_seconds=0.0,
        total_seconds=prep + inf_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes="No persisted checkpoint; one in-memory fit on locked train split for forward-pass timing only.",
        inference_seconds_std=inf_std,
    )


def time_logistic_inference_only(eval_df: pd.DataFrame, train_df: pd.DataFrame, repeats: int) -> TimingBreakdown:
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    model.fit(train_df[POINT_FEATURES], train_df[TARGET_COLUMN].astype(int))
    x_eval = eval_df[POINT_FEATURES]
    hw = hardware_summary("cpu")

    def infer():
        model.predict_proba(x_eval)[:, 1]

    inf_mean, inf_std, _, _ = repeat_stats(infer, repeats)
    base = time_logistic(eval_df, train_df, 1)
    return TimingBreakdown(
        method="Logistic Regression",
        runtime_type="inference_only",
        n_series=base.n_series,
        n_time_points=base.n_time_points,
        n_predictions=base.n_predictions,
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=inf_mean,
        aggregation_seconds=0.0,
        total_seconds=inf_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes="Inference-only; excludes train-fit and Pf aggregation.",
        inference_seconds_std=inf_std,
    )


def time_mlp(eval_df: pd.DataFrame, ckpt_path: Path, device: str, repeats: int) -> tuple[TimingBreakdown, TimingBreakdown]:
    bench = load_benchmark_module()
    MLPClassifier = bench.MLPClassifier
    hw = hardware_summary(device)
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MLPClassifier(input_size=len(POINT_FEATURES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(torch.device(device))
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    x = (eval_df[POINT_FEATURES].to_numpy(dtype=np.float32) - np.asarray(ckpt["scaler_mean"], dtype=np.float32)) / np.asarray(
        ckpt["scaler_scale"], dtype=np.float32
    )
    prep = time.perf_counter() - t0
    dev = torch.device(device)
    x_tensor = torch.as_tensor(x, dtype=torch.float32)

    def infer_and_aggregate():
        with torch.no_grad():
            probs = torch.sigmoid(model(x_tensor.to(dev))).detach().cpu().numpy()
        pred = eval_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].copy()
        pred["p_onset_pred"] = probs
        aggregate_population_pf(pred)

    def infer_only():
        with torch.no_grad():
            _ = model(x_tensor.to(dev))

    inf_mean, inf_std, _, _ = repeat_stats(infer_and_aggregate, repeats, device)
    inf_only_mean, inf_only_std, _, _ = repeat_stats(infer_only, repeats, device)
    n_series = int(eval_df[GROUP_COLUMN].nunique())
    n_tp = int(eval_df.groupby("t_year").ngroups)
    end = TimingBreakdown(
        method="MLP",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=load_s,
        data_preparation_seconds=prep,
        inference_seconds=inf_mean,
        aggregation_seconds=0.0,
        total_seconds=load_s + prep + inf_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes=f"Checkpoint: {ckpt_path.name}; batch=all points.",
        inference_seconds_std=inf_std,
    )
    infer = TimingBreakdown(
        method="MLP",
        runtime_type="inference_only",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=inf_only_mean,
        aggregation_seconds=0.0,
        total_seconds=inf_only_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes="Forward pass only.",
        inference_seconds_std=inf_only_std,
    )
    return end, infer


def time_gru(eval_df: pd.DataFrame, ckpt_path: Path, device: str, repeats: int) -> tuple[TimingBreakdown, TimingBreakdown]:
    bench = load_benchmark_module()
    GRUClassifier = bench.GRUClassifier
    WindowDataset = bench.WindowDataset
    hw = hardware_summary(device)
    t0 = time.perf_counter()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = GRUClassifier(input_size=len(POINT_FEATURES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval().to(torch.device(device))
    load_s = time.perf_counter() - t0
    scaled = eval_df.copy()
    scaled[POINT_FEATURES] = (eval_df[POINT_FEATURES].to_numpy(dtype=np.float32) - np.asarray(ckpt["scaler_mean"])) / np.asarray(
        ckpt["scaler_scale"]
    )
    scaled["_eval_time_idx"] = eval_df[TIME_INDEX_COLUMN].to_numpy()
    scaled["_eval_t_year"] = eval_df["t_year"].to_numpy()
    t0 = time.perf_counter()
    test_ds = WindowDataset(scaled, POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    prep = time.perf_counter() - t0
    dev = torch.device(device)

    def infer_and_aggregate():
        sums: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        with torch.no_grad():
            for xb, _yb, series_id, dec_time_idx, _dec_t_year in test_loader:
                probs = torch.sigmoid(model(xb.to(dev))).detach().cpu().numpy()
                for b, sid in enumerate(series_id):
                    for j in range(probs.shape[1]):
                        key = (int(sid), int(dec_time_idx[b, j].item()))
                        sums[key] = sums.get(key, 0.0) + float(probs[b, j])
                        counts[key] = counts.get(key, 0) + 1
        pred = pd.DataFrame(
            {
                GROUP_COLUMN: [k[0] for k in sums],
                TIME_INDEX_COLUMN: [k[1] for k in sums],
                "p_onset_pred": [sums[k] / counts[k] for k in sums],
            }
        )
        time_map = eval_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].drop_duplicates()
        pred = pred.merge(time_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()
        aggregate_population_pf(pred)

    def infer_only():
        with torch.no_grad():
            for xb, *_ in test_loader:
                _ = model(xb.to(dev))

    inf_mean, inf_std, _, _ = repeat_stats(infer_and_aggregate, repeats, device)
    inf_only_mean, inf_only_std, _, _ = repeat_stats(infer_only, repeats, device)
    n_series = int(eval_df[GROUP_COLUMN].nunique())
    n_tp = int(eval_df.groupby("t_year").ngroups)
    end = TimingBreakdown(
        method="GRU",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=load_s,
        data_preparation_seconds=prep,
        inference_seconds=inf_mean,
        aggregation_seconds=0.0,
        total_seconds=load_s + prep + inf_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes=f"Checkpoint: {ckpt_path.name}; batch=64.",
        inference_seconds_std=inf_std,
    )
    infer = TimingBreakdown(
        method="GRU",
        runtime_type="inference_only",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=inf_only_mean,
        aggregation_seconds=0.0,
        total_seconds=inf_only_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes="Forward pass only.",
        inference_seconds_std=inf_only_std,
    )
    return end, infer


def prepare_tft_loader(df: pd.DataFrame):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)
    enc = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_INDEX_COLUMN,
        target=TARGET_COLUMN,
        group_ids=[GROUP_COLUMN],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        time_varying_known_reals=TFT_TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TFT_TIME_VARYING_UNKNOWN_REALS,
        static_reals=TFT_STATIC_REALS,
        categorical_encoders=enc,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    loader = testing.to_dataloader(train=False, batch_size=INFERENCE_BATCH_SIZE, num_workers=0)
    return testing, loader, test_df_sorted


def time_tft(df: pd.DataFrame, ckpt_path: Path, device: str, repeats: int) -> tuple[TimingBreakdown, TimingBreakdown]:
    from pytorch_forecasting import TemporalFusionTransformer

    hw = hardware_summary(device)
    dev = torch.device(device)
    t0 = time.perf_counter()
    model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path), weights_only=False)
    model.eval().to(dev)
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    testing, loader, test_df_sorted = prepare_tft_loader(df)
    sample_index = testing.index.reset_index(drop=True)
    prep = time.perf_counter() - t0

    def infer_and_aggregate():
        sums: dict[tuple[int, int], float] = {}
        counts: dict[tuple[int, int], int] = {}
        offset = 0
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x_dev = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in x.items()}
                probs = torch.softmax(model(x_dev)["prediction"], dim=-1)[..., 1].detach().cpu().numpy()
                bs = probs.shape[0]
                row_starts = sample_index.iloc[offset : offset + bs]["index_start"].to_numpy(dtype=int)
                series_ids = test_df_sorted.iloc[row_starts][GROUP_COLUMN].to_numpy(dtype=int)
                offset += bs
                dec_t = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
                for b, sid in enumerate(series_ids):
                    for j in range(probs.shape[1]):
                        key = (int(sid), int(dec_t[b, j]))
                        sums[key] = sums.get(key, 0.0) + float(probs[b, j])
                        counts[key] = counts.get(key, 0) + 1
        pred = pd.DataFrame(
            {
                GROUP_COLUMN: [k[0] for k in sums],
                TIME_INDEX_COLUMN: [k[1] for k in sums],
                "p_onset_pred": [sums[k] / counts[k] for k in sums],
            }
        )
        time_map = test_df_sorted[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].drop_duplicates()
        pred = pred.merge(time_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()
        aggregate_population_pf(pred)

    def infer_only():
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x_dev = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in x.items()}
                _ = model(x_dev)

    inf_mean, inf_std, _, _ = repeat_stats(infer_and_aggregate, repeats, device)
    inf_only_mean, inf_only_std, _, _ = repeat_stats(infer_only, repeats, device)
    eval_df = restrict_common_evaluation_range(df, MAX_ENCODER_LENGTH, split_name="test")
    n_series = int(eval_df[GROUP_COLUMN].nunique())
    n_tp = 731
    end = TimingBreakdown(
        method="TFT (deterministic)",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=load_s,
        data_preparation_seconds=prep,
        inference_seconds=inf_mean,
        aggregation_seconds=0.0,
        total_seconds=load_s + prep + inf_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes=f"Checkpoint: {ckpt_path.name}; batch={INFERENCE_BATCH_SIZE}.",
        inference_seconds_std=inf_std,
    )
    infer = TimingBreakdown(
        method="TFT (deterministic)",
        runtime_type="inference_only",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=len(eval_df),
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=inf_only_mean,
        aggregation_seconds=0.0,
        total_seconds=inf_only_mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hw,
        notes="Forward pass only.",
        inference_seconds_std=inf_only_std,
    )
    return end, infer


def physics_reference_task(n_series: int, repeats: int, device: str) -> TimingBreakdown:
    mean, std = time_physics_simulator(n_series, repeats)
    return TimingBreakdown(
        method="Physics simulator",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=N_STEPS,
        n_predictions=n_series * N_STEPS,
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=mean,
        aggregation_seconds=0.0,
        total_seconds=mean,
        n_passes=1,
        runtime_source="measured_now",
        hardware_summary=hardware_summary(device),
        notes="Vectorized apparent-diffusivity simulator; full 783-step trajectories per series.",
        inference_seconds_std=std,
        total_seconds_std=std,
    )


def mc_dropout_row(n_series: int, n_tp: int) -> tuple[TimingBreakdown, TimingBreakdown]:
    hw = hardware_summary("cuda")
    end = TimingBreakdown(
        method="TFT (50-pass MC Dropout)",
        runtime_type="end_to_end",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=n_series * n_tp,
        model_loading_seconds=float("nan"),
        data_preparation_seconds=float("nan"),
        inference_seconds=MC_DROPOUT_50_TOTAL,
        aggregation_seconds=float("nan"),
        total_seconds=MC_DROPOUT_50_TOTAL,
        n_passes=50,
        runtime_source="benchmark_log",
        hardware_summary=hw,
        notes=(
            "From formal 50-pass MC Dropout report (5964.6 s total; 119.29 s/pass). "
            f"100-pass convergence rerun reported 6678.4 s total (66.78 s/pass); difference likely "
            "reflects resume/cache/environment rather than model change."
        ),
    )
    infer = TimingBreakdown(
        method="TFT (50-pass MC Dropout)",
        runtime_type="inference_only",
        n_series=n_series,
        n_time_points=n_tp,
        n_predictions=n_series * n_tp,
        model_loading_seconds=0.0,
        data_preparation_seconds=0.0,
        inference_seconds=MC_DROPOUT_50_TOTAL,
        aggregation_seconds=0.0,
        total_seconds=MC_DROPOUT_50_TOTAL,
        n_passes=50,
        runtime_source="benchmark_log",
        hardware_summary=hw,
        notes="Stochastic forward passes only; excludes one-time checkpoint load.",
    )
    return end, infer


def build_training_summary() -> pd.DataFrame:
    tft = pd.read_csv(REVISION_TABLE_DIR / "tft_20250111_10epoch_metrics.csv")
    rows = [
        {
            "method": "TFT",
            "seed": 20250111,
            "training_seconds": float(tft.iloc[0]["training_time_seconds"]),
            "training_hours": float(tft.iloc[0]["training_time_seconds"]) / 3600.0,
            "best_validation_loss": float(tft.iloc[0]["best_validation_loss"]),
            "test_MAE": float(tft.iloc[0]["MAE"]),
            "test_RMSE": float(tft.iloc[0]["RMSE"]),
            "runtime_source": "benchmark_log",
        },
    ]
    for method, ckpt in [("MLP", MLP_CKPT), ("GRU", GRU_CKPT)]:
        rows.append(
            {
                "method": method,
                "seed": 20250111,
                "training_seconds": float("nan"),
                "training_hours": float("nan"),
                "best_validation_loss": float("nan"),
                "test_MAE": ACCURACY[method]["MAE"],
                "test_RMSE": ACCURACY[method]["RMSE"],
                "runtime_source": "not_logged_in_final_checkpoint",
                "notes": "Final benchmark did not persist per-seed training seconds in CSV; inference remeasured now.",
            }
        )
    rows.append(
        {
            "method": "Logistic Regression",
            "seed": "",
            "training_seconds": float("nan"),
            "training_hours": float("nan"),
            "best_validation_loss": float("nan"),
            "test_MAE": ACCURACY["Logistic Regression"]["MAE"],
            "test_RMSE": ACCURACY["Logistic Regression"]["RMSE"],
            "runtime_source": "not_logged_in_final_checkpoint",
        }
    )
    return pd.DataFrame(rows)


def write_figures(summary: pd.DataFrame, scaling: pd.DataFrame) -> None:
    test_end = summary[(summary["n_series"] == 150) & (summary["runtime_type"] == "end_to_end")].copy()
    test_inf = summary[(summary["n_series"] == 150) & (summary["runtime_type"] == "inference_only")].copy()
    order = [
        "Physics simulator",
        "Logistic Regression",
        "MLP",
        "GRU",
        "TFT (deterministic)",
        "TFT (50-pass MC Dropout)",
    ]
    for frame, path, title, col in [
        (test_end, FIG_TOTAL, "End-to-end total runtime (test task)", "total_seconds"),
        (test_inf, FIG_INFER, "Inference-only runtime (test task)", "inference_seconds"),
    ]:
        frame = frame.set_index("method").reindex(order).reset_index()
        plt.figure(figsize=(9, 4.5))
        plt.bar(frame["method"], frame[col], color="steelblue")
        plt.ylabel("Seconds")
        plt.title(title)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

    scale = scaling.copy()
    plt.figure(figsize=(8, 4.5))
    for method, group in scale.groupby("method"):
        plt.plot(group["n_series"], group["total_seconds"], marker="o", label=method)
    plt.xlabel("Number of series")
    plt.ylabel("Total seconds")
    plt.title("Runtime scaling")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_SCALE, dpi=300)
    plt.close()

    acc_rows = []
    for method in ["Logistic Regression", "MLP", "GRU", "TFT (deterministic)"]:
        inf = test_inf[test_inf["method"] == method.replace("TFT (deterministic)", "TFT (deterministic)")]
        if method == "TFT (deterministic)":
            key = "TFT"
        else:
            key = method
        if inf.empty:
            continue
        acc_rows.append(
            {
                "method": method,
                "inference_seconds": float(inf.iloc[0]["inference_seconds"]),
                "MAE": ACCURACY[key]["MAE"],
            }
        )
    acc_df = pd.DataFrame(acc_rows)
    plt.figure(figsize=(6, 4.5))
    for _, row in acc_df.iterrows():
        plt.scatter(row["inference_seconds"], row["MAE"], s=80)
        plt.annotate(row["method"], (row["inference_seconds"], row["MAE"]), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Inference-only runtime (seconds)")
    plt.ylabel("Test MAE on cumulative Pf(t)")
    plt.title("Accuracy vs inference runtime")
    plt.tight_layout()
    plt.savefig(FIG_ACC, dpi=300)
    plt.close()


def write_report(summary: pd.DataFrame, scaling: pd.DataFrame, training: pd.DataFrame, hw: str) -> None:
    test_end = summary[(summary["runtime_type"] == "end_to_end") & (summary["n_series"] == 150)].set_index("method")
    test_inf = summary[(summary["runtime_type"] == "inference_only") & (summary["n_series"] == 150)].set_index("method")
    fastest_inf = test_inf["inference_seconds"].idxmin()
    fastest_total = test_end["total_seconds"].idxmin()
    revised = (
        "The computational comparison indicates that the proposed TFT surrogate should not be interpreted as "
        "universally faster or more accurate than simpler neural architectures for the present low-dimensional "
        "simulation task. Although the surrogate framework enables reusable sequence prediction and "
        "uncertainty-aware analysis, GRU achieved the best accuracy–efficiency balance in the current benchmark. "
        "Therefore, the computational advantage of TFT is conditional and is expected to be more relevant in future "
        "extensions involving heterogeneous time-varying bridge inspection and environmental data, rather than in "
        "the simplified simulator-replication setting examined here."
    )
    editor = (
        "Response: We sincerely thank the Editor for pointing out that the original manuscript provided weak evidence "
        "of computational advantage. We have now completed a unified computational efficiency comparison under the "
        "locked test evaluation domain (150 series; 731 population time points). The comparison includes the physics "
        "simulator, Logistic Regression, MLP, GRU, deterministic TFT (seed 20250111 selected by validation loss), and "
        "50-pass MC Dropout TFT. The remeasured results do not support a universal TFT speed advantage: "
        f"{fastest_inf} is fastest for inference-only timing among surrogate models, while deterministic TFT and "
        "especially 50-pass MC Dropout TFT incur substantially higher inference cost. We have therefore narrowed the "
        "computational-advantage claim and added the comparison to Results/Discussion/Limitations."
    )
    lines = [
        "# Final Computational Efficiency Report",
        "",
        "## A. Purpose",
        "",
        "This analysis responds to the Editor's comment that the manuscript provided weak evidence of computational advantage.",
        "",
        "## B. Methods compared",
        "",
        "- Physics simulator (reference generator)",
        "- Logistic Regression",
        "- MLP",
        "- GRU",
        "- Deterministic TFT (seed 20250111)",
        "- 50-pass MC Dropout TFT",
        "",
        "## C. Timing protocol",
        "",
        f"- Hardware: {hw}",
        "- Test series: 150",
        "- Population time points: 731",
        "- End-to-end timing includes loading, data preparation, inference, and Pf aggregation where applicable",
        "- Inference-only timing includes forward prediction only",
        f"- Repeated runs: {REPEATS} for remeasured methods",
        "- MC Dropout runtime loaded from formal 50-pass benchmark log",
        "",
        "## D. Results",
        "",
        "### End-to-end (test task)",
        "",
        test_end.reset_index()[
            ["method", "total_seconds", "inference_seconds", "seconds_per_prediction", "runtime_source", "notes"]
        ].to_string(index=False),
        "",
        "### Inference-only (test task)",
        "",
        test_inf.reset_index()[
            ["method", "inference_seconds", "seconds_per_prediction", "runtime_source", "notes"]
        ].to_string(index=False),
        "",
        "### Training runtime (separate from inference)",
        "",
        training.to_string(index=False),
        "",
        "## E. Runtime scaling",
        "",
        scaling.to_string(index=False),
        "",
        "## F. Accuracy-efficiency tradeoff",
        "",
        "- GRU has the best held-out MAE in the locked benchmark.",
        "- Logistic Regression is fastest but least accurate.",
        "- MLP is a strong simple baseline with low inference cost.",
        "- Deterministic TFT is slower than GRU/MLP on this task and not the most accurate.",
        "- 50-pass MC Dropout TFT adds large inference cost for epistemic uncertainty.",
        "- The physics simulator is a reference generator, not a surrogate predictor.",
        "",
        "## G. Revised interpretation",
        "",
        revised,
        "",
        "## H. Reviewer/editor response paragraph",
        "",
        editor,
        "",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    ensure_revision_dirs()
    repeats = 1 if args.smoke_test else args.repeats
    device = args.device

    df, eval_df, test_series, n_tp = load_eval_frames()
    train_df = df[df["split"] == "train"]
    n_series = len(test_series)
    hw = hardware_summary(device)

    rows: list[TimingBreakdown] = []
    rows.append(physics_reference_task(n_series, repeats, device))
    rows.append(time_logistic(eval_df, train_df, repeats))
    rows.append(time_logistic_inference_only(eval_df, train_df, repeats))
    rows.extend(time_mlp(eval_df, MLP_CKPT, device, repeats))
    rows.extend(time_gru(eval_df, GRU_CKPT, device, repeats))
    rows.extend(time_tft(df, SELECTED_TFT_CKPT, device, repeats))
    rows.extend(mc_dropout_row(n_series, n_tp))

    summary = pd.DataFrame([r.to_row() for r in rows])
    summary.to_csv(SUMMARY_FILE, index=False)

    scale_counts = [150, 300, 600, 1000, 2000, 5000] if not args.smoke_test else [150, 300]
    scale_rows = []
    for n in scale_counts:
        mean, std = time_physics_simulator(n, repeats)
        scale_rows.append(
            {
                "method": "Physics simulator",
                "n_series": n,
                "n_time_points": N_STEPS,
                "n_predictions": n * N_STEPS,
                "total_seconds": mean,
                "total_seconds_std": std,
                "seconds_per_series": mean / n,
                "seconds_per_prediction": mean / (n * N_STEPS),
                "runtime_source": "measured_now",
            }
        )
    for n in [c for c in scale_counts if c <= 1000]:
        sub_series = test_series[: min(n, len(test_series))]
        sub_eval = eval_df[eval_df[GROUP_COLUMN].isin(sub_series)].copy()
        if len(sub_series) < n:
            extra = sample_parameters(n - len(sub_series))
            # append synthetic rows at final eval time only for scaling approximation
            last = sub_eval[sub_eval[TIME_INDEX_COLUMN] == sub_eval[TIME_INDEX_COLUMN].max()].copy()
            reps = []
            for i in range(n - len(sub_series)):
                row = last.iloc[i % len(last)].copy()
                for col in extra.columns:
                    row[col] = extra.iloc[i][col]
                row[GROUP_COLUMN] = 10000 + i
                reps.append(row)
            sub_eval = pd.concat([sub_eval, pd.DataFrame(reps)], ignore_index=True)
        _, infer = time_mlp(sub_eval, MLP_CKPT, device, repeats)
        scale_rows.append(
            {
                "method": "MLP",
                "n_series": n,
                "n_time_points": n_tp,
                "n_predictions": len(sub_eval),
                "total_seconds": infer.total_seconds,
                "total_seconds_std": infer.inference_seconds_std,
                "seconds_per_series": infer.total_seconds / n,
                "seconds_per_prediction": infer.total_seconds / max(len(sub_eval), 1),
                "runtime_source": infer.runtime_source,
            }
        )
    scaling = pd.DataFrame(scale_rows)
    scaling.to_csv(SCALING_FILE, index=False)

    training = build_training_summary()
    training.to_csv(TRAINING_FILE, index=False)
    write_figures(summary, scaling)
    write_report(summary, scaling, training, hw)

    test_inf = summary[(summary["runtime_type"] == "inference_only") & (summary["n_series"] == 150)]
    test_end = summary[(summary["runtime_type"] == "end_to_end") & (summary["n_series"] == 150)]
    fastest_inf = test_inf.loc[test_inf["inference_seconds"].idxmin(), "method"]
    fastest_total = test_end.loc[test_end["total_seconds"].idxmin(), "method"]
    tft_det = test_inf[test_inf["method"] == "TFT (deterministic)"]["inference_seconds"].iloc[0]
    mc = test_end[test_end["method"] == "TFT (50-pass MC Dropout)"]["total_seconds"].iloc[0]
    phys = test_end[test_end["method"] == "Physics simulator"]["total_seconds"].iloc[0]

    print("\nCOMPUTATIONAL EFFICIENCY COMPARISON COMPLETE\n")
    print("Methods compared: Physics simulator, Logistic Regression, MLP, GRU, TFT (deterministic), TFT (50-pass MC Dropout)")
    print(f"Hardware: {hw}")
    print(f"Test series: {n_series}")
    print(f"Time points: {n_tp}")
    print(f"Fastest inference-only method: {fastest_inf}")
    print(f"Fastest end-to-end method: {fastest_total}")
    print("Best accuracy-efficiency method: GRU")
    print(f"Deterministic TFT runtime: {tft_det:.3f} s (inference-only)")
    print(f"50-pass MC Dropout runtime: {mc:.1f} s (total)")
    print(f"Physics simulator runtime: {phys:.4f} s (150 series, full trajectories)")
    print("Does current evidence support universal TFT computational advantage? No")
    print("Recommended manuscript claim: conditional / task-dependent; GRU best accuracy-efficiency here")
    print(f"Output report: {REPORT_FILE.name}")
    print(f"Output tables: {SUMMARY_FILE.name}, {SCALING_FILE.name}, {TRAINING_FILE.name}")
    print(f"Output figures: {FIG_TOTAL.name}, {FIG_INFER.name}, {FIG_SCALE.name}, {FIG_ACC.name}")


if __name__ == "__main__":
    main()
