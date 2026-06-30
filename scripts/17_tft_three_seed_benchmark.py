"""
Three-seed TFT 10-epoch benchmark: train, evaluate, summarize, rebuild comparison.

Seed 20250113 uses the completed diagnostic outputs under 20250113_10epoch_test.
Seeds 20250111 and 20250112 train sequentially with identical configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    REVISION_FIGURE_DIR,
    REVISION_LOG_DIR,
    REVISION_OUTPUT_DIR,
    REVISION_PREDICTION_DIR,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TFT_STATIC_REALS,
    TFT_TIME_VARYING_KNOWN_REALS,
    TFT_TIME_VARYING_UNKNOWN_REALS,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)
from revision_data import attach_split, load_or_create_series_split, validate_series_split
from revision_metrics import evaluate_point_predictions, parameter_count

MAX_EPOCHS = 10
MIN_EPOCHS = 5
PATIENCE = 3
TRAIN_BATCH = 32
INFER_BATCH = 64
LEARNING_RATE = 3e-4
ENCODER_LENGTH = 52
PREDICTION_HORIZON = 13
HIDDEN_SIZE = 32
ATTENTION_HEADS = 4
DROPOUT = 0.1
ALL_SEEDS = (20250111, 20250112, 20250113)


@dataclass
class SeedPaths:
    seed: int
    ckpt_dir: Path
    train_log: Path
    eval_log: Path
    epoch_csv: Path
    pred_pf: Path
    pred_points: Path
    metrics_csv: Path


@dataclass
class SeedReport:
    seed: int
    training_completed: bool = False
    evaluation_completed: bool = False
    epochs_completed: int = 0
    early_stopping_triggered: bool = False
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    final_training_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    max_error: float | None = None
    year_of_max_error: float | None = None
    final_year_error: float | None = None
    total_training_seconds: float | None = None
    avg_seconds_per_epoch: float | None = None
    pure_inference_seconds: float | None = None
    parameter_count: int | None = None
    evaluation_time_points: int | None = None
    peak_gpu_memory_gb: float | None = None
    exit_code: int = 1
    failure_stage: str | None = None
    progress_bar_ok: bool = False


def seed_paths(seed: int) -> SeedPaths:
    ckpt_name = "20250113_10epoch_test" if seed == 20250113 else f"{seed}_10epoch"
    ckpt_dir = REVISION_OUTPUT_DIR / "checkpoints" / "tft" / ckpt_name
    tag = f"{seed}_10epoch"
    return SeedPaths(
        seed=seed,
        ckpt_dir=ckpt_dir,
        train_log=REVISION_LOG_DIR / f"tft_{tag}_training.log",
        eval_log=REVISION_LOG_DIR / f"tft_{tag}_evaluation.log",
        epoch_csv=ckpt_dir / "epoch_metrics.csv",
        pred_pf=REVISION_PREDICTION_DIR / f"tft_{tag}.csv",
        pred_points=REVISION_PREDICTION_DIR / f"tft_{tag}_points.csv",
        metrics_csv=REVISION_TABLE_DIR / f"tft_{tag}_metrics.csv",
    )


def setup_logger(path: Path, name: str) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    if hasattr(sh.stream, "reconfigure"):
        try:
            sh.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def gpu_snapshot() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info(0)
    return {
        "cuda_available": True,
        "device_name": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 4),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 4),
        "free_gb": round(free / 1024**3, 4),
        "total_gb": round(total / 1024**3, 4),
    }


def assert_no_other_tft_processes(current_pid: int, logger: logging.Logger) -> None:
    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "CommandLine like '%python%'", "get", "ProcessId,CommandLine", "/format:csv"],
            text=True,
            errors="replace",
        )
        others = []
        for ln in out.splitlines():
            if not ln.strip() or "ProcessId" in ln:
                continue
            if str(current_pid) in ln:
                continue
            if any(k in ln for k in ("14_tft_stable", "07_train_benchmarks", "16_tft_10epoch", "17_tft_three_seed")):
                if str(current_pid) in ln:
                    continue
                others.append(ln)
        if others:
            raise RuntimeError(f"Other TFT processes detected: {others}")
        logger.info("Pre-flight: no conflicting TFT processes.")
    except FileNotFoundError:
        logger.info("Pre-flight: wmic unavailable; skipped process scan.")


def preflight_gpu(logger: logging.Logger) -> None:
    logger.info("GPU snapshot: %s", gpu_snapshot())
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            timeout=30,
        ).strip()
        logger.info("nvidia-smi: %s", smi)
    except Exception as exc:
        logger.warning("nvidia-smi failed: %s", exc)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available.")
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    logger.info("CUDA tensor test: PASS")
    del x, y
    torch.cuda.empty_cache()


def safe_copy(src: Path, dst: Path) -> None:
    import shutil

    s, d = src.resolve(), dst.resolve()
    if s == d:
        return
    if s.exists():
        shutil.copy2(s, d)


def load_data() -> pd.DataFrame:
    ensure_revision_dirs()
    df = pd.read_parquet(FINAL_LABELED_DATA)
    split = load_or_create_series_split(df)
    validate_series_split(df, split)
    return attach_split(df, split)


def build_datasets(df: pd.DataFrame):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    enc = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    training = TimeSeriesDataSet(
        train_df,
        time_idx=TIME_INDEX_COLUMN,
        target=TARGET_COLUMN,
        group_ids=[GROUP_COLUMN],
        max_encoder_length=ENCODER_LENGTH,
        max_prediction_length=PREDICTION_HORIZON,
        time_varying_known_reals=TFT_TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=TFT_TIME_VARYING_UNKNOWN_REALS,
        static_reals=TFT_STATIC_REALS,
        categorical_encoders=enc,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    return training, validation, testing, test_df


class EpochMetricsCallback:
    def __init__(self, logger: logging.Logger, csv_path: Path, peak_mem: list[float]) -> None:
        import lightning.pytorch as pl

        self._logger = logger
        self._csv_path = csv_path
        self._peak_mem = peak_mem
        self._rows: list[dict] = []
        self._best_val = float("inf")
        self._best_epoch: int | None = None
        self._train_start = time.perf_counter()
        self._epoch_start = self._train_start
        self._Base = pl.Callback

    def make(self):
        outer = self

        class _Cb(outer._Base):
            def on_train_epoch_start(self, trainer, pl_module) -> None:
                outer._epoch_start = time.perf_counter()

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                metrics = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in trainer.callback_metrics.items()}
                gpu = gpu_snapshot()
                if gpu.get("cuda_available"):
                    outer._peak_mem[0] = max(outer._peak_mem[0], float(gpu["reserved_gb"]))
                val_loss = metrics.get("val_loss", float("nan"))
                train_loss = metrics.get("train_loss_epoch", metrics.get("train_loss", float("nan")))
                if val_loss < outer._best_val:
                    outer._best_val = val_loss
                    outer._best_epoch = int(trainer.current_epoch)
                elapsed = time.perf_counter() - outer._train_start
                epoch_elapsed = time.perf_counter() - outer._epoch_start
                epochs_done = int(trainer.current_epoch) + 1
                eta = epoch_elapsed * max(MAX_EPOCHS - epochs_done, 0)
                row = {
                    "epoch": int(trainer.current_epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": outer._best_val,
                    "best_epoch": outer._best_epoch,
                    "elapsed_seconds": round(elapsed, 1),
                    "epoch_seconds": round(epoch_elapsed, 1),
                    "eta_seconds": round(eta, 1),
                    "gpu_allocated_gb": gpu.get("allocated_gb"),
                    "gpu_reserved_gb": gpu.get("reserved_gb"),
                }
                outer._rows.append(row)
                with outer._csv_path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    w.writeheader()
                    w.writerows(outer._rows)
                outer._logger.info(
                    "Epoch %s/%s | train_loss=%.6f val_loss=%.6f best_val_loss=%.6f (epoch %s) | "
                    "epoch_time=%.0fs eta=%.0fs | checkpoint saved if improved",
                    epochs_done,
                    MAX_EPOCHS,
                    train_loss,
                    val_loss,
                    outer._best_val,
                    outer._best_epoch,
                    epoch_elapsed,
                    eta,
                )
                for h in outer._logger.handlers:
                    h.flush()

        return _Cb()


def train_seed(seed: int) -> SeedReport:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import CrossEntropy

    paths = seed_paths(seed)
    report = SeedReport(seed=seed)
    logger = setup_logger(paths.train_log, f"tft_train_{seed}")

    try:
        import os

        assert_no_other_tft_processes(os.getpid(), logger)
        preflight_gpu(logger)

        paths.ckpt_dir.mkdir(parents=True, exist_ok=True)
        pl.seed_everything(seed, workers=True)
        training, validation, _, _ = build_datasets(load_data())

        train_loader = training.to_dataloader(train=True, batch_size=TRAIN_BATCH, num_workers=0, persistent_workers=False)
        val_loader = validation.to_dataloader(train=False, batch_size=TRAIN_BATCH, num_workers=0, persistent_workers=False)

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=LEARNING_RATE,
            hidden_size=HIDDEN_SIZE,
            attention_head_size=ATTENTION_HEADS,
            dropout=DROPOUT,
            hidden_continuous_size=16,
            output_size=2,
            loss=CrossEntropy(),
        )

        best_cb = ModelCheckpoint(dirpath=str(paths.ckpt_dir), filename="best", monitor="val_loss", mode="min", save_top_k=1)
        last_cb = ModelCheckpoint(dirpath=str(paths.ckpt_dir), filename="last", save_top_k=0, save_last=True)
        early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")
        peak = [0.0]
        epoch_cb = EpochMetricsCallback(logger, paths.epoch_csv, peak).make()

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            min_epochs=MIN_EPOCHS,
            accelerator="gpu",
            devices=1,
            callbacks=[best_cb, last_cb, early_stop, epoch_cb],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
            num_sanity_val_steps=0,
            log_every_n_steps=10,
            precision="32-true",
        )

        logger.info("Training seed=%s ckpt_dir=%s", seed, paths.ckpt_dir)
        t0 = time.perf_counter()
        trainer.fit(model, train_loader, val_loader, weights_only=False)
        report.total_training_seconds = time.perf_counter() - t0
        report.epochs_completed = int(trainer.current_epoch) + 1
        report.early_stopping_triggered = report.epochs_completed < MAX_EPOCHS
        report.best_validation_loss = float(best_cb.best_model_score.detach().cpu()) if best_cb.best_model_score is not None else None
        report.best_epoch = int(pd.read_csv(paths.epoch_csv)["best_epoch"].iloc[-1]) if paths.epoch_csv.exists() else None
        metrics = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in trainer.callback_metrics.items()}
        report.final_training_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
        report.peak_gpu_memory_gb = peak[0] or gpu_snapshot().get("reserved_gb")
        report.avg_seconds_per_epoch = report.total_training_seconds / max(report.epochs_completed, 1)
        report.parameter_count = parameter_count(model)

        best_path = paths.ckpt_dir / "best.ckpt"
        last_path = paths.ckpt_dir / "last.ckpt"
        if best_cb.best_model_path:
            safe_copy(Path(best_cb.best_model_path), best_path)
        if last_cb.last_model_path:
            safe_copy(Path(last_cb.last_model_path), last_path)

        for req in (best_path, last_path, paths.epoch_csv):
            if not req.exists():
                raise FileNotFoundError(f"Missing after training: {req}")

        report.training_completed = True
        report.progress_bar_ok = True
        report.exit_code = 0
        logger.info("Training complete seed=%s epochs=%s best_epoch=%s", seed, report.epochs_completed, report.best_epoch)
    except Exception:
        report.failure_stage = "training"
        logger.error("Training failed:\n%s", traceback.format_exc())
        report.exit_code = 1
    finally:
        for h in logger.handlers:
            h.flush()
    return report


def evaluate_seed(seed: int, training_seconds: float | None = None) -> SeedReport:
    from pytorch_forecasting import TemporalFusionTransformer

    paths = seed_paths(seed)
    report = SeedReport(seed=seed)
    logger = setup_logger(paths.eval_log, f"tft_eval_{seed}")

    try:
        best_path = paths.ckpt_dir / "best.ckpt"
        if not best_path.exists():
            raise FileNotFoundError(f"best.ckpt missing: {best_path}")

        if paths.epoch_csv.exists():
            hist = pd.read_csv(paths.epoch_csv)
            report.epochs_completed = len(hist)
            report.best_epoch = int(hist["best_epoch"].iloc[-1])
            report.best_validation_loss = float(hist["best_val_loss"].iloc[-1])
            report.total_training_seconds = float(hist["elapsed_seconds"].iloc[-1])
            report.avg_seconds_per_epoch = report.total_training_seconds / max(report.epochs_completed, 1)
            report.early_stopping_triggered = report.epochs_completed < MAX_EPOCHS

        if training_seconds is not None:
            report.total_training_seconds = training_seconds

        df = load_data()
        _, _, testing, test_df = build_datasets(df)
        test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)

        model = TemporalFusionTransformer.load_from_checkpoint(str(best_path), weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        report.parameter_count = parameter_count(model)

        test_loader = testing.to_dataloader(train=False, batch_size=INFER_BATCH, num_workers=0, persistent_workers=False)
        sums: dict[tuple[int, int], float] = defaultdict(float)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        sample_index = testing.index.reset_index(drop=True)
        offset = 0
        t0 = time.perf_counter()

        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
                probs = torch.softmax(model(x_dev)["prediction"], dim=-1)[..., 1].detach().cpu().numpy()
                if not np.isfinite(probs).all():
                    raise ValueError("Non-finite predictions")
                bs = probs.shape[0]
                row_starts = sample_index.iloc[offset : offset + bs]["index_start"].to_numpy(dtype=int)
                series_ids = test_df_sorted.iloc[row_starts][GROUP_COLUMN].to_numpy(dtype=int)
                offset += bs
                dec_t = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
                for b, sid in enumerate(series_ids):
                    for j in range(probs.shape[1]):
                        key = (int(sid), int(dec_t[b, j]))
                        sums[key] += float(probs[b, j])
                        counts[key] += 1

        report.pure_inference_seconds = time.perf_counter() - t0
        pred = pd.DataFrame(
            {GROUP_COLUMN: [k[0] for k in sums], TIME_INDEX_COLUMN: [k[1] for k in sums], "p_onset_pred": [sums[k] / counts[k] for k in sums]}
        )
        pred = pred.merge(test_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].drop_duplicates(), on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()

        result = evaluate_point_predictions(
            pred,
            model_name="TFT",
            seed=seed,
            parameter_count_value=report.parameter_count,
            training_time_seconds=report.total_training_seconds,
            inference_time_seconds=report.pure_inference_seconds,
        )
        pf = result.pf_curve
        m = result.metrics
        report.evaluation_time_points = int(m["evaluation_time_points"])
        if report.evaluation_time_points != 731:
            raise ValueError(f"Expected 731 evaluation points, got {report.evaluation_time_points}")

        pf.to_csv(paths.pred_pf, index=False)
        pred.to_csv(paths.pred_points, index=False)
        row = dict(m)
        row["best_epoch"] = report.best_epoch
        row["best_validation_loss"] = report.best_validation_loss
        row["epochs_completed"] = report.epochs_completed
        row["early_stopping_triggered"] = report.early_stopping_triggered
        pd.DataFrame([row]).to_csv(paths.metrics_csv, index=False)

        report.test_mae = float(m["MAE"])
        report.test_rmse = float(m["RMSE"])
        report.max_error = float(m["max_abs_error"])
        report.year_of_max_error = float(m["year_of_max_error"])
        report.final_year_error = float(m["final_year_abs_error"])
        report.evaluation_completed = True
        report.training_completed = True
        report.exit_code = 0

        # also write legacy final_pf path for benchmark rebuild
        legacy = REVISION_PREDICTION_DIR / f"final_pf_tft_seed{seed}.csv"
        pf.to_csv(legacy, index=False)

        logger.info("Eval seed=%s MAE=%.6f RMSE=%.6f points=%s", seed, report.test_mae, report.test_rmse, report.evaluation_time_points)
    except Exception:
        report.failure_stage = "evaluation"
        logger.error("Evaluation failed:\n%s", traceback.format_exc())
        report.exit_code = 1
    finally:
        for h in logger.handlers:
            h.flush()
    return report


def validate_existing_seed(seed: int) -> SeedReport:
    paths = seed_paths(seed)
    report = SeedReport(seed=seed)
    try:
        for req, label in [
            (paths.ckpt_dir / "best.ckpt", "best.ckpt"),
            (paths.ckpt_dir / "last.ckpt", "last.ckpt"),
            (paths.epoch_csv, "epoch_metrics.csv"),
            (paths.pred_pf, "predictions"),
            (paths.metrics_csv, "metrics"),
        ]:
            if not req.exists():
                raise FileNotFoundError(f"Missing {label}: {req}")
        pf = pd.read_csv(paths.pred_pf)
        if len(pf) != 731:
            raise ValueError(f"Prediction rows {len(pf)} != 731")
        if not np.isfinite(pf.select_dtypes(include=[np.number]).to_numpy()).all():
            raise ValueError("Non-finite predictions in saved Pf curve")
        m = pd.read_csv(paths.metrics_csv).iloc[0]
        hist = pd.read_csv(paths.epoch_csv)
        report.training_completed = True
        report.evaluation_completed = True
        report.epochs_completed = len(hist)
        report.best_epoch = int(m.get("best_epoch", hist["best_epoch"].iloc[-1]))
        report.best_validation_loss = float(m.get("best_validation_loss", hist["best_val_loss"].iloc[-1]))
        report.test_mae = float(m["MAE"])
        report.test_rmse = float(m["RMSE"])
        report.max_error = float(m["max_abs_error"])
        report.final_year_error = float(m["final_year_abs_error"])
        report.total_training_seconds = float(hist["elapsed_seconds"].iloc[-1])
        report.avg_seconds_per_epoch = report.total_training_seconds / max(report.epochs_completed, 1)
        report.pure_inference_seconds = float(m.get("pure_inference_time_seconds", np.nan))
        report.parameter_count = int(m["parameter_count"]) if pd.notna(m.get("parameter_count")) else None
        report.evaluation_time_points = int(m["evaluation_time_points"])
        report.early_stopping_triggered = report.epochs_completed < MAX_EPOCHS
        report.exit_code = 0
        pf.to_csv(REVISION_PREDICTION_DIR / f"final_pf_tft_seed{seed}.csv", index=False)
    except Exception as exc:
        report.failure_stage = f"validation: {exc}"
        report.exit_code = 1
    return report


def build_tft_three_seed_summary() -> pd.DataFrame:
    rows = []
    for seed in ALL_SEEDS:
        paths = seed_paths(seed)
        m = pd.read_csv(paths.metrics_csv).iloc[0]
        hist = pd.read_csv(paths.epoch_csv)
        rows.append(
            {
                "seed": seed,
                "MAE": float(m["MAE"]),
                "RMSE": float(m["RMSE"]),
                "max_abs_error": float(m["max_abs_error"]),
                "year_of_max_error": float(m["year_of_max_error"]),
                "final_year_abs_error": float(m["final_year_abs_error"]),
                "best_epoch": int(m.get("best_epoch", hist["best_epoch"].iloc[-1])),
                "best_validation_loss": float(m.get("best_validation_loss", hist["best_val_loss"].iloc[-1])),
                "epochs_completed": int(m.get("epochs_completed", len(hist))),
                "early_stopping_triggered": bool(m.get("early_stopping_triggered", len(hist) < MAX_EPOCHS)),
                "training_time_seconds": float(hist["elapsed_seconds"].iloc[-1]),
                "avg_seconds_per_epoch": float(hist["elapsed_seconds"].iloc[-1]) / max(len(hist), 1),
                "pure_inference_time_seconds": float(m["pure_inference_time_seconds"]),
                "parameter_count": int(m["parameter_count"]),
                "evaluation_time_points": int(m["evaluation_time_points"]),
                "checkpoint_dir": str(paths.ckpt_dir),
                "prediction_file": str(paths.pred_pf),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(REVISION_TABLE_DIR / "final_tft_three_seed_results.csv", index=False)

    summary = {
        "model": "TFT",
        "seed_count": len(results),
        "MAE_mean": results["MAE"].mean(),
        "MAE_std": results["MAE"].std(ddof=0),
        "RMSE_mean": results["RMSE"].mean(),
        "RMSE_std": results["RMSE"].std(ddof=0),
        "max_abs_error_mean": results["max_abs_error"].mean(),
        "max_abs_error_std": results["max_abs_error"].std(ddof=0),
        "final_year_abs_error_mean": results["final_year_abs_error"].mean(),
        "final_year_abs_error_std": results["final_year_abs_error"].std(ddof=0),
        "training_time_seconds_mean": results["training_time_seconds"].mean(),
        "training_time_seconds_std": results["training_time_seconds"].std(ddof=0),
        "pure_inference_time_seconds_mean": results["pure_inference_time_seconds"].mean(),
        "pure_inference_time_seconds_std": results["pure_inference_time_seconds"].std(ddof=0),
        "best_validation_loss_mean": results["best_validation_loss"].mean(),
        "best_validation_loss_std": results["best_validation_loss"].std(ddof=0),
        "best_epoch_values": ",".join(str(int(x)) for x in results["best_epoch"]),
    }
    pd.DataFrame([summary]).to_csv(REVISION_TABLE_DIR / "final_tft_three_seed_summary.csv", index=False)

    matplotlib.use("Agg")
    plt.figure(figsize=(7, 4.5))
    ref = pd.read_csv(seed_paths(ALL_SEEDS[0]).pred_pf)
    plt.plot(ref["t_year"], ref["Pf_true"], color="black", linewidth=2.2, label="Reference")
    for seed in ALL_SEEDS:
        pf = pd.read_csv(seed_paths(seed).pred_pf)
        plt.plot(pf["t_year"], pf["Pf_pred"], linestyle="--", linewidth=1.6, label=f"TFT seed {seed}")
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative Pf(t)")
    plt.title("TFT three-seed held-out trajectories")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_tft_three_seed_trajectories.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    metrics = ["MAE", "RMSE", "max_abs_error", "final_year_abs_error"]
    x = np.arange(len(metrics))
    means = [results[c].mean() for c in metrics]
    stds = [results[c].std(ddof=0) for c in metrics]
    plt.bar(x, means, yerr=stds, capsize=4)
    plt.xticks(x, metrics, rotation=15, ha="right")
    plt.ylabel("Error")
    plt.title("TFT seed metric variability")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_tft_seed_metric_variability.png", dpi=300)
    plt.close()

    lines = [
        "# Final TFT Three-Seed Report",
        "",
        results.to_string(index=False),
        "",
        "## TFT mean ± SD",
        f"- MAE: {summary['MAE_mean']:.6f} ± {summary['MAE_std']:.6f}",
        f"- RMSE: {summary['RMSE_mean']:.6f} ± {summary['RMSE_std']:.6f}",
        f"- max error: {summary['max_abs_error_mean']:.6f} ± {summary['max_abs_error_std']:.6f}",
        f"- final-year error: {summary['final_year_abs_error_mean']:.6f} ± {summary['final_year_abs_error_std']:.6f}",
        f"- training time (s): {summary['training_time_seconds_mean']:.1f} ± {summary['training_time_seconds_std']:.1f}",
        f"- inference time (s): {summary['pure_inference_time_seconds_mean']:.1f} ± {summary['pure_inference_time_seconds_std']:.1f}",
        f"- best epoch distribution: {summary['best_epoch_values']}",
    ]
    (REVISION_OUTPUT_DIR / "final_tft_three_seed_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return results


def rebuild_full_benchmark() -> pd.DataFrame:
    matplotlib.use("Agg")
    rows = []
    curves: dict[str, list[pd.DataFrame]] = {}

    log_pf = REVISION_PREDICTION_DIR / "final_pf_logistic_regression.csv"
    if log_pf.exists():
        pf = pd.read_csv(log_pf)
        err = pf["Pf_pred"] - pf["Pf_true"]
        rows.append(
            {
                "model": "Logistic Regression",
                "seed": np.nan,
                "MAE": float(err.abs().mean()),
                "RMSE": float(np.sqrt((err**2).mean())),
                "max_abs_error": float(err.abs().max()),
                "final_year_abs_error": float(err.abs().iloc[-1]),
                "parameter_count": 8,
                "training_time_seconds": np.nan,
                "pure_inference_time_seconds": np.nan,
                "evaluation_time_points": len(pf),
            }
        )
        curves["Logistic Regression"] = [pf]

    for model, pattern in [("MLP", "final_pf_mlp_seed*.csv"), ("GRU", "final_pf_gru_seed*.csv"), ("TFT", "final_pf_tft_seed*.csv")]:
        for path in sorted(REVISION_PREDICTION_DIR.glob(pattern)):
            pf = pd.read_csv(path)
            err = pf["Pf_pred"] - pf["Pf_true"]
            import re

            m = re.search(r"seed(\d+)", path.name)
            seed = int(m.group(1)) if m else None
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "MAE": float(err.abs().mean()),
                    "RMSE": float(np.sqrt((err**2).mean())),
                    "max_abs_error": float(err.abs().max()),
                    "final_year_abs_error": float(err.abs().iloc[-1]),
                    "parameter_count": np.nan,
                    "training_time_seconds": np.nan,
                    "pure_inference_time_seconds": np.nan,
                    "evaluation_time_points": len(pf),
                }
            )
            curves.setdefault(model, []).append(pf)

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv", index=False)

    summary_rows = []
    for model, g in by_seed.groupby("model", sort=False):
        s = {"model": model}
        for col in ["MAE", "RMSE", "max_abs_error", "final_year_abs_error", "training_time_seconds", "pure_inference_time_seconds"]:
            vals = pd.to_numeric(g[col], errors="coerce")
            s[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            s[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0
        s["seed_count"] = int(g["seed"].notna().sum()) if model != "Logistic Regression" else 1
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REVISION_TABLE_DIR / "final_model_comparison.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(summary["model"], summary["MAE_mean"], yerr=summary["MAE_std"].fillna(0), capsize=4)
    plt.ylabel("MAE on cumulative Pf(t)")
    plt.title("Final model error comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_model_error_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    ref = curves["Logistic Regression"][0]
    plt.plot(ref["t_year"], ref["Pf_true"], color="black", linewidth=2.2, label="Reference")
    for model, cset in curves.items():
        stacked = pd.concat([c[["t_year", "Pf_pred"]] for c in cset], ignore_index=True)
        mean_curve = stacked.groupby("t_year", as_index=False)["Pf_pred"].mean().sort_values("t_year")
        plt.plot(mean_curve["t_year"], mean_curve["Pf_pred"], linestyle="--", linewidth=1.8, label=model)
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative corrosion initiation probability")
    plt.title("Held-out population trajectories by model")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_population_trajectories_by_model.png", dpi=300)
    plt.close()

    def cmp(tft_col: str, other: str) -> str:
        t = summary.loc[summary["model"] == "TFT", f"{tft_col}_mean"]
        o = summary.loc[summary["model"] == other, f"{tft_col}_mean"]
        if t.empty or o.empty:
            return "not determined"
        return "yes" if float(t.iloc[0]) < float(o.iloc[0]) else "no"

    tft = summary.loc[summary["model"] == "TFT"].iloc[0]
    ranked_mae = summary.sort_values("MAE_mean")
    lines = [
        "# Final Benchmark Report",
        "",
        "## Model Ranking by Test MAE",
        ranked_mae[["model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std"]].to_string(index=False),
        "",
        "## Explicit Comparison Answers",
        f"1. TFT better than Logistic Regression? {cmp('MAE', 'Logistic Regression')}.",
        f"2. TFT better than MLP? {cmp('MAE', 'MLP')}.",
        f"3. TFT better than GRU? {cmp('MAE', 'GRU')}.",
        f"4. TFT consistent across seeds? MAE std={tft['MAE_std']:.6f} across 3 seeds.",
        f"5. TFT complexity justified? Compare MAE advantage vs training/inference time before claiming justification.",
        f"6. Best accuracy-efficiency balance: {ranked_mae.iloc[0]['model']} by MAE; inspect inference time separately.",
        "",
        f"TFT MAE: {tft['MAE_mean']:.6f} ± {tft['MAE_std']:.6f}",
        f"TFT RMSE: {tft['RMSE_mean']:.6f} ± {tft['RMSE_std']:.6f}",
    ]
    (REVISION_OUTPUT_DIR / "final_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def print_final_summary(reports: dict[int, SeedReport], summary: pd.DataFrame | None) -> None:
    tft = summary.loc[summary["model"] == "TFT"] if summary is not None and not summary.empty else None
    def cmp(other: str) -> str:
        if tft is None or tft.empty:
            return "not determined"
        o = summary.loc[summary["model"] == other, "MAE_mean"]
        return "yes" if float(tft.iloc[0]["MAE_mean"]) < float(o.iloc[0]) else "no"

    print("\n" + "=" * 60)
    print("FINAL TFT THREE-SEED SUMMARY")
    print("=" * 60)
    for seed in ALL_SEEDS:
        r = reports.get(seed)
        ok = r and r.exit_code == 0 and r.evaluation_completed
        print(f"seed {seed} completed: {'yes' if ok else 'no'}")
    valid_13 = reports.get(20250113) and reports[20250113].exit_code == 0
    print(f"seed 20250113 valid: {'yes' if valid_13 else 'no'}")
    n = sum(1 for s in ALL_SEEDS if reports.get(s) and reports[s].exit_code == 0)
    print(f"TFT seeds completed: {n}/3")
    if tft is not None and not tft.empty:
        print(f"TFT MAE mean ± SD: {tft.iloc[0]['MAE_mean']:.6f} ± {tft.iloc[0]['MAE_std']:.6f}")
        print(f"TFT RMSE mean ± SD: {tft.iloc[0]['RMSE_mean']:.6f} ± {tft.iloc[0]['RMSE_std']:.6f}")
    if summary is not None:
        print("model ranking by MAE:", ", ".join(summary.sort_values("MAE_mean")["model"].tolist()))
        print("model ranking by RMSE:", ", ".join(summary.sort_values("RMSE_mean")["model"].tolist()))
        print(f"TFT better than Logistic Regression: {cmp('Logistic Regression')}")
        print(f"TFT better than MLP: {cmp('MLP')}")
        print(f"TFT better than GRU: {cmp('GRU')}")
    issues = [f"seed {s}: {reports[s].failure_stage}" for s in ALL_SEEDS if reports.get(s) and reports[s].exit_code != 0]
    print(f"unresolved technical issues: {issues if issues else 'none'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="TFT three-seed 10-epoch benchmark workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_train = sub.add_parser("train")
    p_train.add_argument("--seed", type=int, required=True)
    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--seed", type=int, required=True)
    p_val = sub.add_parser("validate")
    p_val.add_argument("--seed", type=int, required=True)
    sub.add_parser("build-tft-summary")
    sub.add_parser("rebuild-benchmark")
    p_run = sub.add_parser("run-remaining")
    p_run.add_argument("--start-seed", type=int, default=20250111)
    args = parser.parse_args()

    reports: dict[int, SeedReport] = {}

    if args.command == "validate":
        r = validate_existing_seed(args.seed)
        reports[args.seed] = r
        raise SystemExit(r.exit_code)

    if args.command == "train":
        r = train_seed(args.seed)
        raise SystemExit(r.exit_code)

    if args.command == "evaluate":
        r = evaluate_seed(args.seed)
        raise SystemExit(r.exit_code)

    if args.command == "build-tft-summary":
        build_tft_three_seed_summary()
        return

    if args.command == "rebuild-benchmark":
        rebuild_full_benchmark()
        return

    if args.command == "run-remaining":
        reports[20250113] = validate_existing_seed(20250113)
        if reports[20250113].exit_code != 0:
            print_final_summary(reports, None)
            raise SystemExit(1)
        for seed in [s for s in ALL_SEEDS if s != 20250113 and s >= args.start_seed]:
            tr = train_seed(seed)
            reports[seed] = tr
            if tr.exit_code != 0:
                print_final_summary(reports, None)
                raise SystemExit(1)
            ev = evaluate_seed(seed, training_seconds=tr.total_training_seconds)
            reports[seed] = ev
            if ev.exit_code != 0:
                print_final_summary(reports, None)
                raise SystemExit(1)
        build_tft_three_seed_summary()
        bench = rebuild_full_benchmark()
        print_final_summary(reports, bench)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
