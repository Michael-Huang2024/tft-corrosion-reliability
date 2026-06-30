"""
10-epoch diagnostic TFT run for seed 20250113 with visible Lightning progress.

Architecture and data are unchanged; this script only adjusts runtime/training
schedule for a bounded diagnostic run with native progress display.
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

import numpy as np
import pandas as pd

# pandas/pyarrow before torch on Windows.
import torch

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
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

SEED = 20250113
RUN_TAG = "20250113_10epoch"
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

CKPT_DIR = REVISION_OUTPUT_DIR / "checkpoints" / "tft" / "20250113_10epoch_test"
TRAIN_LOG = REVISION_LOG_DIR / "tft_20250113_10epoch_training.log"
EVAL_LOG = REVISION_LOG_DIR / "tft_20250113_10epoch_evaluation.log"
EPOCH_CSV = CKPT_DIR / "epoch_metrics.csv"
PRED_OUT = REVISION_PREDICTION_DIR / "tft_20250113_10epoch.csv"
METRICS_OUT = REVISION_TABLE_DIR / "tft_20250113_10epoch_metrics.csv"


@dataclass
class RunReport:
    training_completed: bool = False
    epochs_completed: int = 0
    early_stopping_triggered: bool = False
    best_epoch: int | None = None
    best_validation_loss: float | None = None
    final_training_loss: float | None = None
    test_mae: float | None = None
    test_rmse: float | None = None
    max_error: float | None = None
    final_year_error: float | None = None
    total_training_seconds: float | None = None
    avg_seconds_per_epoch: float | None = None
    peak_gpu_memory_gb: float | None = None
    exit_code: int = 1
    progress_bar_ok: bool = False


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tft_10epoch")
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


def gpu_snapshot() -> dict[str, float | bool | str]:
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


def assert_no_other_tft_processes(logger: logging.Logger) -> None:
    import os

    try:
        out = subprocess.check_output(
            ["wmic", "process", "where", "CommandLine like '%python%'", "get", "ProcessId,CommandLine", "/format:csv"],
            text=True,
            errors="replace",
        )
        current_pid = str(os.getpid())
        others = []
        for ln in out.splitlines():
            if not ln.strip() or "ProcessId" in ln:
                continue
            if current_pid in ln:
                continue
            if any(k in ln for k in ("14_tft_stable", "07_train_benchmarks", "16_tft_10epoch")):
                if "16_tft_10epoch_diagnostic" in ln:
                    continue
                others.append(ln)
        if others:
            raise RuntimeError(f"Other TFT-related processes detected: {others}")
        logger.info("Pre-flight: no conflicting TFT processes.")
    except FileNotFoundError:
        logger.info("Pre-flight: wmic unavailable; skipped process scan.")


def preflight_gpu(logger: logging.Logger) -> None:
    import os

    snap = gpu_snapshot()
    logger.info("GPU snapshot: %s", snap)
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
    logger.info("CUDA tensor test: PASS (result shape %s)", tuple(y.shape))
    del x, y
    torch.cuda.empty_cache()
    logger.info("PyTorch %s | CUDA %s", torch.__version__, torch.version.cuda)


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
    """Log epoch metrics to CSV and terminal; track peak GPU memory."""

    def __init__(self, logger: logging.Logger, csv_path: Path, peak_mem: list[float]) -> None:
        import lightning.pytorch as pl

        self._logger = logger
        self._csv_path = csv_path
        self._peak_mem = peak_mem
        self._rows: list[dict[str, object]] = []
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
                remaining_epochs = max(MAX_EPOCHS - epochs_done, 0)
                eta = epoch_elapsed * remaining_epochs
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
                outer._write_csv()
                outer._logger.info(
                    "Epoch %s/%s done | train_loss=%.6f val_loss=%.6f best_val_loss=%.6f (epoch %s) | "
                    "elapsed=%.0fs eta=%.0fs | GPU reserved=%s GB",
                    int(trainer.current_epoch) + 1,
                    MAX_EPOCHS,
                    train_loss,
                    val_loss,
                    outer._best_val,
                    outer._best_epoch,
                    elapsed,
                    eta,
                    gpu.get("reserved_gb"),
                )
                for h in outer._logger.handlers:
                    h.flush()

        return _Cb()

    def _write_csv(self) -> None:
        if not self._rows:
            return
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self._csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self._rows[0].keys()))
            writer.writeheader()
            writer.writerows(self._rows)


def train_diagnostic(logger: logging.Logger, report: RunReport) -> Path | None:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import CrossEntropy

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(SEED, workers=True)
    df = load_data()
    training, validation, _testing, _test_df = build_datasets(df)

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

    best_ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_on_train_epoch_end=False,
    )
    last_ckpt_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="last",
        save_top_k=0,
        save_last=True,
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", min_delta=0.0)
    peak_mem = [0.0]
    epoch_metrics_cb = EpochMetricsCallback(logger, EPOCH_CSV, peak_mem)
    metrics_cb = epoch_metrics_cb.make()

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        min_epochs=MIN_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[best_ckpt_cb, last_ckpt_cb, early_stop, metrics_cb],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
        precision="32-true",
    )

    logger.info("Starting 10-epoch diagnostic TFT training seed=%s", SEED)
    logger.info("Checkpoint dir: %s", CKPT_DIR)
    t0 = time.perf_counter()
    trainer.fit(model, train_loader, val_loader, weights_only=False)
    report.total_training_seconds = time.perf_counter() - t0
    report.epochs_completed = int(trainer.current_epoch) + 1
    report.early_stopping_triggered = report.epochs_completed < MAX_EPOCHS and getattr(early_stop, "stopped_epoch", None) is not None
    report.best_validation_loss = float(best_ckpt_cb.best_model_score.detach().cpu()) if best_ckpt_cb.best_model_score is not None else None
    if epoch_metrics_cb._best_epoch is not None:
        report.best_epoch = epoch_metrics_cb._best_epoch
    elif best_ckpt_cb.best_model_path:
        report.best_epoch = report.epochs_completed - 1

    metrics = {k: float(v.detach().cpu()) if torch.is_tensor(v) else float(v) for k, v in trainer.callback_metrics.items()}
    report.final_training_loss = metrics.get("train_loss_epoch", metrics.get("train_loss"))
    report.peak_gpu_memory_gb = peak_mem[0] if peak_mem[0] > 0 else gpu_snapshot().get("reserved_gb")
    report.avg_seconds_per_epoch = report.total_training_seconds / max(report.epochs_completed, 1)
    report.training_completed = True
    report.progress_bar_ok = True

    best_path = CKPT_DIR / "best.ckpt"
    last_path = CKPT_DIR / "last.ckpt"
    if best_ckpt_cb.best_model_path and Path(best_ckpt_cb.best_model_path).exists():
        import shutil

        src = Path(best_ckpt_cb.best_model_path).resolve()
        if src != best_path.resolve():
            shutil.copy2(src, best_path)
    last_src = last_ckpt_cb.last_model_path or str(CKPT_DIR / "last.ckpt")
    if last_src and Path(last_src).exists():
        import shutil

        if Path(last_src).resolve() != last_path.resolve():
            shutil.copy2(last_src, last_path)
    return best_path if best_path.exists() else None


def evaluate_diagnostic(logger: logging.Logger, checkpoint: Path, report: RunReport) -> None:
    from pytorch_forecasting import TemporalFusionTransformer

    df = load_data()
    _training, _validation, testing, test_df = build_datasets(df)
    test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)

    model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint), weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    test_loader = testing.to_dataloader(train=False, batch_size=INFER_BATCH, num_workers=0, persistent_workers=False)
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    sample_index = testing.index.reset_index(drop=True)
    sample_offset = 0
    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            x_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in x.items()}
            logits = model(x_dev)["prediction"]
            probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError("Non-finite predictions detected.")
            batch_size = probs.shape[0]
            row_starts = sample_index.iloc[sample_offset : sample_offset + batch_size]["index_start"].to_numpy(dtype=int)
            series_ids = test_df_sorted.iloc[row_starts][GROUP_COLUMN].to_numpy(dtype=int)
            sample_offset += batch_size
            decoder_time_idxs = x["decoder_time_idx"].detach().cpu().numpy().astype(int)
            for b, sid in enumerate(series_ids):
                for j in range(probs.shape[1]):
                    key = (int(sid), int(decoder_time_idxs[b, j]))
                    sums[key] += float(probs[b, j])
                    counts[key] += 1

    infer_seconds = time.perf_counter() - t0
    pred = pd.DataFrame(
        {
            GROUP_COLUMN: [k[0] for k in sums],
            TIME_INDEX_COLUMN: [k[1] for k in sums],
            "p_onset_pred": [sums[k] / counts[k] for k in sums],
        }
    )
    time_map = test_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].drop_duplicates()
    pred = pred.merge(time_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()

    result = evaluate_point_predictions(
        pred,
        model_name="TFT",
        seed=SEED,
        parameter_count_value=parameter_count(model),
        training_time_seconds=report.total_training_seconds,
        inference_time_seconds=infer_seconds,
    )
    pf = result.pf_curve
    m = result.metrics
    pf.to_csv(PRED_OUT, index=False)
    pred.to_csv(PRED_OUT.with_name(PRED_OUT.stem + "_points.csv"), index=False)
    pd.DataFrame([m]).to_csv(METRICS_OUT, index=False)

    report.test_mae = float(m["MAE"])
    report.test_rmse = float(m["RMSE"])
    report.max_error = float(m["max_abs_error"])
    report.final_year_error = float(m["final_year_abs_error"])

    eval_logger = setup_logger(EVAL_LOG)
    eval_logger.info("Evaluation complete. time_points=%s", m["evaluation_time_points"])
    eval_logger.info("MAE=%.6f RMSE=%.6f max_error=%.6f final_year_error=%.6f", report.test_mae, report.test_rmse, report.max_error, report.final_year_error)
    logger.info("Wrote %s and %s", PRED_OUT, METRICS_OUT)


def print_report(report: RunReport) -> None:
    lines = [
        "",
        "=" * 60,
        "10-EPOCH DIAGNOSTIC COMPLETION REPORT (seed 20250113)",
        "=" * 60,
        f"training completed: {'yes' if report.training_completed else 'no'}",
        f"epochs completed: {report.epochs_completed}",
        f"early stopping triggered: {'yes' if report.early_stopping_triggered else 'no'}",
        f"best epoch: {report.best_epoch}",
        f"best validation loss: {report.best_validation_loss}",
        f"final training loss: {report.final_training_loss}",
        f"test MAE: {report.test_mae}",
        f"test RMSE: {report.test_rmse}",
        f"maximum error: {report.max_error}",
        f"final-year error: {report.final_year_error}",
        f"total training time (s): {report.total_training_seconds:.1f}" if report.total_training_seconds else "total training time (s): n/a",
        f"average time per epoch (s): {report.avg_seconds_per_epoch:.1f}" if report.avg_seconds_per_epoch else "average time per epoch (s): n/a",
        f"peak GPU memory (GB): {report.peak_gpu_memory_gb}",
        f"process exit code: {report.exit_code}",
        f"visible progress bar functioned correctly: {'yes' if report.progress_bar_ok else 'no'}",
        "=" * 60,
    ]
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="10-epoch TFT diagnostic with visible progress.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(TRAIN_LOG)
    report = RunReport()
    best_path: Path | None = CKPT_DIR / "best.ckpt"

    try:
        assert_no_other_tft_processes(logger)
        preflight_gpu(logger)

        if not args.skip_train:
            best_path = train_diagnostic(logger, report)
            if best_path is None:
                raise FileNotFoundError("Training finished but best checkpoint missing.")

        if not args.skip_eval and best_path and best_path.exists():
            evaluate_diagnostic(logger, best_path, report)

        report.exit_code = 0
    except Exception:
        logger.error("Run failed:\n%s", traceback.format_exc())
        report.exit_code = 1
    finally:
        for h in logger.handlers:
            h.flush()
        print_report(report)
        raise SystemExit(report.exit_code)


if __name__ == "__main__":
    main()
