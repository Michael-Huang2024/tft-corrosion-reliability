"""
Stable TFT training and evaluation for reviewer-revision benchmarks.

Training and evaluation are separate commands. Scientific architecture and
data are unchanged; only runtime stability settings differ from the original
07_train_benchmarks.py TFT path.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pandas/pyarrow must initialize before torch on Windows (native DLL conflict).
import torch

_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    LEARNING_RATE,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    REVISION_FIGURE_DIR,
    REVISION_LOG_DIR,
    REVISION_OUTPUT_DIR,
    REVISION_PREDICTION_DIR,
    REVISION_SEEDS,
    REVISION_TABLE_DIR,
    TARGET_COLUMN,
    TFT_ATTENTION_HEADS,
    TFT_DROPOUT,
    TFT_HIDDEN_SIZE,
    TFT_STATIC_REALS,
    TFT_TIME_VARYING_KNOWN_REALS,
    TFT_TIME_VARYING_UNKNOWN_REALS,
    TIME_INDEX_COLUMN,
    ensure_revision_dirs,
)
from revision_data import attach_split, load_or_create_series_split, validate_series_split
from revision_metrics import evaluate_point_predictions, parameter_count, restrict_common_evaluation_range

TFT_TRAIN_BATCH_SIZE = 32
TFT_INFERENCE_BATCH_SIZE = 64
TFT_MAX_EPOCHS = 40
TFT_PATIENCE = 6
TFT_CHECKPOINT_ROOT = REVISION_OUTPUT_DIR / "checkpoints" / "tft"


def seed_checkpoint_dir(seed: int) -> Path:
    path = TFT_CHECKPOINT_ROOT / str(seed)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_log_path(seed: int, phase: str) -> Path:
    REVISION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    return REVISION_LOG_DIR / f"tft_{seed}_{phase}.log"


def setup_logger(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
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


def gpu_memory_snapshot() -> dict[str, float | bool]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    free, total = torch.cuda.mem_get_info(0)
    return {
        "cuda_available": True,
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 4),
        "reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 4),
        "free_gb": round(free / 1024**3, 4),
        "total_gb": round(total / 1024**3, 4),
    }


def load_revision_data() -> pd.DataFrame:
    ensure_revision_dirs()
    df = pd.read_parquet(FINAL_LABELED_DATA)
    split = load_or_create_series_split(df)
    validate_series_split(df, split)
    return attach_split(df, split)


def build_tft_datasets(df: pd.DataFrame):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder

    categorical_encoders = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()

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
        categorical_encoders=categorical_encoders,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    return training, validation, testing, test_df


class EpochDiagnosticsCallback:
    """Lightning callback with explicit logging, GPU memory, and live progress."""

    TRAIN_BATCHES_PER_EPOCH = 15728
    VAL_BATCHES_PER_EPOCH = 700  # approximate; updated at train start

    def __init__(self, logger: logging.Logger, history_path: Path, progress_path: Path) -> None:
        import lightning.pytorch as pl

        self._logger = logger
        self._history_path = history_path
        self._progress_path = progress_path
        self._history: list[dict[str, object]] = []
        self._Base = pl.Callback
        self._train_batches = self.TRAIN_BATCHES_PER_EPOCH
        self._val_batches = self.VAL_BATCHES_PER_EPOCH

    def make(self):
        outer = self

        class _Callback(outer._Base):
            def on_train_start(self, trainer, pl_module) -> None:
                if trainer.num_training_batches:
                    outer._train_batches = int(trainer.num_training_batches)
                if trainer.num_val_batches:
                    outer._val_batches = int(trainer.num_val_batches[0]) if isinstance(trainer.num_val_batches, list) else int(trainer.num_val_batches)

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
                outer._write_progress(
                    epoch=int(trainer.current_epoch),
                    phase="train",
                    batch_idx=int(batch_idx) + 1,
                    total_batches=outer._train_batches,
                )

            def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> None:
                outer._write_progress(
                    epoch=int(trainer.current_epoch),
                    phase="validation",
                    batch_idx=int(batch_idx) + 1,
                    total_batches=outer._val_batches,
                )

            def on_train_epoch_end(self, trainer, pl_module) -> None:
                metrics = {k: float(v) if torch.is_tensor(v) else v for k, v in trainer.callback_metrics.items()}
                row = {
                    "epoch": int(trainer.current_epoch),
                    "phase": "train_epoch_end",
                    "metrics": metrics,
                    "gpu_memory": gpu_memory_snapshot(),
                }
                outer._history.append(row)
                outer._logger.info("epoch=%s metrics=%s gpu=%s", row["epoch"], metrics, row["gpu_memory"])
                outer._flush_history()
                outer._write_progress(epoch=int(trainer.current_epoch), phase="train_epoch_end", batch_idx=outer._train_batches, total_batches=outer._train_batches)

            def on_validation_epoch_end(self, trainer, pl_module) -> None:
                metrics = {k: float(v) if torch.is_tensor(v) else v for k, v in trainer.callback_metrics.items()}
                row = {
                    "epoch": int(trainer.current_epoch),
                    "phase": "validation_epoch_end",
                    "metrics": metrics,
                    "gpu_memory": gpu_memory_snapshot(),
                }
                outer._history.append(row)
                outer._logger.info("validation epoch=%s metrics=%s gpu=%s", row["epoch"], metrics, row["gpu_memory"])
                outer._flush_history()
                outer._write_progress(epoch=int(trainer.current_epoch), phase="validation_epoch_end", batch_idx=outer._val_batches, total_batches=outer._val_batches)

        return _Callback()

    def _write_progress(self, epoch: int, phase: str, batch_idx: int, total_batches: int) -> None:
        if phase == "train":
            epoch_frac = batch_idx / max(total_batches, 1)
        elif phase == "validation":
            epoch_frac = 0.85 + 0.15 * (batch_idx / max(total_batches, 1))
        elif phase == "train_epoch_end":
            epoch_frac = 0.85
        else:
            epoch_frac = 1.0
        payload = {
            "epoch": epoch,
            "phase": phase,
            "batch": batch_idx,
            "total_batches": total_batches,
            "epoch_fraction": round(epoch_frac, 4),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu_memory": gpu_memory_snapshot(),
        }
        self._progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _flush_history(self) -> None:
        self._history_path.write_text(json.dumps(self._history, indent=2), encoding="utf-8")


def train_seed(seed: int, device: str = "cuda", resume: Path | None = None) -> int:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import CrossEntropy

    log_path = seed_log_path(seed, "training")
    logger = setup_logger(log_path, f"tft_train_{seed}")
    ckpt_dir = seed_checkpoint_dir(seed)
    history_path = ckpt_dir / "epoch_history.json"
    completion_marker = ckpt_dir / "training_complete.marker"

    logger.info("Starting TFT training seed=%s device=%s ckpt_dir=%s", seed, device, str(ckpt_dir))
    logger.info("Environment GPU snapshot: %s", gpu_memory_snapshot())

    exit_code = 1
    try:
        pl.seed_everything(seed, workers=True)
        df = load_revision_data()
        training, validation, _testing, _test_df = build_tft_datasets(df)

        train_loader = training.to_dataloader(
            train=True,
            batch_size=TFT_TRAIN_BATCH_SIZE,
            num_workers=0,
            persistent_workers=False,
        )
        val_loader = validation.to_dataloader(
            train=False,
            batch_size=TFT_TRAIN_BATCH_SIZE,
            num_workers=0,
            persistent_workers=False,
        )

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=LEARNING_RATE,
            hidden_size=TFT_HIDDEN_SIZE,
            attention_head_size=TFT_ATTENTION_HEADS,
            dropout=TFT_DROPOUT,
            hidden_continuous_size=16,
            output_size=2,
            loss=CrossEntropy(),
        )

        best_ckpt = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_on_train_epoch_end=False,
        )
        last_ckpt = ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="last-{epoch:02d}",
            save_top_k=-1,
            every_n_epochs=1,
            save_on_train_epoch_end=True,
        )
        epoch_cb = EpochDiagnosticsCallback(logger, history_path, ckpt_dir / "progress.json").make()

        accelerator = "cpu" if device == "cpu" else "gpu"
        trainer = pl.Trainer(
            max_epochs=TFT_MAX_EPOCHS,
            accelerator=accelerator,
            devices=1,
            callbacks=[
                best_ckpt,
                last_ckpt,
                EarlyStopping(monitor="val_loss", patience=TFT_PATIENCE, mode="min"),
                epoch_cb,
            ],
            logger=False,
            enable_checkpointing=True,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
            log_every_n_steps=20,
            precision="32-true",
        )

        logger.info("Entering trainer.fit resume=%s", resume)
        trainer.fit(
            model,
            train_loader,
            val_loader,
            ckpt_path=str(resume) if resume else None,
            weights_only=False,
        )
        logger.info("trainer.fit completed current_epoch=%s", trainer.current_epoch)
        logger.info("best_model_path=%s best_model_score=%s", best_ckpt.best_model_path, best_ckpt.best_model_score)

        import re

        best_epoch = None
        if best_ckpt.best_model_path:
            m = re.search(r"best-(\d+)-", Path(best_ckpt.best_model_path).name)
            if m:
                best_epoch = int(m.group(1))
        summary = {
            "seed": seed,
            "best_checkpoint": best_ckpt.best_model_path,
            "best_validation_loss": float(best_ckpt.best_model_score.detach().cpu()) if best_ckpt.best_model_score is not None else None,
            "best_epoch": best_epoch,
            "epochs_completed": int(trainer.current_epoch) + 1,
            "training_time_seconds": None,
        }
        (ckpt_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        completion_marker.write_text(
            json.dumps({"seed": seed, "status": "complete", "best_checkpoint": best_ckpt.best_model_path}, indent=2),
            encoding="utf-8",
        )
        logger.info("Training completion marker written")
        exit_code = 0
    except Exception:
        logger.error("Training failed:\n%s", traceback.format_exc())
        exit_code = 1
    finally:
        for handler in logger.handlers:
            handler.flush()
    return exit_code


def find_best_checkpoint(seed: int) -> Path | None:
    ckpt_dir = seed_checkpoint_dir(seed)
    marker = ckpt_dir / "training_complete.marker"
    if marker.exists():
        data = json.loads(marker.read_text(encoding="utf-8"))
        path = Path(data.get("best_checkpoint", ""))
        if path.exists():
            return path
    candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
    if candidates:
        return candidates[-1]
    legacy = sorted((REVISION_OUTPUT_DIR / "checkpoints").glob(f"final_tft_seed{seed}*.ckpt"))
    if legacy:
        return min(legacy, key=lambda p: _val_loss_from_name(p.name))
    return None


def _val_loss_from_name(name: str) -> float:
    import re

    m = re.search(r"val_loss=([0-9.]+)", name)
    return float(m.group(1)) if m else float("inf")


def evaluate_seed(seed: int, device: str = "cuda", checkpoint: Path | None = None) -> int:
    from pytorch_forecasting import TemporalFusionTransformer

    log_path = seed_log_path(seed, "evaluation")
    logger = setup_logger(log_path, f"tft_eval_{seed}")
    exit_code = 1

    try:
        ckpt_path = checkpoint or find_best_checkpoint(seed)
        if ckpt_path is None or not ckpt_path.exists():
            raise FileNotFoundError(f"No checkpoint found for seed {seed}")

        logger.info("Loading checkpoint %s", ckpt_path)
        df = load_revision_data()
        training, _validation, testing, test_df = build_tft_datasets(df)
        test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)

        model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path), weights_only=False)
        model.eval()
        torch_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(torch_device)
        logger.info("Model loaded params=%s device=%s", parameter_count(model), torch_device)

        test_loader = testing.to_dataloader(
            train=False,
            batch_size=TFT_INFERENCE_BATCH_SIZE,
            num_workers=0,
            persistent_workers=False,
        )

        sums: dict[tuple[int, int], float] = defaultdict(float)
        counts: dict[tuple[int, int], int] = defaultdict(int)
        sample_index = testing.index.reset_index(drop=True)
        sample_offset = 0
        start_infer = time.perf_counter()

        with torch.no_grad():
            for batch in test_loader:
                x, _ = batch
                x_dev = {key: (value.to(torch_device) if torch.is_tensor(value) else value) for key, value in x.items()}
                logits = model(x_dev)["prediction"]
                probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
                if not np.isfinite(probs).all():
                    raise ValueError("Non-finite predictions detected during inference")
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

        infer_seconds = time.perf_counter() - start_infer
        logger.info("Inference completed in %.3f s", infer_seconds)

        pred = pd.DataFrame(
            {
                GROUP_COLUMN: [key[0] for key in sums],
                TIME_INDEX_COLUMN: [key[1] for key in sums],
                "p_onset_pred": [sums[key] / counts[key] for key in sums],
            }
        )
        time_map = test_df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].drop_duplicates()
        pred = pred.merge(time_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left").dropna()

        result = evaluate_point_predictions(
            pred,
            model_name="TFT",
            seed=seed,
            parameter_count_value=parameter_count(model),
            training_time_seconds=None,
            inference_time_seconds=infer_seconds,
        )
        pf = result.pf_curve
        metrics = result.metrics
        metrics["checkpoint_path"] = str(ckpt_path)
        metrics["pure_inference_time_seconds"] = infer_seconds

        pred_out = REVISION_PREDICTION_DIR / f"tft_{seed}.csv"
        pf_out = REVISION_PREDICTION_DIR / f"final_pf_tft_seed{seed}.csv"
        metrics_out = REVISION_TABLE_DIR / f"tft_{seed}_metrics.csv"

        pred.to_csv(pred_out, index=False)
        pf.to_csv(pf_out, index=False)
        pd.DataFrame([metrics]).to_csv(metrics_out, index=False)

        logger.info("Metrics: MAE=%.6f RMSE=%.6f time_points=%s", metrics["MAE"], metrics["RMSE"], metrics["evaluation_time_points"])
        logger.info("Wrote %s", pred_out)
        logger.info("Wrote %s", pf_out)
        logger.info("Wrote %s", metrics_out)

        if int(metrics["evaluation_time_points"]) != 731:
            logger.error("Expected 731 evaluation time points, got %s", metrics["evaluation_time_points"])
            exit_code = 1
        else:
            exit_code = 0
    except Exception:
        logger.error("Evaluation failed:\n%s", traceback.format_exc())
        exit_code = 1
    finally:
        for handler in logger.handlers:
            handler.flush()
    return exit_code


def run_gpu_audit() -> Path:
    import subprocess

    ensure_revision_dirs()
    out = REVISION_OUTPUT_DIR / "tft_gpu_environment_audit.md"
    info: dict[str, object] = {"python_version": sys.version.replace("\n", " ")}
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["total_gpu_memory_gb"] = round(props.total_memory / 1024**3, 3)
        info["cuda_runtime_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        free, total = torch.cuda.mem_get_info(0)
        info["free_gpu_memory_gb"] = round(free / 1024**3, 3)
        info["allocated_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 4)
        info["reserved_gb"] = round(torch.cuda.memory_reserved(0) / 1024**3, 4)
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.mm(x, x)
        torch.cuda.synchronize()
        info["cuda_tensor_test"] = "PASS"
        del x, y
        torch.cuda.empty_cache()
    try:
        import lightning

        info["lightning_version"] = lightning.__version__
    except Exception as exc:
        info["lightning_version"] = str(exc)
    try:
        import pytorch_forecasting

        info["pytorch_forecasting_version"] = pytorch_forecasting.__version__
    except Exception as exc:
        info["pytorch_forecasting_version"] = str(exc)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        info["nvidia_smi"] = smi.stdout.strip() if smi.returncode == 0 else smi.stderr.strip()
    except Exception as exc:
        info["nvidia_smi"] = str(exc)

    lines = [
        "# TFT GPU Environment Audit",
        "",
        f"- Python: {info['python_version']}",
        f"- PyTorch: {info['torch_version']}",
        f"- Lightning: {info.get('lightning_version')}",
        f"- PyTorch Forecasting: {info.get('pytorch_forecasting_version')}",
        f"- CUDA available: {info['cuda_available']}",
    ]
    if info["cuda_available"]:
        lines.extend(
            [
                f"- GPU model: {info['cuda_device_name']}",
                f"- NVIDIA driver (nvidia-smi): {info.get('nvidia_smi')}",
                f"- CUDA runtime: {info['cuda_runtime_version']}",
                f"- cuDNN: {info['cudnn_version']}",
                f"- Total GPU memory (GB): {info['total_gpu_memory_gb']}",
                f"- Free GPU memory (GB): {info['free_gpu_memory_gb']}",
                f"- Allocated GPU memory (GB): {info['allocated_gb']}",
                f"- Reserved GPU memory (GB): {info['reserved_gb']}",
                f"- Mixed precision bf16 supported: {info['bf16_supported']}",
                f"- CUDA tensor test: {info['cuda_tensor_test']}",
            ]
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return out


def rebuild_benchmark() -> None:
    matplotlib.use("Agg")
    existing_path = REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv"
    existing = pd.read_csv(existing_path) if existing_path.exists() else pd.DataFrame()

    def meta_lookup(model: str, seed: int | None) -> dict:
        if existing.empty:
            return {}
        tmp = existing[existing["model"] == model]
        if seed is None:
            tmp = tmp[tmp["seed"].isna()]
        else:
            tmp = tmp[tmp["seed"].fillna(-1).astype(float) == float(seed)]
        return tmp.iloc[-1].to_dict() if len(tmp) else {}

    rows = []
    curves: dict[str, list[pd.DataFrame]] = {}
    log_path = REVISION_PREDICTION_DIR / "final_pf_logistic_regression.csv"
    if log_path.exists():
        pf = pd.read_csv(log_path)
        err = pf["Pf_pred"] - pf["Pf_true"]
        abs_err = err.abs()
        max_idx = int(abs_err.idxmax())
        meta = meta_lookup("Logistic Regression", None)
        rows.append(
            {
                "model": "Logistic Regression",
                "seed": np.nan,
                "MAE": float(abs_err.mean()),
                "RMSE": float(np.sqrt((err**2).mean())),
                "max_abs_error": float(abs_err.loc[max_idx]),
                "year_of_max_error": float(pf.loc[max_idx, "t_year"]),
                "final_year_abs_error": float(abs_err.iloc[-1]),
                "parameter_count": meta.get("parameter_count", 8),
                "training_time_seconds": meta.get("training_time_seconds", np.nan),
                "pure_inference_time_seconds": meta.get("pure_inference_time_seconds", np.nan),
                "evaluation_start_year": float(pf["t_year"].iloc[0]),
                "evaluation_end_year": float(pf["t_year"].iloc[-1]),
                "evaluation_time_points": int(len(pf)),
                "epochs": meta.get("epochs", np.nan),
                "best_validation_loss": meta.get("best_validation_loss", np.nan),
                "checkpoint_path": meta.get("checkpoint_path", ""),
                "best_epoch": meta.get("best_epoch", np.nan),
            }
        )
        curves["Logistic Regression"] = [pf]

    for model, pattern, pretty in [
        ("MLP", "final_pf_mlp_seed*.csv", "MLP"),
        ("GRU", "final_pf_gru_seed*.csv", "GRU"),
        ("TFT", "final_pf_tft_seed*.csv", "TFT"),
    ]:
        for path in sorted(REVISION_PREDICTION_DIR.glob(pattern)):
            pf = pd.read_csv(path)
            err = pf["Pf_pred"] - pf["Pf_true"]
            abs_err = err.abs()
            max_idx = int(abs_err.idxmax())
            import re

            m = re.search(r"seed(\d+)", path.name)
            seed = int(m.group(1)) if m else None
            meta = meta_lookup(pretty, seed)
            rows.append(
                {
                    "model": pretty,
                    "seed": seed,
                    "MAE": float(abs_err.mean()),
                    "RMSE": float(np.sqrt((err**2).mean())),
                    "max_abs_error": float(abs_err.loc[max_idx]),
                    "year_of_max_error": float(pf.loc[max_idx, "t_year"]),
                    "final_year_abs_error": float(abs_err.iloc[-1]),
                    "parameter_count": meta.get("parameter_count", np.nan),
                    "training_time_seconds": meta.get("training_time_seconds", np.nan),
                    "pure_inference_time_seconds": meta.get("pure_inference_time_seconds", np.nan),
                    "evaluation_start_year": float(pf["t_year"].iloc[0]),
                    "evaluation_end_year": float(pf["t_year"].iloc[-1]),
                    "evaluation_time_points": int(len(pf)),
                    "epochs": meta.get("epochs", np.nan),
                    "best_validation_loss": meta.get("best_validation_loss", np.nan),
                    "checkpoint_path": meta.get("checkpoint_path", ""),
                    "best_epoch": meta.get("best_epoch", np.nan),
                }
            )
            curves.setdefault(pretty, []).append(pf)

    by_seed = pd.DataFrame(rows)
    by_seed.to_csv(REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv", index=False)

    summary_rows = []
    for model, g in by_seed.groupby("model", sort=False):
        s = {"model": model}
        for col in [
            "MAE",
            "RMSE",
            "max_abs_error",
            "final_year_abs_error",
            "training_time_seconds",
            "pure_inference_time_seconds",
            "best_validation_loss",
        ]:
            vals = pd.to_numeric(g[col], errors="coerce")
            s[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            s[f"{col}_std"] = float(vals.std(ddof=0)) if vals.notna().sum() > 1 else 0.0
        pc = pd.to_numeric(g["parameter_count"], errors="coerce")
        s["parameter_count"] = int(pc.dropna().iloc[0]) if pc.notna().any() else np.nan
        epochs = pd.to_numeric(g["epochs"], errors="coerce")
        s["epochs_mean"] = float(epochs.mean()) if epochs.notna().any() else np.nan
        s["seed_count"] = int(g["seed"].notna().sum()) if model != "Logistic Regression" else 1
        summary_rows.append(s)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(REVISION_TABLE_DIR / "final_model_comparison.csv", index=False)

    train_summary_path = REVISION_TABLE_DIR / "final_training_summary.csv"
    if train_summary_path.exists():
        train_summary = pd.read_csv(train_summary_path)
        tft_rows = []
        for seed in REVISION_SEEDS:
            ckpt = find_best_checkpoint(seed)
            metrics_path = REVISION_TABLE_DIR / f"tft_{seed}_metrics.csv"
            val_loss = np.nan
            if ckpt and ckpt.exists():
                val_loss = _val_loss_from_name(ckpt.name)
            if metrics_path.exists():
                pass
            tft_rows.append({"model": "TFT", "seed": seed, "checkpoint": str(ckpt) if ckpt else "", "best_validation_loss": val_loss, "epochs": np.nan})
        if tft_rows:
            tft_df = pd.DataFrame(tft_rows)
            combined = pd.concat([train_summary[train_summary["model"] != "TFT"], tft_df], ignore_index=True)
            combined.to_csv(train_summary_path, index=False)

    plt.figure(figsize=(7, 4))
    plt.bar(summary["model"], summary["MAE_mean"], yerr=summary["MAE_std"].fillna(0), capsize=4)
    plt.ylabel("MAE on cumulative Pf(t)")
    plt.title("Final model error comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_model_error_comparison.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    first = next(iter(curves.values()))[0]
    plt.plot(first["t_year"], first["Pf_true"], color="black", linewidth=2.2, label="Reference cumulative Pf(t)")
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

    ranked = summary.sort_values("MAE_mean").reset_index(drop=True)

    def val(model: str, col: str):
        hit = summary[summary["model"] == model]
        return None if hit.empty else float(hit.iloc[0][col])

    def outperform(other: str) -> str:
        t = val("TFT", "MAE_mean")
        o = val(other, "MAE_mean")
        if t is None or o is None:
            return "not determined"
        return "yes" if t < o else "no"

    tft_count = int(summary.loc[summary["model"].eq("TFT"), "seed_count"].iloc[0]) if summary["model"].eq("TFT").any() else 0
    tft_std = val("TFT", "MAE_std")
    tft_mean = val("TFT", "MAE_mean")
    lines = [
        "# Final Benchmark Report",
        "",
        "Phase 3 held-out evaluation outputs were regenerated after stable TFT training/evaluation.",
        "",
        "## Completion Status",
        "",
        f"- Logistic Regression completed: 1",
        f"- MLP completed seeds: {int(summary.loc[summary.model.eq('MLP'), 'seed_count'].iloc[0]) if summary.model.eq('MLP').any() else 0}/3",
        f"- GRU completed seeds: {int(summary.loc[summary.model.eq('GRU'), 'seed_count'].iloc[0]) if summary.model.eq('GRU').any() else 0}/3",
        f"- TFT completed evaluable prediction seeds: {tft_count}/3",
        "",
        "## Model Ranking by MAE",
        "",
        ranked[
            [
                "model",
                "seed_count",
                "MAE_mean",
                "MAE_std",
                "RMSE_mean",
                "RMSE_std",
                "training_time_seconds_mean",
                "pure_inference_time_seconds_mean",
                "parameter_count",
            ]
        ].to_string(index=False),
        "",
        "## Explicit Comparison Answers",
        "",
        f"1. Is TFT better than Logistic Regression? {outperform('Logistic Regression')}.",
        f"2. Is TFT better than MLP? {outperform('MLP')}.",
        f"3. Is TFT better than GRU? {outperform('GRU')}.",
        f"4. Is the TFT result stable across seeds? {'yes' if tft_count == 3 and tft_std is not None and tft_std < tft_mean else 'see seed-level table' if tft_count == 3 else 'not determined (incomplete seeds)'}.",
        "5. Does TFT justify its additional computational cost? Compare MAE/RMSE advantage against training and inference time before claiming justification.",
        "",
        f"TFT mean ± SD MAE: {tft_mean:.6f} ± {tft_std:.6f}" if tft_mean is not None and tft_std is not None else "TFT mean ± SD MAE: not available",
        "",
        "Do not claim TFT superiority unless supported by complete three-seed held-out metrics.",
    ]
    (REVISION_OUTPUT_DIR / "final_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable TFT diagnostic train/eval workflow.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit", help="Run GPU/environment audit.")
    p_train = sub.add_parser("train", help="Train one TFT seed.")
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_train.add_argument("--resume", type=Path, default=None, help="Resume from a Lightning checkpoint.")
    p_eval = sub.add_parser("evaluate", help="Evaluate one TFT seed from best checkpoint.")
    p_eval.add_argument("--seed", type=int, required=True)
    p_eval.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    sub.add_parser("rebuild-benchmark", help="Regenerate final benchmark tables/figures/report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit":
        run_gpu_audit()
        return
    if args.command == "train":
        code = train_seed(args.seed, device=args.device, resume=args.resume)
        raise SystemExit(code)
    if args.command == "evaluate":
        code = evaluate_seed(args.seed, device=args.device, checkpoint=args.checkpoint)
        raise SystemExit(code)
    if args.command == "rebuild-benchmark":
        rebuild_benchmark()
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
