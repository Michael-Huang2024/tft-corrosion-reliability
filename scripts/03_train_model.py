"""
Train the Temporal Fusion Transformer for onset probability prediction.

This script implements Step 3: TFT model training for corrosion initiation. It
preserves the original onset-flag training setup used to estimate population-
level corrosion initiation probability Pf(t).

Generated files:
- outputs/checkpoints/tft_onset_flag-*.ckpt
- outputs/checkpoints/best_checkpoint.txt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import CrossEntropy


ROOT = Path(__file__).resolve().parents[1]
MODEL_STATIC_REALS = ["Cs", "cover_mm", "D28", "m_aging"]
MODEL_TIME_VARYING_KNOWN_REALS = ["time_idx", "t_year"]
MODEL_TIME_VARYING_UNKNOWN_REALS: list[str] = []
FORBIDDEN_INPUTS = {"chloride_rebar", "C_th", "target_onset", "onset_flag", "binary_label", "onset_raw"}


def ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    if "cover_mm" in df.columns:
        return df
    if "cover_m" in df.columns:
        df = df.copy()
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df
    raise KeyError("Neither 'cover_mm' nor 'cover_m' exists in the dataset.")


def sanity_check_static(df: pd.DataFrame, col: str = "cover_mm") -> None:
    nunique = df.groupby("series_id")[col].nunique()
    if not (nunique == 1).all():
        bad = nunique[nunique != 1].head(10)
        raise ValueError(f"Static feature '{col}' is not constant within some series_id. Examples:\n{bad}")


def validate_model_inputs() -> None:
    model_inputs = (
        set(MODEL_STATIC_REALS)
        | set(MODEL_TIME_VARYING_KNOWN_REALS)
        | set(MODEL_TIME_VARYING_UNKNOWN_REALS)
    )
    leaked_inputs = sorted(model_inputs & FORBIDDEN_INPUTS)
    if leaked_inputs:
        raise ValueError(f"Forbidden leakage-prone model inputs configured: {leaked_inputs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT onset model for Pf(t).")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "chloride_labeled.parquet")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "outputs" / "checkpoints")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-encoder-length", type=int, default=52)
    parser.add_argument("--max-prediction-length", type=int, default=13)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20250111)
    return parser.parse_args()


def main() -> None:
    start_time = time.perf_counter()
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    validate_model_inputs()

    df = pd.read_parquet(args.data)
    df = ensure_cover_mm(df)
    sanity_check_static(df, "cover_mm")
    df["target_onset"] = df["target_onset"].astype(int)

    training_cutoff = df["time_idx"].max() - args.max_prediction_length
    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_onset",
        group_ids=["series_id"],
        max_encoder_length=args.max_encoder_length,
        max_prediction_length=args.max_prediction_length,
        time_varying_known_reals=MODEL_TIME_VARYING_KNOWN_REALS,
        time_varying_unknown_reals=MODEL_TIME_VARYING_UNKNOWN_REALS,
        static_reals=MODEL_STATIC_REALS,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    val_df = df[df.time_idx > training_cutoff - args.max_encoder_length - args.max_prediction_length]
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_dl = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=args.learning_rate,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=2,
        loss=CrossEntropy(),
    )

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(
        dirpath=str(args.checkpoint_dir),
        filename="tft_onset_flag-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint, early_stopping],
        log_every_n_steps=20,
    )
    trainer.fit(tft, train_dl, val_dl)

    best_path = Path(checkpoint.best_model_path)
    pointer = args.checkpoint_dir / "best_checkpoint.txt"
    pointer.write_text(str(best_path), encoding="utf-8")
    print(f"Best model: {best_path}")
    print(f"Saved checkpoint pointer: {pointer}")
    print(f"[Timing] Model training: {time.perf_counter() - start_time:.4f} s")


if __name__ == "__main__":
    main()
