"""
04b_train_tft_onset_flag.py

Train TFT for binary classification: target_onset in {0,1}
Goal: predict P(onset occurred by time t) for future horizon -> Pf(t)

Step-1 update:
- Use cover_mm as the static real covariate (instead of cover_m).
- Ensure cover_mm exists (create from cover_m if needed).
- Add sanity checks for static feature consistency.
"""

from pathlib import Path
import pandas as pd

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import CrossEntropy


def ensure_cover_mm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure cover_mm exists. Prefer cover_mm; otherwise derive from cover_m.
    """
    if "cover_mm" in df.columns:
        return df

    if "cover_m" in df.columns:
        df = df.copy()
        df["cover_mm"] = df["cover_m"].astype(float) * 1000.0
        return df

    raise KeyError("Neither 'cover_mm' nor 'cover_m' exists in the dataset.")


def sanity_check_static(df: pd.DataFrame, col: str = "cover_mm") -> None:
    """
    cover_mm must be constant within each series_id.
    """
    nunique = df.groupby("series_id")[col].nunique()
    if not (nunique == 1).all():
        bad = nunique[nunique != 1].head(10)
        raise ValueError(f"Static feature '{col}' is not constant within some series_id. Examples:\n{bad}")


def main():
    data_path = Path("data/processed/chloride_labeled.parquet")
    df = pd.read_parquet(data_path)

    # Ensure cover_mm exists (and is consistent)
    df = ensure_cover_mm(df)
    sanity_check_static(df, "cover_mm")

    # ✅ Use the cumulative onset flag as target
    # Ensure integer class labels 0/1
    df["target_onset"] = df["target_onset"].astype(int)

    max_encoder_length = 52        # history
    max_prediction_length = 13     # horizon

    training_cutoff = df["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        df[df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="target_onset",
        group_ids=["series_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        # known reals
        time_varying_known_reals=["time_idx", "t_year"],
        # unknown reals (exclude target itself)
        time_varying_unknown_reals=["chloride_rebar"],
        # ✅ Step-1: use cover_mm as static real
        static_reals=["Cs", "cover_mm", "D28", "m_aging", "C_th"],
        # IMPORTANT: classification -> no target normalizer
        add_relative_time_idx=True,
        add_encoder_length=True,
    )

    # Safe validation slice
    val_df = df[df.time_idx > training_cutoff - max_encoder_length - max_prediction_length]
    validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=True, stop_randomization=True)

    train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=3e-4,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        # ✅ 2 classes: {0,1}
        output_size=2,
        loss=CrossEntropy(),
    )

    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="tft_onset_flag-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    es = EarlyStopping(monitor="val_loss", patience=6, mode="min")

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices="auto",
        callbacks=[ckpt, es],
        log_every_n_steps=20,
    )

    trainer.fit(tft, train_dl, val_dl)

    print("\n✅ Best model:", ckpt.best_model_path)


if __name__ == "__main__":
    main()
