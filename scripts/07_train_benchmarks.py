"""
Train revision benchmark models on the corrected cumulative target.

Models:
- Logistic Regression
- MLP
- GRU
- TFT

All models use held-out series-level splits, include C_th/Ccrit, exclude
chloride_rebar, and evaluate on a common time range beginning after the encoder
period. MLP and GRU share the same 52-step encoder and 13-step prediction
horizon with unit stride; Logistic Regression remains a pointwise baseline.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from revision_config import (
    FINAL_LABELED_DATA,
    GROUP_COLUMN,
    INFERENCE_BATCH_SIZE,
    LEARNING_RATE,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    POINT_FEATURES,
    REVISION_CHECKPOINT_DIR,
    REVISION_DATA_DIR,
    REVISION_FIGURE_DIR,
    REVISION_LABELED_DATA,
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
from revision_data import (
    attach_split,
    assert_no_forbidden_predictors,
    create_series_split,
    load_or_create_series_split,
    load_source_labeled,
    validate_series_split,
    write_revision_labeled_data,
)
from revision_metrics import evaluate_point_predictions, parameter_count, restrict_common_evaluation_range


class WindowDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: list[str],
        encoder_length: int,
        prediction_length: int,
    ) -> None:
        self.features = features
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.windows: list[tuple[np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]] = []

        time_idx_column = "_eval_time_idx" if "_eval_time_idx" in df.columns else TIME_INDEX_COLUMN
        time_year_column = "_eval_t_year" if "_eval_t_year" in df.columns else "t_year"

        for series_id, group in df.sort_values([GROUP_COLUMN, time_idx_column]).groupby(GROUP_COLUMN):
            values = group[features].to_numpy(dtype=np.float32)
            target = group[TARGET_COLUMN].to_numpy(dtype=np.float32)
            time_idx = group[time_idx_column].to_numpy(dtype=np.int64)
            t_year = group[time_year_column].to_numpy(dtype=np.float64)
            max_start = len(group) - encoder_length - prediction_length
            for start in range(max_start + 1):
                enc = values[start : start + encoder_length]
                dec_target = target[start + encoder_length : start + encoder_length + prediction_length]
                dec_time_idx = time_idx[start + encoder_length : start + encoder_length + prediction_length]
                dec_t_year = t_year[start + encoder_length : start + encoder_length + prediction_length]
                self.windows.append((enc, dec_target, int(series_id), dec_time_idx, dec_t_year))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        enc, dec_target, series_id, dec_time_idx, dec_t_year = self.windows[idx]
        return (
            torch.as_tensor(enc, dtype=torch.float32),
            torch.as_tensor(dec_target, dtype=torch.float32),
            series_id,
            torch.as_tensor(dec_time_idx, dtype=torch.long),
            torch.as_tensor(dec_t_year, dtype=torch.float64),
        )


class WindowedMLPClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = MAX_ENCODER_LENGTH * len(POINT_FEATURES),
        output_size: int = MAX_PREDICTION_LENGTH,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.reshape(x.shape[0], -1))


class GRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=recurrent_dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, MAX_PREDICTION_LENGTH))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train corrected revision benchmark models.")
    parser.add_argument("--models", nargs="+", default=["logistic", "mlp", "gru", "tft"])
    parser.add_argument("--seeds", nargs="+", type=int, default=REVISION_SEEDS)
    parser.add_argument("--smoke-test", action="store_true", help="Use a small in-memory subset and short training.")
    parser.add_argument("--smoke-series", type=int, default=40)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--tft-max-epochs", type=int, default=40)
    parser.add_argument("--tft-window-stride", type=int, default=1)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def prepare_data(args: argparse.Namespace) -> pd.DataFrame:
    ensure_revision_dirs()
    if FINAL_LABELED_DATA.exists() and not args.smoke_test:
        df = pd.read_parquet(FINAL_LABELED_DATA)
    elif REVISION_LABELED_DATA.exists():
        df = pd.read_parquet(REVISION_LABELED_DATA)
    else:
        df = load_source_labeled()
        write_revision_labeled_data(df, REVISION_LABELED_DATA)

    if args.smoke_test:
        series = sorted(df[GROUP_COLUMN].unique())[: args.smoke_series]
        df = df[df[GROUP_COLUMN].isin(series)].copy()
        split = create_series_split(df, REVISION_DATA_DIR / "series_split_smoke.csv")
    else:
        split = load_or_create_series_split(df)
    validate_series_split(df, split)
    return attach_split(df, split)


def fit_point_scaler(df: pd.DataFrame) -> StandardScaler:
    assert_no_forbidden_predictors(POINT_FEATURES)
    scaler = StandardScaler()
    scaler.fit(df.loc[df["split"] == "train", POINT_FEATURES])
    return scaler


def evaluate_probability_frame(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    model: str,
    seed: int | None,
    train_seconds: float | None,
    infer_seconds: float | None,
    params: int | None,
) -> tuple[dict[str, object], pd.DataFrame]:
    pred = df[[GROUP_COLUMN, TIME_INDEX_COLUMN, "t_year", TARGET_COLUMN]].copy()
    pred["p_onset_pred"] = probabilities
    result = evaluate_point_predictions(
        pred,
        model_name=model,
        seed=seed,
        parameter_count_value=params,
        training_time_seconds=train_seconds,
        inference_time_seconds=infer_seconds,
    )
    return result.metrics, result.pf_curve


def train_logistic(df: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    scaler_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs"))
    train_df = df[df["split"] == "train"]
    eval_df = restrict_common_evaluation_range(df, MAX_ENCODER_LENGTH)
    start = time.perf_counter()
    scaler_model.fit(train_df[POINT_FEATURES], train_df[TARGET_COLUMN].astype(int))
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = scaler_model.predict_proba(eval_df[POINT_FEATURES])[:, 1]
    infer_seconds = time.perf_counter() - start
    metrics, pf = evaluate_probability_frame(
        eval_df,
        probabilities,
        "Logistic Regression",
        None,
        train_seconds,
        infer_seconds,
        parameter_count(scaler_model),
    )
    metrics["epochs"] = None
    metrics["best_validation_loss"] = None
    metrics["checkpoint_path"] = None
    return metrics, pf


def train_windowed_mlp(df: pd.DataFrame, seed: int, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    scaler = fit_point_scaler(df)
    scaled = df.copy()
    scaled[POINT_FEATURES] = scaler.transform(df[POINT_FEATURES])
    scaled["_eval_time_idx"] = df[TIME_INDEX_COLUMN].to_numpy()
    scaled["_eval_t_year"] = df["t_year"].to_numpy()

    train_ds = WindowDataset(scaled[scaled["split"] == "train"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    val_ds = WindowDataset(scaled[scaled["split"] == "validation"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    test_ds = WindowDataset(scaled[scaled["split"] == "test"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = WindowedMLPClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    start_train = time.perf_counter()
    best_epoch = 0
    epochs_ran = 0
    for _epoch in range(1 if args.smoke_test else args.max_epochs):
        epochs_ran = _epoch + 1
        model.train()
        for xb, yb, *_meta in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, *_meta in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = _epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    train_seconds = time.perf_counter() - start_train
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = REVISION_CHECKPOINT_DIR / f"final_mlp_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_type": "windowed_mlp",
            "features": POINT_FEATURES,
            "encoder_length": MAX_ENCODER_LENGTH,
            "prediction_length": MAX_PREDICTION_LENGTH,
            "seed": seed,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        checkpoint_path,
    )

    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    meta_year: dict[tuple[int, int], float] = {}
    start_infer = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for xb, _yb, series_id, dec_time_idx, dec_t_year in test_loader:
            probs = torch.sigmoid(model(xb.to(device))).detach().cpu().numpy()
            sid_np = np.asarray(series_id)
            time_np = dec_time_idx.numpy()
            year_np = dec_t_year.numpy()
            for b, sid in enumerate(sid_np):
                for j in range(probs.shape[1]):
                    key = (int(sid), int(time_np[b, j]))
                    sums[key] += float(probs[b, j])
                    counts[key] += 1
                    meta_year[key] = float(year_np[b, j])
    infer_seconds = time.perf_counter() - start_infer

    pred = pd.DataFrame(
        {
            GROUP_COLUMN: [key[0] for key in sums],
            TIME_INDEX_COLUMN: [key[1] for key in sums],
            "t_year": [meta_year[key] for key in sums],
            "p_onset_pred": [sums[key] / counts[key] for key in sums],
        }
    )
    target_map = df[[GROUP_COLUMN, TIME_INDEX_COLUMN, TARGET_COLUMN]].drop_duplicates()
    pred = pred.merge(target_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left")
    result = evaluate_point_predictions(
        pred,
        model_name="MLP",
        seed=seed,
        parameter_count_value=parameter_count(model),
        training_time_seconds=train_seconds,
        inference_time_seconds=infer_seconds,
    )
    result.metrics["epochs"] = epochs_ran
    result.metrics["best_epoch"] = best_epoch
    result.metrics["best_validation_loss"] = best_val
    result.metrics["checkpoint_path"] = str(checkpoint_path)
    return result.metrics, result.pf_curve, checkpoint_path


def train_gru(df: pd.DataFrame, seed: int, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, Path]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(args.device)
    scaler = fit_point_scaler(df)
    scaled = df.copy()
    scaled[POINT_FEATURES] = scaler.transform(df[POINT_FEATURES])
    scaled["_eval_time_idx"] = df[TIME_INDEX_COLUMN].to_numpy()
    scaled["_eval_t_year"] = df["t_year"].to_numpy()

    train_ds = WindowDataset(scaled[scaled["split"] == "train"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    val_ds = WindowDataset(scaled[scaled["split"] == "validation"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    test_ds = WindowDataset(scaled[scaled["split"] == "test"], POINT_FEATURES, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    model = GRUClassifier(input_size=len(POINT_FEATURES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    start_train = time.perf_counter()
    best_epoch = 0
    epochs_ran = 0
    for _epoch in range(1 if args.smoke_test else args.max_epochs):
        epochs_ran = _epoch + 1
        model.train()
        for xb, yb, *_meta in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, *_meta in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = _epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    train_seconds = time.perf_counter() - start_train
    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = REVISION_CHECKPOINT_DIR / f"final_gru_seed{seed}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "features": POINT_FEATURES,
            "seed": seed,
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        checkpoint_path,
    )

    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    meta_year: dict[tuple[int, int], float] = {}
    start_infer = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for xb, _yb, series_id, dec_time_idx, dec_t_year in test_loader:
            probs = torch.sigmoid(model(xb.to(device))).detach().cpu().numpy()
            sid_np = np.asarray(series_id)
            time_np = dec_time_idx.numpy()
            year_np = dec_t_year.numpy()
            for b, sid in enumerate(sid_np):
                for j in range(probs.shape[1]):
                    key = (int(sid), int(time_np[b, j]))
                    sums[key] += float(probs[b, j])
                    counts[key] += 1
                    meta_year[key] = float(year_np[b, j])
    infer_seconds = time.perf_counter() - start_infer

    pred = pd.DataFrame(
        {
            GROUP_COLUMN: [key[0] for key in sums],
            TIME_INDEX_COLUMN: [key[1] for key in sums],
            "t_year": [meta_year[key] for key in sums],
            "p_onset_pred": [sums[key] / counts[key] for key in sums],
        }
    )
    target_map = df[[GROUP_COLUMN, TIME_INDEX_COLUMN, TARGET_COLUMN]].drop_duplicates()
    pred = pred.merge(target_map, on=[GROUP_COLUMN, TIME_INDEX_COLUMN], how="left")
    result = evaluate_point_predictions(
        pred,
        model_name="GRU",
        seed=seed,
        parameter_count_value=parameter_count(model),
        training_time_seconds=train_seconds,
        inference_time_seconds=infer_seconds,
    )
    result.metrics["epochs"] = epochs_ran
    result.metrics["best_epoch"] = best_epoch
    result.metrics["best_validation_loss"] = best_val
    result.metrics["checkpoint_path"] = str(checkpoint_path)
    return result.metrics, result.pf_curve, checkpoint_path


def train_tft(df: pd.DataFrame, seed: int, args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame, Path]:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data.encoders import NaNLabelEncoder
    from pytorch_forecasting.metrics import CrossEntropy

    pl.seed_everything(seed, workers=True)
    categorical_encoders = {f"__group_id__{GROUP_COLUMN}": NaNLabelEncoder(add_nan=True)}
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()
    test_df = df[df["split"] == "test"].copy()
    test_df_sorted = test_df.sort_values([GROUP_COLUMN, TIME_INDEX_COLUMN]).reset_index(drop=True)

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
    if args.tft_window_stride > 1:
        training.index = training.index.iloc[:: args.tft_window_stride].reset_index(drop=True)
    validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
    testing = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)
    train_loader = training.to_dataloader(train=True, batch_size=args.batch_size, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)
    test_loader = testing.to_dataloader(train=False, batch_size=INFERENCE_BATCH_SIZE, num_workers=0)

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
    ckpt = ModelCheckpoint(
        dirpath=str(REVISION_CHECKPOINT_DIR),
        filename=f"final_tft_seed{seed}" + "-{epoch}-{val_loss:.4f}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        max_epochs=1 if args.smoke_test else args.tft_max_epochs,
        accelerator="cpu" if args.device == "cpu" else "auto",
        devices=1 if args.device == "cpu" else "auto",
        callbacks=[ckpt, EarlyStopping(monitor="val_loss", patience=args.patience, mode="min")],
        logger=False,
        enable_checkpointing=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        log_every_n_steps=20,
    )
    start_train = time.perf_counter()
    trainer.fit(model, train_loader, val_loader)
    train_seconds = time.perf_counter() - start_train
    best_path = Path(ckpt.best_model_path)
    model = TemporalFusionTransformer.load_from_checkpoint(str(best_path), weights_only=False)
    model.eval()

    device = torch.device(args.device)
    model.to(device)
    sums: dict[tuple[int, int], float] = defaultdict(float)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    sample_index = testing.index.reset_index(drop=True)
    sample_offset = 0
    start_infer = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            x, _ = batch
            x_dev = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in x.items()}
            logits = model(x_dev)["prediction"]
            probs = torch.softmax(logits, dim=-1)[..., 1].detach().cpu().numpy()
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
        training_time_seconds=train_seconds,
        inference_time_seconds=infer_seconds,
    )
    result.metrics["epochs"] = int(trainer.current_epoch)
    result.metrics["best_epoch"] = None
    result.metrics["best_validation_loss"] = float(ckpt.best_model_score.detach().cpu()) if ckpt.best_model_score is not None else None
    result.metrics["checkpoint_path"] = str(best_path)
    return result.metrics, result.pf_curve, best_path


def summarize_results(rows: list[dict[str, object]]) -> pd.DataFrame:
    by_seed = pd.DataFrame(rows)
    summary_rows = []
    for model, group in by_seed.groupby("model", sort=False):
        metrics = {"model": model}
        for column in ["MAE", "RMSE", "max_abs_error", "final_year_abs_error", "training_time_seconds", "pure_inference_time_seconds", "best_validation_loss"]:
            metrics[f"{column}_mean"] = float(group[column].mean())
            metrics[f"{column}_std"] = float(group[column].std(ddof=0)) if len(group) > 1 else 0.0
        metrics["parameter_count"] = int(group["parameter_count"].dropna().iloc[0]) if group["parameter_count"].notna().any() else None
        metrics["epochs_mean"] = float(group["epochs"].dropna().mean()) if group["epochs"].notna().any() else None
        summary_rows.append(metrics)
    return pd.DataFrame(summary_rows)


def write_comparison_plot(summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4))
    plt.bar(summary["model"], summary["MAE_mean"], yerr=summary["MAE_std"], capsize=4)
    plt.ylabel("MAE on cumulative Pf(t)")
    plt.title("Revision model comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_model_error_comparison.png", dpi=300)
    plt.close()


def write_population_plot(pf_curves: dict[str, list[pd.DataFrame]]) -> None:
    plt.figure(figsize=(7, 4.5))
    first = next(iter(pf_curves.values()))[0]
    plt.plot(first["t_year"], first["Pf_true"], color="black", linewidth=2.2, label="Reference cumulative Pf(t)")
    for model, curves in pf_curves.items():
        stacked = pd.concat([curve[["t_year", "Pf_pred"]] for curve in curves], ignore_index=True)
        mean_curve = stacked.groupby("t_year", as_index=False)["Pf_pred"].mean().sort_values("t_year")
        plt.plot(mean_curve["t_year"], mean_curve["Pf_pred"], linewidth=1.8, linestyle="--", label=model)
    plt.xlabel("Time (years)")
    plt.ylabel("Cumulative corrosion initiation probability")
    plt.title("Held-out population trajectories by model")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(REVISION_FIGURE_DIR / "final_population_trajectories_by_model.png", dpi=300)
    plt.close()


def write_benchmark_report(summary: pd.DataFrame, by_seed: pd.DataFrame) -> None:
    ranked = summary.sort_values("MAE_mean").reset_index(drop=True)
    tft = summary.loc[summary["model"] == "TFT"].iloc[0] if (summary["model"] == "TFT").any() else None

    def compare_to(other: str) -> str:
        if tft is None or not (summary["model"] == other).any():
            return "not available"
        other_row = summary.loc[summary["model"] == other].iloc[0]
        return "yes" if float(tft["MAE_mean"]) < float(other_row["MAE_mean"]) else "no"

    lines = [
        "# Final Benchmark Report",
        "",
        "All models were evaluated on the independent test split using cumulative `onset_flag`, shared time range, shared population aggregation, and identical physical predictors with `C_th` included. `chloride_rebar` and target-derived fields were not used as predictors. MLP, GRU, and TFT share a 52-step encoder and 13-step prediction horizon with unit stride; Logistic Regression is a pointwise tabular baseline.",
        "",
        "## Model Ranking by Test MAE",
        "",
        ranked[["model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std", "training_time_seconds_mean", "pure_inference_time_seconds_mean", "parameter_count"]].to_string(index=False),
        "",
        "## Explicit Comparison Answers",
        "",
        f"Does TFT outperform Logistic Regression? {compare_to('Logistic Regression')}.",
        f"Does TFT outperform MLP? {compare_to('MLP')}.",
        f"Does TFT outperform GRU? {compare_to('GRU')}.",
        "Is any advantage consistent across seeds? See `outputs/revision/tables/final_model_comparison_by_seed.csv`; neural-model seed dispersion is reported in the final comparison table.",
        "Is any accuracy advantage large enough to justify greater complexity? This should be judged from the held-out MAE/RMSE differences together with parameter count and inference time in the table above.",
        f"Best accuracy-efficiency tradeoff by MAE and deterministic inference time: {ranked.iloc[0]['model']} has the lowest MAE; compare inference time before making a complexity claim.",
        "",
        "No TFT superiority claim should be made unless supported by the final held-out metrics above.",
    ]
    Path("outputs/revision/final_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def merge_existing_by_seed(new_rows: list[dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    existing_path = REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    combined["_seed_key"] = combined["seed"].fillna(-1).astype(int)
    combined = combined.drop_duplicates(subset=["model", "_seed_key"], keep="last").drop(columns="_seed_key")
    return combined


def merge_existing_training_summary(new_rows: list[dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    existing_path = REVISION_TABLE_DIR / "final_training_summary.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
    else:
        combined = new_df
    if combined.empty:
        return combined
    combined["_seed_key"] = combined["seed"].fillna(-1).astype(int)
    combined = combined.drop_duplicates(subset=["model", "_seed_key"], keep="last").drop(columns="_seed_key")
    return combined


def load_available_pf_curves() -> dict[str, list[pd.DataFrame]]:
    files = {
        "Logistic Regression": [REVISION_PREDICTION_DIR / "final_pf_logistic_regression.csv"],
        "MLP": sorted(REVISION_PREDICTION_DIR.glob("final_pf_mlp_seed*.csv")),
        "GRU": sorted(REVISION_PREDICTION_DIR.glob("final_pf_gru_seed*.csv")),
        "TFT": sorted(REVISION_PREDICTION_DIR.glob("final_pf_tft_seed*.csv")),
    }
    curves: dict[str, list[pd.DataFrame]] = {}
    for model, paths in files.items():
        loaded = [pd.read_csv(path) for path in paths if path.exists()]
        if loaded:
            curves[model] = loaded
    return curves


def main() -> None:
    args = parse_args()
    df = prepare_data(args)
    rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    pf_curves: dict[str, list[pd.DataFrame]] = {}

    if "logistic" in args.models:
        metrics, pf = train_logistic(df)
        rows.append(metrics)
        pf_curves.setdefault("Logistic Regression", []).append(pf)
        pf.to_csv(REVISION_PREDICTION_DIR / "final_pf_logistic_regression.csv", index=False)

    for seed in args.seeds:
        if "mlp" in args.models:
            metrics, pf, path = train_windowed_mlp(df, seed, args)
            rows.append(metrics)
            checkpoint_rows.append({"model": "MLP", "seed": seed, "checkpoint": str(path), "best_validation_loss": metrics.get("best_validation_loss"), "epochs": metrics.get("epochs")})
            pf_curves.setdefault("MLP", []).append(pf)
            pf.to_csv(REVISION_PREDICTION_DIR / f"final_pf_mlp_seed{seed}.csv", index=False)
        if "gru" in args.models:
            metrics, pf, path = train_gru(df, seed, args)
            rows.append(metrics)
            checkpoint_rows.append({"model": "GRU", "seed": seed, "checkpoint": str(path), "best_validation_loss": metrics.get("best_validation_loss"), "epochs": metrics.get("epochs")})
            pf_curves.setdefault("GRU", []).append(pf)
            pf.to_csv(REVISION_PREDICTION_DIR / f"final_pf_gru_seed{seed}.csv", index=False)
        if "tft" in args.models:
            metrics, pf, path = train_tft(df, seed, args)
            rows.append(metrics)
            checkpoint_rows.append({"model": "TFT", "seed": seed, "checkpoint": str(path), "best_validation_loss": metrics.get("best_validation_loss"), "epochs": metrics.get("epochs")})
            pf_curves.setdefault("TFT", []).append(pf)
            pf.to_csv(REVISION_PREDICTION_DIR / f"final_pf_tft_seed{seed}.csv", index=False)

    by_seed = merge_existing_by_seed(rows)
    summary = summarize_results(by_seed.to_dict(orient="records"))
    training_summary = merge_existing_training_summary(checkpoint_rows)
    pf_curves = load_available_pf_curves()
    by_seed.to_csv(REVISION_TABLE_DIR / "final_model_comparison_by_seed.csv", index=False)
    summary.to_csv(REVISION_TABLE_DIR / "final_model_comparison.csv", index=False)
    training_summary.to_csv(REVISION_TABLE_DIR / "final_training_summary.csv", index=False)
    write_comparison_plot(summary)
    if pf_curves:
        write_population_plot(pf_curves)
    write_benchmark_report(summary, by_seed)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
