# Windowed Logistic Regression Baseline Report

**Run mode:** full benchmark

## Configuration

- Model: `Windowed Logistic Regression`
- Input: 52-step encoder × 7 covariates → flattened 364 features
- Output: 13-step horizon via `MultiOutputClassifier(LogisticRegression)`
- Preprocessing: `StandardScaler` on flattened windows
- Inference: overlap-averaged window predictions (stride = 1)
- Target: cumulative onset probability Pf(t)

## Results

| Metric | Value |
|---|---|
| MAE | 0.020212 |
| RMSE | 0.023823 |
| Max absolute error | 0.051979 |
| Final-year absolute error | 0.048765 |
| Training time (s) | 37.58 |
| Inference time (s) | 3.43 |
| Parameter count | 4,745 |

## Comparison

- Improves over pointwise Logistic Regression (MAE 0.020652): **Yes**
- Remains worse than Windowed MLP (MAE 0.006975): **Yes**
- Remains worse than GRU (MAE 0.001934): **Yes**
- Remains worse than TFT (MAE 0.004542): **Yes**

## Manuscript Table Row

| Model | MAE (mean ± std) | RMSE (mean ± std) |
|---|---|---|
| Windowed Logistic Regression | 0.020212 ± 0.000000 | 0.023823 ± 0.000000 |

## Artifacts

- Predictions: `outputs/revision/predictions/final_pf_windowed_logistic_regression.csv`
- Checkpoint: `outputs/revision/checkpoints/final_windowed_logistic_regression.joblib`