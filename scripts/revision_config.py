"""
Central configuration for reviewer-revision experiments.

This module keeps revision outputs separate from the original manuscript
artifacts and makes the corrected scientific target explicit:

    Pf(t) = P(Ti <= t)

The repository column implementing this cumulative target is ``onset_flag``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REVISION_OUTPUT_DIR = ROOT / "outputs" / "revision"
REVISION_CHECKPOINT_DIR = REVISION_OUTPUT_DIR / "checkpoints"
REVISION_PREDICTION_DIR = REVISION_OUTPUT_DIR / "predictions"
REVISION_FIGURE_DIR = REVISION_OUTPUT_DIR / "figures"
REVISION_TABLE_DIR = REVISION_OUTPUT_DIR / "tables"
REVISION_LOG_DIR = REVISION_OUTPUT_DIR / "logs"
REVISION_DATA_DIR = ROOT / "data" / "processed" / "revision"

SOURCE_LABELED_DATA = ROOT / "data" / "processed" / "chloride_labeled.parquet"
SOURCE_ONSET_SUMMARY = ROOT / "data" / "processed" / "onset_summary.csv"
REVISION_LABELED_DATA = REVISION_DATA_DIR / "chloride_labeled_revision.parquet"
FINAL_LABELED_DATA = REVISION_DATA_DIR / "final_chloride_labeled.parquet"
FINAL_ONSET_SUMMARY = REVISION_DATA_DIR / "final_onset_summary.csv"
SERIES_SPLIT_PATH = REVISION_DATA_DIR / "series_split.csv"

MANUSCRIPT_SEED = 20250111
REVISION_SEEDS = (20250111, 20250112, 20250113)

TARGET_COLUMN = "onset_flag"
INSTANTANEOUS_TARGET_COLUMN = "target_onset"
RAW_ONSET_COLUMN = "onset_raw"
TIME_COLUMN = "t_year"
TIME_INDEX_COLUMN = "time_idx"
GROUP_COLUMN = "series_id"

PHYSICAL_FEATURES = ["Cs", "D28", "m_aging", "cover_mm", "C_th"]
POINT_FEATURES = [*PHYSICAL_FEATURES, "time_idx", "t_year"]
TFT_STATIC_REALS = PHYSICAL_FEATURES
TFT_TIME_VARYING_KNOWN_REALS = ["time_idx", "t_year"]
TFT_TIME_VARYING_UNKNOWN_REALS: list[str] = []

FORBIDDEN_PREDICTORS = {
    "chloride_rebar",
    "target_onset",
    "onset_raw",
    "onset_flag",
    "time_to_onset",
    "t_init_year",
    "t_init_idx",
    "target_cont",
    "Pf",
    "Pf_true",
    "Pf_pred",
}

MAX_ENCODER_LENGTH = 52
MAX_PREDICTION_LENGTH = 13
BATCH_SIZE = 64
INFERENCE_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
TFT_HIDDEN_SIZE = 32
TFT_ATTENTION_HEADS = 4
TFT_DROPOUT = 0.1

SPLIT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLIT_SEED = MANUSCRIPT_SEED
PARAMETER_LOCK_STATUS = "FINAL_LOCKED_BEFORE_MODEL_TRAINING"
SELECTED_PARAMETER_CANDIDATE = "C"


@dataclass(frozen=True)
class BoundedDistributionSpec:
    name: str
    distribution: str
    lower: float
    upper: float
    mean: float | None = None
    std: float | None = None
    units: str = ""

    @property
    def is_fully_specified(self) -> bool:
        if self.distribution.lower() == "uniform":
            return True
        return self.mean is not None and self.std is not None

    @property
    def lognormal_mu_sigma(self) -> tuple[float, float]:
        if "lognormal" not in self.distribution.lower():
            raise ValueError(f"{self.name} is not lognormal.")
        if self.mean is None or self.std is None:
            raise ValueError(f"{self.name} is missing physical-space mean/std.")
        sigma = math.sqrt(math.log(1.0 + (self.std / self.mean) ** 2))
        mu = math.log(self.mean) - 0.5 * sigma**2
        return mu, sigma


# Bounds and distribution families are manuscript/advisor-confirmed. Shape/central
# parameters are the prespecified Candidate C values selected by simulator-only
# screening before any full model training.
PARAMETER_SPECS = {
    "Cs": BoundedDistributionSpec("Cs", "truncated_lognormal", 2.0, 6.0, mean=4.6, std=0.60, units="kg/m^3"),
    "D28": BoundedDistributionSpec("D28", "truncated_lognormal", 1e-12, 5e-12, mean=4.0e-12, std=0.45e-12, units="m^2/s"),
    "m_aging": BoundedDistributionSpec("m_aging", "truncated_normal", 0.2, 0.6, mean=0.30, std=0.050, units="dimensionless"),
    "cover_mm": BoundedDistributionSpec("cover_mm", "uniform", 40.0, 110.0, units="mm"),
    "C_th": BoundedDistributionSpec("C_th", "truncated_lognormal", 0.6, 1.2, mean=0.75, std=0.090, units="kg/m^3"),
}


def ensure_revision_dirs() -> None:
    for path in [
        REVISION_OUTPUT_DIR,
        REVISION_CHECKPOINT_DIR,
        REVISION_PREDICTION_DIR,
        REVISION_FIGURE_DIR,
        REVISION_TABLE_DIR,
        REVISION_LOG_DIR,
        REVISION_DATA_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def missing_distribution_parameters() -> list[BoundedDistributionSpec]:
    return [spec for spec in PARAMETER_SPECS.values() if not spec.is_fully_specified]
