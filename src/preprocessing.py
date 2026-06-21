"""
preprocessing.py
================
Data loading, cleaning, and validation utilities for the Smart Energy
Consumption Prediction project.

WHAT THIS FILE DOES
--------------------
Real-world sensor data is messy: stray whitespace inside numeric strings,
duplicate readings from the wireless nodes, and physically impossible
outliers (e.g. negative humidity). Before any model can learn from this
data, it has to be loaded, cleaned, and validated. This file is the single
place where that happens, so every other script (training, the notebook,
the Streamlit app) cleans the data in exactly the same way.

WHY IT IS NEEDED
----------------
If cleaning logic were copy-pasted into train.py, evaluate.py, and app.py
separately, the three would eventually drift out of sync (a classic source
of "it worked in training but not in production" bugs). Centralizing it
here means there is one source of truth.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Columns that are random noise variables included by the dataset authors
# specifically to test whether a model can learn to ignore irrelevant
# features. We never want to feed these into the model.
NOISE_COLUMNS = ["rv1", "rv2"]

# Physically sensible ranges used for outlier flagging. These come from the
# original paper's description of the sensors used (ZigBee temperature /
# humidity nodes + an airport weather station), not from looking at the data
# first — that keeps the cleaning logic honest rather than overfit to this
# specific sample.
PLAUSIBLE_RANGES = {
    "Appliances": (0, 2000),       # Wh, sanity ceiling for a household
    "lights": (0, 100),
    "RH_out": (0, 100),
    "Windspeed": (0, 60),
    "Visibility": (0, 70),
}


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the raw UCI Appliances Energy Prediction CSV.

    HOW IT WORKS
    ------------
    The original CSV wraps every value (including numbers) in double
    quotes, which makes pandas read everything as strings unless we
    explicitly convert afterwards. We load it, strip the quoting
    side-effects, and coerce columns to numeric/datetime types.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find dataset at {csv_path}. Download it from "
            "https://github.com/LuisM78/Appliances-energy-prediction-data "
            "and place it at this path, or point load_data() at the right file."
        )

    df = pd.read_csv(csv_path)

    # Strip whitespace from column names (defensive — some mirrors of this
    # dataset have trailing spaces in headers).
    df.columns = [c.strip() for c in df.columns]

    # The 'date' column needs to become an actual datetime so downstream
    # feature engineering (hour, weekday, etc.) can use it.
    df["date"] = pd.to_datetime(df["date"])

    # Every other column should be numeric. Some mirrors store numbers as
    # strings with leading/trailing whitespace (e.g. "  60"); pd.to_numeric
    # handles that coercion safely.
    for col in df.columns:
        if col == "date":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], csv_path)
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows (same timestamp + same readings)."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows", removed)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sensor readings using time-aware interpolation.

    WHY INTERPOLATION (not mean/median fill)
    ------------------------------------------
    This is a 10-minute interval time series. A missing temperature
    reading is almost always very close to the readings immediately
    before and after it — interpolation respects that continuity, whereas
    filling with the column mean would inject discontinuous jumps into a
    naturally smooth signal.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_missing_before = df.isna().sum().sum()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Any column that is still fully empty after interpolation (shouldn't
    # happen on this dataset, but defensive) falls back to median fill.
    remaining = df[numeric_cols].isna().sum()
    for col in remaining[remaining > 0].index:
        df[col] = df[col].fillna(df[col].median())

    n_missing_after = df.isna().sum().sum()
    logger.info("Missing values: %d before -> %d after cleaning", n_missing_before, n_missing_after)
    return df


def flag_outliers(df: pd.DataFrame, columns: list[str] | None = None, z_thresh: float = 5.0) -> pd.DataFrame:
    """
    Cap extreme outliers using a z-score rule combined with physically
    plausible ranges.

    WHY CAPPING INSTEAD OF DELETING ROWS
    -------------------------------------
    Deleting rows breaks the time-series continuity that feature
    engineering (hour-of-day patterns, etc.) relies on. Capping
    (winsorizing) preserves every timestamp while preventing a handful of
    extreme sensor glitches from dominating the loss function of models
    like Linear Regression that are sensitive to outliers.
    """
    df = df.copy()
    columns = columns or [c for c in df.select_dtypes(include=[np.number]).columns if c not in NOISE_COLUMNS]

    for col in columns:
        series = df[col]
        mean, std = series.mean(), series.std()
        if std == 0 or np.isnan(std):
            continue
        z_scores = (series - mean) / std
        lower_z, upper_z = mean - z_thresh * std, mean + z_thresh * std

        lo, hi = PLAUSIBLE_RANGES.get(col, (-np.inf, np.inf))
        lower_bound = max(lower_z, lo)
        upper_bound = min(upper_z, hi)

        n_capped = ((series < lower_bound) | (series > upper_bound)).sum()
        if n_capped:
            df[col] = series.clip(lower=lower_bound, upper=upper_bound)
            logger.info("Capped %d outliers in '%s' to [%.2f, %.2f]", n_capped, col, lower_bound, upper_bound)

    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Run a battery of sanity checks and raise if the data fails them.
    This is a cheap insurance policy against silently training a model on
    broken data.
    """
    assert "date" in df.columns, "Missing 'date' column"
    assert "Appliances" in df.columns, "Missing target column 'Appliances'"
    assert df["date"].is_monotonic_increasing, "Timestamps are not sorted ascending"
    assert df.isna().sum().sum() == 0, "Data still contains missing values after cleaning"
    assert (df["Appliances"] >= 0).all(), "Found negative energy consumption values"
    assert df["date"].duplicated().sum() == 0, "Duplicate timestamps remain in the data"
    logger.info("Validation passed: %d rows, no missing values, timestamps sorted and unique", len(df))


def clean_pipeline(csv_path: str | Path, drop_noise_columns: bool = True) -> pd.DataFrame:
    """
    Run the full load -> dedupe -> impute -> outlier-cap -> validate
    pipeline in one call. This is the function everything else in the
    project should import.
    """
    df = load_data(csv_path)
    df = drop_duplicates(df)
    df = handle_missing_values(df)
    df = flag_outliers(df)
    if drop_noise_columns:
        df = df.drop(columns=[c for c in NOISE_COLUMNS if c in df.columns])
    validate_data(df)
    return df


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    cleaned = clean_pipeline(ROOT / "data" / "energydata_complete.csv")
    out_path = ROOT / "data" / "energy_cleaned.csv"
    cleaned.to_csv(out_path, index=False)
    logger.info("Saved cleaned dataset to %s (%d rows)", out_path, len(cleaned))
