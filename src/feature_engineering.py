"""
feature_engineering.py
=======================
Turns the raw 'date' timestamp into model-friendly features, and adds a
couple of physically-motivated derived features.

WHY THIS MATTERS
----------------
A raw timestamp like "2016-01-15 18:30:00" is useless to a regression
model as-is. Energy consumption has strong daily and weekly rhythms
(people cook dinner around 7pm, use less power at 3am, behave differently
on weekends) — but a model can only learn those rhythms if we expose them
as explicit numeric/categorical columns it can split on.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Indian public/working-day convention isn't relevant here (Belgian house),
# so "working day" is approximated as Mon-Fri, which is what the original
# paper's feature set (NSM, WeekStatus, Day_of_week) effectively encodes.
WEEKEND_DAYS = {5, 6}  # Saturday=5, Sunday=6 (pandas .dayofweek convention)


def add_datetime_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Explode a datetime column into the individual components a tree-based
    or linear model can actually use.

    FEATURES CREATED
    ----------------
    Hour            (0-23)  -> captures intraday usage pattern
    Day             (1-31)  -> weak monthly-cycle signal (e.g. billing days)
    Month           (1-12)  -> captures seasonal heating/cooling load
    Weekday         (0-6)   -> captures weekly routine (Mon vs Sat differ)
    IsWeekend       (0/1)   -> coarser version of Weekday models pick up fast
    IsWorkingDay    (0/1)   -> complement of IsWeekend, kept for readability
    DayPeriod       (categorical) -> Morning/Afternoon/Evening/Night bucket
    """
    df = df.copy()
    dt = df[date_col]

    df["Hour"] = dt.dt.hour
    df["Day"] = dt.dt.day
    df["Month"] = dt.dt.month
    df["Weekday"] = dt.dt.dayofweek
    df["IsWeekend"] = df["Weekday"].isin(WEEKEND_DAYS).astype(int)
    df["IsWorkingDay"] = 1 - df["IsWeekend"]

    df["DayPeriod"] = pd.cut(
        df["Hour"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Night"],
        ordered=False,
    )
    # pd.cut with overlapping label "Night" at both ends needs a manual fix
    # for hour 22-23, which fall in the last bin.
    df.loc[df["Hour"].between(22, 23), "DayPeriod"] = "Night"

    # Cyclical encoding of hour: a plain integer hour tells a linear model
    # that hour 23 and hour 0 are "far apart" (23 vs 0), which is wrong —
    # they are one minute apart. Sine/cosine encoding fixes that by mapping
    # hour onto a circle.
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

    logger.info("Added datetime features: Hour, Day, Month, Weekday, IsWeekend, "
                "IsWorkingDay, DayPeriod, Hour_sin, Hour_cos")
    return df


def add_derived_environmental_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a small number of physically-motivated derived features that
    aggregate the many per-room temperature/humidity sensors.

    WHY THESE HELP
    ---------------
    The dataset has 9 separate room temperature sensors (T1-T9) and 9
    humidity sensors (RH_1-RH_9). Individually they are noisy and highly
    correlated with each other; their *average* is a more stable signal of
    "how warm/humid is the house right now", and the *spread* (max-min)
    captures whether one room (e.g. the kitchen while cooking) is an outlier
    relative to the rest of the house — which is itself informative about
    appliance use.
    """
    df = df.copy()
    temp_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    rh_cols = [c for c in df.columns if c.startswith("RH_") and c[3:].isdigit()]

    if temp_cols:
        df["Temp_avg"] = df[temp_cols].mean(axis=1)
        df["Temp_spread"] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)
    if rh_cols:
        df["RH_avg"] = df[rh_cols].mean(axis=1)
        df["RH_spread"] = df[rh_cols].max(axis=1) - df[rh_cols].min(axis=1)

    if "T_out" in df.columns and "Temp_avg" in df.columns:
        df["Indoor_outdoor_temp_diff"] = df["Temp_avg"] - df["T_out"]

    logger.info("Added derived environmental features: Temp_avg, Temp_spread, "
                "RH_avg, RH_spread, Indoor_outdoor_temp_diff")
    return df


def engineer_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    df = add_datetime_features(df, date_col=date_col)
    df = add_derived_environmental_features(df)
    return df


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from preprocessing import clean_pipeline

    ROOT = Path(__file__).resolve().parent.parent
    cleaned = clean_pipeline(ROOT / "data" / "energydata_complete.csv")
    featured = engineer_features(cleaned)
    featured.to_csv(ROOT / "data" / "energy_featured.csv", index=False)
    logger.info("Saved feature-engineered dataset with %d columns", featured.shape[1])
