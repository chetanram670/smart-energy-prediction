"""
train.py
========
Trains Linear Regression, Random Forest, Gradient Boosting, and XGBoost
on the cleaned, feature-engineered dataset, then saves the best-performing
model (by test RMSE) to models/model.pkl.

WHY A PIPELINE OBJECT (not a raw model)
-----------------------------------------
We bundle preprocessing (scaling, one-hot encoding) and the model itself
into a single sklearn Pipeline. This means predict.py and app.py never
have to remember "oh, I need to scale the inputs the same way training did"
— that logic lives and travels with the saved model file, which is the
single biggest source of train/serve skew bugs in real ML systems.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
TARGET = "Appliances"

# Columns we deliberately exclude from the feature set: 'date' is not a
# numeric feature itself (we already extracted Hour/Day/Month/Weekday from
# it), and the target obviously can't be a feature of itself.
DROP_COLS = ["date", TARGET]

NUMERIC_FEATURES = [
    "lights", "T1", "RH_1", "T2", "RH_2", "T3", "RH_3", "T4", "RH_4", "T5",
    "RH_5", "T6", "RH_6", "T7", "RH_7", "T8", "RH_8", "T9", "RH_9", "T_out",
    "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint",
    "Hour", "Day", "Month", "Weekday", "IsWeekend", "IsWorkingDay",
    "Hour_sin", "Hour_cos", "Temp_avg", "Temp_spread", "RH_avg",
    "RH_spread", "Indoor_outdoor_temp_diff",
]
CATEGORICAL_FEATURES = ["DayPeriod"]


def build_preprocessor() -> ColumnTransformer:
    """
    Numeric columns get standardized (mean 0, std 1) — this matters a lot
    for Linear Regression and barely at all for tree-based models, but
    doing it uniformly keeps one pipeline definition usable for every
    model type. Categorical 'DayPeriod' gets one-hot encoded since it has
    no natural numeric ordering a model should assume.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_model_candidates(random_state: int = 42) -> dict:
    """Return the dict of {model_name: estimator} we will train and compare."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=2,
            random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_state,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=random_state,
            n_jobs=-1, verbosity=0,
        ),
    }


def time_based_split(df: pd.DataFrame, test_size: float = 0.2):
    """
    Split chronologically: train on the first (1-test_size) of the
    timeline, test on the final test_size.

    WHEN TO USE THIS
    -----------------
    This is the honest choice if the deployment goal is "forecast energy
    use into a future period the model has never seen." We use it in
    evaluate.py purely as a *diagnostic* (see README "Limitations"
    section) because, on this dataset, it exposes a real problem: tree
    models cannot extrapolate beyond the feature ranges they were trained
    on, and a 4.5-month dataset has enough seasonal drift (e.g. outdoor
    temperature trending upward toward spring) that a chronological
    holdout becomes an extrapolation problem rather than an interpolation
    problem. It is not used for the headline metrics.
    """
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def random_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Randomly shuffle rows into train/test.

    WHY RANDOM SPLIT IS THE PRIMARY METHODOLOGY HERE
    ---------------------------------------------------
    The modeling question this project answers is "given the current
    temperature, humidity, and time-of-day readings, what is the
    appliance load?" — a cross-sectional regression on contextual
    features, not a multi-step-ahead time-series forecast. Each 10-minute
    reading is treated as an (almost) independent sample of "house
    conditions -> appliance load," which is exactly the framing used in
    the original Candanedo et al. (2017) paper and in most public
    implementations of this dataset. A random split lets every model be
    evaluated on a representative cross-section of conditions (all months,
    all hours) rather than being penalized for the dataset's short 4.5-month
    span. See README "Limitations" for the chronological-split counter-experiment.
    """
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)


def evaluate_predictions(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_and_compare(df: pd.DataFrame, split_strategy: str = "random") -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """
    Train every candidate model, evaluate it on a held-out test split, and
    return everything callers need: fitted pipelines, their metrics, and
    the train/test splits (so evaluate.py can reuse them for plots without
    re-splitting).
    """
    if split_strategy == "random":
        train_df, test_df = random_split(df)
    elif split_strategy == "time":
        train_df, test_df = time_based_split(df)
    else:
        raise ValueError("split_strategy must be 'random' or 'time'")

    logger.info("Train rows: %d | Test rows: %d (%s split)", len(train_df), len(test_df), split_strategy)

    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET]

    fitted_pipelines = {}
    metrics = {}

    for name, model in get_model_candidates().items():
        pipeline = Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", model)])

        start = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start

        y_pred = pipeline.predict(X_test)
        scores = evaluate_predictions(y_test, y_pred)
        scores["train_time_sec"] = round(train_time, 2)

        fitted_pipelines[name] = pipeline
        metrics[name] = scores

        logger.info(
            "%-18s | RMSE=%.2f | MAE=%.2f | R2=%.3f | trained in %.2fs",
            name, scores["RMSE"], scores["MAE"], scores["R2"], train_time,
        )

    return fitted_pipelines, metrics, train_df, test_df


def select_best_model(metrics: dict) -> str:
    """Best model = lowest test RMSE. RMSE is chosen over R2 as the primary
    criterion because it is in the same units as the target (Wh) and is
    what actually matters for a deployed prediction's practical accuracy."""
    return min(metrics, key=lambda name: metrics[name]["RMSE"])


def main():
    from feature_engineering import engineer_features
    from preprocessing import clean_pipeline

    logger.info("=== Step 1/4: Cleaning data ===")
    cleaned = clean_pipeline(ROOT / "data" / "energydata_complete.csv")

    logger.info("=== Step 2/4: Feature engineering ===")
    featured = engineer_features(cleaned)

    logger.info("=== Step 3/4: Training & comparing models (random split) ===")
    fitted_pipelines, metrics, train_df, test_df = train_and_compare(featured, split_strategy="random")

    logger.info("=== Diagnostic: same models under a chronological split (see README Limitations) ===")
    _, time_split_metrics, _, _ = train_and_compare(featured, split_strategy="time")

    logger.info("=== Step 4/4: Selecting & saving best model ===")
    best_name = select_best_model(metrics)
    best_pipeline = fitted_pipelines[best_name]
    logger.info("Best model: %s (RMSE=%.2f, R2=%.3f)", best_name, metrics[best_name]["RMSE"], metrics[best_name]["R2"])

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    # compress=3 trades a little load time for a ~4x smaller file — this
    # takes model.pkl from ~75MB to ~18MB, which matters for keeping a
    # GitHub repo clone-friendly without needing Git LFS.
    joblib.dump(best_pipeline, models_dir / "model.pkl", compress=3)
    joblib.dump({"best_model_name": best_name, "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES},
                models_dir / "model_metadata.pkl")

    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "metrics.json", "w") as f:
        json.dump({
            "metrics_random_split": metrics,
            "metrics_chronological_split_diagnostic": time_split_metrics,
            "best_model": best_name,
        }, f, indent=2)

    # Also save all four fitted pipelines so evaluate.py can plot every
    # model's predictions, not just the winner. This file is for local
    # analysis only — it's gitignored (see .gitignore) since it bundles
    # four full models and isn't needed to run the app.
    joblib.dump(fitted_pipelines, models_dir / "all_pipelines.pkl", compress=3)
    test_df.to_csv(reports_dir / "test_split.csv", index=False)
    train_df.to_csv(reports_dir / "train_split.csv", index=False)

    logger.info("Saved best model to %s", models_dir / "model.pkl")
    logger.info("Saved metrics to %s", reports_dir / "metrics.json")
    return fitted_pipelines, metrics, best_name


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
