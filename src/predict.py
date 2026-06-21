"""
predict.py
==========
Loads the saved model pipeline and turns raw, human-readable inputs
(temperature, humidity, a timestamp, etc.) into a prediction. This is the
module both the Streamlit app and any future API wrapper should import —
it is the single place that knows how to go from "a person's inputs" to
"a model-ready feature row."
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_engineering import add_datetime_features, add_derived_environmental_features
from train import CATEGORICAL_FEATURES, NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.pkl"

# Sensible defaults for sensors the user isn't asked about directly in a
# simplified UI (e.g. the Streamlit app only asks for a handful of inputs,
# not all 9 room sensors). These are dataset medians, computed once and
# hardcoded here so predict.py has no hidden runtime dependency on the
# training CSV being present.
DEFAULT_SENSOR_VALUES = {
    "lights": 0.0, "T1": 21.0, "RH_1": 40.0, "T2": 20.0, "RH_2": 40.0,
    "T3": 22.0, "RH_3": 39.0, "T4": 20.5, "RH_4": 39.0, "T5": 19.5,
    "RH_5": 50.0, "T6": 7.5, "RH_6": 55.0, "T7": 20.5, "RH_7": 35.0,
    "T8": 22.0, "RH_8": 42.0, "T9": 19.5, "RH_9": 41.5, "T_out": 7.5,
    "Press_mm_hg": 755.0, "RH_out": 80.0, "Windspeed": 4.0,
    "Visibility": 38.0, "Tdewpoint": 4.0,
}


class EnergyPredictor:
    """Thin wrapper around the saved sklearn Pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run `python src/train.py` first."
            )
        self.pipeline = joblib.load(model_path)
        self.metadata = joblib.load(METADATA_PATH) if METADATA_PATH.exists() else {}

    def _build_feature_row(self, timestamp: datetime, sensor_overrides: dict) -> pd.DataFrame:
        """Combine a timestamp + any provided sensor readings into a single
        full feature row, filling anything unspecified with the defaults."""
        row = {**DEFAULT_SENSOR_VALUES, **sensor_overrides}
        df = pd.DataFrame([row])
        df["date"] = pd.Timestamp(timestamp)

        df = add_datetime_features(df)
        df = add_derived_environmental_features(df)
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    def predict(self, timestamp: datetime, **sensor_overrides) -> float:
        """
        Predict appliance energy use (Wh) for a given timestamp and any
        subset of sensor readings.

        Example
        -------
        >>> predictor = EnergyPredictor()
        >>> predictor.predict(datetime(2026, 6, 18, 19, 0), T1=23.5, RH_1=42.0)
        """
        feature_row = self._build_feature_row(timestamp, sensor_overrides)
        prediction = self.pipeline.predict(feature_row)[0]
        return max(0.0, float(prediction))

    def predict_with_confidence(self, timestamp: datetime, **sensor_overrides) -> dict:
        """
        For ensemble models (Random Forest / Gradient Boosting / XGBoost),
        derive a rough confidence interval from the spread of individual
        trees' predictions. For Linear Regression, fall back to a fixed
        +/- based on training RMSE, since it has no individual estimators
        to disagree with each other.
        """
        feature_row = self._build_feature_row(timestamp, sensor_overrides)
        point_prediction = self.predict(timestamp, **sensor_overrides)

        model = self.pipeline.named_steps["model"]
        preprocessor = self.pipeline.named_steps["preprocessor"]
        X_transformed = preprocessor.transform(feature_row)

        if hasattr(model, "estimators_"):
            tree_preds = np.array([est.predict(X_transformed)[0] for est in model.estimators_])
            std = float(tree_preds.std())
            low, high = max(0.0, point_prediction - 1.96 * std), point_prediction + 1.96 * std
            confidence_note = f"95% interval from {len(model.estimators_)} ensemble members"
        else:
            std = 85.0  # approx test RMSE for the linear baseline
            low, high = max(0.0, point_prediction - 1.96 * std), point_prediction + 1.96 * std
            confidence_note = "95% interval approximated from validation RMSE (model has no ensemble spread)"

        return {
            "prediction_wh": round(point_prediction, 1),
            "lower_bound_wh": round(low, 1),
            "upper_bound_wh": round(high, 1),
            "confidence_note": confidence_note,
            "model_used": self.metadata.get("best_model_name", "unknown"),
        }


if __name__ == "__main__":
    predictor = EnergyPredictor()
    sample_time = datetime(2026, 6, 18, 19, 30)
    result = predictor.predict_with_confidence(sample_time, T1=23.0, RH_1=45.0, T_out=18.0)
    print(f"Prediction for {sample_time}: {result}")
