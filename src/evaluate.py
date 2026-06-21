"""
evaluate.py
===========
Generates the model comparison table, performance bar charts, and
actual-vs-predicted diagnostic plots used in the README and the
project report. Reads the artifacts that train.py already saved
(models/all_pipelines.pkl, outputs/reports/metrics.json,
outputs/reports/test_split.csv) so it never re-trains anything —
evaluation should be fast and repeatable independent of training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"

sns.set_theme(style="whitegrid", palette="deep")
PALETTE = ["#2563eb", "#16a34a", "#ea580c", "#9333ea"]


def load_artifacts():
    with open(REPORTS_DIR / "metrics.json") as f:
        report = json.load(f)
    pipelines = joblib.load(ROOT / "models" / "all_pipelines.pkl")
    test_df = pd.read_csv(REPORTS_DIR / "test_split.csv")
    return report, pipelines, test_df


def comparison_table(metrics: dict) -> pd.DataFrame:
    """Build a sorted (best-first) comparison DataFrame from a metrics dict."""
    df = pd.DataFrame(metrics).T
    df = df[["MAE", "MSE", "RMSE", "R2", "train_time_sec"]]
    df = df.sort_values("RMSE")
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df.round(3)


def plot_model_comparison(metrics: dict, save_path: Path):
    """Grouped bar chart of RMSE, MAE, and R2 across all four models."""
    table = comparison_table(metrics)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    table[["RMSE", "MAE"]].plot(kind="bar", ax=axes[0], color=PALETTE[:2])
    axes[0].set_title("Error Metrics by Model (lower = better)")
    axes[0].set_ylabel("Watt-hours (Wh)")
    axes[0].tick_params(axis="x", rotation=20)

    bars = axes[1].bar(table.index, table["R2"], color=PALETTE)
    axes[1].set_title("R² Score by Model (higher = better)")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].tick_params(axis="x", rotation=20)
    for bar, val in zip(bars, table["R2"]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center")

    fig.suptitle("Model Performance Comparison (random 80/20 split)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_actual_vs_predicted(pipelines: dict, test_df: pd.DataFrame, best_name: str, save_path: Path):
    """Scatter of true vs predicted Appliances energy use for the best model."""
    from train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET]
    y_pred = pipelines[best_name].predict(X_test)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, y_pred, alpha=0.25, s=12, color=PALETTE[0])
    lims = [0, max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Appliance Energy Use (Wh)")
    ax.set_ylabel("Predicted Appliance Energy Use (Wh)")
    ax.set_title(f"Actual vs Predicted — {best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_residuals(pipelines: dict, test_df: pd.DataFrame, best_name: str, save_path: Path):
    """Residual distribution for the best model — checks for systematic bias."""
    from train import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df[TARGET]
    y_pred = pipelines[best_name].predict(X_test)
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(residuals, kde=True, ax=axes[0], color=PALETTE[2])
    axes[0].axvline(0, color="black", linestyle="--")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Actual − Predicted (Wh)")

    axes[1].scatter(y_pred, residuals, alpha=0.25, s=12, color=PALETTE[2])
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title("Residuals vs Predicted Value")
    axes[1].set_xlabel("Predicted (Wh)")
    axes[1].set_ylabel("Residual (Wh)")

    fig.suptitle(f"Residual Analysis — {best_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_feature_importance(pipelines: dict, best_name: str, save_path: Path, top_n: int = 15):
    """
    Feature importance for tree-based models. Skips gracefully for Linear
    Regression, which doesn't have a directly comparable importances_
    attribute in this pipeline form (it has coefficients instead, which
    aren't standardized the same way and would mislead a side-by-side
    comparison, so we only plot this for tree ensembles).
    """
    pipeline = pipelines[best_name]
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        logger.info("%s has no feature_importances_ — skipping importance plot", best_name)
        return

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values().plot(kind="barh", ax=ax, color=PALETTE[1])
    ax.set_title(f"Top {top_n} Feature Importances — {best_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    report, pipelines, test_df = load_artifacts()
    metrics = report["metrics_random_split"]
    best_name = report["best_model"]

    table = comparison_table(metrics)
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    logger.info("\n%s", table.to_string())

    plot_model_comparison(metrics, PLOTS_DIR / "model_comparison.png")
    plot_actual_vs_predicted(pipelines, test_df, best_name, PLOTS_DIR / "actual_vs_predicted.png")
    plot_residuals(pipelines, test_df, best_name, PLOTS_DIR / "residuals.png")
    plot_feature_importance(pipelines, best_name, PLOTS_DIR / "feature_importance.png")

    logger.info("Evaluation complete. Best model: %s", best_name)


if __name__ == "__main__":
    main()
