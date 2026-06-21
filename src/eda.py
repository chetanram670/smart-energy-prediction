"""
eda.py
======
Generates every exploratory-data-analysis plot referenced in the README:
dataset overview, missing value heatmap, statistical summary, correlation
matrix, histograms, boxplots, scatter plots, and feature distributions.

This is deliberately separate from preprocessing.py / feature_engineering.py:
EDA happens on the *raw* data (before cleaning decisions are locked in) so
that what we see here is what actually motivated the cleaning choices made
later, not data that's already been smoothed into looking nice.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import load_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "outputs" / "plots"
REPORTS_DIR = ROOT / "outputs" / "reports"

sns.set_theme(style="whitegrid")


def dataset_overview(df: pd.DataFrame) -> dict:
    overview = {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "date_range": [str(df["date"].min()), str(df["date"].max())],
        "duration_days": (df["date"].max() - df["date"].min()).days,
        "target_mean_Wh": round(df["Appliances"].mean(), 2),
        "target_median_Wh": round(df["Appliances"].median(), 2),
        "target_std_Wh": round(df["Appliances"].std(), 2),
        "target_max_Wh": round(df["Appliances"].max(), 2),
    }
    logger.info("Dataset overview: %s", overview)
    return overview


def missing_value_heatmap(df: pd.DataFrame, save_path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(df.isna().T, cbar=False, cmap="Reds", ax=ax)
    ax.set_title(f"Missing Value Map (total missing cells: {df.isna().sum().sum()})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def statistical_summary(df: pd.DataFrame, save_path: Path) -> pd.DataFrame:
    summary = df.describe().T
    summary.to_csv(save_path)
    logger.info("Saved statistical summary to %s", save_path)
    return summary


def correlation_matrix(df: pd.DataFrame, save_path: Path):
    numeric = df.select_dtypes(include=[np.number]).drop(columns=["rv1", "rv2"], errors="ignore")
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.3,
                cbar_kws={"shrink": 0.7}, ax=ax)
    ax.set_title("Correlation Matrix — All Numeric Features")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)

    target_corr = corr["Appliances"].drop("Appliances").sort_values(key=abs, ascending=False)
    return target_corr


def plot_target_distribution(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df["Appliances"], bins=50, kde=True, ax=axes[0], color="#2563eb")
    axes[0].set_title("Appliance Energy Use — Distribution")
    axes[0].set_xlabel("Wh per 10-minute interval")

    sns.boxplot(x=df["Appliances"], ax=axes[1], color="#2563eb")
    axes[1].set_title("Appliance Energy Use — Boxplot (outlier view)")
    axes[1].set_xlabel("Wh per 10-minute interval")

    fig.suptitle("Target Variable: Appliances", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_feature_histograms(df: pd.DataFrame, save_path: Path):
    cols = ["T1", "T_out", "RH_1", "RH_out", "Press_mm_hg", "Windspeed", "Visibility", "Tdewpoint"]
    cols = [c for c in cols if c in df.columns]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.flat, cols):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color="#16a34a")
        ax.set_title(col)
    fig.suptitle("Feature Distributions: Temperature, Humidity & Weather", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_boxplots_by_room(df: pd.DataFrame, save_path: Path):
    temp_cols = [c for c in df.columns if c.startswith("T") and c[1:].isdigit()]
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df[temp_cols], ax=ax, palette="Blues")
    ax.set_title("Temperature Spread Across Rooms (T1–T9)")
    ax.set_ylabel("Degrees Celsius")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_scatter_relationships(df: pd.DataFrame, save_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [("T_out", "Appliances", "Outdoor Temp vs Energy Use"),
             ("RH_out", "Appliances", "Outdoor Humidity vs Energy Use"),
             ("lights", "Appliances", "Lights vs Energy Use")]
    for ax, (x, y, title) in zip(axes, pairs):
        ax.scatter(df[x], df[y], alpha=0.15, s=8, color="#9333ea")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def plot_hourly_pattern(df: pd.DataFrame, save_path: Path):
    hourly = df.assign(Hour=df["date"].dt.hour).groupby("Hour")["Appliances"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    hourly.plot(kind="line", marker="o", ax=ax, color="#ea580c")
    ax.set_title("Average Appliance Energy Use by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Mean Wh")
    ax.set_xticks(range(0, 24))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", save_path)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data(ROOT / "data" / "energydata_complete.csv")

    overview = dataset_overview(df)
    import json
    overview = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in overview.items()}
    with open(REPORTS_DIR / "dataset_overview.json", "w") as f:
        json.dump(overview, f, indent=2)

    missing_value_heatmap(df, PLOTS_DIR / "missing_value_map.png")
    statistical_summary(df, REPORTS_DIR / "statistical_summary.csv")
    target_corr = correlation_matrix(df, PLOTS_DIR / "correlation_matrix.png")
    target_corr.to_csv(REPORTS_DIR / "target_correlations.csv", header=["correlation_with_Appliances"])

    plot_target_distribution(df, PLOTS_DIR / "target_distribution.png")
    plot_feature_histograms(df, PLOTS_DIR / "feature_histograms.png")
    plot_boxplots_by_room(df, PLOTS_DIR / "room_temperature_boxplots.png")
    plot_scatter_relationships(df, PLOTS_DIR / "scatter_relationships.png")
    plot_hourly_pattern(df, PLOTS_DIR / "hourly_usage_pattern.png")

    logger.info("Top 5 features most correlated with Appliances:\n%s", target_corr.head(5))
    logger.info("EDA complete. All plots saved to %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
