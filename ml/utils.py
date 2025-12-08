"""
Shared utilities for anomaly detection models.

This module contains model-agnostic evaluation and visualization functions
that can be reused across different anomaly detection approaches.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime


def evaluate(detected_dates: list, ground_truth_date: datetime.date) -> dict:
    """
    Evaluate detection performance against known ground truth.

    Args:
        detected_dates: List of dates flagged as anomalies
        ground_truth_date: The actual anomaly date

    Returns:
        Dictionary with precision, recall, F1 score
    """
    detected_set = set(detected_dates)
    ground_truth_set = {ground_truth_date}

    true_positives = len(detected_set & ground_truth_set)
    false_positives = len(detected_set - ground_truth_set)
    false_negatives = len(ground_truth_set - detected_set)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_results(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecast: pd.DataFrame,
    anomalies: pd.DataFrame,
    output_path: Path,
    model_name: str = "Model",
    interval_width: float = 0.99,
):
    """
    Plot the forecast with detected anomalies.

    Args:
        train_df: Training data with 'timestamp' and 'count' columns
        test_df: Test data with 'timestamp' and 'count' columns
        forecast: Forecast DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
        anomalies: Detected anomalies with 'timestamp' and 'actual' columns
        output_path: Path to save the plot
        model_name: Name of the model for the title
        interval_width: Confidence interval width (for legend label)
    """
    fig, ax = plt.subplots(figsize=(15, 6))

    # Plot training data
    ax.plot(
        train_df["timestamp"],
        train_df["count"],
        color="#333333",
        linewidth=1,
        label="Training Data",
        alpha=0.7,
    )

    # Plot test actual values
    ax.plot(
        test_df["timestamp"],
        test_df["count"],
        color="#1f77b4",
        linewidth=1.5,
        label="Test Data (Actual)",
    )

    # Plot forecast
    ax.plot(
        forecast["ds"],
        forecast["yhat"],
        color="#2ca02c",
        linewidth=1.5,
        linestyle="--",
        label=f"{model_name} Forecast",
    )

    # Plot confidence interval
    ax.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        color="#2ca02c",
        alpha=0.2,
        label=f"{interval_width*100:.0f}% Confidence Interval",
    )

    # Highlight detected anomalies
    if not anomalies.empty:
        ax.scatter(
            anomalies["timestamp"],
            anomalies["actual"],
            color="red",
            s=100,
            zorder=5,
            label=f"Detected Anomalies ({len(anomalies)})",
            edgecolors="darkred",
            linewidths=1.5,
        )

    # Mark train/test split
    split_date = train_df["timestamp"].max()
    ax.axvline(x=split_date, color="gray", linestyle=":", alpha=0.7, label="Train/Test Split")

    # Styling
    ax.set_title(f"{model_name} Anomaly Detection: Daily Ticket Volume", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Tickets")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

