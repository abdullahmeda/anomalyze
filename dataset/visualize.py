"""Generate time-series visualization plots."""

import logging
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "full_dataset.csv"
OUTPUT_DIR = SCRIPT_DIR / "plots"
ANOMALY_DATE = "2023-10-05"


def plot_daily_volume(df: pd.DataFrame, output_dir: Path):
    """Plot daily ticket volume with anomaly highlighted."""
    daily_counts = df.set_index('timestamp').resample('D').size()

    plt.figure(figsize=(15, 6))
    plt.plot(daily_counts.index, daily_counts.values, label='Daily Volume', color='#1f77b4', linewidth=1.5)
    plt.scatter([pd.Timestamp(ANOMALY_DATE)], [daily_counts.loc[ANOMALY_DATE]], color='red', s=100, zorder=5, label='Anomaly (Oct 5)')
    plt.title('Daily Ticket Volume (2023)', fontsize=14)
    plt.ylabel('Number of Tickets')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.savefig(output_dir / "daily_volume.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_priority_share(df: pd.DataFrame, output_dir: Path):
    """Plot daily percentage of high/critical priority tickets."""
    df = df.copy()
    df['is_high_priority'] = df['priority'].isin(['high', 'critical'])
    daily_priority = df.set_index('timestamp').resample('D')['is_high_priority'].mean() * 100

    plt.figure(figsize=(15, 6))
    plt.plot(daily_priority.index, daily_priority.values, color='#ff7f0e', linewidth=1.5, label='% High/Critical Priority')
    plt.scatter([pd.Timestamp(ANOMALY_DATE)], [daily_priority.loc[ANOMALY_DATE]], color='red', s=100, zorder=5, label='Anomaly (Oct 5)')
    plt.title('Daily Share of High/Critical Priority Tickets', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.savefig(output_dir / "priority_share.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])

    plot_daily_volume(df, OUTPUT_DIR)
    plot_priority_share(df, OUTPUT_DIR)

    logger.info(f"\nVisualization complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
