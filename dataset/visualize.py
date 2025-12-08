import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Config
DATA_PATH = Path("data/full_dataset.csv")
OUTPUT_DIR = Path("plots")
ANOMALY_DATE = "2023-10-05"

def load_data():
    print(f"Loading {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_daily_volume(df):
    print("Plotting daily volume...")
    # Resample to daily volume
    daily_counts = df.set_index('timestamp').resample('D').size()
    
    plt.figure(figsize=(15, 6))
    
    # Plot normal days
    plt.plot(daily_counts.index, daily_counts.values, label='Daily Volume', color='#1f77b4', linewidth=1.5)
    
    # Highlight anomaly
    anomaly_val = daily_counts.loc[ANOMALY_DATE]
    plt.scatter([pd.Timestamp(ANOMALY_DATE)], [anomaly_val], color='red', s=100, zorder=5, label='Anomaly (Oct 5)')
    
    # Styling
    plt.title('Daily Ticket Volume (2023)', fontsize=14)
    plt.ylabel('Number of Tickets')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    output_path = OUTPUT_DIR / "daily_volume.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()

def plot_priority_share(df):
    print("Plotting priority distribution over time...")
    # Calculate daily percentage of high/critical tickets
    df['is_high_priority'] = df['priority'].isin(['high', 'critical'])
    daily_priority = df.set_index('timestamp').resample('D')['is_high_priority'].mean() * 100
    
    plt.figure(figsize=(15, 6))
    
    plt.plot(daily_priority.index, daily_priority.values, color='#ff7f0e', linewidth=1.5, label='% High/Critical Priority')
    
    # Highlight anomaly
    anomaly_val = daily_priority.loc[ANOMALY_DATE]
    plt.scatter([pd.Timestamp(ANOMALY_DATE)], [anomaly_val], color='red', s=100, zorder=5, label='Anomaly (Oct 5)')
    
    # Styling
    plt.title('Daily Share of High/Critical Priority Tickets', fontsize=14)
    plt.ylabel('Percentage (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    output_path = OUTPUT_DIR / "priority_share.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved {output_path}")
    plt.close()

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    df = load_data()
    plot_daily_volume(df)
    plot_priority_share(df)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()

