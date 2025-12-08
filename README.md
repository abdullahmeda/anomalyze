# Anomalyze

Hybrid anomaly detection that blends classical time-series methods with LLM-aware, context-aware detectors for seasonality, spikes, step-changes, and drift.

---

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dev dependencies
make install
```

## Quick Reference

```bash
make help        # Show all available commands
make install     # Install package with dev dependencies
make dataset     # Generate raw dataset from HuggingFace
make prepare     # Create train/test splits for ML training
make visualize   # Generate time-series plots
make test        # Run all tests (22 tests)
make clean       # Remove generated files

# Run anomaly detection models
python3 -m ml.run --model prophet              # Prophet model (default 99% CI)
python3 -m ml.run --model arima                # ARIMA model
python3 -m ml.run --model prophet --interval 0.95  # Custom confidence interval
```

---

## Dataset

### Overview

We use the [Customer Support Tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets) dataset from HuggingFace, augmented with synthetic timestamps and controlled anomalies for benchmarking detection algorithms.

| Property | Value |
|----------|-------|
| **Source** | `Tobi-Bueck/customer-support-tickets` |
| **Total Tickets** | 61,763 |
| **Date Range** | Jan 1 – Dec 31, 2023 |
| **Languages** | German (54%), English (46%) |

### Train/Test Split

| Set | Date Range | Tickets | Anomalies |
|-----|------------|---------|-----------|
| **Train** | Jan 1 – Sep 30 | 45,836 | 0 (100% normal) |
| **Test** | Oct 1 – Dec 31 | 15,927 | 589 |

The train set contains only normal data, allowing classical methods to learn baseline patterns. The test set contains both normal data and one anomaly event for evaluation.

### Anomaly: Volume Spike with Characteristic Skew

A single-day anomaly on **October 5, 2023** simulates a major outage or bug:

| Metric | Anomaly Day | Normal Days | Ratio |
|--------|-------------|-------------|-------|
| **Volume** | 589 tickets | ~169/day | **3.0×** |
| **High/Critical Priority** | 84.2% | 38.2% | **2.2×** |
| **Incident Type** | 81.0% | 31.0% | **2.6×** |

The anomaly is detectable via:
- **Classical methods**: Volume spike (3× daily ticket count)
- **LLM/contextual methods**: Characteristic shift (more high-priority incidents, specific tags like Bug/Outage)

### Columns

| Column | Type | Description |
|--------|------|-------------|
| `ticket_id` | int | Unique identifier |
| `timestamp` | datetime | Synthetic timestamp |
| `subject` | str | Ticket title (nullable) |
| `body` | str | Customer message |
| `type` | str | Incident / Request / Problem / Change |
| `queue` | str | Department (52 categories) |
| `priority` | str | critical / high / medium / low / very_low |
| `language` | str | en / de |
| `tag_1` to `tag_4` | str | Classification tags |
| `anomaly_label` | str | "spike_volume" or null |
| `is_anomaly` | bool | True if anomaly ticket |
| `split` | str | "train" or "test" |

### Output Files

```
dataset/data/
├── full_dataset.csv       # Raw ticket dataset (Jan-Dec)
└── anomaly_metadata.json  # Ground truth and statistics

ml/data/
├── train.csv              # Daily time-series (Jan-Sep, normal only)
└── test.csv               # Daily time-series (Oct-Dec, contains anomaly)

ml/plots/
├── prophet_anomaly_detection.png  # Prophet results visualization
└── arima_anomaly_detection.png    # ARIMA results visualization
```

### Generating the Dataset

```bash
# Step 1: Generate raw dataset from HuggingFace
make dataset

# Step 2: Create train/test splits for ML training
make prepare
```

**Step 1** (`make dataset`):
1. Loads data from HuggingFace
2. Generates realistic timestamps with weekly/hourly patterns
3. Injects the anomaly with characteristic skew
4. Exports full_dataset.csv and metadata

**Step 2** (`make prepare`):
1. Reads full_dataset.csv
2. Aggregates tickets to daily time-series
3. Splits into train (Jan-Sep) and test (Oct-Dec)
4. Exports to ml/data/ directory

### Running Tests

```bash
make test
```

22 tests validate:
- File existence and data quality
- Train/test split correctness
- Anomaly volume and characteristics
- Metadata consistency

---

## Models

### Available Models

| Model | Type | Description | Performance |
|-------|------|-------------|-------------|
| **Prophet** | Additive | Facebook's forecasting model with weekly seasonality | 100% F1 |
| **ARIMA** | Classical | SARIMAX(1,0,1)×(1,1,1,7) with weekly seasonality | 100% F1 |

Both models achieve **perfect detection** (100% Precision, 100% Recall, 100% F1) on our synthetic anomaly.

### Running Models

#### Basic Usage

```bash
# Prophet model
python3 -m ml.run --model prophet

# ARIMA model
python3 -m ml.run --model arima
```

#### Custom Confidence Intervals

```bash
# Wider interval (fewer false positives, may miss subtle anomalies)
python3 -m ml.run --model prophet --interval 0.99

# Narrower interval (more sensitive, may increase false positives)
python3 -m ml.run --model prophet --interval 0.95
```

#### Output Example

```
Prophet Anomaly Detection (99% CI)
  Data: 273 train days, 92 test days
  Detected: 1 anomaly(s) | Ground truth: 2023-10-05
    → 2023-10-05: 589 tickets (expected 194, bound 232)
  Metrics: P=100% R=100% F1=100%
  Plot: /path/to/ml/plots/prophet_anomaly_detection.png
```

### Model Architecture

```
ml/
├── models/
│   ├── __init__.py        # Model registry
│   ├── prophet.py         # Prophet implementation
│   └── arima.py           # ARIMA implementation
├── run.py                 # Unified runner with argparse
├── utils.py               # Shared evaluation and plotting
├── features.py            # Data preparation
└── data/                  # Train/test CSVs
```

### Adding New Models

To add a new model:

1. Create `ml/models/your_model.py`:
```python
class AnomalyYourModel:
    name = "YourModel"
    
    def __init__(self, interval_width: float = 0.99):
        self.interval_width = interval_width
    
    def fit(self, train_df: pd.DataFrame) -> "AnomalyYourModel":
        # Train your model
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return forecast with ds, yhat, yhat_lower, yhat_upper, actual
        pass
    
    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        # Return anomalies with timestamp, actual, yhat, yhat_lower, yhat_upper
        pass
```

2. Register in `ml/run.py`:
```python
from ml.models import AnomalyYourModel

MODELS = {
    "prophet": AnomalyProphet,
    "arima": AnomalyARIMA,
    "yourmodel": AnomalyYourModel,  # Add here
}
```

3. Run:
```bash
python3 -m ml.run --model yourmodel
```

---

## Results

### Prophet

- **Model**: Facebook Prophet with weekly seasonality
- **Configuration**: 99% confidence interval
- **Metrics**: Precision 100%, Recall 100%, F1 100%
- **Detection**: Successfully identified October 5, 2023 spike (589 tickets vs expected 194)

### ARIMA

- **Model**: SARIMAX(1, 0, 1) × (1, 1, 1, 7)
- **Configuration**: 99% confidence interval, weekly seasonality (s=7)
- **Metrics**: Precision 100%, Recall 100%, F1 100%
- **Detection**: Successfully identified October 5, 2023 spike (589 tickets vs expected 197)

Both models correctly learned the weekly seasonality patterns from the training data and detected the 3× volume spike with zero false positives.

---

## Development

### Project Structure

```
anomalyze/
├── dataset/               # Data generation and exploration
│   ├── preprocess.py      # HuggingFace loader + anomaly injection
│   ├── explore.py         # Dataset exploration script
│   ├── visualize.py       # Time-series plotting
│   ├── tests.py           # 22 validation tests
│   └── data/              # Generated datasets
│
├── ml/                    # Machine learning models
│   ├── models/
│   │   ├── prophet.py     # Prophet model class
│   │   └── arima.py       # ARIMA model class
│   ├── run.py             # Unified runner with argparse
│   ├── utils.py           # Shared evaluation and plotting
│   ├── features.py        # Data preparation
│   └── data/              # ML-ready time-series
│
├── Makefile               # Convenient workflow commands
├── pyproject.toml         # Package configuration
└── README.md              # This file
```

### Code Style

- **Logging**: Uses Python's `logging` module instead of `print()` statements
- **DRY**: Shared utilities in `ml/utils.py` for evaluation and plotting
- **Modular**: Each model is a self-contained class with consistent interface
- **Typed**: Function signatures include type hints

### Dependencies

- `datasets` - HuggingFace datasets library
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `prophet` - Facebook Prophet forecasting
- `statsmodels` - ARIMA/SARIMAX models
- `pytest` - Testing framework

---

## License

Apache 2.0
