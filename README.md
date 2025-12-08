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
make test        # Run all tests
make clean       # Remove generated files
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

> **Note:** Temporal features (hour, day_of_week, is_weekend, etc.) are computed at training time via `ml/features.py`, not stored in the CSV files.

### Output Files

```
dataset/data/
├── full_dataset.csv       # Raw ticket dataset (Jan-Dec)
└── anomaly_metadata.json  # Ground truth and statistics

ml/data/
├── train.csv              # Daily time-series (Jan-Sep, normal only)
└── test.csv               # Daily time-series (Oct-Dec, contains anomaly)
```

### Data Schema

**`full_dataset.csv`** (Raw Tickets):
See "Columns" table above.

**`train.csv` / `test.csv`** (Aggregated Time-Series):

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Daily timestamp |
| `count` | int | Total tickets per day |

> **Note:** Additional features (day_of_week, is_weekend, etc.) can be computed at training time via `ml/features.py`

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

25 tests validate:
- File existence and data quality
- Train/test split correctness
- Anomaly volume and characteristics
- Metadata consistency

---

## Training

*Coming soon*

---

## Evaluation

*Coming soon*

---

## License

Apache 2.0

