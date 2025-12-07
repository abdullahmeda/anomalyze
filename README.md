# Anomalyze

Hybrid anomaly detection that blends classical time-series methods with LLM-aware, context-aware detectors for seasonality, spikes, step-changes, and drift.

---

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package with dev dependencies
pip install -e ".[dev]"
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
| `hour` | int | Hour of day (0-23) |
| `day_of_week` | int | Day of week (0=Mon, 6=Sun) |
| `day_of_month` | int | Day of month (1-31) |
| `week_of_year` | int | Week number (1-52) |
| `is_weekend` | bool | Saturday or Sunday |
| `is_business_hours` | bool | Mon-Fri, 9am-5pm |
| `text_length` | int | Character count of body |
| `split` | str | "train" or "test" |

### Output Files

```
data/
├── train.csv              # Training set (Jan-Sep, normal only)
├── test.csv               # Test set (Oct-Dec, contains anomaly)
├── full_dataset.csv       # Combined dataset
└── anomaly_metadata.json  # Ground truth and statistics
```

### Generating the Dataset

```bash
# Activate virtual environment
source .venv/bin/activate

# Run preparation pipeline
python prepare_dataset.py
```

The script:
1. Loads data from HuggingFace
2. Generates realistic timestamps with weekly/hourly patterns
3. Injects the anomaly with characteristic skew
4. Adds temporal features
5. Exports train/test splits

### Running Tests

```bash
pytest tests.py -v
```

29 tests validate:
- File existence and data quality
- Train/test split correctness
- Anomaly volume and characteristics
- Metadata consistency
- Temporal feature correctness

---

## Training

*Coming soon*

---

## Evaluation

*Coming soon*

---

## License

Apache 2.0

