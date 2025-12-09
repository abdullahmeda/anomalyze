"""
Tests for the anomaly detection dataset preparation pipeline.

Run with: pytest tests.py -v
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Use paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
ML_DATA_DIR = SCRIPT_DIR.parent / "ml" / "data"
ANOMALY_DATE = datetime(2023, 10, 5).date()
EXPECTED_MULTIPLIER = 3.0


@pytest.fixture
def train_df():
    """Load train dataset (daily aggregated)."""
    return pd.read_csv(ML_DATA_DIR / "train.csv", parse_dates=["timestamp"])


@pytest.fixture
def test_df():
    """Load test dataset (daily aggregated)."""
    return pd.read_csv(ML_DATA_DIR / "test.csv", parse_dates=["timestamp"])


@pytest.fixture
def full_df():
    """Load full dataset (raw tickets)."""
    return pd.read_csv(DATA_DIR / "full_dataset.csv", parse_dates=["timestamp"])


@pytest.fixture
def metadata():
    """Load metadata JSON."""
    with open(DATA_DIR / "anomaly_metadata.json") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE EXISTENCE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestFilesExist:
    def test_train_csv_exists(self):
        assert (ML_DATA_DIR / "train.csv").exists(), f"train.csv not found in {ML_DATA_DIR.absolute()}"

    def test_test_csv_exists(self):
        assert (ML_DATA_DIR / "test.csv").exists(), f"test.csv not found in {ML_DATA_DIR.absolute()}"

    def test_full_dataset_csv_exists(self):
        assert (DATA_DIR / "full_dataset.csv").exists()

    def test_metadata_json_exists(self):
        assert (DATA_DIR / "anomaly_metadata.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataQuality:
    def test_full_columns_exist(self, full_df):
        """Check all required columns are present in full dataset."""
        required = [
            "ticket_id", "subject", "body", "type", "queue", "priority",
            "language", "timestamp", "anomaly_label", "is_anomaly", "split"
        ]
        for col in required:
            assert col in full_df.columns, f"Missing column in full_df: {col}"

    def test_agg_columns_exist(self, train_df, test_df):
        """Check all required columns are present in aggregated datasets."""
        required = ["timestamp", "count"]
        for df in [train_df, test_df]:
            for col in required:
                assert col in df.columns, f"Missing column in agg df: {col}"

    def test_no_duplicate_ticket_ids(self, full_df):
        """All ticket IDs should be unique."""
        assert full_df["ticket_id"].duplicated().sum() == 0

    def test_timestamps_sorted(self, full_df, train_df, test_df):
        """Timestamps should be in ascending order."""
        assert full_df["timestamp"].is_monotonic_increasing
        assert train_df["timestamp"].is_monotonic_increasing
        assert test_df["timestamp"].is_monotonic_increasing


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainTestSplit:
    def test_train_has_no_anomalies(self, train_df, full_df):
        """Train set should be 100% normal data."""
        # Check that no day in train set is the anomaly date
        train_dates = set(train_df["timestamp"].dt.date)
        assert ANOMALY_DATE not in train_dates

    def test_test_has_anomalies(self, test_df):
        """Test set should contain anomaly day."""
        # Check that anomaly date is in test set
        test_dates = set(test_df["timestamp"].dt.date)
        assert ANOMALY_DATE in test_dates

    def test_train_date_range(self, train_df):
        """Train set should be Jan 1 - Sep 30, 2023."""
        assert train_df["timestamp"].min().date() >= datetime(2023, 1, 1).date()
        assert train_df["timestamp"].max().date() <= datetime(2023, 9, 30).date()

    def test_test_date_range(self, test_df):
        """Test set should be Oct 1 - Dec 31, 2023."""
        assert test_df["timestamp"].min().date() >= datetime(2023, 10, 1).date()
        assert test_df["timestamp"].max().date() <= datetime(2023, 12, 31).date()

    def test_no_overlap(self, train_df, test_df):
        """Train and test sets should not overlap in time."""
        assert train_df["timestamp"].max() < test_df["timestamp"].min()

    def test_combined_equals_full(self, train_df, test_df, full_df):
        """Train + test counts should equal full dataset count."""
        total_agg = train_df["count"].sum() + test_df["count"].sum()
        assert total_agg == len(full_df)


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAnomaly:
    def test_anomaly_on_correct_date(self, full_df):
        """Anomalies should only be on the designated date."""
        anomaly_rows = full_df[full_df["is_anomaly"]]
        anomaly_dates = anomaly_rows["timestamp"].dt.date.unique()
        assert len(anomaly_dates) == 1
        assert anomaly_dates[0] == ANOMALY_DATE

    def test_anomaly_volume_spike(self, train_df, test_df):
        """Anomaly day should have ~3x normal volume."""
        # Combine daily counts
        all_daily = pd.concat([train_df, test_df])
        daily_counts = all_daily.set_index("timestamp")["count"]
        
        anomaly_volume = daily_counts[daily_counts.index.date == ANOMALY_DATE].iloc[0]
        normal_volumes = daily_counts[daily_counts.index.date != ANOMALY_DATE]
        avg_normal = normal_volumes.mean()
        
        # Allow 20% tolerance
        expected_min = avg_normal * EXPECTED_MULTIPLIER * 0.8
        expected_max = avg_normal * EXPECTED_MULTIPLIER * 1.2
        
        assert expected_min <= anomaly_volume <= expected_max, \
            f"Anomaly volume {anomaly_volume} not in expected range [{expected_min:.0f}, {expected_max:.0f}]"

    def test_anomaly_label_correct(self, full_df):
        """Anomaly tickets should have correct label."""
        anomaly_rows = full_df[full_df["is_anomaly"]]
        assert (anomaly_rows["anomaly_label"] == "spike_volume").all()

    def test_normal_tickets_no_label(self, full_df):
        """Normal tickets should have null anomaly label."""
        normal_rows = full_df[~full_df["is_anomaly"]]
        assert normal_rows["anomaly_label"].isna().all()


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_metadata_structure(self, metadata):
        """Metadata should have required keys."""
        assert "dataset_info" in metadata
        assert "anomaly" in metadata
        assert "statistics" in metadata

    def test_metadata_ticket_counts(self, metadata, full_df):
        """Metadata counts should match actual data."""
        assert metadata["dataset_info"]["total_tickets"] == len(full_df)
        # Train/test ticket counts from metadata should match full_df split
        train_count = len(full_df[full_df["split"] == "train"])
        test_count = len(full_df[full_df["split"] == "test"])
        assert metadata["dataset_info"]["train_tickets"] == train_count
        assert metadata["dataset_info"]["test_tickets"] == test_count

    def test_metadata_anomaly_count(self, metadata, full_df):
        """Metadata anomaly count should match data."""
        actual_anomalies = full_df["is_anomaly"].sum()
        assert metadata["anomaly"]["affected_tickets"] == actual_anomalies

    def test_metadata_anomaly_date(self, metadata):
        """Metadata should have correct anomaly date."""
        assert metadata["anomaly"]["date"] == ANOMALY_DATE.strftime("%Y-%m-%d")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
