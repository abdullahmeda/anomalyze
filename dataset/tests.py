"""
Tests for the anomaly detection dataset preparation pipeline.

Run with: pytest tests.py -v
"""

import pytest
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("dataset/data")
ANOMALY_DATE = datetime(2023, 10, 5).date()
EXPECTED_MULTIPLIER = 3.0


@pytest.fixture
def train_df():
    """Load train dataset."""
    return pd.read_csv(DATA_DIR / "train.csv", parse_dates=["timestamp"])


@pytest.fixture
def test_df():
    """Load test dataset."""
    return pd.read_csv(DATA_DIR / "test.csv", parse_dates=["timestamp"])


@pytest.fixture
def full_df():
    """Load full dataset."""
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
        assert (DATA_DIR / "train.csv").exists()

    def test_test_csv_exists(self):
        assert (DATA_DIR / "test.csv").exists()

    def test_full_dataset_csv_exists(self):
        assert (DATA_DIR / "full_dataset.csv").exists()

    def test_metadata_json_exists(self):
        assert (DATA_DIR / "anomaly_metadata.json").exists()


# ═══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDataQuality:
    def test_no_null_body(self, full_df):
        """Body field should never be null."""
        assert full_df["body"].isna().sum() == 0

    def test_no_duplicate_ticket_ids(self, full_df):
        """All ticket IDs should be unique."""
        assert full_df["ticket_id"].duplicated().sum() == 0

    def test_timestamps_sorted(self, full_df):
        """Timestamps should be in ascending order."""
        assert full_df["timestamp"].is_monotonic_increasing

    def test_no_duplicate_timestamps(self, full_df):
        """All timestamps should be unique."""
        assert full_df["timestamp"].duplicated().sum() == 0

    def test_required_columns_exist(self, full_df):
        """Check all required columns are present."""
        required = [
            "ticket_id", "subject", "body", "type", "queue", "priority",
            "language", "timestamp", "anomaly_label", "is_anomaly",
            "hour", "day_of_week", "split"
        ]
        for col in required:
            assert col in full_df.columns, f"Missing column: {col}"


# ═══════════════════════════════════════════════════════════════════════════════
# TRAIN/TEST SPLIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTrainTestSplit:
    def test_train_has_no_anomalies(self, train_df):
        """Train set should be 100% normal data."""
        assert train_df["is_anomaly"].sum() == 0

    def test_test_has_anomalies(self, test_df):
        """Test set should contain anomalies."""
        assert test_df["is_anomaly"].sum() > 0

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
        """Train + test should equal full dataset."""
        assert len(train_df) + len(test_df) == len(full_df)


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

    def test_anomaly_volume_spike(self, full_df):
        """Anomaly day should have ~3x normal volume."""
        daily_counts = full_df.groupby(full_df["timestamp"].dt.date).size()
        
        anomaly_volume = daily_counts[ANOMALY_DATE]
        normal_volumes = daily_counts[daily_counts.index != ANOMALY_DATE]
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

    def test_anomaly_has_higher_priority(self, full_df):
        """Anomaly day should have significantly more high/critical priority."""
        anomaly_rows = full_df[full_df["is_anomaly"]]
        normal_rows = full_df[~full_df["is_anomaly"]]
        
        anomaly_high_pct = anomaly_rows["priority"].isin(["high", "critical"]).mean()
        normal_high_pct = normal_rows["priority"].isin(["high", "critical"]).mean()
        
        # Anomaly day should have at least 1.5x the high priority rate
        assert anomaly_high_pct > normal_high_pct * 1.5, \
            f"Anomaly high priority {anomaly_high_pct:.1%} should be > 1.5× normal {normal_high_pct:.1%}"

    def test_anomaly_has_more_incidents(self, full_df):
        """Anomaly day should have significantly more Incident type tickets."""
        anomaly_rows = full_df[full_df["is_anomaly"]]
        normal_rows = full_df[~full_df["is_anomaly"]]
        
        anomaly_incident_pct = (anomaly_rows["type"] == "Incident").mean()
        normal_incident_pct = (normal_rows["type"] == "Incident").mean()
        
        # Anomaly day should have at least 1.5x the Incident rate
        assert anomaly_incident_pct > normal_incident_pct * 1.5, \
            f"Anomaly Incident rate {anomaly_incident_pct:.1%} should be > 1.5× normal {normal_incident_pct:.1%}"


# ═══════════════════════════════════════════════════════════════════════════════
# METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetadata:
    def test_metadata_structure(self, metadata):
        """Metadata should have required keys."""
        assert "dataset_info" in metadata
        assert "anomaly" in metadata
        assert "statistics" in metadata

    def test_metadata_ticket_counts(self, metadata, full_df, train_df, test_df):
        """Metadata counts should match actual data."""
        assert metadata["dataset_info"]["total_tickets"] == len(full_df)
        assert metadata["dataset_info"]["train_tickets"] == len(train_df)
        assert metadata["dataset_info"]["test_tickets"] == len(test_df)

    def test_metadata_anomaly_count(self, metadata, full_df):
        """Metadata anomaly count should match data."""
        actual_anomalies = full_df["is_anomaly"].sum()
        assert metadata["anomaly"]["affected_tickets"] == actual_anomalies

    def test_metadata_anomaly_date(self, metadata):
        """Metadata should have correct anomaly date."""
        assert metadata["anomaly"]["date"] == ANOMALY_DATE.strftime("%Y-%m-%d")


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPORAL FEATURES TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestTemporalFeatures:
    def test_hour_range(self, full_df):
        """Hour should be 0-23."""
        assert full_df["hour"].min() >= 0
        assert full_df["hour"].max() <= 23

    def test_day_of_week_range(self, full_df):
        """Day of week should be 0-6."""
        assert full_df["day_of_week"].min() >= 0
        assert full_df["day_of_week"].max() <= 6

    def test_weekend_flag(self, full_df):
        """Weekend flag should match day_of_week."""
        weekend_rows = full_df[full_df["is_weekend"]]
        assert weekend_rows["day_of_week"].isin([5, 6]).all()

    def test_business_hours_flag(self, full_df):
        """Business hours should be 9-17 on weekdays."""
        biz_rows = full_df[full_df["is_business_hours"]]
        assert (biz_rows["hour"] >= 9).all()
        assert (biz_rows["hour"] < 17).all()
        assert (~biz_rows["is_weekend"]).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

