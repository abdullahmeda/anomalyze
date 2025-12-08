"""Prophet-based Anomaly Detection Model."""

import pandas as pd
from prophet import Prophet


class AnomalyProphet:
    """Prophet wrapper for time-series anomaly detection."""

    name = "Prophet"

    def __init__(self, interval_width: float = 0.99):
        """
        Initialize the anomaly detector.

        Args:
            interval_width: Width of the uncertainty interval (0-1).
                           Higher values = fewer false positives.
        """
        self.interval_width = interval_width
        self.model = None

    def fit(self, train_df: pd.DataFrame) -> "AnomalyProphet":
        """
        Fit Prophet model on training data.

        Args:
            train_df: DataFrame with 'timestamp' and 'count' columns

        Returns:
            self for method chaining
        """
        # Prophet requires columns named 'ds' and 'y'
        df = train_df.rename(columns={"timestamp": "ds", "count": "y"}).copy()

        # Initialize and fit Prophet
        self.model = Prophet(
            interval_width=self.interval_width,
            yearly_seasonality=False,  # Only 1 year of data
            weekly_seasonality=True,   # Strong weekly pattern in our data
            daily_seasonality=False,   # We have daily aggregates
        )
        self.model.fit(df)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with uncertainty intervals.

        Args:
            df: DataFrame with 'timestamp' column

        Returns:
            Forecast DataFrame with ds, yhat, yhat_lower, yhat_upper, actual
        """
        future = df[["timestamp"]].rename(columns={"timestamp": "ds"}).copy()
        forecast = self.model.predict(future)

        # Add actual values for convenience
        forecast["actual"] = df["count"].values

        return forecast

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies where actual values fall outside prediction interval.

        Args:
            df: DataFrame with 'timestamp' and 'count' columns

        Returns:
            DataFrame of detected anomalies with columns:
            - timestamp, actual, yhat, yhat_lower, yhat_upper
        """
        forecast = self.predict(df)

        # Anomaly = actual value outside the confidence interval
        is_anomaly = (forecast["actual"] < forecast["yhat_lower"]) | (
            forecast["actual"] > forecast["yhat_upper"]
        )

        anomalies = forecast[is_anomaly][
            ["ds", "actual", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        anomalies = anomalies.rename(columns={"ds": "timestamp"})

        return anomalies

