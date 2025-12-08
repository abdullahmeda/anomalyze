"""ARIMA-based Anomaly Detection Model."""

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

# Suppress convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class AnomalyARIMA:
    """SARIMAX wrapper for time-series anomaly detection."""

    name = "ARIMA"

    def __init__(
        self,
        order: tuple = (1, 0, 1),
        seasonal_order: tuple = (1, 1, 1, 7),
        interval_width: float = 0.99,
    ):
        """
        Initialize the anomaly detector.

        Args:
            order: ARIMA (p, d, q) order
            seasonal_order: Seasonal (P, D, Q, s) order
            interval_width: Confidence level for prediction intervals (0-1)
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.interval_width = interval_width
        self.model = None
        self.fitted = None

    def fit(self, train_df: pd.DataFrame) -> "AnomalyARIMA":
        """
        Fit SARIMAX model on training data.

        Args:
            train_df: DataFrame with 'timestamp' and 'count' columns

        Returns:
            self for method chaining
        """
        # Prepare data with datetime index
        df = train_df.copy()
        df = df.set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index, freq="D")

        # Fit SARIMAX model
        self.model = SARIMAX(
            df["count"],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.fitted = self.model.fit(disp=False)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with uncertainty intervals.

        Args:
            df: DataFrame with 'timestamp' column

        Returns:
            Forecast DataFrame with ds, yhat, yhat_lower, yhat_upper, actual
        """
        # Generate forecast with confidence intervals
        forecast_result = self.fitted.get_forecast(steps=len(df))
        forecast_mean = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=1 - self.interval_width)

        # Build forecast DataFrame matching Prophet's format
        forecast = pd.DataFrame({
            "ds": df["timestamp"].values,
            "yhat": forecast_mean.values,
            "yhat_lower": conf_int.iloc[:, 0].values,
            "yhat_upper": conf_int.iloc[:, 1].values,
            "actual": df["count"].values,
        })

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

