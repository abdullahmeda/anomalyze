"""LightGBM-based Anomaly Detection Model."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

from ml.features import add_time_series_features


class AnomalyLGBM:
    """LightGBM wrapper for time-series anomaly detection."""

    name = "LightGBM"

    def __init__(self, interval_width: float = 0.99):
        """
        Initialize the anomaly detector.

        Args:
            interval_width: Width of the uncertainty interval (0-1).
        """
        self.interval_width = interval_width
        self.model = None
        self.std_resid = None
        self.last_train_window = None
        self.feature_cols = [
            "day_of_week",
            "is_weekend",
            "month",
            "day_of_month",
            "day_of_year",
            "lag_1",
            "lag_7",
            "lag_14",
            "rolling_mean_7d",
            "rolling_std_7d",
        ]

    def fit(self, train_df: pd.DataFrame) -> "AnomalyLGBM":
        """
        Fit LightGBM model on training data.

        Args:
            train_df: DataFrame with 'timestamp' and 'count' columns

        Returns:
            self for method chaining
        """
        # Store last 14 days for lag generation during prediction
        self.last_train_window = train_df.sort_values("timestamp").iloc[-14:].copy()

        # Prepare features
        df = add_time_series_features(train_df)
        X = df[self.feature_cols]
        y = df["count"]

        # Train model
        self.model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbose=-1,
        )
        self.model.fit(X, y)

        # Estimate uncertainty using time-series cross-validation
        # to get realistic out-of-sample residuals
        val_residuals = []
        n_splits = 5
        min_train_size = len(X) // 2
        
        for i in range(n_splits):
            # Progressive train/validation splits
            val_size = (len(X) - min_train_size) // n_splits
            train_end = min_train_size + i * val_size
            val_start = train_end
            val_end = train_end + val_size
            
            if val_end > len(X):
                break
                
            X_train_cv = X.iloc[:train_end]
            y_train_cv = y.iloc[:train_end]
            X_val_cv = X.iloc[val_start:val_end]
            y_val_cv = y.iloc[val_start:val_end]
            
            # Train on subset
            model_cv = lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            model_cv.fit(X_train_cv, y_train_cv)
            
            # Get validation residuals
            preds_cv = model_cv.predict(X_val_cv)
            val_residuals.extend(y_val_cv - preds_cv)
        
        # Use validation residuals for uncertainty estimation
        # Fallback to inflated training residuals if CV fails
        if len(val_residuals) > 0:
            self.std_resid = np.std(val_residuals)
        else:
            preds = self.model.predict(X)
            residuals = y - preds
            # Inflate by 3x to account for overfitting
            self.std_resid = np.std(residuals) * 3

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with uncertainty intervals.

        Args:
            df: DataFrame with 'timestamp' column

        Returns:
            Forecast DataFrame with ds, yhat, yhat_lower, yhat_upper, actual
        """
        # Prepend history if available to handle lags for the start of the test set
        if self.last_train_window is not None:
            # Concatenate and ensure time order
            combined_df = pd.concat([self.last_train_window, df], axis=0)
            combined_df = combined_df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        else:
            combined_df = df

        df_feats = add_time_series_features(combined_df)
        
        # Filter back to only the rows requested in 'df'
        # We assume 'timestamp' is unique
        df_feats = df_feats[df_feats["timestamp"].isin(df["timestamp"])].copy()

        if df_feats.empty:
            # Fallback if dataset is too small for lags even with history
            raise ValueError("Dataset too small or history missing to generate features")

        X = df_feats[self.feature_cols]
        preds = self.model.predict(X)

        # Calculate confidence interval bounds based on residual std
        z_score = stats.norm.ppf((1 + self.interval_width) / 2)
        margin = z_score * self.std_resid

        forecast = pd.DataFrame({
            "ds": df_feats["timestamp"].values,
            "yhat": preds,
            "yhat_lower": preds - margin,
            "yhat_upper": preds + margin,
            "actual": df_feats["count"].values,
        })

        return forecast

    def detect_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies where actual values fall outside prediction interval.

        Args:
            df: DataFrame with 'timestamp' and 'count' columns

        Returns:
            DataFrame of detected anomalies
        """
        forecast = self.predict(df)

        is_anomaly = (forecast["actual"] < forecast["yhat_lower"]) | (
            forecast["actual"] > forecast["yhat_upper"]
        )

        anomalies = forecast[is_anomaly][
            ["ds", "actual", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        anomalies = anomalies.rename(columns={"ds": "timestamp"})

        return anomalies
