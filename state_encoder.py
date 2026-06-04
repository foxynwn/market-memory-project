"""
state_encoder.py

Encodes continuous market features into discrete market regimes and states.
"""

import os
from typing import Dict

import numpy as np
import pandas as pd


MEMORY_FOLDER = "memory"
STATE_MEMORY_FILE = os.path.join(MEMORY_FOLDER, "state_memory.csv")


class StateEncoder:
    """Encodes market data into discrete state space."""

    def __init__(self, memory_folder=MEMORY_FOLDER):
        self.memory_folder = memory_folder
        os.makedirs(self.memory_folder, exist_ok=True)

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self, df):
        """Validate required columns and format."""
        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # =====================================================
    # FEATURES
    # =====================================================

    def build_features(self, df):
        """Compute feature vectors from OHLCV data."""
        df = self.validate(df)

        # Returns
        df["ret_1"] = df["close"].pct_change() * 100
        df["ret_3"] = df["close"].pct_change(3) * 100
        df["ret_6"] = df["close"].pct_change(6) * 100
        df["ret_12"] = df["close"].pct_change(12) * 100

        # Trend
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma50"] = df["close"].rolling(50).mean()
        df["trend_strength"] = ((df["ma20"] - df["ma50"]) / df["ma50"]) * 100
        df["distance_ma20"] = ((df["close"] - df["ma20"]) / df["ma20"]) * 100

        # Volatility
        df["volatility"] = ((df["high"] - df["low"]) / df["close"]) * 100
        df["volatility_ma"] = df["volatility"].rolling(20).mean()

        # Volume
        volume_ma = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / volume_ma

        # Candle structure
        candle_range = (df["high"] - df["low"]).replace(0, np.nan)
        body = (df["close"] - df["open"]).abs()
        df["body_ratio"] = body / candle_range

        # Momentum
        df["momentum"] = df["close"].diff(3)

        return df

    # =====================================================
    # REGIMES
    # =====================================================

    def build_regimes(self, df):
        """Classify features into market regimes."""
        df = df.copy()

        # Trend regime
        df["trend_regime"] = np.select(
            [
                df["trend_strength"] > 2,
                df["trend_strength"] > 0.5,
                df["trend_strength"] < -2,
                df["trend_strength"] < -0.5,
            ],
            ["BULL_STRONG", "BULL_WEAK", "BEAR_STRONG", "BEAR_WEAK"],
            default="RANGE",
        )

        # Volatility regime
        q20 = df["volatility"].quantile(0.20)
        q40 = df["volatility"].quantile(0.40)
        q60 = df["volatility"].quantile(0.60)
        q80 = df["volatility"].quantile(0.80)

        df["volatility_regime"] = np.select(
            [
                df["volatility"] <= q20,
                df["volatility"] <= q40,
                df["volatility"] <= q60,
                df["volatility"] <= q80,
            ],
            ["V1", "V2", "V3", "V4"],
            default="V5",
        )

        # Volume regime
        df["volume_regime"] = np.select(
            [
                df["volume_ratio"] < 0.8,
                df["volume_ratio"] < 1.5,
            ],
            ["LOW", "NORMAL"],
            default="HIGH",
        )

        # Momentum regime
        df["momentum_regime"] = np.select(
            [
                df["momentum"] > 0,
                df["momentum"] < 0,
            ],
            ["UP", "DOWN"],
            default="FLAT",
        )

        return df

    # =====================================================
    # STATES
    # =====================================================

    def build_states(self, df):
        """Combine regimes into discrete states."""
        df = df.copy()
        df["state"] = (
            df["trend_regime"]
            + "_"
            + df["volatility_regime"]
            + "_"
            + df["volume_regime"]
            + "_"
        )
        return df

    # =====================================================
    # SAVE
    # =====================================================

    def build_state_memory(self, df):
        """Build and save state memory."""
        df = self.build_features(df)
        df = self.build_regimes(df)
        df = self.build_states(df)

        columns = [
            "timestamp",
            "state",
            "trend_regime",
            "volatility_regime",
            "volume_regime",
            "momentum_regime",
            "trend_strength",
            "volatility",
            "volume_ratio",
            "momentum",
            "ret_1",
            "ret_3",
            "ret_6",
            "ret_12",
            "body_ratio",
        ]

        state_memory = df[columns].copy()
        state_memory.to_csv(STATE_MEMORY_FILE, index=False)

        return state_memory

    # =====================================================
    # SUMMARY
    # =====================================================

    def summarize_states(self, df) -> Dict:
        """Summarize state distribution."""
        return {
            "total_rows": int(len(df)),
            "unique_states": int(df["state"].nunique()),
            "top_states": df["state"].value_counts().head(10).to_dict(),
            "average_trend_strength": float(df["trend_strength"].mean()),
            "average_volatility": float(df["volatility"].mean()),
            "average_volume_ratio": float(df["volume_ratio"].mean()),
        }
