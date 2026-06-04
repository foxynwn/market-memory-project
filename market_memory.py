"""
market_memory.py

Core market data fetching, storage, and enrichment module.
Manages episodic, feature store, and working memory layers.
"""

import os
import json
import time
import hashlib

import ccxt
import numpy as np
import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================

CONFIG = {
    "symbol": "BTC/USDT",
    "historical_timeframe": "30m",
    "recent_timeframe": "1m",
    "historical_days": 30,
    "recent_minutes": 720,
    "memory_folder": "memory",
    "max_retries": 3,
    "retry_wait_seconds": 1.5,
    "incremental_overlap_candles": 3,
    "enable_incremental_history": True,
}


# ==========================================================
# MARKET MEMORY
# ==========================================================

class MarketMemory:
    """Fetches, enriches, and stores market data across multiple memory layers."""

    def __init__(self, exchange=None):
        self.exchange = exchange or ccxt.binance()
        self.memory_folder = CONFIG["memory_folder"]

        os.makedirs(self.memory_folder, exist_ok=True)

        self.episodic_path = os.path.join(self.memory_folder, "episodic_memory.csv")
        self.feature_store_path = os.path.join(self.memory_folder, "feature_store.csv")
        self.working_memory_path = os.path.join(self.memory_folder, "working_memory.csv")
        self.metadata_path = os.path.join(self.memory_folder, "market_metadata.json")
        self.checkpoint_path = os.path.join(self.memory_folder, "learning_checkpoint.json")

    # ======================================================
    # TIMEFRAME HELPERS
    # ======================================================

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        tf = timeframe.strip().lower()

        if tf.endswith("m"):
            return int(tf[:-1])
        if tf.endswith("h"):
            return int(tf[:-1]) * 60
        if tf.endswith("d"):
            return int(tf[:-1]) * 1440

        raise ValueError(f"Unsupported timeframe: {timeframe}")

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        return self._timeframe_to_minutes(timeframe) * 60_000

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def _load_checkpoint(self) -> dict:
        """Load training checkpoint if it exists."""
        if not os.path.exists(self.checkpoint_path):
            return {}

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_checkpoint(self, checkpoint: dict) -> None:
        """Save training checkpoint."""
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint, f, indent=4)

    # ======================================================
    # FETCH
    # ======================================================

    def _fetch_ohlcv_safe(self, symbol: str, timeframe: str, since=None, limit=None):
        """Fetch OHLCV data with retry logic."""
        last_error = None

        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                return self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit,
                )
            except Exception as e:
                last_error = e
                if attempt < CONFIG["max_retries"]:
                    time.sleep(CONFIG["retry_wait_seconds"] * attempt)

        raise RuntimeError(
            f"Failed to fetch OHLCV for {symbol} {timeframe}"
        ) from last_error

    # ======================================================
    # DATAFRAME NORMALIZATION
    # ======================================================

    def _to_dataframe(self, data):
        """Convert raw OHLCV data to normalized DataFrame."""
        df = pd.DataFrame(
            data,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df = df.reset_index(drop=True)
        return df

    def _load_existing_csv(self, path: str) -> pd.DataFrame:
        """Load existing CSV file if it exists."""
        if not os.path.exists(path):
            return pd.DataFrame()

        try:
            df = pd.read_csv(path)
            if df.empty:
                return df

            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

            return df
        except Exception:
            return pd.DataFrame()

    def _merge_frames(self, frames):
        """Merge multiple DataFrames and deduplicate."""
        valid = [f for f in frames if f is not None and not f.empty]
        if not valid:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.concat(valid, ignore_index=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
        df = df.reset_index(drop=True)
        return df

    # ======================================================
    # FRESH DATA BUILDERS
    # ======================================================

    def fetch_historical(self):
        """Fetch historical OHLCV data with incremental support."""
        timeframe = CONFIG["historical_timeframe"]
        lookback_minutes = CONFIG["historical_days"] * 24 * 60
        candles_needed = max(1, lookback_minutes // self._timeframe_to_minutes(timeframe))

        checkpoint = self._load_checkpoint()
        since = None

        if CONFIG["enable_incremental_history"]:
            last_ts = checkpoint.get("last_historical_timestamp")
            if last_ts:
                try:
                    last_dt = pd.to_datetime(last_ts, utc=True)
                    since = int(last_dt.value // 1_000_000) - (
                        CONFIG["incremental_overlap_candles"] * self._timeframe_to_ms(timeframe)
                    )
                    since = max(since, 0)
                except Exception:
                    since = None

        raw = self._fetch_ohlcv_safe(
            CONFIG["symbol"],
            timeframe=timeframe,
            since=since,
            limit=candles_needed,
        )

        return self._to_dataframe(raw)

    def fetch_recent(self):
        """Fetch recent OHLCV data."""
        timeframe = CONFIG["recent_timeframe"]
        limit = CONFIG["recent_minutes"]

        raw = self._fetch_ohlcv_safe(
            CONFIG["symbol"],
            timeframe=timeframe,
            since=None,
            limit=limit,
        )

        return self._to_dataframe(raw)

    # ======================================================
    # FEATURES
    # ======================================================

    def enrich_dataframe(self, df):
        """Compute technical features and regimes."""
        if df is None or df.empty:
            return df

        out = df.copy()

        # Returns
        out["pct_change"] = out["close"].pct_change() * 100
        out["log_return"] = np.log(out["close"] / out["close"].shift(1)).replace([np.inf, -np.inf], np.nan)
        out["ret_3"] = out["close"].pct_change(3) * 100
        out["ret_6"] = out["close"].pct_change(6) * 100

        # Moving averages
        out["ma20"] = out["close"].rolling(20, min_periods=20).mean()
        out["ma50"] = out["close"].rolling(50, min_periods=50).mean()
        out["ma200"] = out["close"].rolling(200, min_periods=200).mean()
        out["ma20_slope"] = out["ma20"].diff()
        out["ma50_slope"] = out["ma50"].diff()

        # Volatility
        out["volatility"] = ((out["high"] - out["low"]) / out["close"].replace(0, np.nan)) * 100
        out["volatility_ma20"] = out["volatility"].rolling(20, min_periods=20).mean()

        # Momentum and range
        out["momentum"] = out["close"].diff(3)
        out["range_pct"] = ((out["high"] - out["low"]) / out["close"].replace(0, np.nan)) * 100

        # Volume
        volume_ma20 = out["volume"].rolling(20, min_periods=20).mean()
        out["volume_ratio"] = out["volume"] / volume_ma20.replace(0, np.nan)

        # Trend
        out["trend_strength"] = ((out["ma20"] - out["ma50"]) / out["ma50"].replace(0, np.nan)) * 100
        out["trend_distance"] = ((out["close"] - out["ma20"]) / out["ma20"].replace(0, np.nan)) * 100

        # Candle structure
        out["body"] = (out["close"] - out["open"]).abs()
        out["upper_wick"] = out["high"] - out[["open", "close"]].max(axis=1)
        out["lower_wick"] = out[["open", "close"]].min(axis=1) - out["low"]

        candle_range = (out["high"] - out["low"]).replace(0, np.nan)
        out["body_to_range"] = out["body"] / candle_range
        out["close_location"] = (out["close"] - out["low"]) / candle_range

        # Breakouts
        out["rolling_high_20"] = out["high"].rolling(20, min_periods=20).max()
        out["rolling_low_20"] = out["low"].rolling(20, min_periods=20).min()
        out["breakout_up_distance"] = ((out["close"] - out["rolling_high_20"]) / out["rolling_high_20"].replace(0, np.nan)) * 100
        out["breakout_down_distance"] = ((out["close"] - out["rolling_low_20"]) / out["rolling_low_20"].replace(0, np.nan)) * 100

        # Volatility regime
        vol_q25 = out["volatility"].quantile(0.25)
        vol_q75 = out["volatility"].quantile(0.75)
        out["volatility_regime"] = np.where(
            out["volatility"].isna(),
            "UNKNOWN",
            np.where(out["volatility"] <= vol_q25, "LOW", np.where(out["volatility"] >= vol_q75, "HIGH", "NORMAL")),
        )

        # Trend regime
        trend_q25 = out["trend_strength"].quantile(0.25)
        trend_q75 = out["trend_strength"].quantile(0.75)
        out["trend_regime"] = np.where(
            out["trend_strength"].isna(),
            "UNKNOWN",
            np.where(out["trend_strength"] <= trend_q25, "BEAR", np.where(out["trend_strength"] >= trend_q75, "BULL", "NEUTRAL")),
        )

        # Volume regime
        vol_ratio_q25 = out["volume_ratio"].quantile(0.25)
        vol_ratio_q75 = out["volume_ratio"].quantile(0.75)
        out["volume_regime"] = np.where(
            out["volume_ratio"].isna(),
            "UNKNOWN",
            np.where(out["volume_ratio"] <= vol_ratio_q25, "LOW", np.where(out["volume_ratio"] >= vol_ratio_q75, "HIGH", "NORMAL")),
        )

        # Metadata
        out["episode_id"] = out["timestamp"].dt.strftime("%Y%m%d")
        out["data_quality"] = 1
        out.loc[(out["close"] <= 0) | (out["volume"] < 0), "data_quality"] = 0
        out["feature_complete"] = (
            ~out[
                [
                    "ma20",
                    "ma50",
                    "volatility",
                    "trend_strength",
                    "volume_ratio",
                ]
            ].isna().any(axis=1)
        ).astype(int)

        out = self.add_row_signature(out)
        return out

    # ======================================================
    # TARGETS
    # ======================================================

    def build_future_targets(self, df):
        """Build forward-looking return targets."""
        if df is None or df.empty:
            return df

        out = df.copy()
        base_minutes = self._timeframe_to_minutes(CONFIG["historical_timeframe"])

        horizons = {
            "future_return_30m": 30,
            "future_return_1h": 60,
            "future_return_4h": 240,
            "future_return_24h": 1440,
        }

        for col, minutes in horizons.items():
            steps = max(1, round(minutes / base_minutes))
            out[col] = ((out["close"].shift(-steps) - out["close"]) / out["close"]) * 100

        return out

    # ======================================================
    # QUALITY / SIGNATURES
    # ======================================================

    def add_row_signature(self, df):
        """Add SHA1 hash for each candle row."""
        if df is None or df.empty:
            return df

        out = df.copy()
        signatures = []

        for _, row in out.iterrows():
            raw = (
                f"{row['timestamp']}|"
                f"{row['open']:.8f}|{row['high']:.8f}|{row['low']:.8f}|"
                f"{row['close']:.8f}|{row['volume']:.8f}"
            )
            signatures.append(hashlib.sha1(raw.encode("utf-8")).hexdigest())

        out["candle_hash"] = signatures
        return out

    # ======================================================
    # SAVE
    # ======================================================

    def save_working_memory(self, df):
        """Save recent data for inference."""
        if df is None:
            return

        filepath = self.working_memory_path

        keep_cols = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "pct_change",
            "log_return",
            "ma20",
            "ma50",
            "volatility",
            "momentum",
            "trend_strength",
            "volume_ratio",
            "volatility_regime",
            "trend_regime",
            "volume_regime",
            "data_quality",
            "feature_complete",
            "episode_id",
            "candle_hash",
        ]

        cols = [c for c in keep_cols if c in df.columns]
        df[cols].to_csv(filepath, index=False)

    def save_episodic_memory(self, df):
        """Save full historical data."""
        if df is None:
            return
        df.to_csv(self.episodic_path, index=False)

    def save_feature_store(self, df):
        """Save enriched feature data."""
        if df is None:
            return
        df.to_csv(self.feature_store_path, index=False)

    def save_metadata(self, metadata: dict):
        """Save metadata about the build."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)

    # ======================================================
    # BUILD
    # ======================================================

    def build_memory(self):
        """Main build pipeline: fetch, enrich, save across layers."""
        existing_episodic = self._load_existing_csv(self.episodic_path)

        historical_new = self.fetch_historical()
        historical = self._merge_frames([existing_episodic, historical_new])

        historical = self.enrich_dataframe(historical)
        historical = self.build_future_targets(historical)

        recent = self.fetch_recent()
        recent = self.enrich_dataframe(recent)

        self.save_episodic_memory(historical)
        self.save_feature_store(historical)
        self.save_working_memory(recent)

        latest_ts = None
        if not historical.empty:
            latest_ts = historical["timestamp"].max()

        metadata = {
            "symbol": CONFIG["symbol"],
            "historical_timeframe": CONFIG["historical_timeframe"],
            "recent_timeframe": CONFIG["recent_timeframe"],
            "historical_rows": int(len(historical)),
            "recent_rows": int(len(recent)),
            "feature_count": int(len(historical.columns)) if not historical.empty else 0,
            "last_update_utc": pd.Timestamp.utcnow().isoformat(),
            "latest_historical_timestamp": latest_ts.isoformat() if latest_ts is not None else None,
        }
        self.save_metadata(metadata)

        checkpoint = {
            "last_historical_timestamp": latest_ts.isoformat() if latest_ts is not None else None,
            "historical_rows": int(len(historical)),
            "recent_rows": int(len(recent)),
            "feature_count": int(len(historical.columns)) if not historical.empty else 0,
            "symbol": CONFIG["symbol"],
        }
        self._save_checkpoint(checkpoint)

        return metadata
