"""
pattern_learner.py

Learns state transition patterns from encoded market states.
Builds feature vectors and identifies recurring market regimes.
"""

import os
import json
import hashlib

import numpy as np
import pandas as pd


MEMORY_FOLDER = "memory"

STATE_FILE = os.path.join(MEMORY_FOLDER, "state_memory.csv")
FEATURE_FILE = os.path.join(MEMORY_FOLDER, "feature_store.csv")
PATTERN_FILE = os.path.join(MEMORY_FOLDER, "pattern_memory.csv")
STATS_FILE = os.path.join(MEMORY_FOLDER, "pattern_statistics.json")
CHECKPOINT_FILE = os.path.join(MEMORY_FOLDER, "learning_checkpoint.json")


class PatternLearner:
    """Learns and scores patterns from market state transitions."""

    MIN_SAMPLES = 5

    def __init__(self):
        os.makedirs(MEMORY_FOLDER, exist_ok=True)

    # =====================================================
    # LOAD
    # =====================================================

    def load_data(self):
        """Load and merge state and feature data."""
        try:
            states = pd.read_csv(STATE_FILE)
        except Exception as e:
            print(f"Error loading {STATE_FILE}: {e}")
            return pd.DataFrame()

        try:
            features = pd.read_csv(FEATURE_FILE)
        except Exception as e:
            print(f"Error loading {FEATURE_FILE}: {e}")
            return pd.DataFrame()

        if "timestamp" in states.columns:
            states["timestamp"] = pd.to_datetime(states["timestamp"])
        else:
            raise KeyError("Column 'timestamp' not found in state_memory.csv")

        if "timestamp" in features.columns:
            features["timestamp"] = pd.to_datetime(features["timestamp"])
        else:
            raise KeyError("Column 'timestamp' not found in feature_store.csv")

        states = states.drop_duplicates(subset="timestamp")
        features = features.drop_duplicates(subset="timestamp")

        df = pd.merge(states, features, on="timestamp", how="left")
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    # =====================================================
    # ENTROPY
    # =====================================================

    def entropy(self, values):
        """Calculate Shannon entropy of a probability distribution."""
        probs = np.array(values)
        probs = probs[probs > 0]

        if len(probs) == 0:
            return 0

        return float(-(probs * np.log2(probs)).sum())

    # =====================================================
    # BUILD
    # =====================================================

    def build_patterns(self, df):
        """Learn state transition patterns from historical data."""
        df = df.copy()
        df["next_state"] = df["state"].shift(-1)

        latest_time = df["timestamp"].max()
        patterns = []

        for state in df["state"].dropna().unique():
            subset = df[df["state"] == state]
            samples = len(subset)

            if samples < self.MIN_SAMPLES:
                continue

            transitions = subset["next_state"].value_counts(normalize=True)

            if len(transitions) == 0:
                continue

            top_transitions = transitions.head(3).to_dict()
            most_likely = transitions.idxmax()
            probability = transitions.max()
            entropy = self.entropy(transitions.values)
            stability = 1 / (1 + entropy)

            row = {
                "pattern_id": hashlib.md5(str(state).encode("utf-8")).hexdigest(),
                "state": state,
                "samples": int(samples),
                "most_likely_next_state": most_likely,
                "top_transitions": top_transitions,
                "transition_probability": float(probability),
                "entropy": float(entropy),
                "stability": float(stability),
            }

            # Return metrics
            if "future_return_30m" in subset.columns:
                row["avg_return_30m"] = float(subset["future_return_30m"].mean())
                row["win_rate_30m"] = float((subset["future_return_30m"] > 0).mean())

            if "future_return_1h" in subset.columns:
                row["avg_return_1h"] = float(subset["future_return_1h"].mean())
                row["win_rate_1h"] = float((subset["future_return_1h"] > 0).mean())

            if "future_return_4h" in subset.columns:
                row["avg_return_4h"] = float(subset["future_return_4h"].mean())
                row["win_rate_4h"] = float((subset["future_return_4h"] > 0).mean())

            # Age and confidence
            last_seen = subset["timestamp"].max()
            if pd.isnull(last_seen):
                continue

            age_hours = (latest_time - last_seen).total_seconds() / 3600
            row["age_hours"] = float(age_hours)

            confidence = probability * stability
            row["confidence"] = float(confidence)

            score = confidence * np.sqrt(samples)
            row["global_score"] = float(score)

            patterns.append(row)

        patterns = pd.DataFrame(patterns)
        patterns = patterns.sort_values("global_score", ascending=False)

        return patterns

    # =====================================================
    # SAVE
    # =====================================================

    def save_patterns(self, patterns):
        """Save patterns and compute statistics."""
        patterns.to_csv(PATTERN_FILE, index=False)

        stats = {
            "total_patterns": int(len(patterns)),
            "active_patterns": int((patterns["global_score"] > 1).sum()) if len(patterns) else 0,
            "average_confidence": float(patterns["confidence"].mean()) if len(patterns) else 0.0,
            "best_pattern_score": float(patterns["global_score"].max()) if len(patterns) else 0.0,
        }

        try:
            with open(STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=4)
        except Exception as e:
            print(f"Error writing to {STATS_FILE}: {e}")

        checkpoint = {"patterns_learned": int(len(patterns))}

        try:
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=4)
        except Exception as e:
            print(f"Error writing to {CHECKPOINT_FILE}: {e}")

    # =====================================================
    # MAIN
    # =====================================================

    def learn(self):
        """Main learning pipeline."""
        df = self.load_data()
        patterns = self.build_patterns(df)
        self.save_patterns(patterns)
        return {
            "patterns": len(patterns),
            "top_score": float(patterns["global_score"].max()) if len(patterns) else 0.0,
        }
