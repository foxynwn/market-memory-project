"""
self_critic.py

Self-evaluation module for assessing model quality and generating recommendations.
"""

import os
import json

import numpy as np
import pandas as pd


MEMORY_FOLDER = "memory"
STATE_MEMORY_FILE = os.path.join(MEMORY_FOLDER, "state_memory.csv")
PATTERN_MEMORY_FILE = os.path.join(MEMORY_FOLDER, "pattern_memory.csv")
SELF_CRITIC_FILE = os.path.join(MEMORY_FOLDER, "self_critic.json")


class SelfCritic:
    """Evaluates model health, accuracy, and recommends improvements."""

    def __init__(self):
        os.makedirs(MEMORY_FOLDER, exist_ok=True)

    # =====================================================
    # LOAD
    # =====================================================

    def load_states(self):
        """Load state memory."""
        if not os.path.exists(STATE_MEMORY_FILE):
            raise FileNotFoundError(STATE_MEMORY_FILE)

        df = pd.read_csv(STATE_MEMORY_FILE)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp", "state"])
        return df.sort_values("timestamp").reset_index(drop=True)

    def load_patterns(self):
        """Load pattern memory."""
        if not os.path.exists(PATTERN_MEMORY_FILE):
            raise FileNotFoundError(PATTERN_MEMORY_FILE)

        df = pd.read_csv(PATTERN_MEMORY_FILE)

        required = ["state", "most_likely_next_state", "transition_probability"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        return df

    # =====================================================
    # TRANSITIONS
    # =====================================================

    def build_actual_transitions(self, states_df):
        """Build actual transition history from states."""
        df = states_df.copy()
        df["actual_next_state"] = df["state"].shift(-1)
        df = df.dropna(subset=["state", "actual_next_state"])
        return df

    # =====================================================
    # PREDICTIONS
    # =====================================================

    def build_prediction_map(self, patterns):
        """Build lookup map of state predictions."""
        pred_map = {}

        for _, row in patterns.iterrows():
            pred_map[row["state"]] = {
                "next_state": row["most_likely_next_state"],
                "confidence": float(row.get("confidence", row["transition_probability"])),
                "samples": int(row.get("samples", 0)),
                "score": float(row.get("global_score", 0)),
            }

        return pred_map

    # =====================================================
    # STATE ANALYSIS
    # =====================================================

    def analyze_states(self, states_df):
        """Analyze state distribution and balance."""
        counts = states_df["state"].value_counts()

        return {
            "unique_states": int(counts.shape[0]),
            "largest_state": str(counts.idxmax()),
            "largest_state_samples": int(counts.max()),
            "smallest_state": str(counts.idxmin()),
            "smallest_state_samples": int(counts.min()),
            "state_balance": float(counts.std() / counts.mean()) if counts.mean() > 0 else 0,
        }

    # =====================================================
    # PATTERN ANALYSIS
    # =====================================================

    def analyze_patterns(self, patterns):
        """Analyze learned patterns."""
        if patterns.empty:
            return {}

        return {
            "patterns_seen": int(len(patterns)),
            "avg_confidence": float(patterns["confidence"].mean()),
            "avg_probability": float(patterns["transition_probability"].mean()),
            "avg_score": float(patterns["global_score"].mean()),
            "max_score": float(patterns["global_score"].max()),
        }

    # =====================================================
    # ACCURACY
    # =====================================================

    def evaluate_predictions(self, transitions, prediction_map):
        """Evaluate prediction accuracy."""
        results = []

        for _, row in transitions.iterrows():
            state = row["state"]

            if state not in prediction_map:
                continue

            actual = row["actual_next_state"]
            pred = prediction_map[state]
            hit = int(pred["next_state"] == actual)

            results.append({
                "hit": hit,
                "confidence": pred["confidence"],
                "score": pred["score"],
            })

        if not results:
            return {}

        eval_df = pd.DataFrame(results)
        weighted_accuracy = np.average(eval_df["hit"], weights=eval_df["confidence"])

        return {
            "coverage": float(len(eval_df) / len(transitions)),
            "accuracy": float(eval_df["hit"].mean()),
            "weighted_accuracy": float(weighted_accuracy),
            "avg_confidence": float(eval_df["confidence"].mean()),
        }

    # =====================================================
    # HEALTH
    # =====================================================

    def build_health_score(self, prediction_metrics, state_metrics):
        """Calculate overall system health score."""
        accuracy = prediction_metrics.get("weighted_accuracy", 0)
        diversity = min(state_metrics["unique_states"] / 50, 1)
        balance = max(0, 1 - state_metrics["state_balance"])

        health = accuracy * 0.50 + diversity * 0.30 + balance * 0.20
        return float(round(health, 4))

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    def generate_recommendations(self, health, states):
        """Generate improvement recommendations."""
        recommendations = []

        if health < 0.50:
            recommendations.append("increase_state_resolution")

        if states["unique_states"] < 25:
            recommendations.append("increase_state_diversity")

        if states["state_balance"] > 1.5:
            recommendations.append("reduce_state_fragmentation")

        return recommendations

    # =====================================================
    # MAIN
    # =====================================================

    def evaluate(self):
        """Main evaluation pipeline."""
        states = self.load_states()
        patterns = self.load_patterns()
        transitions = self.build_actual_transitions(states)
        prediction_map = self.build_prediction_map(patterns)
        state_metrics = self.analyze_states(states)
        pattern_metrics = self.analyze_patterns(patterns)
        prediction_metrics = self.evaluate_predictions(transitions, prediction_map)
        health_score = self.build_health_score(prediction_metrics, state_metrics)
        recommendations = self.generate_recommendations(health_score, state_metrics)

        report = {
            "samples_seen": int(len(states)),
            "health_score": health_score,
            "state_metrics": state_metrics,
            "pattern_metrics": pattern_metrics,
            "prediction_metrics": prediction_metrics,
            "recommendations": recommendations,
        }

        with open(SELF_CRITIC_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return report
