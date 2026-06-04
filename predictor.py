"""
predictor.py

Generates trading predictions from learned patterns and current market state.
"""

import os
import json
import pandas as pd


MEMORY_FOLDER = "memory"
PATTERN_FILE = os.path.join(MEMORY_FOLDER, "pattern_memory.csv")
STATE_FILE = os.path.join(MEMORY_FOLDER, "state_memory.csv")
REPORT_FILE = os.path.join(MEMORY_FOLDER, "prediction_report.json")


class Predictor:
    """Generates trading predictions from learned patterns."""

    def __init__(self):
        os.makedirs(MEMORY_FOLDER, exist_ok=True)

    def load_patterns(self):
        """Load learned patterns."""
        if not os.path.exists(PATTERN_FILE):
            return pd.DataFrame()
        return pd.read_csv(PATTERN_FILE)

    def load_states(self):
        """Load current state history."""
        if not os.path.exists(STATE_FILE):
            return pd.DataFrame()
        return pd.read_csv(STATE_FILE)

    def predict(self):
        """Generate prediction for current state."""
        patterns = self.load_patterns()
        states = self.load_states()

        if patterns.empty or states.empty:
            report = {"status": "empty", "decision": "neutral"}
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            return report

        current_state = str(states.iloc[-1]["state"])

        row = patterns[patterns["state"] == current_state]
        if row.empty:
            report = {
                "status": "no_match",
                "current_state": current_state,
                "decision": "neutral",
            }
            with open(REPORT_FILE, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=4)
            return report

        row = row.iloc[0]

        # Calculate directional scores
        bullish = 0.0
        bearish = 0.0

        if "win_rate_1h" in row.index:
            bullish += float(row["win_rate_1h"])
        if "win_rate_4h" in row.index:
            bullish += float(row["win_rate_4h"])
        if "avg_return_1h" in row.index:
            bullish += max(0.0, float(row["avg_return_1h"]))
            bearish += max(0.0, -float(row["avg_return_1h"]))
        if "avg_return_4h" in row.index:
            bullish += max(0.0, float(row["avg_return_4h"]))
            bearish += max(0.0, -float(row["avg_return_4h"]))

        confidence = float(row.get("confidence", 0.0))
        decision = "neutral"

        if bullish > bearish * 1.1:
            decision = "bullish"
        elif bearish > bullish * 1.1:
            decision = "bearish"

        report = {
            "status": "ok",
            "current_state": current_state,
            "most_likely_next_state": row.get("most_likely_next_state", ""),
            "top_transitions": row.get("top_transitions", ""),
            "bullish_score": bullish,
            "bearish_score": bearish,
            "confidence": confidence,
            "decision": decision,
        }

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, default=str)

        return report
