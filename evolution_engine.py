"""
evolution_engine.py

Manages knowledge base evolution, pattern scoring, and system health metrics.
"""

import os
import json

import pandas as pd
import numpy as np


MEMORY_FOLDER = "memory"
PATTERN_FILE = os.path.join(MEMORY_FOLDER, "pattern_memory.csv")
CRITIC_FILE = os.path.join(MEMORY_FOLDER, "self_critic.json")
KNOWLEDGE_FILE = os.path.join(MEMORY_FOLDER, "knowledge_base.json")
REPORT_FILE = os.path.join(MEMORY_FOLDER, "evolution_report.json")


class EvolutionEngine:
    """Evolves and maintains the system knowledge base."""

    def __init__(self):
        os.makedirs(MEMORY_FOLDER, exist_ok=True)
        self.initialize()

    # ==================================================
    # INIT
    # ==================================================

    def initialize(self):
        """Initialize knowledge base if it doesn't exist."""
        if not os.path.exists(KNOWLEDGE_FILE):
            data = {
                "version": 2,
                "active_patterns": [],
                "retired_patterns": [],
                "state_scores": {},
                "pattern_scores": {},
                "active_hypotheses": [],
                "recommended_actions": [],
                "system_health": {},
            }

            with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

    # ==================================================
    # LOADERS
    # ==================================================

    def load_patterns(self):
        """Load patterns from CSV."""
        if not os.path.exists(PATTERN_FILE):
            return pd.DataFrame()
        return pd.read_csv(PATTERN_FILE)

    def load_critic(self):
        """Load critic report."""
        if not os.path.exists(CRITIC_FILE):
            return {}
        with open(CRITIC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_knowledge(self):
        """Load knowledge base."""
        if not os.path.exists(KNOWLEDGE_FILE):
            self.initialize()
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    # ==================================================
    # SYSTEM HEALTH
    # ==================================================

    def evaluate_health(self):
        """Evaluate system health metrics."""
        critic = self.load_critic()
        prediction = critic.get("prediction_metrics", {})
        states = critic.get("state_metrics", {})

        return {
            "accuracy": float(prediction.get("weighted_accuracy", 0)),
            "coverage": float(prediction.get("coverage", 0)),
            "diversity": int(states.get("unique_states", 0)),
            "confidence": float(prediction.get("avg_confidence", 0)),
            "overconfidence": 0.0,
        }

    # ==================================================
    # STATE SCORING
    # ==================================================

    def evaluate_states(self):
        """Score states based on performance."""
        patterns = self.load_patterns()

        if patterns.empty:
            return {}

        scores = {}

        for state, group in patterns.groupby("state"):
            score = 0.0

            if "win_rate_1h" in group.columns:
                score += float(group["win_rate_1h"].mean())

            score += float(group["transition_probability"].mean())
            score += min(float(group["samples"].sum()) / 1000.0, 1.0)

            scores[state] = round(float(score), 4)

        return scores

    # ==================================================
    # PATTERN SCORING
    # ==================================================

    def evaluate_patterns(self):
        """Score and classify patterns."""
        patterns = self.load_patterns()

        if patterns.empty:
            return {}, [], []

        scores = {}
        active = []
        retired = []

        for _, row in patterns.iterrows():
            score = float(row["transition_probability"]) * np.log1p(float(row["samples"]))
            pattern_id = row["pattern_id"]
            scores[pattern_id] = round(float(score), 4)

            if score >= 0.30:
                active.append(pattern_id)
            else:
                retired.append(pattern_id)

        return scores, active, retired

    # ==================================================
    # HYPOTHESIS ENGINE
    # ==================================================

    def generate_hypotheses(self, health, state_scores):
        """Generate improvement hypotheses and actions."""
        hypotheses = []

        if health["accuracy"] < 0.55:
            hypotheses.append("increase_state_resolution")

        if health["coverage"] < 0.90:
            hypotheses.append("learn_missing_transitions")

        if health["diversity"] < 10:
            hypotheses.append("increase_state_diversity")

        if health["overconfidence"] > 0:
            hypotheses.append("reduce_overconfidence")

        for state, score in state_scores.items():
            if score < 0.75:
                hypotheses.append(f"review_state_{state}")

        hypotheses = sorted(list(set(hypotheses)))

        actions = []

        if "increase_state_resolution" in hypotheses:
            actions.append({"action": "state_encoder_refinement", "priority": 1})

        if "increase_state_diversity" in hypotheses:
            actions.append({"action": "state_space_expansion", "priority": 2})

        if "learn_missing_transitions" in hypotheses:
            actions.append({"action": "transition_exploration", "priority": 3})

        return {"hypotheses": hypotheses, "actions": actions}

    # ==================================================
    # UPDATE
    # ==================================================

    def update_knowledge(self):
        """Update knowledge base with latest metrics."""
        knowledge = self.load_knowledge()
        health = self.evaluate_health()
        state_scores = self.evaluate_states()
        pattern_scores, active_patterns, retired_patterns = self.evaluate_patterns()
        evolution_plan = self.generate_hypotheses(health, state_scores)

        knowledge["system_health"] = health
        knowledge["state_scores"] = state_scores
        knowledge["pattern_scores"] = pattern_scores
        knowledge["active_patterns"] = active_patterns
        knowledge["retired_patterns"] = retired_patterns
        knowledge["active_hypotheses"] = evolution_plan["hypotheses"]
        knowledge["recommended_actions"] = evolution_plan["actions"]

        with open(KNOWLEDGE_FILE, "w", encoding="utf-8") as f:
            json.dump(knowledge, f, indent=4)

        return knowledge

    # ==================================================
    # REPORT
    # ==================================================

    def save_report(self, knowledge):
        """Save evolution report."""
        report = {
            "active_patterns": len(knowledge.get("active_patterns", [])),
            "retired_patterns": len(knowledge.get("retired_patterns", [])),
            "states": len(knowledge.get("state_scores", {})),
            "hypotheses": len(knowledge.get("active_hypotheses", [])),
            "recommended_actions": knowledge.get("recommended_actions", []),
            "health": knowledge.get("system_health", {}),
        }

        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return report

    # ==================================================
    # MAIN
    # ==================================================

    def evolve(self):
        """Main evolution pipeline."""
        knowledge = self.update_knowledge()
        return self.save_report(knowledge)
