"""
trainer.py

Orchestrates the complete training pipeline: data collection, state encoding,
pattern learning, evaluation, and architectural refinement.
"""

import os
import json
import traceback

import pandas as pd

from market_memory import MarketMemory
from state_encoder import StateEncoder
from pattern_learner import PatternLearner
from self_critic import SelfCritic
from evolution_engine import EvolutionEngine
from adaptive_state_engine import AdaptiveStateEngine
from architecture_refiner import ArchitectureRefiner


MEMORY_FOLDER = "memory"
TRAINING_REPORT = os.path.join(MEMORY_FOLDER, "training_report.json")
TRAINING_HISTORY = os.path.join(MEMORY_FOLDER, "training_history.csv")


class Trainer:
    """Orchestrates the complete training pipeline."""

    def __init__(self):
        self.market_memory = MarketMemory()
        self.state_encoder = StateEncoder()
        self.pattern_learner = PatternLearner()
        self.self_critic = SelfCritic()
        self.evolution_engine = EvolutionEngine()
        self.adaptive_state_engine = AdaptiveStateEngine()
        self.architecture_refiner = ArchitectureRefiner()
        self.initialize_files()

    # ==================================================
    # INIT
    # ==================================================

    def initialize_files(self):
        """Initialize training history file."""
        if not os.path.exists(TRAINING_HISTORY):
            pd.DataFrame(
                columns=["timestamp", "accuracy", "coverage", "patterns", "states"]
            ).to_csv(TRAINING_HISTORY, index=False)

    # ==================================================
    # MARKET
    # ==================================================

    def build_market_memory(self):
        """Build market memory from exchange data."""
        return self.market_memory.build_memory()

    # ==================================================
    # STATES
    # ==================================================

    def build_states(self):
        """Encode features into states."""
        feature_store = pd.read_csv(os.path.join(MEMORY_FOLDER, "feature_store.csv"))
        states = self.state_encoder.build_state_memory(feature_store)
        return states

    # ==================================================
    # PATTERNS
    # ==================================================

    def learn_patterns(self):
        """Learn state transition patterns."""
        result = self.pattern_learner.learn()
        return {
            "patterns": result["patterns"],
            "top_score": result["top_score"],
        }

    # ==================================================
    # CRITIC
    # ==================================================

    def evaluate(self):
        """Evaluate system health and accuracy."""
        return self.self_critic.evaluate()

    # ==================================================
    # EVOLUTION
    # ==================================================

    def evolve(self):
        """Evolve knowledge base."""
        return self.evolution_engine.evolve()

    # ==================================================
    # ADAPTIVE
    # ==================================================

    def adapt(self):
        """Analyze adaptive state recommendations."""
        return self.adaptive_state_engine.analyze()

    # ==================================================
    # HISTORY
    # ==================================================

    def update_history(self, critic, states_count, patterns_count):
        """Update training history."""
        history = pd.read_csv(TRAINING_HISTORY)
        prediction = critic.get("prediction_metrics", {})

        row = {
            "timestamp": pd.Timestamp.utcnow(),
            "accuracy": prediction.get("weighted_accuracy", 0),
            "coverage": prediction.get("coverage", 0),
            "patterns": patterns_count,
            "states": states_count,
        }

        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
        history.to_csv(TRAINING_HISTORY, index=False)

    # ==================================================
    # IMPROVEMENT CHECK
    # ==================================================

    def detect_improvement(self):
        """Check if accuracy has improved."""
        history = pd.read_csv(TRAINING_HISTORY)

        if len(history) < 2:
            return {"improved": None}

        last = history.iloc[-1]
        prev = history.iloc[-2]
        accuracy_delta = float(last["accuracy"] - prev["accuracy"])

        return {"improved": bool(accuracy_delta > 0), "accuracy_delta": accuracy_delta}

    # ==================================================
    # MAIN
    # ==================================================

    def train_cycle(self):
        """Execute one complete training cycle."""
        report = {"success": True}

        try:
            market = self.build_market_memory()
            states = self.build_states()
            patterns = self.learn_patterns()
            critic = self.evaluate()
            evolution = self.evolve()
            adaptive = self.adapt()

            self.architecture_refiner.refine(states, patterns, critic)
            self.update_history(critic, len(states), patterns["patterns"])
            improvement = self.detect_improvement()

            report = {
                "success": True,
                "market_memory": market,
                "states": len(states),
                "patterns": patterns["patterns"],
                "critic": critic,
                "evolution": evolution,
                "adaptive": adaptive,
                "improvement": improvement,
            }

        except Exception as e:
            report = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }

        with open(TRAINING_REPORT, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4, default=str)

        return report
