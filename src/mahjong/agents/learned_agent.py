"""Learned agent — HybridAgent with a trained deal-in model for defense.

Identical offense, claiming, and dynamic weighting to HybridAgent; the
only change is the danger estimate. Where the hybrid blends four
heuristic signals with hand-tuned weights (0.25/0.25/0.35/0.15), this
agent asks a logistic regression trained on simulated deal-in outcomes:
"what is the probability this exact discard deals in right now?"

The model outputs a calibrated probability (typically 0-15%), which is
far spikier than the heuristic's 0-1 danger scale. Feeding raw
probabilities into the hybrid's offense/defense blend would make every
tile look safe, so the probability is squashed through

    danger = p / (p + SQUASH_K)

which maps p = SQUASH_K to 0.5 danger. SQUASH_K is set near the average
per-discard deal-in probability, so "twice as risky as an average
discard" lands above 0.5 — the same region the heuristic occupies.
"""

from typing import Optional

from mahjong.agents.hybrid_agent import HybridAgent
from mahjong.ml.features import danger_features
from mahjong.ml.model import LinearModel, load_danger_model


class LearnedAgent(HybridAgent):
    """Hybrid strategy, model-driven defense."""

    SQUASH_K = 0.02

    def __init__(self, name: str = "Learned",
                 model: Optional[LinearModel] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model if model is not None else load_danger_model()
        if self.model is None:
            raise RuntimeError(
                "No trained danger model found. Run the ML pipeline first:\n"
                "  PYTHONPATH=src python3 -m mahjong.ml.datagen --games 2000\n"
                "  PYTHONPATH=src python3 -m mahjong.ml.train")

    def deal_in_probability(self, tile_id: int, player_idx: int,
                            game_state, threat_data, visible) -> float:
        """Calibrated P(deal-in) for one candidate discard."""
        x = danger_features(tile_id, player_idx, game_state,
                            visible, threat_data)
        return self.model.predict(x)

    def danger_for(self, tile_id: int, player_idx: int, game_state,
                   threat_data, visible) -> float:
        p = self.deal_in_probability(tile_id, player_idx, game_state,
                                     threat_data, visible)
        return p / (p + self.SQUASH_K)
