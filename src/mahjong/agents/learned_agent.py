"""Learned agent — discards chosen by expected value over trained models.

Claiming and kong logic come from HybridAgent (Phase B learns those).
The discard decision scores each candidate in POINTS:

    EV(tile) = (1 − P(deal-in on tile)) × V(hand after discard)
             −      P(deal-in on tile)  × DEAL_IN_COST

P(deal-in) comes from the danger model. V comes from the Phase A value
model: expected net points from the post-discard state, trained on
realized outcomes with tai-potential features — so a hand tracking a
full flush or holding its concealed ping hu is WORTH more than a
chicken hand at the same shanten, and the agent pushes or folds
accordingly. The earlier formula priced every win at one constant
(8.63), which is exactly why it folded hands worth fighting for.

DEAL_IN_COST stays an empirical constant (mean shooter loss, 8.55):
the cost of feeding a hand depends on the OPPONENT's tiles, which are
hidden. V(after) technically already includes downstream deal-in risk;
the immediate-discard term is split out because V's features cannot see
which tile is about to leave the hand. The overlap is second-order.

History, kept honest: the first integration squashed P(deal-in) into
the hybrid's 0-1 danger slot and LOST to Hybrid (DI 1.90 vs 1.30). The
constant-stakes EV rewrite fixed defense but only reached statistical
parity (1,000-game duplicate evaluation: all differences n.s.). The
squash path survives as `danger_for` for comparison runs.
"""

from typing import List, Optional

from mahjong.agents.hybrid_agent import HybridAgent
from mahjong.hand import evaluate_discards
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.ml.features import danger_features, outcome_features
from mahjong.ml.model import (
    LinearModel, load_danger_model, load_value_model,
)

_TRAIN_HINT = (
    "Run the ML pipeline first:\n"
    "  PYTHONPATH=src python3 -m mahjong.ml.datagen --games 10000\n"
    "  PYTHONPATH=src python3 -m mahjong.ml.train")


class LearnedAgent(HybridAgent):
    """Hybrid claims, value-aware EV discards."""

    # Mean shooter loss over 400 seeded games at this table (tai cap 6,
    # no self-draw tai, tai-only accounting).
    DEAL_IN_COST = 8.55

    # Legacy squash for danger_for (superseded; kept for comparisons).
    SQUASH_K = 0.02

    def __init__(self, name: str = "Learned",
                 model: Optional[LinearModel] = None,
                 value_model: Optional[LinearModel] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model if model is not None else load_danger_model()
        if self.model is None:
            raise RuntimeError("No trained danger model found. " + _TRAIN_HINT)
        self.value_model = (value_model if value_model is not None
                            else load_value_model())
        if self.value_model is None:
            raise RuntimeError("No trained value model found. " + _TRAIN_HINT)

    def deal_in_probability(self, tile_id: int, player_idx: int,
                            game_state, threat_data, visible) -> float:
        """Calibrated P(deal-in) for one candidate discard."""
        x = danger_features(tile_id, player_idx, game_state,
                            visible, threat_data)
        return self.model.predict(x)

    def hand_value(self, player_idx: int, game_state,
                   counts_after: List[int], shanten_after: int,
                   acceptance: int, threat_data) -> float:
        """Expected net points from the post-discard state."""
        x = outcome_features(player_idx, game_state, counts_after,
                             shanten_after, acceptance, threat_data)
        return self.value_model.predict(x)

    def danger_for(self, tile_id: int, player_idx: int, game_state,
                   threat_data, visible) -> float:
        """Legacy hook: squashed probability on the hybrid's 0-1 scale."""
        p = self.deal_in_probability(tile_id, player_idx, game_state,
                                     threat_data, visible)
        return p / (p + self.SQUASH_K)

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]
        visible = game_state.get_visible_counts(player_idx)
        threat_data = estimate_opponent_threats(player_idx, game_state)
        evals = evaluate_discards(hand, visible_counts=visible)
        counts = hand.copy_counts()

        best_tile = None
        best_ev = None
        # evals arrive sorted offense-best-first, so ties keep the
        # more efficient discard.
        for e in evals:
            tid = e["tile_id"]
            counts[tid] -= 1
            value = self.hand_value(player_idx, game_state, counts,
                                    e["shanten"], e["acceptance"],
                                    threat_data)
            counts[tid] += 1
            p_di = self.deal_in_probability(tid, player_idx, game_state,
                                            threat_data, visible)
            ev = (1.0 - p_di) * value - p_di * self.DEAL_IN_COST
            if best_ev is None or ev > best_ev:
                best_tile = tid
                best_ev = ev
        return best_tile
