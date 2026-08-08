"""Learned agent — discards chosen by expected value over two trained models.

Claiming and kong logic come from HybridAgent. The discard decision is
where it differs: instead of blending normalised offense/defense scores
with hand-tuned weights, each candidate discard is scored in POINTS:

    EV(tile) = P(win | state after discard) x WIN_VALUE
             - P(deal-in on tile)           x DEAL_IN_COST

Both probabilities come from models fitted on simulated outcomes
(mahjong/ml/): the deal-in model reads the tile + table context, the
win model reads the hand's shape after the discard (shanten,
acceptance, exposure, game phase). WIN_VALUE and DEAL_IN_COST are the
empirical average winner gain and shooter loss measured over seeded
games at the default table (tai cap 6, base 1).

Because both terms are calibrated probabilities times point stakes,
folding emerges naturally: when the hand is hopeless, P(win) is flat
and near zero for every candidate, so the risk term decides — the
agent discards its safest tile without any explicit "defense mode".

The first integration attempt squashed P(deal-in) into the hybrid's
0-1 danger slot (p / (p + k)); it LOST to the hybrid head-to-head
(21.8% vs 26.2%, deal-ins 1.90 vs 1.30 per 100 discards) because
calibrated ~2% probabilities flattened a defense term balanced around
inflated heuristic scores. That path is kept as `danger_for` for
comparison runs; `choose_discard` no longer uses it.
"""

from typing import Optional

from mahjong.agents.hybrid_agent import HybridAgent
from mahjong.hand import evaluate_discards
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.ml.features import danger_features, outcome_features
from mahjong.ml.model import LinearModel, load_danger_model, load_win_model

_TRAIN_HINT = (
    "Run the ML pipeline first:\n"
    "  PYTHONPATH=src python3 -m mahjong.ml.datagen --games 2000\n"
    "  PYTHONPATH=src python3 -m mahjong.ml.train")


class LearnedAgent(HybridAgent):
    """Hybrid claims, EV-over-trained-models discards."""

    # Empirical point stakes (400 seeded games under the table's rules:
    # tai cap 6, base 1, no self-draw tai, tai-only accounting).
    # Mean winner gain 8.63; mean shooter loss 8.55.
    #
    # Note how close these are now. With instant chip payouts off, a ron
    # is a straight transfer — the shooter pays exactly what the winner
    # collects — so risking a deal-in costs almost precisely what
    # winning pays. Under the previous rules the gap was much wider
    # (7.65 vs 5.94) because bonus-tile payouts inflated winners'
    # takings without touching the shooter.
    WIN_VALUE = 8.63
    DEAL_IN_COST = 8.55

    # Legacy squash for danger_for (the shipped-then-superseded path).
    SQUASH_K = 0.02

    def __init__(self, name: str = "Learned",
                 model: Optional[LinearModel] = None,
                 win_model: Optional[LinearModel] = None, **kwargs):
        super().__init__(name, **kwargs)
        self.model = model if model is not None else load_danger_model()
        if self.model is None:
            raise RuntimeError("No trained danger model found. " + _TRAIN_HINT)
        self.win_model = (win_model if win_model is not None
                          else load_win_model())
        if self.win_model is None:
            raise RuntimeError("No trained win model found. " + _TRAIN_HINT)

    def deal_in_probability(self, tile_id: int, player_idx: int,
                            game_state, threat_data, visible) -> float:
        """Calibrated P(deal-in) for one candidate discard."""
        x = danger_features(tile_id, player_idx, game_state,
                            visible, threat_data)
        return self.model.predict(x)

    def win_probability(self, player_idx: int, game_state,
                        shanten_after: int, acceptance: int,
                        threat_data) -> float:
        """Calibrated P(this seat wins the hand) after a discard."""
        x = outcome_features(player_idx, game_state, shanten_after,
                             acceptance, threat_data)
        return self.win_model.predict(x)

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

        best_tile = None
        best_ev = None
        # evals arrive sorted offense-best-first, so ties keep the
        # more efficient discard.
        for e in evals:
            tid = e["tile_id"]
            p_win = self.win_probability(player_idx, game_state,
                                         e["shanten"], e["acceptance"],
                                         threat_data)
            p_di = self.deal_in_probability(tid, player_idx, game_state,
                                            threat_data, visible)
            ev = p_win * self.WIN_VALUE - p_di * self.DEAL_IN_COST
            if best_ev is None or ev > best_ev:
                best_tile = tid
                best_ev = ev
        return best_tile
