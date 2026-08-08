"""Learned agent — discards AND claims chosen by expected value.

Phase B made the claim decisions model-based. There is deliberately no
supervised "claim model": the heuristic agents that generated our data
only claim at shanten <= 2, so observed claims are hopelessly
confounded — a model fitted on them learns "claiming means you were
already close", not whether claiming helps. Instead every claim window
is a BRANCH COMPARISON under the same value model:

    claim branch:  form the meld, then best-EV discard from the
                   shortened hand (its forced discard carries the usual
                   deal-in risk term)
    pass branch:   V of the hand exactly as it stands

Claim iff the claim branch is worth more points. Everything the value
model learned now prices the claim: shanten gained, shape entered, and
— this table's tax — the concealed bonus a first meld destroys and the
ping hu that a claimed pong forfeits. Passing has no discard-risk term
(you throw nothing), so claims get rarer as the table gets hotter,
with no explicit rule saying so.

Own-turn kong declarations stay on the hybrid's shanten rule: they are
0.3% of decisions, and their upside (a free replacement draw, a
possible robbed kong against us) is invisible to V — a V comparison
would be systematically biased against them.

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
from mahjong.hand import calculate_shanten, evaluate_discards, tile_acceptance
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.tiles import NUM_STANDARD_UNIQUE
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

    # ── Phase B: claims as branch evaluation ─────────────────────────

    def _shape_value(self, player_idx: int, game_state, counts: List[int],
                     exposed, threat_data, visible) -> float:
        """V of a 13-shape state (counts + possibly hypothetical melds)."""
        melds = len(exposed)
        shanten = calculate_shanten(counts, melds)
        _, acceptance = tile_acceptance(counts, melds, visible)
        x = outcome_features(player_idx, game_state, counts, shanten,
                             acceptance, threat_data, exposed=exposed)
        return self.value_model.predict(x)

    def _best_discard_ev(self, player_idx: int, game_state,
                         counts: List[int], exposed, threat_data,
                         visible) -> float:
        """Best EV over all discards from a 14-shape hand — the value of
        a claim branch, forced follow-up discard risk included."""
        melds = len(exposed)
        best = None
        for t in range(NUM_STANDARD_UNIQUE):
            if counts[t] == 0:
                continue
            counts[t] -= 1
            shanten = calculate_shanten(counts, melds)
            _, acceptance = tile_acceptance(counts, melds, visible)
            x = outcome_features(player_idx, game_state, counts, shanten,
                                 acceptance, threat_data, exposed=exposed)
            value = self.value_model.predict(x)
            counts[t] += 1
            p_di = self.deal_in_probability(t, player_idx, game_state,
                                            threat_data, visible)
            ev = (1.0 - p_di) * value - p_di * self.DEAL_IN_COST
            if best is None or ev > best:
                best = ev
        return best if best is not None else -self.DEAL_IN_COST

    def should_claim(self, player_idx: int, tile_id: int,
                     claim_type: str, game_state) -> bool:
        """Pong (or kong) a discard iff the melded branch out-values
        standing pat."""
        hand = game_state.hands[player_idx]
        counts = hand.copy_counts()
        needed = 2 if claim_type == "pong" else 3
        if counts[tile_id] < needed:
            return False
        visible = game_state.get_visible_counts(player_idx)
        threat_data = estimate_opponent_threats(player_idx, game_state)

        pass_value = self._shape_value(player_idx, game_state, counts,
                                       hand.exposed, threat_data, visible)

        counts[tile_id] -= needed
        meld_kind = "pong" if claim_type == "pong" else "kong"
        exposed = list(hand.exposed) + [(meld_kind,
                                         [tile_id] * (needed + 1))]
        if claim_type == "pong":
            # 14-shape: meld formed, must discard
            claim_value = self._best_discard_ev(
                player_idx, game_state, counts, exposed, threat_data, visible)
        else:
            # Exposed kong: 13-shape before the replacement draw. The
            # draw's upside and the forced discard's risk are both
            # unmodelled and roughly offset.
            claim_value = self._shape_value(
                player_idx, game_state, counts, exposed, threat_data, visible)
        return claim_value > pass_value

    def choose_chow(self, player_idx: int, tile_id: int,
                    options, game_state):
        """Chow with the best partner pair iff it out-values passing."""
        hand = game_state.hands[player_idx]
        visible = game_state.get_visible_counts(player_idx)
        threat_data = estimate_opponent_threats(player_idx, game_state)
        counts = hand.copy_counts()

        pass_value = self._shape_value(player_idx, game_state, counts,
                                       hand.exposed, threat_data, visible)

        best_pair = None
        best_value = pass_value
        for p1, p2 in options:
            counts[p1] -= 1
            counts[p2] -= 1
            exposed = list(hand.exposed) + [
                ("chow", sorted([tile_id, p1, p2]))]
            branch = self._best_discard_ev(
                player_idx, game_state, counts, exposed, threat_data, visible)
            counts[p1] += 1
            counts[p2] += 1
            if branch > best_value:
                best_value = branch
                best_pair = (p1, p2)
        return best_pair

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
