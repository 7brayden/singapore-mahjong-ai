"""Learning-stats payload for the web UI.

Bundles what the engine already computes — shanten, per-discard
efficiency and danger breakdowns, opponent threat estimates — into one
JSON-ready dict. This feeds the stats sidebar that teaches players WHY
a discard is good or dangerous while they play.
"""

from typing import Dict

from mahjong.game import GameState
from mahjong.hand import calculate_shanten, evaluate_discards, get_winning_tiles
from mahjong.defense import estimate_danger_detailed
from mahjong.opponent_model import estimate_opponent_threats
from mahjong.ml.features import danger_features, outcome_features
from mahjong.ml.model import (
    load_danger_model, load_value_model, load_win_model,
)

# Trained models (None until the ML pipeline has been run once).
# Unlike the heuristic danger score, their outputs are calibrated:
# "this discard deals in X% of the time", "this hand wins Y% of the
# time from here", "this hand is worth Z points from here".
from mahjong.ml.features import DANGER_FEATURES, OUTCOME_FEATURES


def _fresh(model, expected_features):
    """Reject stale artifacts: a model trained on an older feature list
    must not be fed today's vectors — drop it rather than mis-predict."""
    if model is None or model.features != list(expected_features):
        return None
    return model


_danger_model = _fresh(load_danger_model(), DANGER_FEATURES)
_win_model = _fresh(load_win_model(), OUTCOME_FEATURES)
_value_model = _fresh(load_value_model(), OUTCOME_FEATURES)


def analyze_seat(game: GameState, seat: int,
                 agent_pick: int = None) -> Dict:
    """Full decision analysis from one seat's point of view.

    Uses only information visible to that seat. With 14 tiles (about to
    discard) every discard option is scored; with 13 tiles the hand's
    waits are reported instead.

    ``agent_pick`` is the tile the seat's own agent would discard right
    now. When given, that entry is flagged and moved to the front so
    the UI's starred recommendation is the SAME tile the coach explains
    — the advisor must never contradict the coach. The rest of the list
    keeps its pure-efficiency order (shanten ASC, acceptance DESC),
    which is the teaching contrast: "here is the fastest line, here is
    why the agent didn't take it".
    """
    hand = game.hands[seat]
    visible = game.get_visible_counts(seat)
    shanten = calculate_shanten(hand.copy_counts(), hand.num_exposed_melds)
    threat_data = estimate_opponent_threats(seat, game)

    analysis = {
        "seat": seat,
        "shanten": shanten,
        "opponents": _threat_view(seat, threat_data),
    }

    if len(hand.tiles) % 3 == 2:
        # 14-tile state: evaluate every discard option
        evals = evaluate_discards(hand, visible_counts=visible)
        discards = []
        for e in evals:
            danger = estimate_danger_detailed(e["tile_id"], seat, game, threat_data)
            entry = {
                "tile": e["tile_id"],
                "shanten_after": e["shanten"],
                "acceptance": e["acceptance"],
                "improving_tiles": e["improving_tiles"],
                "danger": round(danger["danger"], 3),
                "danger_components": {k: round(v, 3)
                                      for k, v in danger["components"].items()},
            }
            if _danger_model is not None:
                x = danger_features(e["tile_id"], seat, game, visible, threat_data)
                entry["deal_in_prob"] = round(_danger_model.predict(x), 4)
            if _win_model is not None or _value_model is not None:
                counts_after = hand.copy_counts()
                counts_after[e["tile_id"]] -= 1
                w = outcome_features(seat, game, counts_after,
                                     e["shanten"], e["acceptance"],
                                     threat_data)
                if _win_model is not None:
                    entry["win_prob"] = round(_win_model.predict(w), 4)
                if _value_model is not None:
                    # Expected net points of the hand this discard keeps
                    entry["hand_value"] = round(_value_model.predict(w), 3)
            discards.append(entry)
        if agent_pick is not None:
            for i, entry in enumerate(discards):
                if entry["tile"] == agent_pick:
                    entry["agent_pick"] = True
                    discards.insert(0, discards.pop(i))
                    break
        analysis["discards"] = discards
    elif shanten == 0:
        analysis["waiting_on"] = get_winning_tiles(
            hand.copy_counts(), hand.num_exposed_melds)

    return analysis


def _threat_view(seat: int, threat_data: Dict):
    opponents = []
    for opp in range(4):
        if opp == seat:
            continue
        suit_danger = threat_data["suit_danger"][opp]
        opponents.append({
            "seat": opp,
            "threat": round(threat_data["threats"][opp], 3),
            "suit_danger": {s.name.lower(): round(v, 2)
                            for s, v in suit_danger.items()},
        })
    return opponents
