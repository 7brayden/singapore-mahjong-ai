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
from mahjong.ml.model import load_danger_model, load_win_model

# Trained models (None until the ML pipeline has been run once).
# Unlike the heuristic danger score, their outputs are calibrated
# probabilities: "this discard deals in X% of the time", "this hand
# wins Y% of the time from here".
_danger_model = load_danger_model()
_win_model = load_win_model()


def analyze_seat(game: GameState, seat: int) -> Dict:
    """Full decision analysis from one seat's point of view.

    Uses only information visible to that seat. With 14 tiles (about to
    discard) every discard option is scored; with 13 tiles the hand's
    waits are reported instead.
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
            if _win_model is not None:
                w = outcome_features(seat, game, e["shanten"],
                                     e["acceptance"], threat_data)
                entry["win_prob"] = round(_win_model.predict(w), 4)
            discards.append(entry)
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
