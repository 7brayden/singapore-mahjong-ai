"""Feature extraction for the learned models.

One source of truth for feature vectors: the data generator, the
training script, and live inference all call these functions, so a
model trained on generated data sees identical features at play time.

Everything here is computed from information VISIBLE to the acting
seat — the same constraint the heuristic agents live under. Labels
(which use perfect information) live in datagen.py, never here.

The four heuristic danger signals are included as features on purpose:
the logistic regression then learns, from real deal-in outcomes, the
weights that defense.py currently hard-codes as 0.25/0.25/0.35/0.15 —
and the raw tile/state features let it correct them where they're wrong.
"""

from typing import Dict, List

from mahjong.tiles import is_honor, is_numbered, rank_of
from mahjong.defense import estimate_danger_detailed


# ── Danger model: one vector per (state, candidate tile) ─────────────

DANGER_FEATURES = [
    "is_honor",             # 0/1
    "is_terminal",          # 0/1 — numbered rank 1 or 9
    "centrality",           # 0 honors; 0.2 edge .. 1.0 middle (rank 5)
    "copies_visible",       # visible copies of this tile / 4
    "adj1_visible",         # visible copies at rank ±1 / 8 (edges count as dead)
    "adj2_visible",         # visible copies at rank ±2 / 8
    "opp_river_copies",     # copies of this exact tile in opponent rivers / 3
    "sig_visibility",       # the four heuristic signals from defense.py
    "sig_discard_absence",
    "sig_opponent_threat",
    "sig_suit_safety",
    "turn_frac",            # min(1, turn / 60)
    "wall_frac",            # tiles_remaining / 92
    "max_opp_threat",       # opponent_model's strongest-threat estimate
    "opp_max_melds",        # most exposed melds by any opponent / 4
    "opp_total_melds",      # exposed melds across all opponents / 12
]


def danger_features(tile_id: int, player_idx: int, game,
                    visible: List[int], threat_data: Dict) -> List[float]:
    """Feature vector for "how dangerous is discarding this tile here".

    `visible` and `threat_data` are passed in so the caller computes
    them once per decision, not once per candidate tile.
    """
    detail = estimate_danger_detailed(tile_id, player_idx, game, threat_data)
    sig = detail["components"]

    honor = 1.0 if is_honor(tile_id) else 0.0
    if is_numbered(tile_id):
        rank = rank_of(tile_id)
        terminal = 1.0 if rank in (1, 9) else 0.0
        centrality = (5 - abs(rank - 5)) / 5.0
        adj1 = _adjacent_visible(tile_id, rank, visible, 1)
        adj2 = _adjacent_visible(tile_id, rank, visible, 2)
    else:
        terminal = 0.0
        centrality = 0.0
        adj1 = adj2 = 1.0  # honors have no neighbours to worry about

    river_copies = 0
    for p in range(4):
        if p == player_idx:
            continue
        river_copies += sum(1 for t in game.hands[p].discards if t == tile_id)

    opp_melds = [game.hands[p].num_exposed_melds
                 for p in range(4) if p != player_idx]

    return [
        honor,
        terminal,
        centrality,
        visible[tile_id] / 4.0,
        adj1,
        adj2,
        min(1.0, river_copies / 3.0),
        sig["visibility"],
        sig["discard_absence"],
        sig["opponent_threat"],
        sig["suit_safety"],
        min(1.0, game.turn / 60.0),
        min(1.0, game.tiles_remaining / 92.0),
        threat_data["max_threat"],
        max(opp_melds) / 4.0,
        sum(opp_melds) / 12.0,
    ]


def _adjacent_visible(tile_id: int, rank: int, visible: List[int],
                      distance: int) -> float:
    """Visible copies at rank ± distance, normalised to [0, 1].

    Off-the-end ranks count as fully visible (4 copies): a 1-wan has no
    0-wan for an opponent to be waiting through, which is exactly as
    safe as all four copies being dead.
    """
    total = 0
    for delta in (-distance, distance):
        r = rank + delta
        if 1 <= r <= 9:
            total += visible[tile_id + delta]
        else:
            total += 4
    return total / 8.0


# ── Outcome model: one vector per decision state ─────────────────────

OUTCOME_FEATURES = [
    "shanten",              # best reachable shanten this turn / 6
    "best_acceptance",      # tile acceptance of the best discard / 30
    "turn_frac",
    "wall_frac",
    "own_melds",            # own exposed melds / 4
    "own_bonus",            # own flowers + animals collected / 8
    "max_opp_threat",
    "opp_total_melds",
]


def outcome_features(player_idx: int, game, best_shanten: int,
                     best_acceptance: int, threat_data: Dict) -> List[float]:
    """Feature vector for "how is this hand going" — trains P(win)."""
    hand = game.hands[player_idx]
    opp_melds = sum(game.hands[p].num_exposed_melds
                    for p in range(4) if p != player_idx)
    return [
        best_shanten / 6.0,
        min(1.0, best_acceptance / 30.0),
        min(1.0, game.turn / 60.0),
        min(1.0, game.tiles_remaining / 92.0),
        hand.num_exposed_melds / 4.0,
        min(1.0, len(hand.flowers) / 8.0),
        threat_data["max_threat"],
        opp_melds / 12.0,
    ]
