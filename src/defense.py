"""
Defense module for Singapore Mahjong.

Estimates the danger score of each candidate discard — how likely it is
to help an opponent complete their hand (deal-in risk).

Danger signals:
  1. Visibility:       Tiles with fewer visible copies are more dangerous
                       (opponents more likely to be holding/waiting for them)
  2. Discard pattern:  If no opponent has discarded a tile or its neighbors,
                       they may be collecting that suit/area
  3. Opponent threat:  Per-tile danger from opponent modelling — estimates
                       how close each opponent is to winning and whether
                       this specific tile is in a suit they're collecting
  4. Suit safety:      If an opponent has been heavily discarding a suit,
                       remaining tiles in that suit are safer

Each signal produces a score in [0, 1], and the final danger score
is a weighted combination.
"""

from typing import List, Dict, Optional, Tuple

from tiles import (
    NUM_STANDARD_UNIQUE, tile_short, is_numbered, is_honor, is_bonus,
    suit_of, rank_of, neighbors, Suit,
    WAN_START, TONG_START, SUO_START, WIND_START, DRAGON_START
)


# ── Danger weights (tunable) ─────────────────────────────────────────

WEIGHT_VISIBILITY = 0.25    # fewer copies visible = more dangerous
WEIGHT_DISCARD_ABSENCE = 0.25  # tile/neighbors not discarded by opponents
WEIGHT_OPPONENT_THREAT = 0.35  # per-tile threat from opponent modelling
WEIGHT_SUIT_SAFETY = 0.15  # opponent dumping a suit = that suit is safer


def estimate_danger(
    tile_id: int,
    player_idx: int,
    game_state,
    threat_data=None,
) -> float:
    """Estimate the danger score of discarding a specific tile.

    Args:
        tile_id: the tile being considered for discard
        player_idx: which player is discarding
        game_state: current GameState object
        threat_data: pre-computed opponent threat data (optional)

    Returns:
        Danger score in [0.0, 1.0] where higher = more dangerous
    """
    from opponent_model import tile_threat_score

    scores = {
        "visibility": _score_visibility(tile_id, player_idx, game_state),
        "discard_absence": _score_discard_absence(tile_id, player_idx, game_state),
        "opponent_threat": tile_threat_score(tile_id, player_idx, game_state, threat_data),
        "suit_safety": _score_suit_safety(tile_id, player_idx, game_state),
    }

    danger = (
        WEIGHT_VISIBILITY * scores["visibility"]
        + WEIGHT_DISCARD_ABSENCE * scores["discard_absence"]
        + WEIGHT_OPPONENT_THREAT * scores["opponent_threat"]
        + WEIGHT_SUIT_SAFETY * scores["suit_safety"]
    )

    return min(1.0, max(0.0, danger))


def estimate_danger_detailed(
    tile_id: int,
    player_idx: int,
    game_state,
    threat_data=None,
) -> Dict:
    """Same as estimate_danger but returns the full breakdown."""
    from opponent_model import tile_threat_score

    scores = {
        "visibility": _score_visibility(tile_id, player_idx, game_state),
        "discard_absence": _score_discard_absence(tile_id, player_idx, game_state),
        "opponent_threat": tile_threat_score(tile_id, player_idx, game_state, threat_data),
        "suit_safety": _score_suit_safety(tile_id, player_idx, game_state),
    }

    danger = (
        WEIGHT_VISIBILITY * scores["visibility"]
        + WEIGHT_DISCARD_ABSENCE * scores["discard_absence"]
        + WEIGHT_OPPONENT_THREAT * scores["opponent_threat"]
        + WEIGHT_SUIT_SAFETY * scores["suit_safety"]
    )

    return {
        "tile_id": tile_id,
        "danger": min(1.0, max(0.0, danger)),
        "components": scores,
    }


# ── Signal 1: Visibility ─────────────────────────────────────────────

def _score_visibility(tile_id: int, player_idx: int, game_state) -> float:
    """Tiles with fewer visible copies are more dangerous.

    If 3 of 4 copies are visible (in discards, melds, your hand),
    the 4th is unlikely to be a critical wait. But if only 0-1 copies
    are visible, opponents could easily be holding or waiting for it.

    Score: 0 copies visible → 1.0, 4 copies → 0.0
    """
    visible = game_state.get_visible_counts(player_idx)
    copies_visible = visible[tile_id]
    # 4 copies total for standard tiles
    return max(0.0, (4 - copies_visible) / 4)


# ── Signal 2: Discard absence ────────────────────────────────────────

def _score_discard_absence(tile_id: int, player_idx: int, game_state) -> float:
    """If opponents haven't discarded this tile or its neighbors, it's dangerous.

    Logic: if an opponent never discards 4wan, 5wan, or 6wan, they might
    be building a sequence in that area. Discarding 5wan into that is risky.

    For honor tiles, we only check the tile itself (no neighbors).
    """
    # Collect all opponent discards into a set for fast lookup
    opponent_discards = set()
    for p in range(4):
        if p == player_idx:
            continue
        for tile in game_state.hands[p].discards:
            if not is_bonus(tile):
                opponent_discards.add(tile)

    # Check if this tile or its neighbors appear in opponent discards
    tiles_to_check = [tile_id] + neighbors(tile_id)
    found = sum(1 for t in tiles_to_check if t in opponent_discards)

    # If none found among tile + neighbors → very dangerous (1.0)
    # If all found → safe (0.0)
    return 1.0 - (found / len(tiles_to_check))


# ── Signal 3: Suit safety ────────────────────────────────────────────

def _score_suit_safety(tile_id: int, player_idx: int, game_state) -> float:
    """If opponents are dumping tiles in this suit, the suit is safer.

    Logic: if an opponent discards many wan tiles, they probably aren't
    collecting wan, so your wan discards are less likely to help them.

    For honor tiles, we check how many of the same honor are in discards.
    """
    if is_honor(tile_id):
        # For honors: how many copies in all opponent discards?
        copies_in_discards = 0
        for p in range(4):
            if p == player_idx:
                continue
            copies_in_discards += sum(1 for t in game_state.hands[p].discards if t == tile_id)
        # More copies discarded → safer (lower danger)
        # 0 discarded → dangerous (1.0), 3 discarded → safe (0.0)
        return max(0.0, 1.0 - copies_in_discards / 3)

    # For numbered suits: count how many tiles of this suit
    # opponents have discarded
    tile_suit = suit_of(tile_id)
    suit_discard_count = 0
    total_opponent_discards = 0

    for p in range(4):
        if p == player_idx:
            continue
        for t in game_state.hands[p].discards:
            if not is_bonus(t):
                total_opponent_discards += 1
                if suit_of(t) == tile_suit:
                    suit_discard_count += 1

    if total_opponent_discards == 0:
        return 0.5  # no info yet

    # If this suit makes up a large fraction of opponent discards → safer
    # Expected fraction for a random player: ~9/34 ≈ 0.26 for each numbered suit
    suit_fraction = suit_discard_count / total_opponent_discards
    expected_fraction = 9 / 34

    if suit_fraction > expected_fraction * 1.5:
        return 0.2  # opponents dumping this suit — safe
    elif suit_fraction > expected_fraction:
        return 0.4
    elif suit_fraction > expected_fraction * 0.5:
        return 0.6
    else:
        return 0.9  # opponents rarely discard this suit — dangerous


# ── Batch evaluation ──────────────────────────────────────────────────

def evaluate_danger_all(
    candidate_tiles: List[int],
    player_idx: int,
    game_state,
) -> List[Dict]:
    """Evaluate danger for all candidate discard tiles.

    Pre-computes opponent threat data once and reuses it across all tiles.

    Returns list of dicts sorted by danger (safest first):
        [{"tile_id": int, "danger": float, "components": {...}}, ...]
    """
    from opponent_model import estimate_opponent_threats

    # Compute threat data once for all tiles
    threat_data = estimate_opponent_threats(player_idx, game_state)

    seen = set()
    results = []

    for tile_id in candidate_tiles:
        if tile_id in seen or is_bonus(tile_id):
            continue
        seen.add(tile_id)
        results.append(estimate_danger_detailed(tile_id, player_idx, game_state, threat_data))

    results.sort(key=lambda r: r["danger"])
    return results


# ══════════════════════════════════════════════════════════════════════
# SANITY TESTS
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))

    from game import GameState, BaseAgent

    class _RandomAgent(BaseAgent):
        def choose_discard(self, player_idx, game_state):
            return random.choice(game_state.hands[player_idx].tiles)

    print("=== Defense Module Test ===\n")

    # Play a game partway, then evaluate danger for P0's hand
    agents = [_RandomAgent(f"R{i}") for i in range(4)]
    game = GameState(agents, seed=42)
    game.setup()

    # Simulate 30 turns to build up discard history
    for _ in range(30):
        if game.game_over:
            break
        game._execute_turn()

    print(f"Turn: {game.turn}, Wall remaining: {game.tiles_remaining}")
    print(f"P0 hand: {', '.join(tile_short(t) for t in sorted(game.hands[0].tiles))}")
    print(f"P0 discards: {', '.join(tile_short(t) for t in game.hands[0].discards)}")

    # Evaluate danger for each tile in P0's hand
    print(f"\nDanger evaluation for P0's hand:")
    print(f"  {'Tile':<6} {'Danger':<8} {'Visib':<7} {'Absent':<7} "
          f"{'Threat':<7} {'Suit'}")
    print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")

    results = evaluate_danger_all(game.hands[0].tiles, 0, game)
    for r in results:
        c = r["components"]
        print(f"  {tile_short(r['tile_id']):<6} {r['danger']:<8.3f} "
              f"{c['visibility']:<7.2f} {c['discard_absence']:<7.2f} "
              f"{c['opponent_threat']:<7.2f} "
              f"{c['suit_safety']:.2f}")

    print("\n=== Done ===")