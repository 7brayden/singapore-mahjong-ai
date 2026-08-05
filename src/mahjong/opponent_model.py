"""
Opponent threat modelling for Singapore Mahjong.

Estimates how close each opponent is to winning and how dangerous
a specific tile is to each opponent, using only visible information.

Threat signals per opponent:
  1. Exposed melds: more exposed melds = fewer concealed tiles needed
  2. Discard efficiency: keeping more drawn tiles = hand is improving
  3. Late safe discards: clearing honors/terminals late = likely tenpai
  4. Suit concentration: avoiding a suit in discards = collecting it

The per-tile threat score combines:
  - Each opponent's threat level (how close to winning?)
  - Whether THIS tile is in a suit that opponent wants
"""

from typing import List, Dict, Optional, Tuple

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, tile_short, is_numbered, is_honor, is_bonus,
    suit_of, rank_of, Suit,
    WAN_START, TONG_START, SUO_START, WIND_START, DRAGON_START
)


# ── Per-opponent threat estimation ────────────────────────────────────

def estimate_opponent_threats(player_idx: int, game_state) -> Dict:
    """Compute threat data for all opponents.

    Returns a dict with:
        "threats": list of 4 floats (threat level per player, 0.0-1.0)
        "suit_danger": list of 4 dicts mapping Suit -> danger score
        "max_threat": float (highest opponent threat)
        "most_threatening": int (player index of most threatening opponent)
    """
    threats = [0.0] * 4
    suit_danger = [{} for _ in range(4)]

    for opp in range(4):
        if opp == player_idx:
            continue

        opp_hand = game_state.hands[opp]
        opp_discards = opp_hand.discards

        # ── Signal A: Exposed melds (strongest signal) ────────────
        # More exposed melds = fewer tiles needed to complete
        exposed_score = min(0.9, opp_hand.num_exposed_melds * 0.3)

        # ── Signal B: Discard efficiency ──────────────────────────
        # Fewer discards than expected = keeping more drawn tiles
        expected_discards = game_state.turn // 4
        actual_discards = len(opp_discards)
        if expected_discards > 0:
            efficiency_ratio = actual_discards / expected_discards
            efficiency_score = max(0.0, min(1.0, 1.0 - efficiency_ratio))
        else:
            efficiency_score = 0.0

        # ── Signal C: Late safe tile discards ─────────────────────
        # Discarding honors/terminals late = clearing safe tiles while tenpai
        late_safe_score = 0.0
        if len(opp_discards) >= 5:
            recent = opp_discards[-3:]
            safe_count = sum(1 for t in recent
                           if is_honor(t) or (is_numbered(t) and rank_of(t) in (1, 9)))
            late_safe_score = safe_count / 3.0
            if game_state.turn < 20:
                late_safe_score *= 0.3

        # ── Signal D: Suit concentration ──────────────────────────
        # Track which suits opponent avoids discarding (= collecting)
        suit_counts = {Suit.WAN: 0, Suit.TONG: 0, Suit.SUO: 0}
        honor_count = 0
        total_discards = 0

        for t in opp_discards:
            if is_bonus(t):
                continue
            total_discards += 1
            s = suit_of(t)
            if s in suit_counts:
                suit_counts[s] += 1
            else:
                honor_count += 1

        # Per-suit danger: low discard fraction = likely collecting
        for s in [Suit.WAN, Suit.TONG, Suit.SUO]:
            if total_discards >= 4:
                fraction = suit_counts[s] / total_discards
                expected = 9 / 34  # ~0.265
                if fraction < expected * 0.3:
                    suit_danger[opp][s] = 0.9
                elif fraction < expected * 0.7:
                    suit_danger[opp][s] = 0.6
                elif fraction < expected:
                    suit_danger[opp][s] = 0.3
                else:
                    suit_danger[opp][s] = 0.1
            else:
                suit_danger[opp][s] = 0.5

        # Honor danger
        if total_discards >= 4:
            honor_fraction = honor_count / total_discards
            expected_honor = 7 / 34
            if honor_fraction < expected_honor * 0.3:
                honor_danger_val = 0.8
            elif honor_fraction > expected_honor * 1.5:
                honor_danger_val = 0.2
            else:
                honor_danger_val = 0.5
            suit_danger[opp][Suit.WIND] = honor_danger_val
            suit_danger[opp][Suit.DRAGON] = honor_danger_val
        else:
            suit_danger[opp][Suit.WIND] = 0.5
            suit_danger[opp][Suit.DRAGON] = 0.5

        # ── Combine into overall threat ───────────────────────────
        threat = (
            0.35 * exposed_score
            + 0.20 * efficiency_score
            + 0.25 * late_safe_score
            + 0.20 * _suit_concentration_score(suit_counts, total_discards)
        )

        # Game phase multiplier: threats more meaningful later
        phase = min(1.0, game_state.turn / 60.0)
        threat = threat * (0.3 + 0.7 * phase)

        threats[opp] = min(1.0, max(0.0, threat))

    # Find most threatening opponent
    max_threat = 0.0
    most_threatening = -1
    for opp in range(4):
        if opp == player_idx:
            continue
        if threats[opp] > max_threat:
            max_threat = threats[opp]
            most_threatening = opp

    return {
        "threats": threats,
        "suit_danger": suit_danger,
        "max_threat": max_threat,
        "most_threatening": most_threatening,
    }


def _suit_concentration_score(suit_counts: Dict, total_discards: int) -> float:
    """How concentrated are discards? High variance = collecting one suit."""
    if total_discards < 4:
        return 0.0

    fractions = [suit_counts.get(s, 0) / total_discards
                 for s in [Suit.WAN, Suit.TONG, Suit.SUO]]

    mean = sum(fractions) / len(fractions)
    variance = sum((f - mean) ** 2 for f in fractions) / len(fractions)

    # Normalize: max variance ≈ 0.07 when all discards from one suit
    return min(1.0, variance / 0.05)


# ── Per-tile threat score ─────────────────────────────────────────────

def tile_threat_score(tile_id: int, player_idx: int, game_state,
                      threat_data: Optional[Dict] = None) -> float:
    """Score how dangerous a specific tile is based on opponent threats.

    Combines each opponent's threat level with how much they want
    this tile's suit. A high-threat opponent collecting this suit
    = very dangerous discard.

    Returns danger score in [0.0, 1.0].
    """
    if threat_data is None:
        threat_data = estimate_opponent_threats(player_idx, game_state)

    tile_suit = suit_of(tile_id)
    max_danger = 0.0

    for opp in range(4):
        if opp == player_idx:
            continue

        opp_threat = threat_data["threats"][opp]
        suit_want = threat_data["suit_danger"][opp].get(tile_suit, 0.5)

        # Opponent threat × how much they want this suit
        tile_danger = opp_threat * suit_want
        max_danger = max(max_danger, tile_danger)

    return min(1.0, max_danger)


# ══════════════════════════════════════════════════════════════════════
# SANITY TESTS
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random

    from mahjong.game import GameState, BaseAgent, advance_turns
    from mahjong.tiles import hand_to_str

    class _RandomAgent(BaseAgent):
        def choose_discard(self, player_idx, game_state):
            return random.choice(game_state.hands[player_idx].tiles)

    print("=== Opponent Threat Model Test ===\n")

    agents = [_RandomAgent(f"R{i}") for i in range(4)]
    game = GameState(agents, seed=42)
    game.setup()

    # Simulate 35 turns
    advance_turns(game, 35)

    print(f"Turn: {game.turn}, Wall remaining: {game.tiles_remaining}\n")

    for p in range(4):
        h = game.hands[p]
        discards_str = ", ".join(tile_short(t) for t in h.discards)
        print(f"  P{p} discards ({len(h.discards)}): {discards_str}")
        print(f"  P{p} exposed melds: {h.num_exposed_melds}")

    # Estimate threats from P0's perspective
    print(f"\n--- Threat estimation (P0's view) ---")
    threat_data = estimate_opponent_threats(0, game)

    for opp in range(4):
        if opp == 0:
            continue
        t = threat_data["threats"][opp]
        sd = threat_data["suit_danger"][opp]
        suit_strs = []
        for s_name, s_val in [("Wan", Suit.WAN), ("Tong", Suit.TONG),
                               ("Suo", Suit.SUO), ("Wind", Suit.WIND)]:
            suit_strs.append(f"{s_name}={sd.get(s_val, 0):.2f}")
        print(f"  P{opp}: threat={t:.3f} | {', '.join(suit_strs)}")

    print(f"\n  Max threat: {threat_data['max_threat']:.3f} "
          f"(P{threat_data['most_threatening']})")

    # Per-tile threat scores for P0's hand
    print(f"\n--- Per-tile threat for P0's hand ---")
    print(f"  Hand: {hand_to_str(game.hands[0].tiles, short=True)}")
    for tile_id in sorted(set(game.hands[0].tiles)):
        if is_bonus(tile_id):
            continue
        score = tile_threat_score(tile_id, 0, game, threat_data)
        print(f"  {tile_short(tile_id):<5} threat={score:.3f}")

    print("\n=== Done ===")