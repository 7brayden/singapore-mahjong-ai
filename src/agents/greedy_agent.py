"""
Greedy hand-efficiency agent — baseline #2.

Strategy: always discard the tile that leaves the hand in the
best state, measured by:
  1. Lowest shanten (closest to winning)
  2. Highest tile acceptance (most tiles that further improve the hand)

This is a pure offense agent — it only optimizes its own hand
with zero regard for what opponents are doing. No defense,
no danger awareness, no opponent modelling.

This is the baseline the hybrid agent (Week 2) needs to beat
on deal-in rate and overall win rate.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game import BaseAgent
from hand import evaluate_discards, calculate_shanten


class GreedyAgent(BaseAgent):
    """Discards the tile that minimizes shanten, then maximizes acceptance."""

    def __init__(self, name: str = "Greedy"):
        super().__init__(name)

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]
        visible = game_state.get_visible_counts(player_idx)
        evals = evaluate_discards(hand, visible_counts=visible)
        return evals[0]["tile_id"]

    def should_claim(self, player_idx: int, tile_id: int,
                     claim_type: str, game_state) -> bool:
        """Claim if it would improve shanten."""
        hand = game_state.hands[player_idx]
        current_shanten = calculate_shanten(hand.copy_counts(), hand.num_exposed_melds)

        # Simulate the claim
        counts = hand.copy_counts()
        if claim_type == "pong":
            # Remove 2 from hand, add the meld (net: -2 concealed, +1 meld)
            if counts[tile_id] < 2:
                return False
            counts[tile_id] -= 2
            new_shanten = calculate_shanten(counts, hand.num_exposed_melds + 1)
        elif claim_type == "chow":
            # Find best chow combination
            from tiles import is_numbered, suit_of, rank_of
            if not is_numbered(tile_id):
                return False
            best_chow_shanten = current_shanten
            rank = rank_of(tile_id)
            found = False
            for p1, p2 in _chow_partners(tile_id, counts):
                test_counts = counts[:]
                test_counts[p1] -= 1
                test_counts[p2] -= 1
                s = calculate_shanten(test_counts, hand.num_exposed_melds + 1)
                if s < best_chow_shanten:
                    best_chow_shanten = s
                    found = True
            if found:
                new_shanten = best_chow_shanten
            else:
                return False
        else:
            return False

        return new_shanten < current_shanten


def _chow_partners(tile_id: int, counts):
    """Yield (partner1, partner2) pairs for possible chows with tile_id."""
    from tiles import rank_of, suit_of
    rank = rank_of(tile_id)
    suit_start = (tile_id // 9) * 9

    # tile is leftmost: [tile, tile+1, tile+2]
    if rank <= 7:
        p1, p2 = tile_id + 1, tile_id + 2
        if suit_of(p1) == suit_of(tile_id) and counts[p1] >= 1 and counts[p2] >= 1:
            yield (p1, p2)
    # tile is middle: [tile-1, tile, tile+1]
    if rank >= 2 and rank <= 8:
        p1, p2 = tile_id - 1, tile_id + 1
        if suit_of(p1) == suit_of(tile_id) and counts[p1] >= 1 and counts[p2] >= 1:
            yield (p1, p2)
    # tile is rightmost: [tile-2, tile-1, tile]
    if rank >= 3:
        p1, p2 = tile_id - 2, tile_id - 1
        if suit_of(p1) == suit_of(tile_id) and counts[p1] >= 1 and counts[p2] >= 1:
            yield (p1, p2)


if __name__ == "__main__":
    import random
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from game import GameState
    from random_agent import RandomAgent

    total = 200

    # Test 1: 4 greedy agents
    print("--- Test 1: 4x Greedy ---")
    wins = 0
    win_turns = []
    for i in range(total):
        agents = [GreedyAgent(f"G{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is not None:
            wins += 1
            win_turns.append(result.turns)

    print(f"  Games: {total}")
    print(f"  Wins:  {wins} ({100*wins/total:.1f}%)")
    if win_turns:
        print(f"  Avg win turn: {sum(win_turns)/len(win_turns):.1f}")

    # Test 2: 1 greedy vs 3 random
    print("\n--- Test 2: 1x Greedy vs 3x Random ---")
    greedy_wins = 0
    random_wins = 0
    draws = 0
    for i in range(total):
        agents = [GreedyAgent("Greedy")] + [RandomAgent(f"Rand_{j}") for j in range(3)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is None:
            draws += 1
        elif result.winner == 0:
            greedy_wins += 1
        else:
            random_wins += 1

    print(f"  Games: {total}")
    print(f"  Greedy wins: {greedy_wins} ({100*greedy_wins/total:.1f}%)")
    print(f"  Random wins: {random_wins} ({100*random_wins/total:.1f}%)")
    print(f"  Draws: {draws} ({100*draws/total:.1f}%)")