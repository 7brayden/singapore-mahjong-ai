"""
Random discard agent — baseline #1.

Discards a uniformly random tile from the hand.
This is the weakest possible agent and serves as the
bottom-line baseline for all experiments.

Expected behavior:
  - Win rate: ~0% (confirmed over 1000 games)
  - Final shanten: typically 3-4
  - No strategic reasoning whatsoever
"""

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from game import BaseAgent


class RandomAgent(BaseAgent):
    """Discards a random tile from the hand."""

    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]
        return random.choice(hand.tiles)


if __name__ == "__main__":
    # Quick test: run 100 games of 4 random agents
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from game import GameState

    wins = 0
    total = 100
    for i in range(total):
        agents = [RandomAgent(f"Rand_{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is not None:
            wins += 1

    print(f"RandomAgent: {wins}/{total} wins ({100*wins/total:.1f}%)")