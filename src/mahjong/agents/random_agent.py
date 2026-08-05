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

from mahjong.game import BaseAgent


class RandomAgent(BaseAgent):
    """Discards a random tile from the hand.

    Uses the game's seeded RNG so that seeded games stay reproducible.
    """

    def __init__(self, name: str = "Random"):
        super().__init__(name)

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]
        return game_state.rng.choice(hand.tiles)


if __name__ == "__main__":
    # Quick test: run 100 games of 4 random agents
    from mahjong.game import GameState

    wins = 0
    total = 100
    for i in range(total):
        agents = [RandomAgent(f"Rand_{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is not None:
            wins += 1

    print(f"RandomAgent: {wins}/{total} wins ({100*wins/total:.1f}%)")
