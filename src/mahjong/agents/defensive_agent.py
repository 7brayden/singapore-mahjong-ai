"""
Defensive discard agent — baseline #3.

Strategy: always discard the tile with the lowest danger score,
completely ignoring hand efficiency (shanten / acceptance).

This agent will almost never win because it doesn't try to build
a winning hand. Its purpose is to:
  1. Establish a defense-only baseline
  2. Measure how much deal-in rate drops when playing purely safe
  3. Provide the contrast for the hybrid agent (offense + defense)

Expected behavior:
  - Win rate: very low (not trying to win)
  - Deal-in rate: lowest of all agents
  - Final shanten: high (hand doesn't improve efficiently)
"""

from mahjong.game import BaseAgent
from mahjong.defense import evaluate_danger_all


class DefensiveAgent(BaseAgent):
    """Discards the tile with the lowest danger score (safest discard)."""

    def __init__(self, name: str = "Defensive"):
        super().__init__(name)

    def choose_discard(self, player_idx: int, game_state) -> int:
        hand = game_state.hands[player_idx]

        # Evaluate danger for all tiles in hand — sorted safest first
        danger_evals = evaluate_danger_all(hand.tiles, player_idx, game_state)

        # Pick the safest discard
        return danger_evals[0]["tile_id"]


if __name__ == "__main__":
    from mahjong.game import GameState
    from mahjong.agents.random_agent import RandomAgent
    from mahjong.agents.greedy_agent import GreedyAgent

    total = 100

    # Test 1: 4x Defensive
    print("--- Test 1: 4x Defensive ---")
    wins = 0
    for i in range(total):
        agents = [DefensiveAgent(f"D{j}") for j in range(4)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is not None:
            wins += 1
    print(f"  Games: {total}, Wins: {wins} ({100*wins/total:.1f}%)")

    # Test 2: 1x Defensive vs 3x Random
    print("\n--- Test 2: 1x Defensive vs 3x Random ---")
    def_wins = 0
    rand_wins = 0
    draws = 0
    for i in range(total):
        agents = [DefensiveAgent("Def")] + [RandomAgent(f"R{j}") for j in range(3)]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is None:
            draws += 1
        elif result.winner == 0:
            def_wins += 1
        else:
            rand_wins += 1
    print(f"  Defensive wins: {def_wins} ({100*def_wins/total:.1f}%)")
    print(f"  Random wins:    {rand_wins} ({100*rand_wins/total:.1f}%)")
    print(f"  Draws:          {draws} ({100*draws/total:.1f}%)")

    # Test 3: 1x Greedy vs 1x Defensive vs 2x Random
    print("\n--- Test 3: 1x Greedy vs 1x Defensive vs 2x Random ---")
    greedy_w = 0
    def_w = 0
    rand_w = 0
    draws = 0
    for i in range(total):
        agents = [GreedyAgent("Greedy"), DefensiveAgent("Def"),
                  RandomAgent("R0"), RandomAgent("R1")]
        game = GameState(agents, seed=i)
        result = game.play()
        if result.winner is None:
            draws += 1
        elif result.winner == 0:
            greedy_w += 1
        elif result.winner == 1:
            def_w += 1
        else:
            rand_w += 1
    print(f"  Greedy wins:    {greedy_w} ({100*greedy_w/total:.1f}%)")
    print(f"  Defensive wins: {def_w} ({100*def_w/total:.1f}%)")
    print(f"  Random wins:    {rand_w} ({100*rand_w/total:.1f}%)")
    print(f"  Draws:          {draws} ({100*draws/total:.1f}%)")
