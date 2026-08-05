"""
Greedy hand-efficiency agent — baseline #2.

Strategy: always discard the tile that leaves the hand in the
best state, measured by:
  1. Lowest shanten (closest to winning)
  2. Highest tile acceptance (most tiles that further improve the hand)

This is a pure offense agent — it only optimizes its own hand
with zero regard for what opponents are doing. No defense,
no danger awareness, no opponent modelling.

This is the baseline the hybrid agent needs to beat
on deal-in rate and overall win rate.
"""

from mahjong.game import BaseAgent
from mahjong.hand import evaluate_discards, calculate_shanten, best_chow_option


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
        """Claim a pong if it would improve shanten."""
        if claim_type != "pong":
            return False

        hand = game_state.hands[player_idx]
        counts = hand.copy_counts()
        if counts[tile_id] < 2:
            return False

        current_shanten = calculate_shanten(counts, hand.num_exposed_melds)
        counts[tile_id] -= 2
        new_shanten = calculate_shanten(counts, hand.num_exposed_melds + 1)
        return new_shanten < current_shanten

    def choose_chow(self, player_idx: int, tile_id: int,
                    options, game_state):
        """Claim the chow combination that improves shanten the most."""
        hand = game_state.hands[player_idx]
        best_pair, _ = best_chow_option(hand.copy_counts(),
                                        hand.num_exposed_melds, options)
        return best_pair


if __name__ == "__main__":
    from mahjong.game import GameState
    from mahjong.agents.random_agent import RandomAgent

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
