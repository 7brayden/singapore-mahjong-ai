"""Behavioral tests for the AI agents."""

from mahjong.game import GameState
from mahjong.agents import RandomAgent, GreedyAgent, HybridAgent


def test_greedy_beats_random():
    greedy_wins = 0
    random_wins = 0
    for seed in range(10):
        agents = [GreedyAgent("G")] + [RandomAgent(f"R{j}") for j in range(3)]
        result = GameState(agents, seed=seed).play()
        if result.winner == 0:
            greedy_wins += 1
        elif result.winner is not None:
            random_wins += 1
    assert greedy_wins > random_wins


def test_hybrid_declines_chow_when_far_from_winning():
    # Scattered hand (shanten well above the claim gate) with a valid
    # chow option for a discarded 5w — hybrid should still pass.
    game = GameState([HybridAgent(f"H{i}") for i in range(4)], seed=0)
    hand = game.hands[1]
    for t in [5, 6, 9, 13, 17, 18, 22, 26, 27, 28, 31, 32, 33]:
        hand.add_tile(t)

    choice = game.agents[1].choose_chow(1, 4, [(5, 6)], game)
    assert choice is None


def test_greedy_accepts_improving_chow():
    # Same shape as the engine-level chow test, from the agent's view:
    # only (6w,7w) improves shanten, and greedy has no claim gate.
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    hand = game.hands[1]
    for t in [1, 2, 3, 5, 6, 8, 9, 9, 9, 20, 21, 22, 25]:
        hand.add_tile(t)

    options = [(2, 3), (3, 5), (5, 6)]
    choice = game.agents[1].choose_chow(1, 4, options, game)
    assert choice == (5, 6)


def test_agents_only_discard_tiles_they_hold():
    # The engine raises if an agent discards a tile not in its hand,
    # so completing games without errors covers discard legality.
    for seed in range(3):
        agents = [GreedyAgent("G"), HybridAgent("H"),
                  RandomAgent("R0"), RandomAgent("R1")]
        result = GameState(agents, seed=seed).play()
        assert result is not None
