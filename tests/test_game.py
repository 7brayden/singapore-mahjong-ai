"""Tests for the game engine: dealing, determinism, claims, and golden games.

Some tests call private GameState methods (_resolve_discard, _check_chow)
directly to exercise claim mechanics with constructed hands.
"""

import pytest

from mahjong.game import GameState, BaseAgent
from mahjong.agents import RandomAgent, GreedyAgent, DefensiveAgent, HybridAgent


class ScriptedAgent(BaseAgent):
    """Discards from a fixed script and pongs a fixed set of tiles."""

    def __init__(self, name, discards=None, pong_tiles=()):
        super().__init__(name)
        self.discard_script = list(discards or [])
        self.pong_tiles = set(pong_tiles)

    def choose_discard(self, player_idx, game_state):
        hand = game_state.hands[player_idx]
        if self.discard_script:
            return self.discard_script.pop(0)
        return hand.tiles[0]

    def should_claim(self, player_idx, tile_id, claim_type, game_state):
        return claim_type == "pong" and tile_id in self.pong_tiles


# ── Dealing ───────────────────────────────────────────────────────────

def test_deal_gives_each_player_13_tiles():
    game = GameState([RandomAgent(f"R{i}") for i in range(4)], seed=5)
    game.setup()
    total_flowers = sum(len(h.flowers) for h in game.hands)
    for hand in game.hands:
        assert len(hand.tiles) == 13
        assert sum(hand.counts) == 13
    # Wall consumed: 52 dealt standard tiles + one replacement per flower
    assert game.tiles_remaining == 148 - 52 - total_flowers


# ── Determinism ───────────────────────────────────────────────────────

def _play(seed, agent_factories):
    game = GameState([f(f"P{i}") for i, f in enumerate(agent_factories)], seed=seed)
    result = game.play()
    return game, result


def test_same_seed_reproduces_entire_game():
    factories = [GreedyAgent, HybridAgent, DefensiveAgent, RandomAgent]
    g1, r1 = _play(11, factories)
    g2, r2 = _play(11, factories)
    assert g1.wall == g2.wall
    assert (r1.winner, r1.win_type, r1.turns, r1.tiles_remaining) == \
           (r2.winner, r2.win_type, r2.turns, r2.tiles_remaining)
    assert [h.discards for h in g1.hands] == [h.discards for h in g2.hands]


def test_different_seeds_give_different_walls():
    g1 = GameState([RandomAgent(f"R{i}") for i in range(4)], seed=1)
    g2 = GameState([RandomAgent(f"R{i}") for i in range(4)], seed=2)
    g1.setup()
    g2.setup()
    assert g1.wall != g2.wall


# ── Chow claim interface ──────────────────────────────────────────────

def test_check_chow_executes_agents_chosen_combination():
    # P1 holds 2w3w4w (complete run) + 6w7w. For a discarded 5w the FIRST
    # option (3w,4w) would break the run without improving shanten; the
    # improving option is (6w,7w). The engine must take the agent's pick,
    # not the first valid combination.
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    hand = game.hands[1]
    for t in [1, 2, 3, 5, 6, 8, 9, 9, 9, 20, 21, 22, 25]:
        hand.add_tile(t)

    result = game._check_chow(0, 4)  # P0 discards 5w; P1 may chow
    assert result == (1, [5, 6])


def test_check_chow_passes_when_no_option_improves():
    # P1 holds only a completed run around the discard — claiming would
    # break it, so a shanten-driven agent should pass.
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    hand = game.hands[1]
    for t in [1, 2, 3, 9, 9, 9, 14, 14, 14, 20, 21, 22, 25]:
        hand.add_tile(t)

    assert game._check_chow(0, 4) is None


def test_check_chow_rejects_invalid_agent_choice():
    class CheatingAgent(BaseAgent):
        def choose_chow(self, player_idx, tile_id, options, game_state):
            return (0, 1)  # not a valid option for the discarded tile

    agents = [CheatingAgent(f"C{i}") for i in range(4)]
    game = GameState(agents, seed=0)
    hand = game.hands[1]
    for t in [5, 6, 9, 13, 17, 18, 22, 26, 27, 28, 31, 32, 33]:
        hand.add_tile(t)

    with pytest.raises(ValueError):
        game._check_chow(0, 4)


# ── Nested claim windows ──────────────────────────────────────────────

def test_claim_window_opens_on_claimers_discard():
    # P0 discards 1w → P1 pongs and discards 2t → P2 pongs 2t and discards
    # West Wind. Before the fix, P2's pong window never opened.
    agents = [
        ScriptedAgent("P0"),
        ScriptedAgent("P1", discards=[10], pong_tiles={0}),
        ScriptedAgent("P2", discards=[29], pong_tiles={10}),
        ScriptedAgent("P3"),
    ]
    game = GameState(agents, seed=0)

    hands = [
        [0, 2, 6, 9, 13, 17, 18, 22, 26, 27, 31, 4, 24],
        [0, 0, 10, 1, 5, 12, 16, 19, 23, 28, 32, 8, 25],
        [10, 10, 3, 7, 14, 15, 20, 26, 29, 33, 21, 6, 27],
        [2, 4, 8, 9, 12, 18, 22, 25, 30, 31, 5, 13, 17],
    ]
    for p, tiles in enumerate(hands):
        for t in tiles:
            game.hands[p].add_tile(t)

    game.hands[0].discard(0)
    assert game._resolve_discard(0, 0) is True

    assert game.hands[1].exposed == [("pong", [0, 0, 0])]
    assert game.hands[2].exposed == [("pong", [10, 10, 10])]
    # Claimed tiles were removed from the discard piles
    assert game.hands[0].discards == []
    assert game.hands[1].discards == []
    assert game.hands[2].discards == [29]
    # Claimers hold 10 concealed tiles after pong + discard
    assert len(game.hands[1].tiles) == 10
    assert len(game.hands[2].tiles) == 10
    # Play continues after the last claimer
    assert not game.game_over
    assert game.active_player == 3


# ── Golden regression games ───────────────────────────────────────────
#
# Pinned results of seeded games. If engine behavior changes on purpose
# (new rules, fixed bugs), regenerate these values and note why in the
# commit message.

GOLDEN_GREEDY_MIRROR = [
    (0, 3, "ron", 53, 31),
    (1, 1, "ron", 22, 69),
    (2, 3, "ron", 29, 59),
]

GOLDEN_MIXED_FIELD = [
    (0, 0, "ron", 50, 34),
    (1, 1, "ron", 50, 38),
]


@pytest.mark.parametrize("seed,winner,win_type,turns,remaining", GOLDEN_GREEDY_MIRROR)
def test_golden_greedy_mirror(seed, winner, win_type, turns, remaining):
    result = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=seed).play()
    assert (result.winner, result.win_type, result.turns, result.tiles_remaining) == \
           (winner, win_type, turns, remaining)


@pytest.mark.parametrize("seed,winner,win_type,turns,remaining", GOLDEN_MIXED_FIELD)
def test_golden_mixed_field(seed, winner, win_type, turns, remaining):
    agents = [GreedyAgent("G"), HybridAgent("H"), DefensiveAgent("D"), RandomAgent("R")]
    result = GameState(agents, seed=seed).play()
    assert (result.winner, result.win_type, result.turns, result.tiles_remaining) == \
           (winner, win_type, turns, remaining)
