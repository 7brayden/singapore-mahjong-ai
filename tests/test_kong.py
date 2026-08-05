"""Tests for kong mechanics: declarations, claims, robbing, and kong tai.

Engine tests construct game states by hand (no setup()) and give them a
padded wall so draws and last-tile logic behave normally.
"""

import pytest

from mahjong.hand import Hand
from mahjong.game import GameState, BaseAgent, run_decision
from mahjong.agents import GreedyAgent
from mahjong.scoring import ScoreConfig, score_win
from mahjong.tiles import WIND_START


def make_hand(tiles):
    hand = Hand()
    for t in tiles:
        hand.add_tile(t)
    return hand


def make_game(agents=None, score_config=None, wall=None):
    game = GameState(agents or [GreedyAgent(f"G{i}") for i in range(4)],
                     seed=0, score_config=score_config)
    game.wall = wall if wall is not None else [0] * 40
    return game


# ── Hand-level kong operations ────────────────────────────────────────

def test_declare_concealed_kong():
    hand = make_hand([4, 4, 4, 4, 9, 10, 11, 20, 21, 22, 27, 27, 31])
    assert ("concealed", 4) in hand.kong_options()
    hand.declare_kong("concealed", 4)
    assert hand.counts[4] == 0
    assert hand.exposed == [("concealed_kong", [4, 4, 4, 4])]
    assert hand.num_exposed_melds == 1
    assert len(hand.tiles) == 9


def test_declare_added_kong():
    hand = make_hand([4, 9, 10, 11, 20, 21, 22, 27, 27, 31])
    hand.add_exposed_meld("pong", [4, 4, 4])
    assert ("added", 4) in hand.kong_options()
    hand.declare_kong("added", 4)
    assert hand.exposed == [("kong", [4, 4, 4, 4])]
    assert hand.num_exposed_melds == 1
    assert hand.counts[4] == 0


def test_declare_exposed_kong_from_claim():
    hand = make_hand([4, 4, 4, 9, 10, 11, 20, 21, 22, 27, 27, 31, 5])
    hand.declare_kong("exposed", 4)  # fourth copy comes from the discard
    assert hand.exposed == [("kong", [4, 4, 4, 4])]
    assert hand.counts[4] == 0
    assert len(hand.tiles) == 10


# ── Agent kong policy ─────────────────────────────────────────────────

def test_greedy_declares_harmless_kong():
    game = make_game()
    hand = game.hands[1]
    # Four 5w plus two runs, a pair, and isolated honors: the fourth 5w
    # serves nothing, so the kong costs no shanten
    for t in [4, 4, 4, 4, 9, 10, 11, 20, 21, 22, 27, 27, 31, 33]:
        hand.add_tile(t)
    choice = game.agents[1].choose_kong(1, hand.kong_options(), game)
    assert choice == ("concealed", 4)


def test_greedy_declines_kong_that_breaks_hand():
    game = make_game()
    hand = game.hands[1]
    # 3w x4 works as chow(1w2w3w) + chow(3w4w5w) + pair 3w3w — konging
    # all four destroys two runs and the pair.
    for t in [2, 2, 2, 2, 0, 1, 3, 4, 9, 9, 9, 20, 21, 24]:
        hand.add_tile(t)
    choice = game.agents[1].choose_kong(1, hand.kong_options(), game)
    assert choice is None


# ── Engine: kong claim from a discard ─────────────────────────────────

def test_kong_claim_draws_replacement_and_discards():
    game = make_game()
    # P1 holds three 5w; P0 discards the fourth
    for t in [4, 4, 4, 9, 13, 17, 18, 22, 26, 28, 30, 31, 33]:
        game.hands[1].add_tile(t)
    for t in [4, 0, 2, 6, 10, 14, 19, 23, 25, 27, 29, 32, 5]:
        game.hands[0].add_tile(t)

    wall_before = game.tiles_remaining
    game.hands[0].discard(4)
    assert run_decision(game, game._resolve_discard(0, 4)) is True

    assert game.hands[1].exposed == [("kong", [4, 4, 4, 4])]
    # Drew one replacement, then discarded: 13 - 3 + 1 - 1 = 10 concealed
    assert len(game.hands[1].tiles) == 10
    assert game.tiles_remaining == wall_before - 1
    assert len(game.hands[1].discards) == 1
    # Kong payout: 1 base unit from each player
    assert game.payments[1] == 3
    assert sum(game.payments) == 0
    assert game.active_player == 2


# ── Engine: kong-draw win (杠上开花) ──────────────────────────────────

def test_kong_replacement_win_scores_kong_draw():
    game = make_game(wall=[24, 23] + [0] * 40)  # draw 8s, replacement 6s
    game.active_player = 0
    # P0: 5w x4 + 1w2w3w + 1t pong + 9s pair + 8s → after drawing a second
    # 8s: kong 5w, replacement 6s completes 6s7s8s... build the wait:
    for t in [4, 4, 4, 4, 0, 1, 2, 9, 9, 9, 26, 26, 25]:
        game.hands[0].add_tile(t)

    assert run_decision(game, game._execute_turn()) is False
    assert game.result.winner == 0
    assert game.result.win_type == "tsumo"
    rules = [item.rule for item in game.result.winning_score.items]
    assert "kong_draw" in rules
    assert "self_draw" in rules
    # Concealed kong payout (2 units each) happened before the win
    assert sum(game.result.payments) == 0


# ── Engine: robbing the kong (抢杠) ───────────────────────────────────

def test_added_kong_can_be_robbed():
    game = make_game(wall=[33] + [0] * 40)  # P1 draws junk (White Dragon)
    game.active_player = 1
    # P1: exposed pong of 3t, holds the fourth 3t (isolated — no
    # neighbours in hand, so the added kong costs nothing) plus junk
    game.hands[1].add_exposed_meld("pong", [11, 11, 11])
    for t in [11, 0, 4, 8, 15, 17, 20, 24, 28, 32]:
        game.hands[1].add_tile(t)
    # P2: pure ping-hu shape waiting on 3t (1t2t + 5s pair + three runs)
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 22, 22, 9, 10]:
        game.hands[2].add_tile(t)

    assert run_decision(game, game._execute_turn()) is False
    assert game.result.winner == 2
    assert game.result.win_type == "ron"
    assert game.result.dealt_in_by == 1
    rules = [item.rule for item in game.result.winning_score.items]
    assert "rob_kong" in rules
    assert "ping_hu" in rules
    assert game.deal_ins[1] == 1
    # The robbed tile moved from P1's hand to P2's
    assert game.hands[1].counts[11] == 0
    assert game.hands[2].counts[11] == 1


# ── Last tile (海底捞月) ──────────────────────────────────────────────

def test_last_tile_tsumo_scores_last_tile():
    # Wall of exactly 16: one live draw remains (dead wall = 15)
    game = make_game(wall=[10] + [0] * 15)
    game.active_player = 0
    # P0 tenpai waiting on 2t (pair wait after 1w-9w runs + 1t pong)
    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10]:
        game.hands[0].add_tile(t)

    assert run_decision(game, game._execute_turn()) is False
    assert game.result.winner == 0
    rules = [item.rule for item in game.result.winning_score.items]
    assert "last_tile" in rules
    assert "self_draw" in rules


# ── Scoring units ─────────────────────────────────────────────────────

def test_kong_counts_as_pong_for_patterns():
    # All-triplets hand where one triplet is a concealed kong
    hand = make_hand([2, 2, 2, 6, 6, 6, 31, 31, 31, 8, 8])
    hand.add_tile(0)
    hand.counts[0] += 3  # simulate 4 copies then kong
    hand.tiles.extend([0, 0, 0])
    hand.declare_kong("concealed", 0)
    score = score_win(hand, 8, False, 0, WIND_START, ScoreConfig())
    rules = [item.rule for item in score.items]
    assert "all_triplets" in rules
    assert "dragon_triplet" in rules


def test_kong_payouts_can_be_disabled():
    cfg = ScoreConfig(instant_kong_payouts=False)
    game = make_game(score_config=cfg)
    for t in [4, 4, 4, 4, 9, 10, 11, 20, 21, 22, 27, 27, 31, 5]:
        game.hands[1].add_tile(t)
    game.hands[1].declare_kong("concealed", 4)
    game._apply_kong_payout(1, "concealed")
    assert game.payments == [0, 0, 0, 0]


# ── Visible counts include own melds (regression) ─────────────────────

def test_visible_counts_include_own_exposed_melds():
    game = make_game()
    hand = game.hands[0]
    for t in [9, 10, 11, 20, 21, 22, 27, 27, 31, 5]:
        hand.add_tile(t)
    hand.add_exposed_meld("pong", [4, 4, 4])
    visible = game.get_visible_counts(0)
    assert visible[4] == 3