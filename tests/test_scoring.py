"""Tests for tai scoring, payments, win gating, and instant bonuses."""

import pytest

from mahjong.hand import Hand
from mahjong.game import GameState, BaseAgent, run_decision
from mahjong.agents import GreedyAgent
from mahjong.scoring import (
    ScoreConfig, score_win, is_legal_win, compute_win_payments,
)
from mahjong.tiles import WIND_START


def make_hand(tiles, flowers=()):
    hand = Hand()
    for t in tiles:
        hand.add_tile(t)
    for f in flowers:
        hand.add_tile(f)  # bonus tiles are set aside automatically
    return hand


def rules_of(score):
    return sorted(item.rule for item in score.items)


CFG = ScoreConfig()


# ── Individual tai rules ──────────────────────────────────────────────

PING_HU_TILES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 21, 22, 23, 25, 25]


def test_ping_hu_concealed_clean():
    # All chows, non-honor pair, zero bonus tiles, no claimed melds:
    # ping hu 4 + concealed 1 = 5 tai
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["ping_hu", "ping_hu_concealed"]
    assert score.tai == 5
    assert score.value == 16  # 2^(5-1)


def test_ping_hu_exposed_clean():
    # Same shape with a claimed chow: 4 tai, no concealed bonus
    hand = make_hand(PING_HU_TILES[3:])
    hand.add_exposed_meld("chow", [0, 1, 2])
    score = score_win(hand, 5, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["ping_hu"]
    assert score.tai == 4


def test_chou_ping_hu():
    # Ping hu shape holding a flower: degrades to chou ping hu (臭平胡),
    # 1 tai (+1 concealed here = 2)
    hand = make_hand(PING_HU_TILES, flowers=[35])
    score = score_win(hand, 2, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["chou_ping_hu", "ping_hu_concealed"]
    assert score.tai == 2

    exposed = make_hand(PING_HU_TILES[3:], flowers=[35])
    exposed.add_exposed_meld("chow", [0, 1, 2])
    score = score_win(exposed, 5, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["chou_ping_hu"]
    assert score.tai == 1


def test_chicken_hand_scores_zero():
    # Mixed chows and a pong, honor pair, holds a flower: nothing scores
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[40])
    score = score_win(hand, 27, False, 0, WIND_START, CFG)
    assert score.items == []
    assert score.total_tai == 0
    assert not is_legal_win(score, CFG)
    assert is_legal_win(score, ScoreConfig(allow_chicken_hand=True))


def test_all_triplets_half_flush_stack_and_cap():
    # Pongs of 1w/3w/5w/Rd + 7w pair, self-drawn, no flowers:
    # all_triplets 2 + half_flush 2 + dragon 1 + no_bonus 1 = 6 → at cap
    # (self-draw adds no tai at this table — it only changes who pays)
    hand = make_hand([0, 0, 0, 2, 2, 2, 4, 4, 4, 31, 31, 31, 6, 6])
    score = score_win(hand, 4, True, 0, WIND_START, CFG)
    assert score.total_tai == 6
    assert score.tai == 6
    assert score.is_limit
    assert score.value == 32


def test_tai_cap_is_configurable():
    cfg = ScoreConfig(tai_cap=5)
    hand = make_hand([0, 0, 0, 2, 2, 2, 4, 4, 4, 31, 31, 31, 6, 6])
    score = score_win(hand, 4, True, 0, WIND_START, cfg)
    assert score.tai == 5
    assert score.value == 16


def test_base_unit_scales_values():
    cfg = ScoreConfig(base_unit=5)
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, False, 0, WIND_START, cfg)  # concealed ping hu, 5 tai
    assert score.value == 5 * 2 ** 4  # 80 chips


def test_full_flush():
    hand = make_hand([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8],
                     flowers=[35])  # non-seat flower blocks no_bonus only
    score = score_win(hand, 8, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["full_flush"]
    assert score.tai == 4


def test_seat_and_prevailing_wind_stack():
    # Dealer (East seat) with East prevailing: an East pong scores both
    hand = make_hand([27, 27, 27, 0, 1, 2, 3, 4, 5, 9, 10, 11, 22, 22],
                     flowers=[35])
    score = score_win(hand, 22, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["prevailing_wind_triplet", "seat_wind_triplet"]
    assert score.tai == 2


def test_non_seat_wind_pong_scores_nothing():
    # South pong for the East-seat player with East prevailing: 0 tai
    hand = make_hand([28, 28, 28, 0, 1, 2, 3, 4, 5, 9, 10, 11, 22, 22],
                     flowers=[35])
    score = score_win(hand, 22, False, 0, WIND_START, CFG)
    assert score.total_tai == 0


def test_little_three_dragons():
    # Two dragon pongs + dragon pair = 2 (pongs) + 2 (bonus) = 4 tai
    hand = make_hand([31, 31, 31, 32, 32, 32, 33, 33, 0, 1, 2, 12, 13, 14],
                     flowers=[35])
    score = score_win(hand, 14, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["dragon_triplet", "dragon_triplet",
                               "little_three_dragons"]
    assert score.tai == 4


def test_big_three_dragons_is_limit():
    hand = make_hand([31, 31, 31, 32, 32, 32, 33, 33, 33, 0, 1, 2, 9, 9])
    score = score_win(hand, 2, False, 0, WIND_START, CFG)
    assert "big_three_dragons" in rules_of(score)
    assert score.is_limit
    assert score.tai == CFG.tai_cap


def test_all_honors_is_limit():
    hand = make_hand([27, 27, 27, 28, 28, 28, 29, 29, 29, 31, 31, 31, 30, 30])
    score = score_win(hand, 30, False, 0, WIND_START, CFG)
    assert score.is_limit


def test_flowers_and_animals():
    # Seat 0: matching flowers are F1 (34) and F5 (38); Cat (42) is an animal
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 5, 5],
                     flowers=[34, 38, 42])
    score = score_win(hand, 5, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["animal", "seat_flower", "seat_flower"]
    assert score.tai == 3


def test_self_draw_adds_no_tai():
    # House rule: 自摸 changes who pays, not the tai. A chicken shape won
    # by self-draw is still 0 tai — and still can't win.
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[40])
    score = score_win(hand, 27, True, 0, WIND_START, CFG)
    assert score.items == []
    assert not is_legal_win(score, CFG)

    # The old convention is one config line away
    tsumo_tai = ScoreConfig(tai_values={**CFG.tai_values, "self_draw": 1})
    score = score_win(hand, 27, True, 0, WIND_START, tsumo_tai)
    assert rules_of(score) == ["self_draw"]
    assert is_legal_win(score, tsumo_tai)


def test_all_four_animals_bonus():
    # Each animal is 1 tai; the complete set adds one more: 4 + 1 = 5
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[42, 43, 44, 45])
    score = score_win(hand, 27, False, 1, WIND_START, CFG)
    assert rules_of(score) == ["animal", "animal", "animal", "animal",
                               "complete_animals"]
    assert score.tai == 5


def test_ping_hu_tsumo_is_five_tai():
    # Concealed clean ping hu won by self-draw: ping hu 4 + concealed 1
    # = 5 tai, NOT 6 — tsumo adds payment reach, not tai.
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, True, 0, WIND_START, CFG)
    assert rules_of(score) == ["ping_hu", "ping_hu_concealed"]
    assert score.tai == 5


# ── Payments ──────────────────────────────────────────────────────────

def test_tsumo_payments():
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[40])
    score = score_win(hand, 27, True, 0, WIND_START, CFG)  # 1 tai, value 1
    payments = compute_win_payments(score, winner=2, dealt_in_by=None, config=CFG)
    assert payments == [-1, -1, 3, -1]
    assert sum(payments) == 0


def test_ron_shooter_pays_all():
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, False, 0, WIND_START, CFG)  # concealed ping hu, value 16
    payments = compute_win_payments(score, winner=1, dealt_in_by=3, config=CFG)
    assert payments == [0, 48, 0, -48]


def test_ron_shooter_pays_own_share_only():
    cfg = ScoreConfig(shooter_pays_all=False)
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, False, 0, WIND_START, cfg)
    payments = compute_win_payments(score, winner=1, dealt_in_by=3, config=cfg)
    assert payments == [0, 16, 0, -16]


# ── Engine integration: the minimum-tai gate ──────────────────────────

def _chicken_gate_game(score_config=None):
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)],
                     seed=0, score_config=score_config)
    # A live wall, so the discard isn't a "last tile" (which would add tai)
    game.wall = [0] * 40
    # P1: tenpai chicken hand waiting to pair East Wind (has a flower)
    for t in [0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27]:
        game.hands[1].add_tile(t)
    game.hands[1].add_tile(40)
    # P0: junk hand holding the East Wind it will discard
    for t in [27, 3, 7, 9, 15, 17, 20, 22, 28, 29, 30, 31, 5]:
        game.hands[0].add_tile(t)
    return game


def test_chicken_hand_cannot_ron():
    game = _chicken_gate_game()
    game.hands[0].discard(27)
    assert run_decision(game, game._resolve_discard(0, 27)) is True  # continues
    assert not game.game_over
    assert game.deal_ins[0] == 0  # not a legal win → not a deal-in


def test_chicken_hand_can_ron_when_house_allows():
    game = _chicken_gate_game(ScoreConfig(allow_chicken_hand=True))
    game.hands[0].discard(27)
    assert run_decision(game, game._resolve_discard(0, 27)) is False
    assert game.result.winner == 1
    assert game.result.winning_score.total_tai == 0
    # Chicken value = base unit; shooter pays all three shares
    assert game.result.payments == [-3, 3, 0, 0]


# ── Instant bonus payouts ─────────────────────────────────────────────

_PAYOUTS_ON = ScoreConfig(instant_bonus_payouts=True)


def test_animal_pays_instantly():
    # Instant chips are off by default (tai-only accounting); opt in to
    # keep the payout machinery covered.
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0,
                     score_config=_PAYOUTS_ON)
    game.wall = [42] + [0] * 30  # Cat, then padding
    game.wall_idx = 0
    drawn = game._deal_tile_to(0)
    assert drawn == 0
    assert 42 in game.hands[0].flowers
    assert game.payments == [3, -1, -1, -1]


def test_completed_flower_series_pays_instantly():
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0,
                     score_config=_PAYOUTS_ON)
    for f in (34, 35, 36):
        game.hands[0].add_tile(f)
    game.wall = [37] + [0] * 30  # F4 completes the series
    game.wall_idx = 0
    game._deal_tile_to(0)
    assert game.payments == [3, -1, -1, -1]


def test_instant_payouts_can_be_disabled():
    cfg = ScoreConfig(instant_bonus_payouts=False)
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)],
                     seed=0, score_config=cfg)
    game.wall = [42] + [0] * 30
    game.wall_idx = 0
    game._deal_tile_to(0)
    assert game.payments == [0, 0, 0, 0]
