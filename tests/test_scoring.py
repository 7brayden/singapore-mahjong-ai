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


def test_ping_hu_wait_restriction():
    # This concealed shape waits on ONE tile (3w completing 1w2w).
    # PDF rule: a single wait earns ping hu only on self-draw — by
    # discard the 4 tai do not count, leaving a 0-tai hand.
    hand = make_hand(PING_HU_TILES)
    ron = score_win(hand, 2, False, 0, WIND_START, CFG)
    assert ron.items == []

    # Self-drawn: ping hu 4 + fully concealed (门清) 1 = 5 tai
    tsumo = score_win(hand, 2, True, 0, WIND_START, CFG)
    assert rules_of(tsumo) == ["fully_concealed", "ping_hu"]
    assert tsumo.tai == 5
    assert tsumo.value == 16  # 2^(5-1)


def test_ping_hu_exposed_clean():
    # Same shape with a claimed chow: 4 tai, no concealed bonus
    hand = make_hand(PING_HU_TILES[3:])
    hand.add_exposed_meld("chow", [0, 1, 2])
    score = score_win(hand, 5, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["ping_hu"]
    assert score.tai == 4


def test_chou_ping_hu():
    # Ping hu shape holding a flower degrades to chou ping hu (臭平胡).
    # Single wait → only by self-draw; concealed tsumo adds 门清: 1+1=2
    hand = make_hand(PING_HU_TILES, flowers=[35])
    score = score_win(hand, 2, True, 0, WIND_START, CFG)
    assert rules_of(score) == ["chou_ping_hu", "fully_concealed"]
    assert score.tai == 2

    # Claimed chow, two-sided wait (3w/6w): chou ping hu by discard OK
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
    score = score_win(hand, 2, True, 0, WIND_START, cfg)  # ping hu + 门清, 5 tai
    assert score.value == 5 * 2 ** 4  # 80 chips


def test_full_flush():
    # Deliberately NOT the nine-gates pattern (see next test)
    hand = make_hand([0, 0, 0, 1, 2, 3, 3, 4, 5, 5, 6, 7, 8, 8],
                     flowers=[35])
    score = score_win(hand, 8, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["full_flush"]
    assert score.tai == 4


def test_nine_gates_is_limit():
    # 1112345678999 + one extra of the suit, fully concealed: 九连宝灯
    hand = make_hand([0, 0, 0, 1, 2, 3, 4, 5, 6, 6, 7, 8, 8, 8])
    score = score_win(hand, 6, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["nine_gates"]
    assert score.is_limit


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
    # PDF: 3 total — 1 per dragon pong + 1 for the dragon pair
    hand = make_hand([31, 31, 31, 32, 32, 32, 33, 33, 0, 1, 2, 12, 13, 14],
                     flowers=[35])
    score = score_win(hand, 14, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["dragon_triplet", "dragon_triplet",
                               "little_three_dragons"]
    assert score.tai == 3


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
    # House rule: 自摸 changes who pays, not the tai. But a concealed
    # self-drawn hand now earns 门清 (PDF) — that's the only tai here.
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[40])
    score = score_win(hand, 27, True, 0, WIND_START, CFG)
    assert rules_of(score) == ["fully_concealed"]
    assert score.tai == 1

    # Restoring tsumo-as-tai is one config line away
    tsumo_tai = ScoreConfig(tai_values={**CFG.tai_values, "self_draw": 1})
    score = score_win(hand, 27, True, 0, WIND_START, tsumo_tai)
    assert rules_of(score) == ["fully_concealed", "self_draw"]
    assert score.tai == 2


def test_all_four_animals_bonus():
    # Each animal is 1 tai; the complete set adds one more: 4 + 1 = 5
    hand = make_hand([0, 1, 2, 12, 13, 14, 19, 19, 19, 24, 25, 26, 27, 27],
                     flowers=[42, 43, 44, 45])
    score = score_win(hand, 27, False, 1, WIND_START, CFG)
    assert rules_of(score) == ["animal", "animal", "animal", "animal",
                               "complete_animals"]
    assert score.tai == 5


def test_ping_hu_tsumo_is_five_tai():
    # Concealed clean ping hu won by self-draw: ping hu 4 + 门清 1
    # = 5 tai, NOT 6 — tsumo adds payment reach, not tai.
    hand = make_hand(PING_HU_TILES)
    score = score_win(hand, 2, True, 0, WIND_START, CFG)
    assert rules_of(score) == ["fully_concealed", "ping_hu"]
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
    # Two-sided wait (3w/6w) so ping hu stands on a ron: 4 tai, value 8
    hand = make_hand(PING_HU_TILES[3:])
    hand.add_exposed_meld("chow", [0, 1, 2])
    score = score_win(hand, 5, False, 0, WIND_START, CFG)
    payments = compute_win_payments(score, winner=1, dealt_in_by=3, config=CFG)
    assert payments == [0, 24, 0, -24]


def test_ron_shooter_pays_own_share_only():
    cfg = ScoreConfig(shooter_pays_all=False)
    hand = make_hand(PING_HU_TILES[3:])
    hand.add_exposed_meld("chow", [0, 1, 2])
    score = score_win(hand, 5, False, 0, WIND_START, cfg)
    payments = compute_win_payments(score, winner=1, dealt_in_by=3, config=cfg)
    assert payments == [0, 8, 0, -8]


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
    # Simulate mid-game state: without this, P1 winning before their
    # "first draw" would legitimately be a Humanly Hand (人胡, limit)
    # under the PDF rules — which is exactly what this fixture is NOT
    # trying to test.
    game.draws_made = [1, 1, 1, 1]
    game.total_discards = 4
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


# ── PDF rules audit (Aug 2026): new hands and events ─────────────────

def test_neutral_wind_pair_allowed_in_ping_hu():
    # PDF: the ping hu pair may be a wind that is NEITHER seat nor
    # prevailing. Seat 1 (South), prevailing East: a West pair is fine.
    tiles = [0, 1, 2, 3, 4, 5, 9, 10, 11, 21, 22, 23, 29, 29]
    hand = make_hand(tiles)
    score = score_win(hand, 2, True, 1, WIND_START, CFG)
    assert "ping_hu" in rules_of(score)

    # The same pair for the WEST-seat player is their seat wind: no ping hu
    score = score_win(hand, 2, True, 2, WIND_START, CFG)
    assert "ping_hu" not in rules_of(score)


def test_mixed_terminals_stacks_with_all_triplets():
    # 混老头: terminal + honor pongs, terminal pair → 2 + 2 = 4 tai
    hand = make_hand([0, 0, 0, 8, 8, 8, 27, 27, 27, 31, 31, 31, 17, 17],
                     flowers=[35])
    score = score_win(hand, 17, False, 1, WIND_START, CFG)
    assert "mixed_terminals" in rules_of(score)
    assert "all_triplets" in rules_of(score)
    assert score.total_tai >= 4


def test_small_four_winds_is_not_limit():
    # PDF: 小四喜 = 2 for the hand plus whatever else stacks — not an
    # automatic limit. Seat 1 (South) holding S/W/N pongs + East pair,
    # prevailing East (flower 36 = West's, so no seat-flower tai).
    hand = make_hand([28, 28, 28, 29, 29, 29, 30, 30, 30, 0, 1, 2, 27, 27],
                     flowers=[36])
    score = score_win(hand, 2, False, 1, WIND_START, CFG)
    # 小四喜 2 + seat wind (South) 1 + half flush 2 = 5 — below the cap,
    # which a limit hand could never be
    assert rules_of(score) == ["half_flush", "seat_wind_triplet",
                               "small_four_winds"]
    assert score.total_tai == 5
    assert not score.is_limit


def test_hidden_treasure_needs_tsumo():
    # 四暗刻: four concealed triplets — limit on self-draw only
    tiles = [0, 0, 0, 2, 2, 2, 12, 12, 12, 19, 19, 19, 6, 6]
    hand = make_hand(tiles, flowers=[35])
    tsumo = score_win(hand, 6, True, 0, WIND_START, CFG)
    assert "hidden_treasure" in rules_of(tsumo)
    assert tsumo.is_limit
    ron = score_win(hand, 6, False, 0, WIND_START, CFG)
    assert "hidden_treasure" not in rules_of(ron)  # just a triplets hand


def test_thirteen_wonders_scores_limit_and_double_payout():
    # One of each terminal/honor, pair of East
    tiles = [0, 8, 9, 17, 18, 26, 27, 27, 28, 29, 30, 31, 32, 33]
    hand = make_hand(tiles)
    score = score_win(hand, 27, False, 0, WIND_START, CFG)
    assert rules_of(score) == ["thirteen_wonders"]
    assert score.is_limit
    assert score.payout_multiplier == 2
    payments = compute_win_payments(score, winner=0, dealt_in_by=2, config=CFG)
    assert payments[0] == 2 * score.value * 3  # everyone pays double


def test_thirteen_wonders_is_a_recognized_win():
    from mahjong.hand import is_winning_hand, get_winning_tiles
    counts = [0] * 34
    for t in [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33]:
        counts[t] += 1
    # 13-sided wait: every orphan tile completes it
    assert len(get_winning_tiles(counts)) == 13
    counts[27] += 1
    assert is_winning_hand(counts)


def test_flower_draw_scores_one_tai(monkeypatch):
    # 花上: winning on the replacement after a bonus tile draw
    hand = make_hand(PING_HU_TILES, flowers=[35])
    score = score_win(hand, 2, True, 0, WIND_START, CFG, is_flower_draw=True)
    assert "flower_draw" in rules_of(score)


def test_last_tile_cancelled_by_replacement_draw():
    hand = make_hand(PING_HU_TILES, flowers=[35])
    normal = score_win(hand, 2, True, 0, WIND_START, CFG, is_last_tile=True)
    assert "last_tile" in rules_of(normal)
    via_flower = score_win(hand, 2, True, 0, WIND_START, CFG,
                           is_last_tile=True, is_flower_draw=True)
    assert "last_tile" not in rules_of(via_flower)


def test_kong_on_kong_is_limit():
    hand = make_hand([0, 1, 2, 3, 4, 5, 9, 10, 11, 25, 25], flowers=[35])
    hand.add_exposed_meld("kong", [21] * 4)
    score = score_win(hand, 2, True, 0, WIND_START, CFG,
                      is_kong_draw=True, is_kong_on_kong=True)
    assert "kong_on_kong" in rules_of(score)
    assert score.is_limit


def test_heavenly_earthly_humanly_are_limit():
    hand = make_hand(PING_HU_TILES, flowers=[35])
    for flag in ("heavenly", "earthly", "humanly"):
        score = score_win(hand, 2, True, 0, WIND_START, CFG,
                          first_turn_flag=flag)
        assert f"{flag}_hand" in rules_of(score)
        assert score.is_limit


def test_eighteen_arhats_is_limit():
    # Four kongs + a pair
    hand = make_hand([6, 6])
    for t in (0, 9, 18, 27):
        for _ in range(4):
            hand.add_tile(t)
        hand.declare_kong("concealed", t)
    hand.add_tile(6)  # win tile completes the pair... already paired; adjust
    hand.remove_tile(6)
    score = score_win(hand, 6, False, 0, WIND_START, CFG)
    assert "eighteen_arhats" in rules_of(score)
    assert score.is_limit


def test_eight_flowers_instant_win():
    from mahjong.game import GameState
    from mahjong.agents import GreedyAgent
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    game.setup()
    game.hands[1].flowers = list(range(34, 41))  # 7 of 8 series flowers
    game._check_eight_flowers(1, 41)  # ...but the drawer completes them
    game.hands[1].flowers.append(41)
    game._check_eight_flowers(1, 41)
    assert game._flower_win == (1, None)
    assert game._settle_flower_win()
    assert game.result.win_type == "flowers"
    assert game.result.winner == 1
    assert game.result.winning_score.is_limit


def test_robbing_the_eighth():
    from mahjong.game import GameState
    from mahjong.agents import GreedyAgent
    game = GameState([GreedyAgent(f"G{i}") for i in range(4)], seed=0)
    game.setup()
    game.hands[2].flowers = list(range(34, 41))   # P2 holds 7
    game.hands[0].flowers = [41]                  # P0 draws the 8th
    game._check_eight_flowers(0, 41)
    assert game._flower_win == (2, 0)             # P2 robs, P0 is shooter
    assert 41 in game.hands[2].flowers and 41 not in game.hands[0].flowers
