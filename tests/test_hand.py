"""Tests for hand representation, shanten calculation, and discard evaluation."""

from mahjong.hand import (
    Hand, calculate_shanten, is_winning_hand, get_winning_tiles,
    evaluate_discards, tile_acceptance, best_chow_option,
)


def counts_of(tiles):
    counts = [0] * 34
    for t in tiles:
        counts[t] += 1
    return counts


def test_complete_hand():
    # 1w-9w (three runs) + 1t pong + 2t pair
    counts = counts_of(list(range(9)) + [9, 9, 9, 10, 10])
    assert calculate_shanten(counts, 0) == -1
    assert is_winning_hand(counts, 0)


def test_tenpai_two_sided_wait():
    # 1w-9w + 1t pair + 2t3t: waits on 1t (pair becomes pong) or 4t
    counts = counts_of(list(range(9)) + [9, 9, 10, 11])
    assert calculate_shanten(counts, 0) == 0
    assert set(get_winning_tiles(counts, 0)) == {9, 12}


def test_tenpai_pair_wait():
    # Four complete melds + a lone 1t: tenpai waiting to pair it
    counts = counts_of([0, 0, 0, 1, 2, 3, 4, 5, 6, 8, 8, 8, 9])
    assert calculate_shanten(counts, 0) == 0
    assert get_winning_tiles(counts, 0) == [9]


def test_no_pair_hand_is_not_tenpai():
    # 1w-9w + 1t 2t 4t 5t: three melds + two partials but NO pair anywhere.
    # No single tile completes 4 melds + a pair, so shanten must be 1.
    counts = counts_of(list(range(9)) + [9, 10, 12, 13])
    assert calculate_shanten(counts, 0) == 1


def test_scattered_hand_high_shanten():
    counts = counts_of([0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 28, 29, 30])
    assert calculate_shanten(counts, 0) >= 4


def test_exposed_melds_reduce_requirement():
    # 3 concealed melds + pair with 1 exposed meld = complete hand
    counts = counts_of(list(range(9)) + [9, 9])
    assert calculate_shanten(counts, 1) == -1


def test_evaluate_discards_keeps_tenpai():
    # 1w-9w + 1t pong + 2t + Rd (14 tiles): discarding Rd or 2t keeps tenpai
    hand = Hand()
    for t in list(range(9)) + [9, 9, 9, 10, 31]:
        hand.add_tile(t)
    evals = evaluate_discards(hand)
    assert evals[0]["shanten"] == 0
    top_two = {evals[0]["tile_id"], evals[1]["tile_id"]}
    assert top_two == {10, 31}


def test_tile_acceptance_respects_visible_counts():
    # Tenpai waiting on 2t (id 10) after discarding Rd
    counts = counts_of(list(range(9)) + [9, 9, 9, 10])
    improving, acceptance = tile_acceptance(counts, 0)
    assert 10 in improving  # wait: pairing the 2t
    assert acceptance > 0

    # If 3 copies of every winning tile are already visible, acceptance drops
    visible = counts[:]
    for t in improving:
        visible[t] = 3
    _, reduced = tile_acceptance(counts, 0, visible_counts=visible)
    assert reduced < acceptance


def test_best_chow_option_prefers_improving_pair():
    # Hand: 2w3w4w run + 6w7w partial + 9w + 1t pong + 3s4s5s run + 8s.
    # Claiming a discarded 5w with (6w,7w) reaches tenpai; claiming with
    # (3w,4w) would break the completed 2w3w4w run and NOT improve shanten.
    counts = counts_of([1, 2, 3, 5, 6, 8, 9, 9, 9, 20, 21, 22, 25])
    assert calculate_shanten(counts, 0) == 1

    options = [(2, 3), (3, 5), (5, 6)]
    best_pair, best_shanten = best_chow_option(counts, 0, options)
    assert best_pair == (5, 6)
    assert best_shanten == 0


def test_best_chow_option_returns_none_when_no_improvement():
    # Completed hand region — any chow claim would only break it up
    counts = counts_of([1, 2, 3, 9, 9, 9, 14, 14, 14, 20, 21, 22, 25])
    current = calculate_shanten(counts, 0)
    best_pair, best_shanten = best_chow_option(counts, 0, [(2, 3)])
    assert best_pair is None
    assert best_shanten == current
