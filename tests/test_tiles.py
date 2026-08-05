"""Tests for tile encoding, wall creation, and tile relationships."""

from collections import Counter

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, NUM_TOTAL_TILES,
    create_wall, tile_name, tile_short,
    suit_of, rank_of, is_numbered, is_honor, is_terminal, is_bonus,
    possible_chow_partners, neighbors, Suit,
)


def test_wall_composition():
    wall = create_wall()
    assert len(wall) == NUM_TOTAL_TILES == 148
    counts = Counter(wall)
    for tile_id in range(NUM_STANDARD_UNIQUE):
        assert counts[tile_id] == 4, f"standard tile {tile_id} should have 4 copies"
    for tile_id in range(34, 46):
        assert counts[tile_id] == 1, f"bonus tile {tile_id} should have 1 copy"


def test_wall_is_deterministic():
    # create_wall must NOT shuffle — GameState shuffles with its seeded RNG.
    assert create_wall() == create_wall()


def test_tile_names():
    assert tile_name(0) == "1 Wan"
    assert tile_name(17) == "9 Tong"
    assert tile_name(27) == "East Wind"
    assert tile_name(33) == "White Dragon"
    assert tile_short(4) == "5w"
    assert tile_short(31) == "Rd"


def test_tile_properties():
    assert suit_of(0) == Suit.WAN
    assert suit_of(13) == Suit.TONG
    assert suit_of(30) == Suit.WIND
    assert rank_of(4) == 5
    assert rank_of(27) is None
    assert is_numbered(26) and not is_numbered(27)
    assert is_honor(31) and not is_honor(8)
    assert is_terminal(0) and is_terminal(26) and not is_terminal(1)
    assert is_bonus(34) and is_bonus(45) and not is_bonus(33)


def test_chow_partners_middle_rank():
    # 5wan (id 4) can be rightmost, middle, or leftmost of a run
    assert possible_chow_partners(4) == [(2, 3), (3, 5), (5, 6)]


def test_chow_partners_edge_ranks():
    # 1wan can only be leftmost; 9wan can only be rightmost
    assert possible_chow_partners(0) == [(1, 2)]
    assert possible_chow_partners(8) == [(6, 7)]
    # honors form no chows
    assert possible_chow_partners(27) == []


def test_chow_partners_stay_in_suit():
    # 9wan (id 8) must not chain into 1tong (id 9)
    for p1, p2 in possible_chow_partners(8):
        assert suit_of(p1) == suit_of(p2) == Suit.WAN


def test_neighbors():
    assert neighbors(4) == [3, 5]
    assert neighbors(0) == [1]
    assert neighbors(8) == [7]
    assert neighbors(27) == []
