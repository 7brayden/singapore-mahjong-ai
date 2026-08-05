"""
Tile representation for Singapore Mahjong.

Encoding scheme:
- Each unique tile gets an integer ID (0-45)
- Standard tiles (34 unique x 4 copies = 136 tiles):
    - Wan (Characters):  0-8   → 1wan to 9wan
    - Tong (Dots):       9-17  → 1tong to 9tong
    - Suo (Bamboo):      18-26 → 1suo to 9suo
    - Winds:             27-30 → East, South, West, North
    - Dragons:           31-33 → Red, Green, White
- Bonus tiles (1 copy each = 12 tiles):
    - Flowers:           34-41 → Flower 1-8
    - Animals:           42-45 → Cat, Mouse, Rooster, Centipede

Total in wall: 136 standard + 12 bonus = 148 tiles

For the AI/game logic, we mostly work with tile IDs (0-33) since
the 4 copies of each standard tile are interchangeable.
Bonus tiles are set aside immediately when drawn.
"""

from enum import IntEnum
from typing import List, Dict, Tuple, Optional


# ── Suits and Categories ─────────────────────────────────────────────

class Suit(IntEnum):
    WAN = 0    # Characters (万)
    TONG = 1   # Dots/Circles (筒)
    SUO = 2    # Bamboo (索)
    WIND = 3   # Winds (风)
    DRAGON = 4 # Dragons (箭)
    FLOWER = 5 # Flowers (花)
    ANIMAL = 6 # Animals (动物)


# ── Tile ID ranges ────────────────────────────────────────────────────

# Standard tiles: 0-33 (34 unique, 4 copies each)
WAN_START = 0     # 0-8:   1wan through 9wan
TONG_START = 9    # 9-17:  1tong through 9tong
SUO_START = 18    # 18-26: 1suo through 9suo
WIND_START = 27   # 27-30: East, South, West, North
DRAGON_START = 31 # 31-33: Red, Green, White

NUM_STANDARD_UNIQUE = 34  # unique standard tile types
NUM_STANDARD_COPIES = 4   # copies of each standard tile
NUM_STANDARD_TOTAL = NUM_STANDARD_UNIQUE * NUM_STANDARD_COPIES  # 136

# Bonus tiles: 34-45 (1 copy each)
FLOWER_START = 34  # 34-41: Flower 1 through 8
ANIMAL_START = 42  # 42-45: Cat, Mouse, Rooster, Centipede

NUM_FLOWERS = 8
NUM_ANIMALS = 4
NUM_BONUS = NUM_FLOWERS + NUM_ANIMALS  # 12

NUM_TOTAL_UNIQUE = NUM_STANDARD_UNIQUE + NUM_BONUS  # 46
NUM_TOTAL_TILES = NUM_STANDARD_TOTAL + NUM_BONUS     # 148


# ── Tile names ────────────────────────────────────────────────────────

WIND_NAMES = ["East", "South", "West", "North"]
DRAGON_NAMES = ["Red", "Green", "White"]
ANIMAL_NAMES = ["Cat", "Mouse", "Rooster", "Centipede"]


def tile_name(tile_id: int) -> str:
    """Return a human-readable name for a tile ID."""
    if 0 <= tile_id <= 8:
        return f"{tile_id + 1} Wan"
    elif 9 <= tile_id <= 17:
        return f"{tile_id - 8} Tong"
    elif 18 <= tile_id <= 26:
        return f"{tile_id - 17} Suo"
    elif 27 <= tile_id <= 30:
        return f"{WIND_NAMES[tile_id - 27]} Wind"
    elif 31 <= tile_id <= 33:
        return f"{DRAGON_NAMES[tile_id - 31]} Dragon"
    elif 34 <= tile_id <= 41:
        return f"Flower {tile_id - 33}"
    elif 42 <= tile_id <= 45:
        return ANIMAL_NAMES[tile_id - 42]
    else:
        raise ValueError(f"Invalid tile ID: {tile_id}")


def tile_short(tile_id: int) -> str:
    """Return a compact string representation (useful for logging/debug).

    Examples: '1w', '9t', '5s', 'Ew', 'Rd', 'F1', 'Cat'
    """
    if 0 <= tile_id <= 8:
        return f"{tile_id + 1}w"
    elif 9 <= tile_id <= 17:
        return f"{tile_id - 8}t"
    elif 18 <= tile_id <= 26:
        return f"{tile_id - 17}s"
    elif 27 <= tile_id <= 30:
        return f"{'ESWN'[tile_id - 27]}w"
    elif 31 <= tile_id <= 33:
        return f"{'RGW'[tile_id - 31]}d"
    elif 34 <= tile_id <= 41:
        return f"F{tile_id - 33}"
    elif 42 <= tile_id <= 45:
        return ANIMAL_NAMES[tile_id - 42][:3]
    else:
        raise ValueError(f"Invalid tile ID: {tile_id}")


# ── Tile property helpers ─────────────────────────────────────────────

def suit_of(tile_id: int) -> Suit:
    """Return the suit/category of a tile."""
    if tile_id < 0 or tile_id > 45:
        raise ValueError(f"Invalid tile ID: {tile_id}")
    if tile_id <= 8:
        return Suit.WAN
    elif tile_id <= 17:
        return Suit.TONG
    elif tile_id <= 26:
        return Suit.SUO
    elif tile_id <= 30:
        return Suit.WIND
    elif tile_id <= 33:
        return Suit.DRAGON
    elif tile_id <= 41:
        return Suit.FLOWER
    else:
        return Suit.ANIMAL


def rank_of(tile_id: int) -> Optional[int]:
    """Return the rank (1-9) for numbered suit tiles, None for honors/bonus."""
    if tile_id <= 26:
        return (tile_id % 9) + 1
    return None


def is_numbered(tile_id: int) -> bool:
    """True if tile is a numbered suit tile (wan/tong/suo)."""
    return 0 <= tile_id <= 26


def is_honor(tile_id: int) -> bool:
    """True if tile is a wind or dragon."""
    return 27 <= tile_id <= 33


def is_terminal(tile_id: int) -> bool:
    """True if tile is a 1 or 9 of any numbered suit."""
    return is_numbered(tile_id) and rank_of(tile_id) in (1, 9)


def is_simple(tile_id: int) -> bool:
    """True if tile is a 2-8 of any numbered suit."""
    return is_numbered(tile_id) and rank_of(tile_id) not in (1, 9)


def is_bonus(tile_id: int) -> bool:
    """True if tile is a flower or animal (set aside when drawn)."""
    return tile_id >= 34


def is_wind(tile_id: int) -> bool:
    return 27 <= tile_id <= 30


def is_dragon(tile_id: int) -> bool:
    return 31 <= tile_id <= 33


# ── Tile relationship helpers (for melds and shanten) ─────────────────

def same_suit_tiles(tile_id: int) -> List[int]:
    """Return all tile IDs in the same suit (for numbered suits only)."""
    if not is_numbered(tile_id):
        return [tile_id]  # honors don't form sequences
    suit_start = (tile_id // 9) * 9
    return list(range(suit_start, suit_start + 9))


def possible_chow_partners(tile_id: int) -> List[Tuple[int, int]]:
    """Return pairs of tile IDs that form a chow (sequence) with this tile.

    E.g., for 5wan (ID 4): returns pairs for [3w,4w], [4w,6w], [6w,7w]

    Only numbered tiles can form chows.
    """
    # check if tile can even form a chow, return empty list if non-numbered chow
    if not is_numbered(tile_id):
        return []

    rank = rank_of(tile_id)
    # find which suit does the tile belong to: 0 -> wan, 9 -> tong, 18 -> suo
    suit_start = (tile_id // 9) * 9
    # store valid partner pairs
    partners = []

    # tile is the rightmost: [rank-2, rank-1, rank]
    if rank >= 3:
        partners.append((suit_start + rank - 3, suit_start + rank - 2))
    # tile is the middle: [rank-1, rank, rank+1]
    if rank >= 2 and rank <= 8:
        partners.append((suit_start + rank - 2, suit_start + rank))
    # tile is the leftmost: [rank, rank+1, rank+2]
    if rank <= 7:
        partners.append((suit_start + rank, suit_start + rank + 1))

    return partners


def neighbors(tile_id: int) -> List[int]:
    """Return tile IDs that are adjacent in rank (useful for shanten).

    E.g., 5wan → [4wan, 6wan], 1wan → [2wan], East Wind → []
    """
    if not is_numbered(tile_id):
        return []

    rank = rank_of(tile_id)
    suit_start = (tile_id // 9) * 9
    result = []
    if rank > 1:
        result.append(suit_start + rank - 2)
    if rank < 9:
        result.append(suit_start + rank)
    return result


# ── Wall creation ─────────────────────────────────────────────────────

def create_wall() -> List[int]:
    """Create a full, unshuffled wall of 148 tiles.

    Returns a list of tile IDs where each standard tile (0-33) appears
    4 times and each bonus tile (34-45) appears once.

    The caller is responsible for shuffling. GameState shuffles with its
    own seeded RNG so that a given seed always produces the same game.
    """
    wall = []
    # 4 copies of each standard tile
    for tile_id in range(NUM_STANDARD_UNIQUE):
        wall.extend([tile_id] * NUM_STANDARD_COPIES)
    # 1 copy of each bonus tile
    for tile_id in range(FLOWER_START, FLOWER_START + NUM_FLOWERS):
        wall.append(tile_id)
    for tile_id in range(ANIMAL_START, ANIMAL_START + NUM_ANIMALS):
        wall.append(tile_id)

    assert len(wall) == NUM_TOTAL_TILES  # 148
    return wall


# ── Display helpers ───────────────────────────────────────────────────

def hand_to_str(tile_ids: List[int], short: bool = False) -> str:
    """Pretty-print a list of tile IDs as a readable hand.

    Sorts by suit then rank for readability.
    """
    fn = tile_short if short else tile_name
    sorted_tiles = sorted(tile_ids, key=lambda t: (suit_of(t), t))
    return " | ".join(fn(t) for t in sorted_tiles)


def tile_count_str(counts: Dict[int, int]) -> str:
    """Pretty-print a tile count dictionary (tile_id → count)."""
    parts = []
    for tid in sorted(counts.keys()):
        if counts[tid] > 0:
            parts.append(f"{tile_short(tid)}x{counts[tid]}")
    return ", ".join(parts)


# ── Quick sanity check ────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Singapore Mahjong Tile Set ===\n")

    print("--- Numbered Suits ---")
    for suit_name, start in [("Wan", WAN_START), ("Tong", TONG_START), ("Suo", SUO_START)]:
        tiles = [f"{tile_short(i):>3}" for i in range(start, start + 9)]
        print(f"  {suit_name}: {' '.join(tiles)}")

    print("\n--- Honors ---")
    winds = [tile_short(i) for i in range(WIND_START, WIND_START + 4)]
    dragons = [tile_short(i) for i in range(DRAGON_START, DRAGON_START + 3)]
    print(f"  Winds:   {' '.join(winds)}")
    print(f"  Dragons: {' '.join(dragons)}")

    print("\n--- Bonus Tiles ---")
    flowers = [tile_short(i) for i in range(FLOWER_START, FLOWER_START + NUM_FLOWERS)]
    animals = [tile_short(i) for i in range(ANIMAL_START, ANIMAL_START + NUM_ANIMALS)]
    print(f"  Flowers: {' '.join(flowers)}")
    print(f"  Animals: {' '.join(animals)}")

    print(f"\n--- Wall ---")
    import random
    wall = create_wall()
    random.shuffle(wall)  # demo only; GameState shuffles with a seeded RNG
    print(f"  Total tiles: {len(wall)}")
    print(f"  First 20: {hand_to_str(wall[:20], short=True)}")

    print("\n--- Sample Hand (13 tiles) ---")
    sample = sorted(wall[:13])
    print(f"  {hand_to_str(sample)}")
    print(f"  {hand_to_str(sample, short=True)}")

    print("\n--- Chow Partners for 5 Wan (ID 4) ---")
    for a, b in possible_chow_partners(4):
        print(f"  {tile_short(a)} + {tile_short(4)} + {tile_short(b)}")

    print("\n--- Neighbors of 1 Tong (ID 9) ---")
    print(f"  {[tile_short(n) for n in neighbors(9)]}")