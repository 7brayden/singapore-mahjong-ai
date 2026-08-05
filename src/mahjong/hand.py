"""
Hand representation and shanten calculation for Singapore Mahjong.

A standard winning hand: 4 melds + 1 pair (from 14 tiles).
Melds can be:
  - Chow (chi):  3 consecutive tiles in the same numbered suit
  - Pong:        3 identical tiles
  - Kong:        4 identical tiles (counts as 1 meld, player draws replacement)

Shanten number = minimum tiles needed to swap to reach tenpai (ready).
  - Shanten -1 = complete (winning) hand
  - Shanten  0 = tenpai (one tile away from winning)
  - Shanten  1 = one away from tenpai
  - etc.

We use a count-based representation: an array of 34 ints where
counts[tile_id] = number of copies of that tile in hand.
This makes meld extraction and shanten calculation efficient.
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import Counter

from mahjong.tiles import (
    NUM_STANDARD_UNIQUE, tile_short, tile_name, hand_to_str,
    suit_of, rank_of, is_numbered, is_honor, is_bonus,
    Suit, WAN_START, TONG_START, SUO_START, WIND_START, DRAGON_START
)


class Hand:
    """Represents a player's hand in Singapore Mahjong.

    Core state:
        tiles:     list of tile IDs currently in hand (13 normally, 14 after draw)
        counts:    array of 34 ints, counts[i] = how many of tile i in hand
        exposed:   list of exposed melds [(meld_type, tile_ids), ...]
        flowers:   list of bonus tile IDs collected
        discards:  list of tile IDs discarded (in order)

    The hand normally holds 13 tiles. After drawing, it holds 14 tiles
    and the player must discard one (or declare a win/kong).
    """

    def __init__(self):
        self.tiles: List[int] = []
        self.counts: List[int] = [0] * NUM_STANDARD_UNIQUE  # 34 slots
        self.exposed: List[Tuple[str, List[int]]] = []  # ("chow"/"pong"/"kong", [tile_ids])
        self.flowers: List[int] = []
        self.discards: List[int] = []

    def add_tile(self, tile_id: int) -> bool:
        """Add a tile to hand. Returns True if it's a bonus tile (set aside)."""
        if is_bonus(tile_id):
            self.flowers.append(tile_id)
            return True
        self.tiles.append(tile_id)
        self.counts[tile_id] += 1
        return False

    def remove_tile(self, tile_id: int):
        """Remove one copy of a tile from hand."""
        if tile_id not in self.tiles:
            raise ValueError(f"Tile {tile_short(tile_id)} not in hand")
        self.tiles.remove(tile_id)
        self.counts[tile_id] -= 1

    def discard(self, tile_id: int):
        """Discard a tile from hand."""
        self.remove_tile(tile_id)
        self.discards.append(tile_id)

    def add_exposed_meld(self, meld_type: str, tile_ids: List[int]):
        """Record an exposed meld (from pong/chow/kong calls)."""
        self.exposed.append((meld_type, tile_ids))

    @property
    def num_exposed_melds(self) -> int:
        return len(self.exposed)

    @property
    def concealed_count(self) -> int:
        """Number of concealed (non-exposed) tiles in hand."""
        return len(self.tiles)

    def copy_counts(self) -> List[int]:
        """Return a copy of the counts array (for shanten calculation)."""
        return self.counts[:]

    def __repr__(self):
        return f"Hand({hand_to_str(self.tiles, short=True)})"


# ══════════════════════════════════════════════════════════════════════
# SHANTEN CALCULATION
# ══════════════════════════════════════════════════════════════════════
#
# Algorithm: recursive decomposition.
#
# A winning hand = 4 melds + 1 pair (standard form).
# With N exposed melds, we need (4 - N) melds + 1 pair from concealed tiles.
#
# We try each possible pair, then greedily/recursively extract melds
# from the remaining tiles. Shanten = (melds_needed - melds_found)
# + (partial_melds_needed - partial_melds_found) ... but the standard
# formula is:
#
#   shanten = 2 * melds_needed - melds_found - partial_found
#
# where:
#   melds_needed = 4 - num_exposed_melds
#   melds_found = complete melds in concealed tiles
#   partial_found = partial melds (pairs, two-sided waits, etc.)
#     but capped so melds_found + partial_found <= melds_needed
#
# We iterate over all possible pair choices and meld extractions
# to find the minimum shanten.
# ══════════════════════════════════════════════════════════════════════


def calculate_shanten(counts: List[int], num_exposed_melds: int = 0) -> int:
    """Calculate the shanten number for a hand.

    Args:
        counts: array of 34 ints (tile counts for concealed tiles)
        num_exposed_melds: number of already-exposed melds

    Returns:
        Shanten number (-1 = complete, 0 = tenpai, 1+ = further away)
    """
    num_melds_needed = 4 - num_exposed_melds
    # assume hand is very far
    best_shanten = 8  # worst case: 4 melds + 1 pair = 8 shanten

    # Try each tile as the pair
    for pair_tile in range(NUM_STANDARD_UNIQUE):
        if counts[pair_tile] >= 2:
            counts[pair_tile] -= 2
            melds, partials = _extract_melds(counts, num_melds_needed)
            # Formula: 2 * melds_needed - 2*melds - partials - 1 (for pair)
            shanten_with_pair = (2 * num_melds_needed) - (2 * melds) - partials - 1
            best_shanten = min(best_shanten, shanten_with_pair)
            counts[pair_tile] += 2

    # Also try without a pair (maybe no pair yet)
    melds, partials = _extract_melds(counts, num_melds_needed)
    shanten_no_pair = (2 * num_melds_needed) - (2 * melds) - partials
    best_shanten = min(best_shanten, shanten_no_pair)

    return best_shanten


def _extract_melds(counts: List[int], melds_needed: int) -> Tuple[int, int]:
    """Extract maximum melds and partial melds from counts.

    Uses recursive backtracking: try extracting each possible meld type,
    then count remaining partial melds.

    Returns: (complete_melds, partial_melds) where the total
             melds + partials <= melds_needed.
    """
    best = (0, 0)  # (melds, partials)

    # Try to find at least one complete meld
    found_meld = False
    for tile_id in range(NUM_STANDARD_UNIQUE):
        # Try pong (triplet)
        if counts[tile_id] >= 3:
            counts[tile_id] -= 3
            # recursively call for the leftover hand for the number of melds and partials
            m, p = _extract_melds(counts, melds_needed)
            m += 1
            # reduce partials if necessary
            if m + p > melds_needed:
                p = melds_needed - m
            # updating
            if (m, p) > best:
                best = (m, p)
            counts[tile_id] += 3
            found_meld = True

        # Try chow (sequence) - only for numbered tiles
        if is_numbered(tile_id) and rank_of(tile_id) <= 7:
            # Check for 3 consecutive tiles in same suit
            t1, t2, t3 = tile_id, tile_id + 1, tile_id + 2
            # Make sure all 3 are in the same suit
            if suit_of(t1) == suit_of(t2) == suit_of(t3):
                if counts[t1] >= 1 and counts[t2] >= 1 and counts[t3] >= 1:
                    counts[t1] -= 1
                    counts[t2] -= 1
                    counts[t3] -= 1
                    m, p = _extract_melds(counts, melds_needed)
                    m += 1
                    if m + p > melds_needed:
                        p = melds_needed - m
                    if (m, p) > best:
                        best = (m, p)
                    counts[t1] += 1
                    counts[t2] += 1
                    counts[t3] += 1
                    found_meld = True

    # If no more complete melds, count partial melds
    if not found_meld:
        partials = _count_partials(counts)
        melds = 0
        if partials > melds_needed:
            partials = melds_needed
        best = max(best, (melds, partials))

    return best


def _count_partials(counts: List[int]) -> int:
    """Count partial melds (pairs and connected tiles) in remaining counts.

    A partial meld is any 2 tiles that could become a meld with one more tile:
      - A pair (two identical tiles) -> needs a third for pong
      - Two consecutive tiles (e.g., 3w+4w) -> needs adjacent for chow
      - Two tiles with one gap (e.g., 3w+5w) -> needs middle for chow

    Each tile can only contribute to one partial.
    """
    temp = counts[:]
    partials = 0

    for tile_id in range(NUM_STANDARD_UNIQUE):
        # Pair partial
        if temp[tile_id] >= 2:
            temp[tile_id] -= 2
            partials += 1

    for tile_id in range(NUM_STANDARD_UNIQUE):
        if not is_numbered(tile_id):
            continue
        rank = rank_of(tile_id)

        # Adjacent partial (e.g., 3w+4w)
        if rank <= 8:
            next_tile = tile_id + 1
            if suit_of(tile_id) == suit_of(next_tile):
                if temp[tile_id] >= 1 and temp[next_tile] >= 1:
                    temp[tile_id] -= 1
                    temp[next_tile] -= 1
                    partials += 1

        # Gap partial (e.g., 3w+5w)
        if rank <= 7:
            gap_tile = tile_id + 2
            if suit_of(tile_id) == suit_of(gap_tile):
                if temp[tile_id] >= 1 and temp[gap_tile] >= 1:
                    temp[tile_id] -= 1
                    temp[gap_tile] -= 1
                    partials += 1

    return partials


# ══════════════════════════════════════════════════════════════════════
# TILE ACCEPTANCE (ukeire) - how many tiles improve shanten
# ══════════════════════════════════════════════════════════════════════

def tile_acceptance(counts: List[int], num_exposed_melds: int = 0,
                    visible_counts: Optional[List[int]] = None) -> Tuple[List[int], int]:
    """Calculate which tiles would improve (lower) the shanten number.

    Args:
        counts: current concealed tile counts (34 ints)
        num_exposed_melds: number of exposed melds
        visible_counts: counts of all visible tiles (for calculating remaining copies)
                       If None, assumes 4 copies of each tile available.

    Returns:
        (improving_tiles, total_acceptance)
        improving_tiles: list of tile IDs that reduce shanten
        total_acceptance: sum of remaining copies of those tiles
    """
    current_shanten = calculate_shanten(counts, num_exposed_melds)
    improving_tiles = []
    total_acceptance = 0

    for tile_id in range(NUM_STANDARD_UNIQUE):
        # How many copies could we still draw?
        if visible_counts is not None:
            remaining = 4 - visible_counts[tile_id]
        
        # assume only my hand is known, everything else is potentially out there to draw from the wall
        else:
            remaining = 4 - counts[tile_id]

        if remaining <= 0:
            continue

        # Temporarily add this tile and check shanten
        counts[tile_id] += 1
        new_shanten = calculate_shanten(counts, num_exposed_melds)
        counts[tile_id] -= 1

        if new_shanten < current_shanten:
            improving_tiles.append(tile_id)
            total_acceptance += remaining

    return improving_tiles, total_acceptance


# ══════════════════════════════════════════════════════════════════════
# CLAIM EVALUATION
# ══════════════════════════════════════════════════════════════════════

def best_chow_option(counts: List[int], num_exposed_melds: int,
                     options) -> Tuple[Optional[Tuple[int, int]], int]:
    """Find the chow combination that lowers shanten the most.

    Args:
        counts: concealed tile counts before claiming
        num_exposed_melds: exposed melds before claiming
        options: iterable of (partner1, partner2) tile-id pairs

    Returns:
        (best_pair, best_shanten): the pair giving the lowest post-claim
        shanten, or (None, current_shanten) if no option improves it.
    """
    current = calculate_shanten(counts, num_exposed_melds)
    best_pair = None
    best_shanten = current
    for p1, p2 in options:
        test = counts[:]
        test[p1] -= 1
        test[p2] -= 1
        s = calculate_shanten(test, num_exposed_melds + 1)
        if s < best_shanten:
            best_shanten = s
            best_pair = (p1, p2)
    return best_pair, best_shanten


# ══════════════════════════════════════════════════════════════════════
# WIN CHECKING
# ══════════════════════════════════════════════════════════════════════

def is_winning_hand(counts: List[int], num_exposed_melds: int = 0) -> bool:
    """Check if the concealed tiles + exposed melds form a complete hand.

    A complete hand has shanten = -1, i.e., 4 melds + 1 pair total.
    """
    return calculate_shanten(counts, num_exposed_melds) == -1


def get_winning_tiles(counts: List[int], num_exposed_melds: int = 0) -> List[int]:
    """If tenpai (shanten=0), return which tiles complete the hand."""
    if calculate_shanten(counts, num_exposed_melds) != 0:
        return []

    winning = []
    for tile_id in range(NUM_STANDARD_UNIQUE):
        if counts[tile_id] >= 4:
            continue
        counts[tile_id] += 1
        if is_winning_hand(counts, num_exposed_melds):
            winning.append(tile_id)
        counts[tile_id] -= 1
    return winning


# ══════════════════════════════════════════════════════════════════════
# WINNING HAND DECOMPOSITION (for scoring)
# ══════════════════════════════════════════════════════════════════════

def decompose_winning_hand(counts: List[int],
                           melds_needed: int) -> List[Tuple[int, List[Tuple[str, int]]]]:
    """Decompose a complete concealed hand into pair + melds.

    Args:
        counts: concealed tile counts INCLUDING the winning tile
        melds_needed: 4 minus the number of exposed melds

    Returns:
        All distinct decompositions as (pair_tile, melds) where melds is
        a list of ("chow", start_tile) or ("pong", tile) entries.
        Empty list if the tiles do not form a complete hand.

    Scoring evaluates every decomposition and keeps the best, since some
    hands decompose multiple ways (e.g. 111222333 as pongs or chows).
    """
    results = []
    for pair_tile in range(NUM_STANDARD_UNIQUE):
        if counts[pair_tile] < 2:
            continue
        counts[pair_tile] -= 2
        for melds in _decompose_melds(counts, melds_needed):
            results.append((pair_tile, melds))
        counts[pair_tile] += 2
    return results


def _decompose_melds(counts: List[int], melds_needed: int) -> List[List[Tuple[str, int]]]:
    """Return every way to split counts into exactly melds_needed melds."""
    if melds_needed == 0:
        return [[]] if sum(counts) == 0 else []

    # Anchor on the lowest remaining tile: any full decomposition must
    # consume it as a pong or as the start of a chow (lower tiles are gone).
    first = next((t for t in range(NUM_STANDARD_UNIQUE) if counts[t] > 0), None)
    if first is None:
        return []

    results = []

    if counts[first] >= 3:
        counts[first] -= 3
        for rest in _decompose_melds(counts, melds_needed - 1):
            results.append([("pong", first)] + rest)
        counts[first] += 3

    if is_numbered(first) and rank_of(first) <= 7:
        t2, t3 = first + 1, first + 2
        if suit_of(t2) == suit_of(first) == suit_of(t3):
            if counts[t2] >= 1 and counts[t3] >= 1:
                counts[first] -= 1
                counts[t2] -= 1
                counts[t3] -= 1
                for rest in _decompose_melds(counts, melds_needed - 1):
                    results.append([("chow", first)] + rest)
                counts[first] += 1
                counts[t2] += 1
                counts[t3] += 1

    return results


# ══════════════════════════════════════════════════════════════════════
# DISCARD EVALUATION (for greedy agent)
# ══════════════════════════════════════════════════════════════════════

def evaluate_discards(hand: Hand,
                      visible_counts: Optional[List[int]] = None
                      ) -> List[Dict]:
    """Evaluate each possible discard from the hand.

    For each tile we could discard, compute:
      - resulting shanten
      - tile acceptance count (how many tiles improve shanten after discard)

    Returns a list of dicts sorted by (shanten ASC, acceptance DESC):
        [{"tile_id": int, "shanten": int, "acceptance": int, "improving_tiles": [int]}, ...]
    """
    results = []
    seen = set()  # avoid evaluating duplicate tiles

    for tile_id in hand.tiles:
        if tile_id in seen:
            continue
        if is_bonus(tile_id):
            continue
        seen.add(tile_id)

        # Temporarily remove this tile
        counts = hand.copy_counts()
        counts[tile_id] -= 1

        shanten = calculate_shanten(counts, hand.num_exposed_melds)
        improving, acceptance = tile_acceptance(counts, hand.num_exposed_melds, visible_counts)

        results.append({
            "tile_id": tile_id,
            "shanten": shanten,
            "acceptance": acceptance,
            "improving_tiles": improving,
        })

    # Sort: lower shanten first, then higher acceptance
    results.sort(key=lambda r: (r["shanten"], -r["acceptance"]))
    return results


# ══════════════════════════════════════════════════════════════════════
# SANITY TESTS
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== Hand & Shanten Tests ===\n")

    # Test 1: Complete hand (shanten = -1)
    # 1-2-3wan, 4-5-6wan, 7-8-9wan, 1-1-1tong, 2-2tong (pair)
    print("--- Test 1: Complete hand ---")
    counts = [0] * 34
    for i in range(9):  # 1-9 wan
        counts[i] = 1
    counts[9] = 3   # 1tong x3 (pong)
    counts[10] = 2  # 2tong x2 (pair)
    shanten = calculate_shanten(counts, 0)
    print(f"  Shanten: {shanten} (expected: -1)")
    print(f"  Is winning: {is_winning_hand(counts, 0)}")

    # Test 2: Tenpai hand (shanten = 0)
    # 1-2-3wan, 4-5-6wan, 7-8-9wan, 1-1tong (pair), 2-3tong (waiting for 1t or 4t)
    print("\n--- Test 2: Tenpai hand ---")
    counts2 = [0] * 34
    for i in range(9):
        counts2[i] = 1
    counts2[9] = 2   # 1tong pair
    counts2[10] = 1  # 2tong
    counts2[11] = 1  # 3tong
    shanten2 = calculate_shanten(counts2, 0)
    print(f"  Shanten: {shanten2} (expected: 0)")
    winning = get_winning_tiles(counts2, 0)
    print(f"  Winning tiles: {[tile_short(t) for t in winning]}")

    # Test 3: Hand evaluation
    print("\n--- Test 3: Discard evaluation ---")
    h = Hand()
    # 1-9wan + 1tong x3 + 2tong + Red Dragon = 14 tiles
    test_tiles = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 10, 31]
    for t in test_tiles:
        h.add_tile(t)
    print(f"  Hand: {hand_to_str(h.tiles, short=True)}")
    print(f"  Concealed: {h.concealed_count} tiles")

    evals = evaluate_discards(h)
    print(f"\n  All discards (sorted best to worst):")
    for e in evals:
        print(f"    Discard {tile_short(e['tile_id']):>3}: "
              f"shanten={e['shanten']}, acceptance={e['acceptance']}")

    # Test 4: Shanten for scattered hand
    print("\n--- Test 4: Scattered hand ---")
    counts4 = [0] * 34
    scattered = [0, 4, 8, 9, 13, 17, 18, 22, 26, 27, 28, 29, 30]
    for t in scattered:
        counts4[t] = 1
    shanten4 = calculate_shanten(counts4, 0)
    print(f"  Tiles: {hand_to_str(scattered, short=True)}")
    print(f"  Shanten: {shanten4} (expected: high, ~5-6)")

    # Test 5: Hand with exposed meld
    print("\n--- Test 5: With exposed meld ---")
    counts5 = [0] * 34
    # 3 melds concealed + 1 exposed = need 0 more melds from concealed... wait
    # Actually: 1 exposed meld means we need 3 melds + 1 pair from concealed (10 tiles)
    # Let's do: 1-2-3wan, 4-5-6wan, 7-8-9wan, 1-1tong = 3 melds + pair = complete
    for i in range(9):
        counts5[i] = 1
    counts5[9] = 2  # 1tong pair
    shanten5 = calculate_shanten(counts5, 1)  # 1 exposed meld
    print(f"  Shanten with 1 exposed meld: {shanten5} (expected: -1)")

    print("\n=== All tests done ===")